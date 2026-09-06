"""
Build a TensorRT engine from ONNX. Must run ON the target device.

TensorRT auto-tunes by timing candidate kernels on the actual GPU, so the engine
embeds choices specific to that architecture and TensorRT version. An engine built
elsewhere will not deserialise.

    python3 models/build_trt_engine.py --precision fp16
    python3 models/build_trt_engine.py --precision fp32 --workspace 512
"""

import argparse
import os
import sys
import time

import tensorrt as trt
import yaml


def build(onnx_path, engine_path, precision, workspace_mb, verbose=False):
    logger = trt.Logger(trt.Logger.VERBOSE if verbose else trt.Logger.WARNING)
    builder = trt.Builder(logger)

    # EXPLICIT_BATCH is required for ONNX; the implicit-batch path is legacy.
    flag = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    network = builder.create_network(flag)
    parser = trt.OnnxParser(network, logger)

    with open(onnx_path, "rb") as f:
        if not parser.parse(f.read()):
            for i in range(parser.num_errors):
                print("parser error:", parser.get_error(i))
            raise RuntimeError("failed to parse %s" % onnx_path)

    config = builder.create_builder_config()
    config.max_workspace_size = workspace_mb * 1024 * 1024

    if precision == "fp16":
        if not builder.platform_has_fast_fp16:
            raise RuntimeError("platform reports no fast FP16 support")
        config.set_flag(trt.BuilderFlag.FP16)

    # Reported for the record. INT8 needs compute capability 6.1 for the DP4A
    # instruction; this board is SM 5.3, so it is False here by design.
    print("platform_has_fast_fp16 :", builder.platform_has_fast_fp16)
    print("platform_has_fast_int8 :", builder.platform_has_fast_int8)

    inp = network.get_input(0)
    out = network.get_output(0)
    print("input  : %s %s" % (inp.name, inp.shape))
    print("output : %s %s" % (out.name, out.shape))
    print("layers : %d" % network.num_layers)
    print("")
    print("building %s engine, workspace %d MB..." % (precision, workspace_mb))
    print("this times candidate kernels on the GPU and takes several minutes")

    t0 = time.perf_counter()
    engine = builder.build_serialized_network(network, config)
    elapsed = time.perf_counter() - t0

    if engine is None:
        raise RuntimeError("engine build failed")

    with open(engine_path, "wb") as f:
        f.write(engine)

    size_mb = os.path.getsize(engine_path) / (1024.0 * 1024.0)
    print("")
    print("built in   : %.1f s" % elapsed)
    print("engine     : %s  (%.1f MB)" % (engine_path, size_mb))
    return elapsed


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/params.yaml")
    p.add_argument("--precision", default="fp16", choices=["fp32", "fp16"])
    p.add_argument("--workspace", type=int, default=None, help="MB")
    p.add_argument("--onnx", default=None)
    p.add_argument("--out", default=None)
    p.add_argument("--verbose", action="store_true")
    args = p.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    onnx_path = args.onnx or cfg["model"]["onnx"]
    workspace = args.workspace or cfg["engine"]["workspace_mb"]
    out_path = args.out or "models/%s_%s.engine" % (cfg["model"]["name"], args.precision)

    if not os.path.isfile(onnx_path):
        raise SystemExit("%s not found" % onnx_path)

    print("TensorRT   : %s" % trt.__version__)
    print("onnx       : %s" % onnx_path)
    print("")

    build(onnx_path, out_path, args.precision, workspace, args.verbose)


if __name__ == "__main__":
    sys.exit(main())
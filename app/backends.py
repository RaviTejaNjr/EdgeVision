"""
Inference backends behind a common interface.

benchmark.py calls load() and infer() without knowing which runtime it holds, so
differences between results are differences between runtimes rather than between
measurement methods.

Imports are deferred into load(): the Jetson has tensorrt but not onnxruntime,
the laptop the reverse.
"""

import numpy as np


class Backend(object):

    name = "base"

    def load(self):
        raise NotImplementedError

    def infer(self, tensor):
        raise NotImplementedError

    def close(self):
        pass


class TorchBackend(Backend):

    name = "pytorch"

    def __init__(self, weights, device="cuda", precision="fp32"):
        self.weights = weights
        self.device = device
        self.precision = precision
        self.model = None
        self.torch = None

    def load(self):
        import torch
        from ultralytics import YOLO

        self.torch = torch

        if self.device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but torch.cuda.is_available() is False")

        # .model is the raw nn.Module; YOLO.predict() would add Ultralytics'
        # own pre/postprocessing, which is timed separately here.
        yolo = YOLO(self.weights)
        self.model = yolo.model.to(self.device).eval()

        if self.precision == "fp16":
            self.model = self.model.half()

    def infer(self, tensor):
        t = self.torch.from_numpy(tensor).to(self.device)
        if self.precision == "fp16":
            t = t.half()

        with self.torch.no_grad():
            out = self.model(t)

        if isinstance(out, (list, tuple)):
            out = out[0]

        # CUDA work is queued asynchronously; without this the timer would stop
        # when the work was submitted rather than finished.
        if self.device == "cuda":
            self.torch.cuda.synchronize()

        return out.float().cpu().numpy()

class TorchScriptBackend(Backend):
    """
    Frozen TorchScript graph. Needs only torch -- no Ultralytics -- which is why
    this is the PyTorch path on the Jetson, where Ultralytics cannot run.
    """

    name = "torchscript"

    def __init__(self, path, device="cuda", precision="fp32"):
        self.path = path
        self.device = device
        self.precision = precision
        self.model = None
        self.torch = None

    def load(self):
        import torch

        self.torch = torch

        if self.device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but torch.cuda.is_available() is False")

        self.model = torch.jit.load(self.path, map_location=self.device).eval()

        if self.precision == "fp16":
            self.model = self.model.half()

    def infer(self, tensor):
        t = self.torch.from_numpy(tensor).to(self.device)
        if self.precision == "fp16":
            t = t.half()

        with self.torch.no_grad():
            out = self.model(t)

        if isinstance(out, (list, tuple)):
            out = out[0]

        if self.device == "cuda":
            self.torch.cuda.synchronize()

        return out.float().cpu().numpy()

class OnnxBackend(Backend):

    name = "onnxruntime"

    def __init__(self, onnx_path, device="cuda"):
        self.onnx_path = onnx_path
        self.device = device
        self.session = None
        self.input_name = None

    def load(self):
        import onnxruntime as ort

        if self.device == "cuda":
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        else:
            providers = ["CPUExecutionProvider"]

        self.session = ort.InferenceSession(self.onnx_path, providers=providers)
        self.input_name = self.session.get_inputs()[0].name

        # ORT falls back to CPU silently if the CUDA provider fails to load,
        # which would be recorded as a GPU result.
        active = self.session.get_providers()
        if self.device == "cuda" and "CUDAExecutionProvider" not in active:
            raise RuntimeError(
                "CUDAExecutionProvider unavailable; ORT fell back to %s" % active
            )

    def infer(self, tensor):
        return self.session.run(None, {self.input_name: tensor})[0]


class TensorRTBackend(Backend):
    """
    Manages the CUDA context explicitly. pycuda.autoinit destroys the context at
    interpreter exit, before TensorRT's engine is garbage-collected, which
    produces 'context is destroyed' errors and a segfault on teardown.
    """

    name = "tensorrt"

    def __init__(self, engine_path):
        self.engine_path = engine_path
        self.engine = None
        self.context = None
        self.cuda_ctx = None
        self.stream = None
        self.d_input = None
        self.d_output = None
        self.h_output = None
        self.bindings = None
        self.cuda = None

    def load(self):
        import tensorrt as trt
        import pycuda.driver as cuda

        self.cuda = cuda
        cuda.init()
        self.cuda_ctx = cuda.Device(0).make_context()

        logger = trt.Logger(trt.Logger.WARNING)
        with open(self.engine_path, "rb") as f:
            self.engine = trt.Runtime(logger).deserialize_cuda_engine(f.read())

        if self.engine is None:
            raise RuntimeError(
                "Failed to deserialise %s. Engines are specific to the GPU "
                "architecture and TensorRT version that built them." % self.engine_path
            )

        self.context = self.engine.create_execution_context()

        # Allocated once here, not per frame, so the loop does not time cudaMalloc.
        for i in range(self.engine.num_bindings):
            shape = self.engine.get_binding_shape(i)
            size = int(np.prod(shape))
            nbytes = size * np.float32().itemsize

            if self.engine.binding_is_input(i):
                self.d_input = cuda.mem_alloc(nbytes)
                self.input_shape = tuple(shape)
            else:
                self.d_output = cuda.mem_alloc(nbytes)
                self.output_shape = tuple(shape)
                # Pinned host memory: transfers can use DMA.
                self.h_output = cuda.pagelocked_empty(size, dtype=np.float32)

        self.bindings = [int(self.d_input), int(self.d_output)]
        self.stream = cuda.Stream()

    def infer(self, tensor):
        cuda = self.cuda

        cuda.memcpy_htod_async(self.d_input, tensor, self.stream)
        self.context.execute_async_v2(
            bindings=self.bindings, stream_handle=self.stream.handle
        )
        cuda.memcpy_dtoh_async(self.h_output, self.d_output, self.stream)

        # Everything above is queued; nothing is complete until this returns.
        self.stream.synchronize()

        return self.h_output.reshape(self.output_shape).copy()

    def close(self):
        # Release TensorRT objects before the context they allocated from.
        self.context = None
        self.engine = None
        if self.cuda_ctx is not None:
            self.cuda_ctx.pop()
            self.cuda_ctx.detach()
            self.cuda_ctx = None


def build_backend(runtime, cfg, device, precision):
    if runtime == "pytorch":
        return TorchBackend(cfg["model"]["weights"], device=device, precision=precision)
    if runtime == "onnxruntime":
        return OnnxBackend(cfg["model"]["onnx"], device=device)
    if runtime == "torchscript":
        return TorchScriptBackend(cfg["model"]["torchscript"], device=device,
                                  precision=precision)
    if runtime == "tensorrt":
        return TensorRTBackend(cfg["model"]["engine"])
    raise ValueError("unknown runtime: %s" % runtime)

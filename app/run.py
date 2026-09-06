"""
Container entrypoint: run inference continuously and report throughput.

This is the deployment path, not the measurement harness. It shares
preprocess/postprocess/backends with benchmarks/benchmark.py, so the numbers are
comparable -- but it is a service that runs, not a tool that measures once.

Deliberately has no dependency on benchmarks/ or evaluation/, which are not
copied into the image.

    python3 app/run.py --runtime tensorrt --precision fp16 --frames 500
    python3 app/run.py --runtime tensorrt --precision fp16   # runs until stopped
"""

import argparse
import json
import os
import signal
import sys
import time

import cv2
import numpy as np
import yaml

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from app import postprocess, preprocess          # noqa: E402
from app.backends import build_backend           # noqa: E402


_stop = {"now": False}


def _handle_signal(signum, frame):
    """
    Docker sends SIGTERM on `docker stop`, then SIGKILL after a grace period.
    Handling it means the run summary still gets printed instead of the process
    vanishing mid-frame.
    """
    _stop["now"] = True


def main():
    p = argparse.ArgumentParser(description="EdgeVision inference service")
    p.add_argument("--config", default="configs/params.yaml")
    p.add_argument("--runtime", default="tensorrt",
                   choices=["pytorch", "torchscript", "onnxruntime", "tensorrt"])
    p.add_argument("--precision", default="fp16", choices=["fp32", "fp16"])
    p.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    p.add_argument("--source", default=None)
    p.add_argument("--warmup", type=int, default=50)
    p.add_argument("--frames", type=int, default=None,
                   help="stop after N frames; omit to run until stopped")
    p.add_argument("--report-every", dest="report_every", type=int, default=100)
    p.add_argument("--sink", default=None,
                   help="write detections as JSONL to this path")
    args = p.parse_args()

    signal.signal(signal.SIGTERM, _handle_signal)
    signal.signal(signal.SIGINT, _handle_signal)

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    res = cfg["model"]["input_res"]
    conf_th = cfg["runtime"]["conf_threshold"]
    iou_th = cfg["runtime"]["nms_iou"]
    source = args.source or cfg["runtime"]["source"]

    print("EdgeVision inference service")
    print("  runtime   : %s %s" % (args.runtime, args.precision))
    print("  source    : %s" % source)
    print("  threshold : conf %.2f, nms iou %.2f" % (conf_th, iou_th))
    print("")

    backend = build_backend(args.runtime, cfg, args.device, args.precision)

    t0 = time.perf_counter()
    backend.load()
    print("cold start  : %.1f ms" % ((time.perf_counter() - t0) * 1000.0))

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print("ERROR: cannot open %s" % source)
        return 1

    sink = open(args.sink, "w") if args.sink else None

    # First inferences are slow for reasons unrelated to steady state: CUDA
    # context setup, kernel selection, cache population.
    for _ in range(args.warmup):
        ok, frame = cap.read()
        if not ok:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ok, frame = cap.read()
        tensor, info = preprocess.to_tensor(frame, res)
        out = backend.infer(tensor)
        postprocess.decode(out, info, conf_th, iou_th)

    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    print("warmup      : %d frames" % args.warmup)
    print("")

    frame_times = []
    infer_times = []
    n = 0
    loop_start = time.perf_counter()

    while not _stop["now"]:
        if args.frames and n >= args.frames:
            break

        frame_start = time.perf_counter()

        ok, frame = cap.read()
        if not ok:
            # Loop the video rather than exiting: this is a service, and a
            # finite source should not end it.
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ok, frame = cap.read()
            if not ok:
                print("ERROR: source produced no frames")
                break

        tensor, info = preprocess.to_tensor(frame, res)

        t = time.perf_counter()
        raw = backend.infer(tensor)
        infer_times.append((time.perf_counter() - t) * 1000.0)

        boxes, scores, class_ids = postprocess.decode(raw, info, conf_th, iou_th)

        frame_times.append((time.perf_counter() - frame_start) * 1000.0)
        n += 1

        if sink is not None:
            sink.write(json.dumps({
                "frame": n,
                "timestamp": time.time(),
                "detections": [
                    {"box": [round(float(v), 1) for v in box],
                     "score": round(float(sc), 4),
                     "class": postprocess.COCO_CLASSES[int(ci)]}
                    for box, sc, ci in zip(boxes, scores, class_ids)
                ],
            }) + "\n")
            sink.flush()

        if n % args.report_every == 0:
            recent = frame_times[-args.report_every:]
            print("  %6d frames  |  %.1f FPS  |  %.1f ms/frame  |  %d objects"
                  % (n, 1000.0 / np.mean(recent), np.mean(recent), len(boxes)))

    elapsed = time.perf_counter() - loop_start
    cap.release()
    if sink is not None:
        sink.close()
    backend.close()

    if n == 0:
        print("no frames processed")
        return 1

    print("")
    print("=" * 54)
    print("frames            : %d" % n)
    print("elapsed           : %.1f s" % elapsed)
    print("-" * 54)
    print("inference    mean : %8.2f ms" % float(np.mean(infer_times)))
    print("             p50  : %8.2f ms" % float(np.percentile(infer_times, 50)))
    print("             p95  : %8.2f ms" % float(np.percentile(infer_times, 95)))
    print("-" * 54)
    print("end-to-end   mean : %8.2f ms" % float(np.mean(frame_times)))
    print("             p50  : %8.2f ms" % float(np.percentile(frame_times, 50)))
    print("             p95  : %8.2f ms" % float(np.percentile(frame_times, 95)))
    print("-" * 54)
    print("FPS engine        : %8.2f" % (1000.0 / float(np.mean(infer_times))))
    print("FPS end-to-end    : %8.2f" % (n / elapsed))
    print("=" * 54)
    return 0


if __name__ == "__main__":
    sys.exit(main())

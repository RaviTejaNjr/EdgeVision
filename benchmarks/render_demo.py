"""
Render an annotated video: detections drawn, live FPS burned in.

Runs on the Jetson through the same backends as the benchmark, so the frame times
shown are the ones actually measured -- not a replay of numbers from elsewhere.

    python3 benchmarks/render_demo.py --runtime tensorrt --precision fp16
    python3 benchmarks/render_demo.py --runtime torchscript --precision fp32

Writes assets/demo_<runtime>_<precision>.mp4 and prints the measured mean FPS,
which is what the side-by-side compositor needs for real-time playback.
"""

import argparse
import json
import os
import sys
import time

import cv2
import numpy as np
import yaml

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from app import postprocess, preprocess          # noqa: E402
from app.backends import build_backend           # noqa: E402


# Distinct colours per class, so overlapping detections stay readable.
def class_colour(class_id):
    rng = np.random.RandomState(int(class_id) * 7919)
    return tuple(int(c) for c in rng.randint(60, 255, size=3))


def draw(frame, boxes, scores, class_ids, label, fps, frame_ms, accent):
    out = frame.copy()

    for box, score, cls in zip(boxes, scores, class_ids):
        x1, y1, x2, y2 = [int(v) for v in box]
        colour = class_colour(cls)
        cv2.rectangle(out, (x1, y1), (x2, y2), colour, 2)

        text = "%s %.2f" % (postprocess.COCO_CLASSES[int(cls)], score)
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        # Filled label background: text over a busy scene is otherwise unreadable.
        cv2.rectangle(out, (x1, max(y1 - th - 6, 0)), (x1 + tw + 4, y1), colour, -1)
        cv2.putText(out, text, (x1 + 2, max(y1 - 4, th)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

    # Header band. Deliberately oversized: the composed video is downscaled for
    # the README GIF, and text rendered at normal size does not survive that.
    h, w = out.shape[:2]
    band = 96
    cv2.rectangle(out, (0, 0), (w, band), (18, 18, 18), -1)
    cv2.line(out, (0, band), (w, band), accent, 4)

    cv2.putText(out, label, (24, 64), cv2.FONT_HERSHEY_SIMPLEX,
                1.6, accent, 3, cv2.LINE_AA)

    stat = "%.1f FPS" % fps
    (sw, _), _ = cv2.getTextSize(stat, cv2.FONT_HERSHEY_SIMPLEX, 1.8, 4)
    cv2.putText(out, stat, (w - sw - 24, 66), cv2.FONT_HERSHEY_SIMPLEX,
                1.8, (255, 255, 255), 4, cv2.LINE_AA)

    detail = "%.0f ms/frame   %d objects" % (frame_ms, len(boxes))
    (dw, _), _ = cv2.getTextSize(detail, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
    cv2.putText(out, detail, (w - dw - 24, 88), cv2.FONT_HERSHEY_SIMPLEX,
                0.8, (170, 170, 170), 2, cv2.LINE_AA)

    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/params.yaml")
    p.add_argument("--runtime", required=True,
                   choices=["pytorch", "torchscript", "onnxruntime", "tensorrt"])
    p.add_argument("--precision", default="fp32", choices=["fp32", "fp16"])
    p.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    p.add_argument("--frames", type=int, default=300,
                   help="frames to render; the full clip is 1040")
    p.add_argument("--warmup", type=int, default=30)
    p.add_argument("--label", default=None, help="header text; defaults to runtime")
    p.add_argument("--out", default=None)
    args = p.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    res = cfg["model"]["input_res"]
    conf_th = cfg["runtime"]["conf_threshold"]
    iou_th = cfg["runtime"]["nms_iou"]

    label = args.label or "%s %s" % (args.runtime.upper(), args.precision.upper())

    # BGR. Green marks the optimised runtime, amber the baseline, so the two
    # panes are distinguishable at a glance even in a small GIF.
    accent = (120, 255, 120) if args.runtime == "tensorrt" else (80, 180, 255)
    out_path = args.out or "assets/demo_%s_%s.mp4" % (args.runtime, args.precision)
    if not os.path.isdir("assets"):
        os.makedirs("assets")

    backend = build_backend(args.runtime, cfg, args.device, args.precision)
    print("loading %s..." % label)
    backend.load()

    cap = cv2.VideoCapture(cfg["runtime"]["source"])
    if not cap.isOpened():
        raise RuntimeError("cannot open %s" % cfg["runtime"]["source"])

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Warm up before recording: the first frames are slow for reasons unrelated
    # to steady-state performance, and burning those numbers into the video would
    # misrepresent it.
    for _ in range(args.warmup):
        ok, frame = cap.read()
        if not ok:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ok, frame = cap.read()
        tensor, info = preprocess.to_tensor(frame, res)
        out = backend.infer(tensor)
        postprocess.decode(out, info, conf_th, iou_th)

    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    # Written at 30 fps; the compositor retimes to real speed afterwards.
    writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"),
                             30.0, (width, height))

    frame_times = []
    smoothed_fps = None
    n = 0

    print("rendering %d frames -> %s" % (args.frames, out_path))

    while n < args.frames:
        t0 = time.perf_counter()

        ok, frame = cap.read()
        if not ok:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ok, frame = cap.read()
            if not ok:
                break

        tensor, info = preprocess.to_tensor(frame, res)
        raw = backend.infer(tensor)
        boxes, scores, class_ids = postprocess.decode(raw, info, conf_th, iou_th)

        elapsed = time.perf_counter() - t0
        frame_times.append(elapsed)

        # Exponential smoothing: a raw per-frame FPS number flickers too much to
        # read on screen.
        inst = 1.0 / elapsed if elapsed > 0 else 0.0
        smoothed_fps = inst if smoothed_fps is None else 0.9 * smoothed_fps + 0.1 * inst

        writer.write(draw(frame, boxes, scores, class_ids,
                          label, smoothed_fps, elapsed * 1000.0, accent))

        n += 1
        if n % 50 == 0:
            print("  %4d / %d   %.1f FPS" % (n, args.frames, smoothed_fps))

    writer.release()
    cap.release()
    backend.close()

    mean_fps = len(frame_times) / sum(frame_times)
    mean_ms = 1000.0 * sum(frame_times) / len(frame_times)

    meta = {
        "runtime": args.runtime,
        "precision": args.precision,
        "label": label,
        "frames": n,
        "mean_fps": round(mean_fps, 3),
        "mean_ms": round(mean_ms, 2),
        "video": out_path,
    }
    meta_path = out_path.replace(".mp4", ".json")
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    print("")
    print("wrote %s" % out_path)
    print("frames    : %d" % n)
    print("mean FPS  : %.2f" % mean_fps)
    print("mean ms   : %.2f" % mean_ms)
    print("metadata  : %s" % meta_path)


if __name__ == "__main__":
    main()

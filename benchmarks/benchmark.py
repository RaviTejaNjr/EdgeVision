"""
Benchmark harness.

Written before any optimisation and reused unchanged across every runtime, so
that differences between results reflect the runtimes rather than the
measurement method.

Reports cold start separately from steady state, and engine throughput
separately from end-to-end throughput. Python 3.6 compatible: it runs unchanged
on the Jetson.
"""

import argparse
import csv
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
import uuid

import cv2
import numpy as np
import yaml

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from app import postprocess, preprocess          # noqa: E402
from app.backends import build_backend           # noqa: E402


def git_sha():
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL
        )
        sha = out.decode().strip()
    except Exception:
        return "unknown"

    # A dirty tree means the code differs from the commit it claims.
    try:
        dirty = subprocess.check_output(
            ["git", "status", "--porcelain"], stderr=subprocess.DEVNULL
        ).decode().strip()
        if dirty:
            sha += "-dirty"
    except Exception:
        pass

    return sha


def config_hash(fields):
    """Join key between speed.csv and accuracy.csv, so ordering must be stable."""
    blob = json.dumps(fields, sort_keys=True).encode()
    return hashlib.sha1(blob).hexdigest()[:8]


def peak_memory_mb():
    try:
        import psutil
        return psutil.Process().memory_info().rss / (1024.0 * 1024.0)
    except ImportError:
        pass
    try:
        import resource
        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
    except Exception:
        return None


def gpu_temp_c():
    """Jetson exposes thermal zones under /sys; returns None elsewhere."""
    base = "/sys/devices/virtual/thermal"
    if not os.path.isdir(base):
        return None

    hottest = None
    for zone in sorted(os.listdir(base)):
        if not zone.startswith("thermal_zone"):
            continue
        try:
            with open(os.path.join(base, zone, "type")) as f:
                kind = f.read().strip()
            if "GPU" not in kind.upper():
                continue
            with open(os.path.join(base, zone, "temp")) as f:
                c = int(f.read().strip()) / 1000.0
            if hottest is None or c > hottest:
                hottest = c
        except Exception:
            continue
    return hottest


def power_mode():
    try:
        out = subprocess.check_output(
            ["nvpmodel", "-q"], stderr=subprocess.DEVNULL
        ).decode()
        for line in out.splitlines():
            if "NV Power Mode" in line:
                return line.split(":")[-1].strip()
    except Exception:
        pass
    return "n/a"


def percentile(values, p):
    if len(values) == 0:
        return None
    return float(np.percentile(np.asarray(values), p))


def run(args, cfg):
    run_id = uuid.uuid4().hex[:12]

    cfg_fields = {
        "model": cfg["model"]["name"],
        "runtime": args.runtime,
        "precision": args.precision,
        "device": args.device,
        "input_res": cfg["model"]["input_res"],
        "batch_size": 1,
        "power_mode": args.power_mode,
        "clocks_locked": args.clocks_locked,
        "fan": args.fan,
        "host_profile": args.host_profile,
    }
    cfg_hash = config_hash(cfg_fields)

    print("run_id      : %s" % run_id)
    print("config_hash : %s" % cfg_hash)
    print("config      : %s" % json.dumps(cfg_fields))
    print("")

    backend = build_backend(args.runtime, cfg, args.device, args.precision)

    t0 = time.perf_counter()
    backend.load()
    cold_start_ms = (time.perf_counter() - t0) * 1000.0
    print("cold start  : %.1f ms" % cold_start_ms)

    res = cfg["model"]["input_res"]
    conf_th = cfg["runtime"]["conf_threshold"]
    iou_th = cfg["runtime"]["nms_iou"]

    cap = cv2.VideoCapture(args.source)
    if not cap.isOpened():
        raise RuntimeError("cannot open %s" % args.source)

    # First inferences are slow for reasons unrelated to steady state: context
    # setup, kernel JIT, cache population, clock ramp-up.
    print("warmup      : %d frames" % args.warmup)
    for _ in range(args.warmup):
        ok, frame = cap.read()
        if not ok:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ok, frame = cap.read()
        tensor, info = preprocess.to_tensor(frame, res)
        out = backend.infer(tensor)
        postprocess.decode(out, info, conf_th, iou_th)

    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    t_capture, t_pre, t_infer, t_post, t_total = [], [], [], [], []
    frame_wall_times = []
    detections_per_frame = []
    max_temp = gpu_temp_c()

    print("measuring   : %s"
          % ("%d s" % args.duration if args.duration else "%d frames" % args.frames))

    loop_start = time.perf_counter()
    n = 0

    while True:
        if args.duration and (time.perf_counter() - loop_start) >= args.duration:
            break
        if args.frames and n >= args.frames:
            break

        frame_start = time.perf_counter()

        t = time.perf_counter()
        ok, frame = cap.read()
        if not ok:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ok, frame = cap.read()
            if not ok:
                break
        t_capture.append((time.perf_counter() - t) * 1000.0)

        t = time.perf_counter()
        tensor, info = preprocess.to_tensor(frame, res)
        t_pre.append((time.perf_counter() - t) * 1000.0)

        t = time.perf_counter()
        out = backend.infer(tensor)
        t_infer.append((time.perf_counter() - t) * 1000.0)

        t = time.perf_counter()
        boxes, scores, class_ids = postprocess.decode(out, info, conf_th, iou_th)
        t_post.append((time.perf_counter() - t) * 1000.0)

        frame_end = time.perf_counter()
        t_total.append((frame_end - frame_start) * 1000.0)
        frame_wall_times.append(frame_end - loop_start)
        detections_per_frame.append(len(boxes))

        n += 1

        if n % 100 == 0:
            temp = gpu_temp_c()
            if temp is not None and (max_temp is None or temp > max_temp):
                max_temp = temp
            print("  %5d frames  |  e2e %.1f ms  |  engine %.1f ms  |  %d det"
                  % (n, t_total[-1], t_infer[-1], len(boxes)))

    elapsed = time.perf_counter() - loop_start
    cap.release()

    peak_mb = peak_memory_mb()
    backend.close()

    # Throughput over the final 60 s only. On a thermally constrained board this
    # is materially lower than the overall mean.
    sustained_fps = None
    if elapsed > 60.0:
        cutoff = elapsed - 60.0
        tail = [w for w in frame_wall_times if w >= cutoff]
        if len(tail) > 1:
            sustained_fps = len(tail) / (tail[-1] - tail[0])

    fps_engine = 1000.0 / float(np.mean(t_infer)) if t_infer else 0.0
    fps_e2e = n / elapsed if elapsed > 0 else 0.0

    row = {
        "run_id": run_id,
        "config_hash": cfg_hash,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "git_commit": git_sha(),
        "hostname": platform.node(),

        "model": cfg_fields["model"],
        "runtime": args.runtime,
        "precision": args.precision,
        "device": args.device,
        "input_res": res,
        "batch_size": 1,

        "power_mode": args.power_mode,
        "clocks_locked": args.clocks_locked,
        "fan": args.fan,
        "host_profile": args.host_profile,

        "warmup_frames": args.warmup,
        "n_frames": n,
        "duration_s": round(elapsed, 2),

        "cold_start_ms": round(cold_start_ms, 2),

        "engine_p50_ms": round(percentile(t_infer, 50), 3),
        "engine_p95_ms": round(percentile(t_infer, 95), 3),
        "engine_mean_ms": round(float(np.mean(t_infer)), 3),

        "e2e_p50_ms": round(percentile(t_total, 50), 3),
        "e2e_p95_ms": round(percentile(t_total, 95), 3),
        "e2e_mean_ms": round(float(np.mean(t_total)), 3),

        "capture_ms": round(float(np.mean(t_capture)), 3),
        "preprocess_ms": round(float(np.mean(t_pre)), 3),
        "inference_ms": round(float(np.mean(t_infer)), 3),
        "postprocess_ms": round(float(np.mean(t_post)), 3),

        "fps_engine": round(fps_engine, 2),
        "fps_end_to_end": round(fps_e2e, 2),
        "fps_sustained_last_60s": round(sustained_fps, 2) if sustained_fps else "",

        "mean_detections": round(float(np.mean(detections_per_frame)), 2),
        "peak_mem_mb": round(peak_mb, 1) if peak_mb else "",
        "gpu_temp_max_c": round(max_temp, 1) if max_temp else "",
        "throttled": "",
        "notes": args.notes,
    }

    # Per-frame arrays for percentile recomputation and thermal plots.
    raw_dir = os.path.join(args.results_dir, "raw")
    if not os.path.isdir(raw_dir):
        os.makedirs(raw_dir)
    np.savez(
        os.path.join(raw_dir, "%s.npz" % run_id),
        capture_ms=np.asarray(t_capture, dtype=np.float32),
        preprocess_ms=np.asarray(t_pre, dtype=np.float32),
        inference_ms=np.asarray(t_infer, dtype=np.float32),
        postprocess_ms=np.asarray(t_post, dtype=np.float32),
        total_ms=np.asarray(t_total, dtype=np.float32),
        wall_time_s=np.asarray(frame_wall_times, dtype=np.float32),
        detections=np.asarray(detections_per_frame, dtype=np.int32),
    )

    csv_path = os.path.join(args.results_dir, "speed.csv")
    write_header = not os.path.isfile(csv_path)
    with open(csv_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if write_header:
            writer.writeheader()
        writer.writerow(row)

    print("")
    print("=" * 62)
    print("cold start        : %8.1f ms" % cold_start_ms)
    print("-" * 62)
    print("capture           : %8.2f ms" % row["capture_ms"])
    print("preprocess        : %8.2f ms" % row["preprocess_ms"])
    print("inference         : %8.2f ms" % row["inference_ms"])
    print("postprocess + nms : %8.2f ms" % row["postprocess_ms"])
    print("-" * 62)
    print("engine    p50/p95 : %8.2f / %.2f ms" % (row["engine_p50_ms"], row["engine_p95_ms"]))
    print("end-to-end p50/p95: %8.2f / %.2f ms" % (row["e2e_p50_ms"], row["e2e_p95_ms"]))
    print("-" * 62)
    print("FPS engine        : %8.2f" % row["fps_engine"])
    print("FPS end-to-end    : %8.2f" % row["fps_end_to_end"])
    if sustained_fps:
        print("FPS sustained 60s : %8.2f" % sustained_fps)
    print("-" * 62)
    print("mean detections   : %8.2f" % row["mean_detections"])
    if peak_mb:
        print("peak memory       : %8.1f MB" % peak_mb)
    if max_temp:
        print("max GPU temp      : %8.1f C" % max_temp)
    print("=" * 62)
    print("")
    print("appended to %s" % csv_path)
    print("raw arrays  %s/%s.npz" % (raw_dir, run_id))


def main():
    p = argparse.ArgumentParser(description="EdgeVision benchmark harness")
    p.add_argument("--config", default="configs/params.yaml")
    p.add_argument("--runtime", required=True,
                   choices=["pytorch", "torchscript", "onnxruntime", "tensorrt"])
    p.add_argument("--precision", default="fp32", choices=["fp32", "fp16"])
    p.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    p.add_argument("--source", default=None)
    p.add_argument("--warmup", type=int, default=None)
    p.add_argument("--duration", type=float, default=None, help="seconds")
    p.add_argument("--frames", type=int, default=None)
    p.add_argument("--power-mode", dest="power_mode", default=None)
    p.add_argument("--clocks-locked", dest="clocks_locked", default="false")
    p.add_argument("--fan", default="true")
    p.add_argument("--host-profile", dest="host_profile", default="unknown",
                   help="host power/performance state, e.g. best-performance, "
                        "whisper, jetson-10w. Part of the config hash: the same "
                        "code on the same GPU differs 5x between profiles.")
    p.add_argument("--results-dir", dest="results_dir", default="results")
    p.add_argument("--notes", default="")
    args = p.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    if args.source is None:
        args.source = cfg["runtime"]["source"]
    if args.warmup is None:
        args.warmup = cfg["benchmark"]["warmup_frames"]
    if args.power_mode is None:
        args.power_mode = power_mode()

    if args.duration is None and args.frames is None:
        args.frames = 500

    run(args, cfg)


if __name__ == "__main__":
    main()

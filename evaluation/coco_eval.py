"""
Score COCO detections and append a row to results/accuracy.csv.

Runs on the laptop, where pycocotools is available. Takes the JSON written by
run_coco_detections.py -- so every runtime is scored by one implementation, and
differences between rows reflect the runtimes rather than the evaluation.

    python evaluation/coco_eval.py --detections results/detections_tensorrt_fp16.json \
        --runtime tensorrt --precision fp16 --device nano

    pip install pycocotools
"""

import argparse
import csv
import hashlib
import json
import os
import subprocess
import sys
import time


def git_sha():
    try:
        sha = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:
        return "unknown"
    try:
        if subprocess.check_output(
            ["git", "status", "--porcelain"], stderr=subprocess.DEVNULL
        ).decode().strip():
            sha += "-dirty"
    except Exception:
        pass
    return sha


def config_hash(fields):
    """Must match benchmark.py's hashing, or the join to speed.csv breaks."""
    return hashlib.sha1(
        json.dumps(fields, sort_keys=True).encode()).hexdigest()[:8]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--detections", required=True)
    p.add_argument("--runtime", required=True)
    p.add_argument("--precision", default="fp32")
    p.add_argument("--device", default="nano")
    p.add_argument("--config", default="configs/params.yaml")
    p.add_argument("--coco-root", dest="coco_root", default="data/coco")
    p.add_argument("--subset", type=int, default=500)
    p.add_argument("--results-dir", dest="results_dir", default="results")
    p.add_argument("--notes", default="")
    args = p.parse_args()

    try:
        from pycocotools.coco import COCO
        from pycocotools.cocoeval import COCOeval
    except ImportError:
        raise SystemExit("pycocotools not installed -- pip install pycocotools")

    import yaml
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    gt_path = os.path.join(
        args.coco_root, "instances_val2017_subset%d.json" % args.subset)
    if not os.path.isfile(gt_path):
        raise SystemExit("%s not found -- run evaluation/prepare_coco.py" % gt_path)
    if not os.path.isfile(args.detections):
        raise SystemExit("%s not found" % args.detections)

    with open(args.detections) as f:
        dets = json.load(f)
    if not dets:
        raise SystemExit("%s contains no detections" % args.detections)

    coco_gt = COCO(gt_path)
    coco_dt = coco_gt.loadRes(args.detections)

    ev = COCOeval(coco_gt, coco_dt, "bbox")
    ev.evaluate()
    ev.accumulate()
    ev.summarize()

    # COCOeval.stats is a fixed 12-element vector; the indices are its documented
    # order and there is no named accessor.
    stats = ev.stats
    metrics = {
        "mAP50_95": float(stats[0]),
        "mAP50": float(stats[1]),
        "mAP75": float(stats[2]),
        "mAP_small": float(stats[3]),
        "mAP_medium": float(stats[4]),
        "mAP_large": float(stats[5]),
        "AR_max1": float(stats[6]),
        "AR_max10": float(stats[7]),
        "AR_max100": float(stats[8]),
    }

    # Mirrors benchmark.py's field set so the hashes match and the two CSVs join.
    cfg_fields = {
        "model": cfg["model"]["name"],
        "runtime": args.runtime,
        "precision": args.precision,
        "device": args.device,
        "input_res": cfg["model"]["input_res"],
        "batch_size": 1,
    }

    row = {
        "config_hash": config_hash(cfg_fields),
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "git_commit": git_sha(),
        "model": cfg_fields["model"],
        "runtime": args.runtime,
        "precision": args.precision,
        "device": args.device,
        "input_res": cfg_fields["input_res"],
        "n_images": args.subset,
        "n_detections": len(dets),
    }
    for k, v in sorted(metrics.items()):
        row[k] = round(v, 5)
    row["notes"] = args.notes

    csv_path = os.path.join(args.results_dir, "accuracy.csv")
    write_header = not os.path.isfile(csv_path)
    with open(csv_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if write_header:
            writer.writeheader()
        writer.writerow(row)

    print("")
    print("=" * 56)
    print("%s %s on %s" % (args.runtime, args.precision, args.device))
    print("-" * 56)
    print("mAP@50-95        : %.4f" % metrics["mAP50_95"])
    print("mAP@50           : %.4f" % metrics["mAP50"])
    print("mAP@75           : %.4f" % metrics["mAP75"])
    print("-" * 56)
    print("mAP small        : %.4f" % metrics["mAP_small"])
    print("mAP medium       : %.4f" % metrics["mAP_medium"])
    print("mAP large        : %.4f" % metrics["mAP_large"])
    print("-" * 56)
    print("AR @100          : %.4f" % metrics["AR_max100"])
    print("detections       : %d over %d images" % (len(dets), args.subset))
    print("=" * 56)
    print("")
    print("appended to %s" % csv_path)


if __name__ == "__main__":
    sys.exit(main())

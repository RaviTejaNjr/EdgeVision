"""
Run a backend over the COCO subset and write detections in COCO result format.

Runs on the Jetson (or the laptop). Deliberately does NOT compute mAP -- scoring
needs pycocotools, which compiles a C extension, and keeping that on the laptop
avoids another build on the board. It also means one scoring implementation is
used for every runtime.

    python3 evaluation/run_coco_detections.py --runtime tensorrt --precision fp16

Writes results/detections_<runtime>_<precision>.json.
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


# COCO category ids are not 0..79 -- there are gaps where categories were removed
# from the original 91-class set. Model class index 0 is category 1, index 11 is
# category 13, and so on. Getting this wrong silently scores every detection
# against the wrong class.
COCO91 = [
    1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 18, 19, 20, 21,
    22, 23, 24, 25, 27, 28, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42,
    43, 44, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61,
    62, 63, 64, 65, 67, 70, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 84,
    85, 86, 87, 88, 89, 90,
]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/params.yaml")
    p.add_argument("--runtime", required=True,
                   choices=["pytorch", "torchscript", "onnxruntime", "tensorrt"])
    p.add_argument("--precision", default="fp32", choices=["fp32", "fp16"])
    p.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    p.add_argument("--coco-root", dest="coco_root", default="data/coco")
    p.add_argument("--subset", type=int, default=500)
    p.add_argument("--conf", type=float, default=None,
                   help="confidence threshold. Defaults to 0.001 for evaluation, "
                        "NOT the runtime threshold -- mAP integrates over the "
                        "precision-recall curve and needs low-confidence "
                        "detections to trace it.")
    p.add_argument("--out", default=None)
    args = p.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    res = cfg["model"]["input_res"]
    iou_th = cfg["runtime"]["nms_iou"]

    # 0.001 is the COCO convention. Using the runtime's 0.25 would truncate the
    # PR curve and understate mAP for every runtime equally -- consistent, but
    # not comparable with published figures.
    conf_th = args.conf if args.conf is not None else 0.001

    subset_path = os.path.join(
        args.coco_root, "instances_val2017_subset%d.json" % args.subset)
    img_dir = os.path.join(args.coco_root, "val2017")

    if not os.path.isfile(subset_path):
        raise SystemExit("%s not found -- run evaluation/prepare_coco.py first"
                         % subset_path)

    with open(subset_path) as f:
        subset = json.load(f)
    images = subset["images"]

    out_path = args.out or "results/detections_%s_%s.json" % (
        args.runtime, args.precision)

    backend = build_backend(args.runtime, cfg, args.device, args.precision)
    print("loading %s %s..." % (args.runtime, args.precision))
    backend.load()

    print("conf threshold : %.3f" % conf_th)
    print("images         : %d" % len(images))
    print("")

    detections = []
    t0 = time.perf_counter()

    for i, im in enumerate(images, 1):
        path = os.path.join(img_dir, im["file_name"])
        frame = cv2.imread(path)
        if frame is None:
            print("skipping unreadable %s" % im["file_name"])
            continue

        tensor, info = preprocess.to_tensor(frame, res)
        raw = backend.infer(tensor)
        boxes, scores, class_ids = postprocess.decode(raw, info, conf_th, iou_th)

        for box, score, cls in zip(boxes, scores, class_ids):
            x1, y1, x2, y2 = [float(v) for v in box]
            detections.append({
                "image_id": int(im["id"]),
                "category_id": int(COCO91[int(cls)]),
                # COCO expects xywh with the top-left corner, not corners.
                "bbox": [round(x1, 2), round(y1, 2),
                         round(x2 - x1, 2), round(y2 - y1, 2)],
                "score": round(float(score), 5),
            })

        if i % 50 == 0 or i == len(images):
            elapsed = time.perf_counter() - t0
            sys.stdout.write("\r  %d / %d   %.1f img/s   %d detections"
                             % (i, len(images), i / elapsed, len(detections)))
            sys.stdout.flush()

    print("")
    backend.close()

    if not os.path.isdir("results"):
        os.makedirs("results")
    with open(out_path, "w") as f:
        json.dump(detections, f)

    print("")
    print("detections : %d" % len(detections))
    print("per image  : %.1f" % (len(detections) / float(len(images))))
    print("wrote      : %s" % out_path)
    print("")
    print("Score it on the laptop:")
    print("  python evaluation/coco_eval.py --detections %s \\" % out_path)
    print("      --runtime %s --precision %s" % (args.runtime, args.precision))


if __name__ == "__main__":
    sys.exit(main())

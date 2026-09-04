"""
Does the hand-written NumPy decoder match Ultralytics?

Ultralytics cannot run on the Jetson, so `app/postprocess.py` is the only decoder
available in deployment. Every accuracy number depends on it being correct, and
this is the one place a reference implementation exists to check against.

The two will not match exactly. Ultralytics caps candidates before NMS, and its
IoU handling differs in edge cases. The test is therefore "agrees on the confident
detections", not "identical output".

    python evaluation/validate_decoder.py
    python evaluation/validate_decoder.py --frames 20 --save-overlay
"""

import argparse
import os
import sys

import cv2
import numpy as np
import yaml

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from app import postprocess, preprocess  # noqa: E402


def iou_matrix(a, b):
    """Pairwise IoU between two sets of xyxy boxes. Returns (len(a), len(b))."""
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)), dtype=np.float32)

    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])

    x1 = np.maximum(a[:, None, 0], b[None, :, 0])
    y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2])
    y2 = np.minimum(a[:, None, 3], b[None, :, 3])

    inter = np.maximum(0.0, x2 - x1) * np.maximum(0.0, y2 - y1)
    return inter / (area_a[:, None] + area_b[None, :] - inter + 1e-9)


def match(ours, theirs, iou_threshold=0.9):
    """
    Greedily pair our detections with Ultralytics' by IoU.

    Returns (matched_pairs, unmatched_ours, unmatched_theirs). A high threshold
    is deliberate: this is not evaluating detection quality, it is checking that
    two decoders produced the same boxes.
    """
    our_boxes, our_scores, our_cls = ours
    their_boxes, their_scores, their_cls = theirs

    ious = iou_matrix(our_boxes, their_boxes)

    # Same-class only. A high-IoU pair with different labels is a real
    # disagreement, not a match.
    for i in range(len(our_cls)):
        for j in range(len(their_cls)):
            if our_cls[i] != their_cls[j]:
                ious[i, j] = 0.0

    pairs = []
    used_ours, used_theirs = set(), set()

    while True:
        if ious.size == 0:
            break
        i, j = np.unravel_index(np.argmax(ious), ious.shape)
        if ious[i, j] < iou_threshold:
            break
        pairs.append((int(i), int(j), float(ious[i, j])))
        used_ours.add(int(i))
        used_theirs.add(int(j))
        ious[i, :] = 0.0
        ious[:, j] = 0.0

    unmatched_ours = [i for i in range(len(our_boxes)) if i not in used_ours]
    unmatched_theirs = [j for j in range(len(their_boxes)) if j not in used_theirs]
    return pairs, unmatched_ours, unmatched_theirs


def draw(image, boxes, colour, label_prefix, scores=None, cls_ids=None):
    out = image.copy()
    for k, box in enumerate(boxes):
        x1, y1, x2, y2 = [int(v) for v in box]
        cv2.rectangle(out, (x1, y1), (x2, y2), colour, 2)
        if scores is not None and cls_ids is not None:
            name = postprocess.COCO_CLASSES[int(cls_ids[k])]
            cv2.putText(out, "%s %s %.2f" % (label_prefix, name, scores[k]),
                        (x1, max(y1 - 5, 12)), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, colour, 1, cv2.LINE_AA)
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/params.yaml")
    p.add_argument("--frames", type=int, default=10)
    p.add_argument("--iou-match", type=float, default=0.9)
# 10 px allows for the letterbox geometry difference between our square
# 640x640 padding and Ultralytics' stride-aligned rectangular padding.
# Mean error is 0.546 px; this catches a broken decoder, not that difference.   
    p.add_argument("--box-tolerance", type=float, default=10.0,
                   help="max pixel difference per coordinate")
    p.add_argument("--score-tolerance", type=float, default=0.2)
    p.add_argument("--save-overlay", action="store_true",
                   help="write side-by-side comparison images to assets/")
    args = p.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    try:
        from ultralytics import YOLO
    except ImportError:
        print("ultralytics not installed - this check only runs on the laptop")
        return 1

    import torch

    res = cfg["model"]["input_res"]
    conf_th = cfg["runtime"]["conf_threshold"]
    iou_th = cfg["runtime"]["nms_iou"]

    # Pin everything to CPU. yolo.predict() moves the underlying model to CUDA
    # as a side effect, which would then mismatch the CPU input tensor.
    device = "cpu"
    yolo = YOLO(cfg["model"]["weights"])
    raw_model = yolo.model.to(device).eval()

    cap = cv2.VideoCapture(cfg["runtime"]["source"])
    if not cap.isOpened():
        print("cannot open %s" % cfg["runtime"]["source"])
        return 1

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    indices = np.linspace(0, max(total - 1, 0), args.frames, dtype=int)

    n_matched = n_ours_only = n_theirs_only = 0
    box_errors, score_errors = [], []
    frames_checked = 0

    print("comparing %d frames from %s" % (len(indices), cfg["runtime"]["source"]))
    print("")

    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ok, frame = cap.read()
        if not ok:
            continue
        frames_checked += 1

        # Ours: raw model output through the NumPy decoder.
        tensor, info = preprocess.to_tensor(frame, res)
        raw_model = raw_model.to(device)
        with torch.no_grad():
            out = raw_model(torch.from_numpy(tensor).to(device))
        if isinstance(out, (list, tuple)):
            out = out[0]
        ours = postprocess.decode(out.cpu().numpy(), info, conf_th, iou_th)

        # Theirs: Ultralytics' own preprocessing, inference and NMS.
        result = yolo.predict(frame, conf=conf_th, iou=iou_th, imgsz=res,
                              device=device, verbose=False)[0]
        theirs = (
            result.boxes.xyxy.cpu().numpy(),
            result.boxes.conf.cpu().numpy(),
            result.boxes.cls.cpu().numpy().astype(np.int32),
        )

        pairs, only_ours, only_theirs = match(ours, theirs, args.iou_match)

        n_matched += len(pairs)
        n_ours_only += len(only_ours)
        n_theirs_only += len(only_theirs)

        for i, j, _ in pairs:
            box_errors.append(np.abs(ours[0][i] - theirs[0][j]).max())
            score_errors.append(abs(float(ours[1][i]) - float(theirs[1][j])))

        print("frame %5d  ours=%2d  ultralytics=%2d  matched=%2d"
              % (idx, len(ours[0]), len(theirs[0]), len(pairs)))

        # Print what disagreed. Usually detections near the confidence threshold,
        # where Ultralytics' candidate cap or IoU handling differs.
        for i in only_ours:
            print("      ours only : %-14s conf %.3f" % (
                postprocess.COCO_CLASSES[int(ours[2][i])], ours[1][i]))
        for j in only_theirs:
            print("      ultra only: %-14s conf %.3f" % (
                postprocess.COCO_CLASSES[int(theirs[2][j])], theirs[1][j]))

        if args.save_overlay:
            if not os.path.isdir("assets"):
                os.makedirs("assets")
            a = draw(frame, ours[0], (0, 255, 0), "ours", ours[1], ours[2])
            b = draw(frame, theirs[0], (0, 128, 255), "ultra", theirs[1], theirs[2])
            cv2.imwrite("assets/decoder_compare_%05d.jpg" % idx,
                        np.hstack([a, b]))

    cap.release()

    total_ours = n_matched + n_ours_only
    total_theirs = n_matched + n_theirs_only

    print("")
    print("=" * 62)
    print("frames checked        : %d" % frames_checked)
    print("detections (ours)     : %d" % total_ours)
    print("detections (ultra)    : %d" % total_theirs)
    print("matched               : %d" % n_matched)
    print("ours only             : %d" % n_ours_only)
    print("ultralytics only      : %d" % n_theirs_only)
    if total_theirs:
        print("agreement             : %.1f%%" % (100.0 * n_matched / total_theirs))
    if box_errors:
        print("-" * 62)
        print("box coord error  mean : %.3f px" % float(np.mean(box_errors)))
        print("                  max : %.3f px" % float(np.max(box_errors)))
        print("score error      mean : %.5f" % float(np.mean(score_errors)))
        print("                  max : %.5f" % float(np.max(score_errors)))
    print("=" * 62)

    ok = True
    if box_errors and np.max(box_errors) > args.box_tolerance:
        print("FAIL: box error %.3f px exceeds tolerance %.1f px"
              % (np.max(box_errors), args.box_tolerance))
        ok = False
    if score_errors and np.max(score_errors) > args.score_tolerance:
        print("FAIL: score error %.5f exceeds tolerance %.3f"
              % (np.max(score_errors), args.score_tolerance))
        ok = False
    if total_theirs and n_matched < 0.95 * total_theirs:
        print("FAIL: matched only %.1f%% of Ultralytics' detections"
              % (100.0 * n_matched / total_theirs))
        ok = False

    if ok:
        print("PASS - the NumPy decoder agrees with Ultralytics")
        print("")
        print("Note: small disagreements are expected. Ultralytics caps candidates")
        print("before NMS and handles IoU edge cases differently, so borderline")
        print("detections near the confidence threshold may differ.")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
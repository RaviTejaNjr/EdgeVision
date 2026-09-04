"""
Decode raw YOLO output and suppress duplicate boxes.

Output tensor is (1, 84, 8400): 8400 candidates across strides 8/16/32
(80x80 + 40x40 + 20x20), each with 4 box values and 80 COCO class scores.
Written in NumPy because Ultralytics requires Python 3.8+ and the Jetson's
TensorRT bindings are 3.6-only.
"""

import numpy as np


COCO_CLASSES = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck",
    "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench",
    "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra",
    "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
    "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove",
    "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
    "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
    "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
    "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse",
    "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
    "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
    "hair drier", "toothbrush",
]


def xywh_to_xyxy(boxes):
    """(cx, cy, w, h) -> (x1, y1, x2, y2)."""
    out = np.empty_like(boxes)
    half_w = boxes[:, 2] / 2.0
    half_h = boxes[:, 3] / 2.0
    out[:, 0] = boxes[:, 0] - half_w
    out[:, 1] = boxes[:, 1] - half_h
    out[:, 2] = boxes[:, 0] + half_w
    out[:, 3] = boxes[:, 1] + half_h
    return out


def nms(boxes, scores, iou_threshold):
    """Greedy non-maximum suppression. Returns indices of kept boxes."""
    if len(boxes) == 0:
        return []

    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = (x2 - x1) * (y2 - y1)

    order = scores.argsort()[::-1]
    keep = []

    while order.size > 0:
        i = order[0]
        keep.append(i)

        if order.size == 1:
            break

        rest = order[1:]

        xx1 = np.maximum(x1[i], x1[rest])
        yy1 = np.maximum(y1[i], y1[rest])
        xx2 = np.minimum(x2[i], x2[rest])
        yy2 = np.minimum(y2[i], y2[rest])

        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        inter = w * h

        iou = inter / (areas[i] + areas[rest] - inter + 1e-9)
        order = rest[iou <= iou_threshold]

    return keep


def undo_letterbox(boxes, info):
    """Map boxes from padded 640x640 space back to original image pixels."""
    boxes = boxes.copy()
    boxes[:, [0, 2]] -= info.pad_x
    boxes[:, [1, 3]] -= info.pad_y
    boxes /= info.scale

    boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]], 0, info.orig_w)
    boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]], 0, info.orig_h)
    return boxes


def decode(output, info, conf_threshold=0.25, iou_threshold=0.45, class_agnostic=False):
    """
    Raw model output -> (boxes, scores, class_ids) in original image coordinates.
    """
    if output.ndim == 3:
        output = output[0]

    # (84, 8400) -> (8400, 84)
    preds = output.transpose(1, 0)

    boxes_xywh = preds[:, :4]

    # Anchor-free head: no objectness column. Confidence is the best class score.
    class_scores = preds[:, 4:]
    class_ids = np.argmax(class_scores, axis=1)
    scores = class_scores[np.arange(class_scores.shape[0]), class_ids]

    mask = scores >= conf_threshold
    if not np.any(mask):
        return (np.empty((0, 4), dtype=np.float32),
                np.empty((0,), dtype=np.float32),
                np.empty((0,), dtype=np.int32))

    boxes_xywh = boxes_xywh[mask]
    scores = scores[mask]
    class_ids = class_ids[mask]

    boxes = xywh_to_xyxy(boxes_xywh)

    if class_agnostic:
        keep = nms(boxes, scores, iou_threshold)
    else:
        # Per class: overlapping boxes of different classes are both valid.
        keep = []
        for cls in np.unique(class_ids):
            idx = np.where(class_ids == cls)[0]
            kept_local = nms(boxes[idx], scores[idx], iou_threshold)
            keep.extend(idx[k] for k in kept_local)
        keep = sorted(keep, key=lambda i: -scores[i])

    keep = np.asarray(keep, dtype=np.int64)
    boxes = undo_letterbox(boxes[keep], info)

    return (boxes.astype(np.float32),
            scores[keep].astype(np.float32),
            class_ids[keep].astype(np.int32))

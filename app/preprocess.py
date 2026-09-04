"""Letterbox resize and tensor preparation for YOLO inference."""

import cv2
import numpy as np


PAD_VALUE = 114


class LetterboxInfo(object):
    """Records how an image was padded, so boxes can be mapped back afterwards."""

    def __init__(self, scale, pad_x, pad_y, orig_w, orig_h):
        self.scale = scale
        self.pad_x = pad_x
        self.pad_y = pad_y
        self.orig_w = orig_w
        self.orig_h = orig_h

    def __repr__(self):
        return "LetterboxInfo(scale=%.4f, pad=(%.1f, %.1f), orig=%dx%d)" % (
            self.scale, self.pad_x, self.pad_y, self.orig_w, self.orig_h
        )


def letterbox(image, target=640):
    """Resize preserving aspect ratio, padding the shorter dimension with grey."""
    h, w = image.shape[:2]

    scale = min(target / float(w), target / float(h))
    new_w = int(round(w * scale))
    new_h = int(round(h * scale))

    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    pad_w = target - new_w
    pad_h = target - new_h
    top = pad_h // 2
    bottom = pad_h - top
    left = pad_w // 2
    right = pad_w - left

    padded = cv2.copyMakeBorder(
        resized, top, bottom, left, right,
        cv2.BORDER_CONSTANT, value=(PAD_VALUE, PAD_VALUE, PAD_VALUE)
    )

    return padded, LetterboxInfo(scale, left, top, w, h)


def to_tensor(image_bgr, target=640):
    """BGR frame -> NCHW float32 tensor, plus the letterbox parameters."""
    padded, info = letterbox(image_bgr, target)

    rgb = cv2.cvtColor(padded, cv2.COLOR_BGR2RGB)
    normalised = rgb.astype(np.float32) / 255.0
    chw = np.transpose(normalised, (2, 0, 1))
    nchw = np.expand_dims(chw, axis=0)

    # TensorRT copies raw memory to the device, so the buffer must be contiguous.
    return np.ascontiguousarray(nchw), info

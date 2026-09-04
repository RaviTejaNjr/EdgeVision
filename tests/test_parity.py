"""
Does the ONNX export preserve the model?

Runs identical input through PyTorch and ONNX Runtime and compares the raw
(1, 84, 8400) tensors. If they diverge, the export is wrong and every downstream
number inherits the error.

Laptop only: ONNX Runtime has no CUDA wheel for the Jetson's Python 3.6.

    pytest tests/test_parity.py -v
"""

import os
import sys

import cv2
import numpy as np
import pytest
import yaml

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from app import preprocess  # noqa: E402


CONFIG_PATH = "configs/params.yaml"

# FP32 arithmetic is not associative, so two runtimes that fuse or reorder
# operations differently will not produce bit-identical results. These bounds
# allow for that while still catching a genuinely broken export.
ATOL = 1e-3
RTOL = 1e-3
MAX_MEAN_ABS_DIFF = 1e-4


@pytest.fixture(scope="module")
def cfg():
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)


@pytest.fixture(scope="module")
def frames(cfg):
    """Five frames spread across the clip, so the test is not one lucky sample."""
    cap = cv2.VideoCapture(cfg["runtime"]["source"])
    if not cap.isOpened():
        pytest.skip("test video not available")

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    out = []
    for idx in np.linspace(0, max(total - 1, 0), 5, dtype=int):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ok, frame = cap.read()
        if ok:
            out.append(frame)
    cap.release()

    if not out:
        pytest.skip("could not read frames")
    return out


@pytest.fixture(scope="module")
def torch_outputs(cfg, frames):
    torch = pytest.importorskip("torch")
    ultralytics = pytest.importorskip("ultralytics")

    model = ultralytics.YOLO(cfg["model"]["weights"]).model.eval()

    res = cfg["model"]["input_res"]
    outs = []
    for frame in frames:
        tensor, _ = preprocess.to_tensor(frame, res)
        with torch.no_grad():
            out = model(torch.from_numpy(tensor))
        if isinstance(out, (list, tuple)):
            out = out[0]
        outs.append(out.numpy())
    return outs


@pytest.fixture(scope="module")
def onnx_outputs(cfg, frames):
    ort = pytest.importorskip("onnxruntime")

    onnx_path = cfg["model"]["onnx"]
    if not os.path.isfile(onnx_path):
        pytest.skip("%s not found - run models/export_to_onnx.py" % onnx_path)

    session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name

    res = cfg["model"]["input_res"]
    outs = []
    for frame in frames:
        tensor, _ = preprocess.to_tensor(frame, res)
        outs.append(session.run(None, {input_name: tensor})[0])
    return outs


def test_output_shape(onnx_outputs):
    """Shape must be (1, 84, 8400): 8400 candidates, 4 box values + 80 classes."""
    for out in onnx_outputs:
        assert out.shape == (1, 84, 8400), "unexpected shape %s" % (out.shape,)


def test_output_dtype(onnx_outputs):
    """
    Output is float32 even when the engine computes internally in FP16.
    The NumPy decoder relies on this.
    """
    for out in onnx_outputs:
        assert out.dtype == np.float32


def test_no_nan_or_inf(onnx_outputs):
    for i, out in enumerate(onnx_outputs):
        assert np.isfinite(out).all(), "frame %d contains NaN or Inf" % i


def test_pytorch_onnx_parity(torch_outputs, onnx_outputs):
    """The core check: do the two runtimes agree?"""
    for i, (t, o) in enumerate(zip(torch_outputs, onnx_outputs)):
        assert t.shape == o.shape, "frame %d shape mismatch" % i

        abs_diff = np.abs(t - o)
        mean_diff = abs_diff.mean()
        max_diff = abs_diff.max()

        assert mean_diff < MAX_MEAN_ABS_DIFF, (
            "frame %d: mean abs diff %.3e exceeds %.3e (max %.3e)"
            % (i, mean_diff, MAX_MEAN_ABS_DIFF, max_diff)
        )
        assert np.allclose(t, o, atol=ATOL, rtol=RTOL), (
            "frame %d: max abs diff %.3e" % (i, max_diff)
        )


def test_box_coordinates_are_plausible(onnx_outputs, cfg):
    """
    Box centres should fall inside the letterboxed input. Catches a transposed
    or misinterpreted output layout, which would still produce finite numbers.
    """
    res = cfg["model"]["input_res"]
    for i, out in enumerate(onnx_outputs):
        preds = out[0].transpose(1, 0)      # (8400, 84)
        cx, cy = preds[:, 0], preds[:, 1]
        assert (cx >= -res).all() and (cx <= 2 * res).all(), "frame %d: cx out of range" % i
        assert (cy >= -res).all() and (cy <= 2 * res).all(), "frame %d: cy out of range" % i


def test_class_scores_are_probabilities(onnx_outputs):
    """
    The 80 class columns are sigmoid outputs, so they must lie in [0, 1]. If the
    objectness assumption were wrong, the columns would be offset by one and this
    would fail.
    """
    for i, out in enumerate(onnx_outputs):
        scores = out[0].transpose(1, 0)[:, 4:]
        assert scores.min() >= -1e-4, "frame %d: score below 0 (%.4f)" % (i, scores.min())
        assert scores.max() <= 1.0 + 1e-4, "frame %d: score above 1 (%.4f)" % (i, scores.max())

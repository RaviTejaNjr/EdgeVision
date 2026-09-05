"""Verify the TorchScript file loads with torch alone, as the Nano will."""

import torch

m = torch.jit.load("models/yolov5nu.torchscript")
m.eval()

with torch.no_grad():
    out = m(torch.zeros(1, 3, 640, 640))

if isinstance(out, (list, tuple)):
    print("returns a", type(out).__name__, "of", len(out))
    for i, o in enumerate(out):
        print("  [%d]" % i, type(o).__name__, getattr(o, "shape", None))
else:
    print("returns a tensor:", tuple(out.shape), out.dtype)
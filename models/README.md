# Models

Model choice, export settings, and the reasoning behind both.

## Which model, and what it actually is

**`yolov5nu`** — Ultralytics' retrofit of the YOLOv5 backbone with an
**anchor-free, decoupled detection head** taken from YOLOv8. The `u` suffix is
the marker.

**It is not the original 2020 YOLOv5.** Worth stating plainly, because the
difference changes how the output is decoded.

| Property | Value |
|---|---|
| Parameters | 2,649,200 |
| GFLOPs @ 640×640 | 7.7 |
| Checkpoint | 5.3 MB |
| ONNX | 10.3 MB |
| TorchScript | 10.6 MB |
| TensorRT FP32 engine | 22.9 MB |
| TensorRT FP16 engine | 11.8 MB |
| Published mAP@50-95 | 0.343 (Ultralytics) |
| Measured here | **0.3343** — see `results/README.md` |

### Why not YOLOv8n

Not a performance judgement — a platform constraint.

**Ultralytics requires Python 3.8+. TensorRT's bindings on JetPack 4.6 are built
for the system Python 3.6 only.** Installing Python 3.8 to get Ultralytics leaves
`No module named tensorrt`, and often `Illegal instruction (core dumped)` on torch
import.

TensorRT is the point of the project, so the Nano stays on 3.6 and Ultralytics
never touches it. YOLOv5 runs there under 3.6 with NVIDIA's torch wheel.

*(The `yolov5nu` checkpoint is still fetched **on the laptop** via Ultralytics —
only the on-device path is constrained.)*

---

## The output tensor: `(1, 84, 8400)`

The single most important thing to understand here, because `app/postprocess.py`
decodes it by hand.

### Where 8400 comes from

Predictions at three scales, with strides 8, 16 and 32:

| Stride | Grid | Cells |
|---|---|---|
| 8 | 80 × 80 | 6400 |
| 16 | 40 × 40 | 1600 |
| 32 | 20 × 20 | 400 |
| | **Total** | **8400** |

Each cell is one candidate detection. Three scales so small objects are caught by
the fine grid and large ones by the coarse.

### Where 84 comes from

```
84 = 4 box values + 80 COCO class scores
```

**There is no objectness column.** Classic YOLOv5 output was `5 + num_classes` —
4 box values, 1 objectness, then classes. The anchor-free head **drops objectness
entirely** and uses the maximum class score as confidence.

Assuming the old layout shifts all 80 class scores by one position. The result
still produces detections, so it fails silently. `tests/test_parity.py` asserts
the 80 class columns lie in [0, 1] — which they would not if the layout were
offset, since the last column would then hold unbounded box data.

### Layout and format

**Channel-first:** 84 rows of 8400 values, not 8400 rows of 84. Transpose before
iterating.

**Box format is `(cx, cy, w, h)`** — centre, not corners — in pixels relative to
the 640×640 letterboxed input. Converting to `x1,y1,x2,y2` and undoing the
letterbox is `app/postprocess.py`'s job.

### The DFL layers in the build log

`trtexec` output shows `/model.24/dfl/Reshape`, `/dfl/Softmax`, `/dfl/conv/Conv`.

That is **Distribution Focal Loss** decoding: instead of regressing a box edge as
a single number, the model predicts a probability distribution over 16 discrete
bins per edge and takes the expected value. Better gradients during training,
sub-pixel accuracy at inference.

**It happens inside the exported graph.** The `(1, 84, 8400)` output is already
decoded — there is nothing to implement.

---

## Export scripts

### `export_to_onnx.py`

```python
model.export(format="onnx", opset=13, imgsz=640, simplify=True, dynamic=False)
```

| Argument | Why |
|---|---|
| `opset=13` | TensorRT 8.2 supports roughly opset 13–14. A newer opset exports cleanly on the laptop and then **fails at engine build** — a failure one step removed from its cause |
| `dynamic=False` | Fixed 640×640. Dynamic shapes require TensorRT optimisation profiles and complicate everything for no benefit when the input size is known |
| `simplify=True` | onnxslim constant-folds and removes redundant nodes |

### `export_torchscript.py`

```python
model.export(format="torchscript", imgsz=640)
```

**Uses Ultralytics' own exporter rather than a manual `torch.jit.trace`.** Two
manual attempts failed:

1. `RuntimeError: Tracer cannot infer type of (tensor, {'boxes':..., 'feats':[...]})`
   — the model returns a dict of mixed tensors and lists that the tracer cannot
   type.
2. After wrapping to return only the prediction tensor:
   `ERROR: Tensor-valued Constant nodes differed in value across invocations`
   — **Ultralytics caches anchor points after the first forward pass**, so the two
   trace runs produce different graphs.

Ultralytics warms the model before tracing, so the anchor cache is populated and
both invocations match.

**Lesson: when a library ships its own exporter, use it — it knows about internal
state you do not.**

TorchScript exists because it is the **only PyTorch path available on the Nano**.
`torch.jit.load` needs nothing but torch — no Ultralytics, and as it turns out no
torchvision either, since YOLOv5n is pure convolutions with no torchvision
operators in the graph.

### `build_trt_engine.py`

```bash
python3 models/build_trt_engine.py --precision fp16
python3 models/build_trt_engine.py --precision fp32
```

**Must run on the Jetson.** TensorRT auto-tunes by timing candidate kernels on the
actual GPU, so the engine embeds choices specific to that architecture and
TensorRT version. An engine built elsewhere will not deserialise.

It prints the platform capability flags:

```
platform_has_fast_fp16 : True
platform_has_fast_int8 : False
```

**That second line is TensorRT's own API declaring INT8 unavailable** — the board
is SM 5.3 and the INT8 path needs compute capability 6.1 for the DP4A
instruction. A stronger citation than documentation, and reproducible by anyone
with the hardware.

| Precision | Build time | Engine size |
|---|---|---|
| FP32 | 151.0 s | 22.9 MB |
| FP16 | 362.6 s | 11.8 MB |

**FP16 takes 2.4× longer to build** — the auto-tuner has both precisions available
per layer, so a larger search space. The engine is half the size because FP32
weights are 4 bytes and FP16 are 2.

⚠️ **Engines are not a pure function of the ONNX input.** Rebuilding on an idle
headless board produced an engine **2.2% faster** than the same conversion run
during initial setup (50.64 vs 51.76 ms — five times the run-to-run CV). With more
free memory the auto-tuner can consider tactics it would otherwise skip; the build
log warns about exactly this: *"Some tactics do not have sufficient workspace
memory to run."*

### `check_onnx.py`, `check_torchscript.py`

Verify the artifacts **before** transferring them to the Nano — cheaper than
discovering a problem after an 11-minute engine build on the board.

`check_onnx.py` confirms opset 13 and that shapes are fixed rather than dynamic.
`check_torchscript.py` loads the file **importing only torch**, which is exactly
the situation on the Nano.

---

## Artifacts

All gitignored — regenerable, and the engines are useless on any other machine.

| File | Regenerate with | Time |
|---|---|---|
| `yolov5nu.pt` | downloaded by Ultralytics on first use | seconds |
| `yolov5nu.onnx` | `export_to_onnx.py` | ~10 s |
| `yolov5nu.torchscript` | `export_torchscript.py` | ~2 s |
| `yolov5nu_fp32.engine` | `build_trt_engine.py --precision fp32` | 151 s, **on the Nano** |
| `yolov5nu_fp16.engine` | `build_trt_engine.py --precision fp16` | 363 s, **on the Nano** |

**The engines in particular must not be committed.** They only load on a Maxwell
GPU with TensorRT 8.2 — 12–23 MB of binary that is useless to anyone else.

---

## Where things run

| Laptop (Python 3.10) | Jetson Nano (Python 3.6) |
|---|---|
| Ultralytics | TensorRT 8.2 bindings |
| ONNX + TorchScript export | PyCUDA |
| `pycocotools` scoring | torch 1.10 (NVIDIA aarch64 wheel) |
| | JetPack OpenCV 4.1.1 |
| | NumPy 1.13.3 |

The split is forced by the Python 3.6 constraint, not chosen — but it happens to
match how deployment works in practice: export on a workstation, run on the device.

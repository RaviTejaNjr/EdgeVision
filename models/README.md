# Models

Model choice, export settings, artifact checks and TensorRT engine build notes.

## Model

The project uses **`yolov5nu`**, Ultralytics' retrofit of the YOLOv5 backbone with an **anchor-free, decoupled detection head**. This is not the original 2020 YOLOv5 detection head, and the difference matters when decoding the exported output.

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

### Why `yolov5nu`

`yolov5nu` is used throughout the project so the same model can be compared across the PyTorch, TorchScript, ONNX and TensorRT paths.

Ultralytics is used on the development machine for checkpoint loading, export and reference validation. The Jetson Nano remains on the JetPack 4.6 system Python 3.6 environment because its TensorRT bindings are provided there. The on-device path therefore uses exported artifacts and does not depend on Ultralytics.

TorchScript provides the on-device PyTorch baseline, while TensorRT is the deployment runtime.

---

## Output tensor: `(1, 84, 8400)`

`app/postprocess.py` decodes the exported model output directly, so the tensor layout needs to be handled correctly.

### Where 8400 comes from

Predictions are produced at three scales with strides 8, 16 and 32:

| Stride | Grid | Cells |
|---|---|---|
| 8 | 80 × 80 | 6400 |
| 16 | 40 × 40 | 1600 |
| 32 | 20 × 20 | 400 |
| | **Total** | **8400** |

Each grid cell contributes one candidate detection.

### Where 84 comes from

```text
84 = 4 box values + 80 COCO class scores
```

There is **no objectness column** in this exported head. The older YOLOv5 layout used `5 + num_classes`: four box values, one objectness value and the class scores. Treating this output as that older layout would offset the class-score columns and produce incorrect detections.

`tests/test_parity.py` checks that the 80 class-score columns remain in the expected range.

### Layout and box format

The tensor is **channel-first**: 84 rows of 8400 values, not 8400 rows of 84. It is transposed before candidate detections are processed.

Boxes are represented as `(cx, cy, w, h)` in pixels relative to the 640×640 letterboxed input. `app/postprocess.py` converts them to corner coordinates and maps them back to the original image.

### DFL layers in the exported graph

The TensorRT build log contains layers such as:

```text
/model.24/dfl/Reshape
/model.24/dfl/Softmax
/model.24/dfl/conv/Conv
```

These are part of Distribution Focal Loss decoding. The DFL computation is already included in the exported graph, so the `(1, 84, 8400)` tensor reaching the application does not require a separate DFL implementation.

---

## Export scripts

### `export_to_onnx.py`

```python
model.export(format="onnx", opset=13, imgsz=640, simplify=True, dynamic=False)
```

| Argument | Reason |
|---|---|
| `opset=13` | Compatible with the TensorRT 8.2 toolchain used on the Nano |
| `dynamic=False` | Keeps the input fixed at 640×640 and avoids TensorRT optimization profiles |
| `simplify=True` | Simplifies the exported ONNX graph |

The exported ONNX model is checked before transfer to the Nano.

### `export_torchscript.py`

```python
model.export(format="torchscript", imgsz=640)
```

Ultralytics' exporter is used instead of a manual `torch.jit.trace`.

Two direct tracing attempts were tested first:

1. The unwrapped model returned mixed tensor/list/dictionary outputs and failed with:

```text
RuntimeError: Tracer cannot infer type of (tensor, {'boxes':..., 'feats':[...]})
```

2. A wrapper that returned only the prediction tensor then failed with:

```text
ERROR: Tensor-valued Constant nodes differed in value across invocations
```

The model caches anchor information after the first forward pass, so the repeated trace invocations did not produce the same graph. Ultralytics' exporter handles that model state before tracing.

TorchScript is used as the on-device PyTorch baseline because `torch.jit.load` only needs PyTorch at runtime. The exported graph used here does not require torchvision operators.

### `build_trt_engine.py`

```bash
python3 models/build_trt_engine.py --precision fp16
python3 models/build_trt_engine.py --precision fp32
```

TensorRT engines are built on the Jetson Nano. The builder benchmarks candidate kernels on the target GPU, so the generated engine depends on the target architecture and TensorRT environment.

The build script records the platform capability flags:

```text
platform_has_fast_fp16 : True
platform_has_fast_int8 : False
```

TensorRT therefore reports a fast FP16 path but no fast INT8 path on this Nano.

| Precision | Build time | Engine size |
|---|---|---|
| FP32 | 151.0 s | 22.9 MB |
| FP16 | 362.6 s | 11.8 MB |

The FP16 engine takes longer to build because TensorRT has a larger set of candidate tactics to evaluate. Its engine file is smaller because the weights use FP16 rather than FP32 storage.

Engine build conditions can also affect the selected TensorRT tactics. Rebuilding the same ONNX model on an idle headless board produced a 50.64 ms result compared with 51.76 ms from the earlier build. The builder log also reported cases where candidate tactics could not run because of workspace limits.

### `check_onnx.py` and `check_torchscript.py`

These scripts verify the exported artifacts before they are copied to the Nano.

`check_onnx.py` confirms the ONNX opset and fixed input shapes.

`check_torchscript.py` loads the exported file while importing only PyTorch, matching the intended Nano runtime environment.

---

## Artifacts

Model artifacts are gitignored because they can be regenerated, and TensorRT engines are tied to the target GPU and TensorRT environment.

| File | Regenerate with | Time |
|---|---|---|
| `yolov5nu.pt` | downloaded by Ultralytics on first use | seconds |
| `yolov5nu.onnx` | `export_to_onnx.py` | ~10 s |
| `yolov5nu.torchscript` | `export_torchscript.py` | ~2 s |
| `yolov5nu_fp32.engine` | `build_trt_engine.py --precision fp32` | 151 s, **on the Nano** |
| `yolov5nu_fp16.engine` | `build_trt_engine.py --precision fp16` | 363 s, **on the Nano** |

The TensorRT engine files are not committed to the repository.

---

## Where things run

| Laptop (Python 3.10) | Jetson Nano (Python 3.6) |
|---|---|
| Ultralytics | TensorRT 8.2 bindings |
| ONNX + TorchScript export | PyCUDA |
| `pycocotools` scoring | torch 1.10 (NVIDIA aarch64 wheel) |
| | JetPack OpenCV 4.1.1 |
| | NumPy 1.13.3 |

Export and reference evaluation stay on the development machine. The Jetson receives the exported artifacts and runs the deployment path with the versions supplied by JetPack 4.6.

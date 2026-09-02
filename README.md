# EdgeVision — Production Edge AI Deployment on NVIDIA Jetson Nano

[![TensorRT](https://img.shields.io/badge/TensorRT-8.2-76B900?style=flat&logo=nvidia&logoColor=white)](https://developer.nvidia.com/tensorrt)
[![Jetson](https://img.shields.io/badge/Jetson%20Nano-2GB-76B900?style=flat&logo=nvidia&logoColor=white)](https://developer.nvidia.com/embedded/jetson-nano)
[![ONNX](https://img.shields.io/badge/ONNX-005CED?style=flat&logo=onnx&logoColor=white)](https://onnx.ai/)
[![Docker](https://img.shields.io/badge/Docker-2496ED?style=flat&logo=docker&logoColor=white)](https://www.docker.com/)
[![PyCUDA](https://img.shields.io/badge/PyCUDA-3776AB?style=flat&logo=python&logoColor=white)](https://documen.tician.de/pycuda/)
[![Python](https://img.shields.io/badge/Python-3.6%20%7C%203.10-blue?style=flat&logo=python)](https://www.python.org/)
[![CI](https://github.com/RaviTejaNjr/edgevision/actions/workflows/ci.yml/badge.svg)](https://github.com/RaviTejaNjr/edgevision/actions)

An end-to-end edge deployment pipeline taking an object detector from a PyTorch
checkpoint to a **containerised, monitored, self-recovering service** on an
NVIDIA Jetson Nano 2GB — with latency, accuracy and thermal behaviour measured
at every step.

---

## Table of Contents

- [Overview](#overview)
- [Results](#results)
- [Architecture](#architecture)
- [Project Structure](#project-structure)
- [Technology Stack](#technology-stack)
- [Pipeline Stages](#pipeline-stages)
- [Getting Started](#getting-started)
- [Running the Pipeline](#running-the-pipeline)
- [Benchmarking & Evaluation](#benchmarking--evaluation)
- [Monitoring & Resilience](#monitoring--resilience)
- [Configuration](#configuration)
- [Limitations](#limitations)

---

## Overview

Most object detection projects stop at "the model works". This one starts there
and answers the questions an embedded team actually asks before shipping:

- How fast is it **after ten minutes**, once the board is hot — not in the first
  thirty seconds?
- How much accuracy did the optimisation actually cost?
- What happens when the process dies at 3 a.m.?
- Does it still hit target frame rate at the 5W power budget?

The target hardware is deliberately constrained: a Jetson Nano 2GB with a Maxwell
GPU, **no tensor cores**, and roughly 1.4 GB of usable shared memory. Constraints
this tight make the engineering decisions visible.

**Key capabilities**

- PyTorch → ONNX → TensorRT FP16 conversion with numerical parity checks at each step
- Reproducible benchmark harness reporting p50/p95 latency, sustained FPS, and thermals
- Accuracy verified after every conversion — speed without accuracy is worthless
- Containerised deployment on `l4t-base`, reproducing bare-metal performance
- Detections published to a pluggable sink (JSONL now, ROS2 as an optional extension)
- Prometheus + Grafana metrics, systemd watchdog, tested failure recovery
- CI with an accuracy regression gate

---

## Results

> **Status: pending.** Tables below are populated automatically by
> `benchmarks/make_report.py` from `results/speed.csv` and `results/accuracy.csv`.
> No numbers are reported here until they have been measured on the target
> hardware.

### Latency and throughput

All Jetson figures at **10W power mode with clocks locked** (`nvpmodel -m 0`,
`jetson_clocks`), active cooling, batch size 1, 640×640 input.

| Device | Runtime | Precision | p50 (ms) | p95 (ms) | FPS (mean) | FPS (sustained, 10 min) | Peak mem (MB) | Max temp (°C) |
|---|---|---|---|---|---|---|---|---|
| Laptop CPU | PyTorch | FP32 | _pending_ | _pending_ | _pending_ | _pending_ | _pending_ | _pending_ |
| Laptop GPU | PyTorch | FP32 | _pending_ | _pending_ | _pending_ | _pending_ | _pending_ | _pending_ |
| Laptop GPU | TensorRT | FP16 | _pending_ | _pending_ | _pending_ | _pending_ | _pending_ | _pending_ |
| Jetson Nano | PyTorch | FP32 | _pending_ | _pending_ | _pending_ | _pending_ | _pending_ | _pending_ |
| Jetson Nano | TensorRT | FP32 | _pending_ | _pending_ | _pending_ | _pending_ | _pending_ | _pending_ |
| Jetson Nano | TensorRT | FP16 | _pending_ | _pending_ | _pending_ | _pending_ | _pending_ | _pending_ |

### Accuracy after optimisation

Evaluated on 500 held-out COCO validation images via `pycocotools`.

| Runtime | Precision | mAP@50 | mAP@50-95 | Δ vs FP32 baseline |
|---|---|---|---|---|
| PyTorch | FP32 | _pending_ | _pending_ | — |
| ONNX Runtime | FP32 | _pending_ | _pending_ | _pending_ |
| TensorRT | FP32 | _pending_ | _pending_ | _pending_ |
| TensorRT | FP16 | _pending_ | _pending_ | _pending_ |

### Sustained throughput and thermal behaviour

<!-- results/plots/thermal_curve.png -->
> Plot pending. FPS over a ten-minute run, overlaid with GPU temperature, with
> and without active cooling.

### Demo

<!-- assets/demo.gif -->
> Demo GIF pending.

---

## Architecture

```
video file / IMX219 camera
        │
        ▼
  capture  (OpenCV + GStreamer / nvarguscamerasrc)
        │
        ▼
  preprocess  (letterbox resize, normalise, HWC→CHW)
        │
        ▼
  TensorRT FP16 engine   ◄── built on-device from ONNX
        │                     (engines are hardware- and version-specific)
        ▼
  postprocess  (box decode + NMS)
        │
        ├──────────────►  detection sink (JSONL; ROS2 optional)
        │
        └──────────────►  metrics exporter
                                │
                                ▼
                          Prometheus → Grafana

  systemd watchdog wraps the process (Restart=always)
```

---

## Project Structure

```
edgevision/
│
├── app/                               # the inference pipeline — this is what deploys
│   ├── capture.py                     # video file / CSI camera source
│   ├── preprocess.py                  # letterbox resize, normalise, HWC→CHW
│   ├── infer_torch.py                 # PyTorch baseline
│   ├── infer_onnx.py                  # ONNX Runtime
│   ├── infer_trt.py                   # TensorRT, buffers pre-allocated
│   ├── postprocess.py                 # decode (1,84,8400) + NMS, hand-written NumPy
│   └── sink.py                        # where detections go — JSONL now, ROS2 later
│
├── models/
│   ├── export_to_onnx.py              # PyTorch → ONNX (fixed shape, opset 13)
│   ├── build_trt_engine.py            # ONNX → TensorRT — run ON the Nano
│   ├── check_onnx.py                  # validate opset and shapes before transfer
│   └── README.md                      # model choice, opset, rationale
│
├── benchmarks/                        # measurement, kept apart from what it measures
│   ├── benchmark.py                   # the harness — reused unchanged throughout
│   ├── run_matrix.sh                  # sweeps benchmark configurations
│   └── make_report.py                 # CSVs → README tables + plots
│
├── evaluation/
│   └── coco_eval.py                   # mAP on the fixed 500-image COCO subset
│
├── monitoring/
│   ├── exporter.py                    # prometheus_client metrics
│   ├── prometheus.yml
│   └── grafana_dashboard.json
│
├── systemd/
│   └── edgevision.service             # Restart=always + watchdog
│
├── docker/
│   ├── Dockerfile.jetson              # nvcr.io/nvidia/l4t-base:r32.7.1
│   └── docker-compose.monitoring.yml  # Prometheus + Grafana
│
├── tests/
│   ├── test_postprocess.py
│   ├── test_parity.py                 # PyTorch vs ONNX numerical agreement
│   └── test_regression.py             # accuracy gate — fails CI on mAP drop
│
├── configs/
│   └── params.yaml                    # single source of configuration
│
├── docs/
│   ├── environment_report.md          # Part 0 output — JetPack, CUDA, TensorRT versions
│   └── SETUP_LOG.md                   # every command, its output, and why
│
├── results/                           # a deliverable, NOT gitignored
│   ├── speed.csv                      # one row per benchmark run
│   ├── accuracy.csv                   # one row per config
│   ├── raw/                           # per-frame latency arrays (.npy)
│   └── plots/
│
├── data/                              # test video (gitignored)
├── assets/                            # demo GIF, diagrams
│
├── .github/workflows/ci.yml
├── requirements/
│   ├── laptop.txt                     # Python 3.10
│   └── jetson.txt                     # Python 3.6 — diverges deliberately
├── PROJECT_PLAN.md
├── NOTES.md                           # running build log
└── README.md
```

**Why `app/`, `benchmarks/` and `evaluation/` are separate** rather than one
`src/`: they have different lifetimes. `app/` is what the Docker container
deploys. `benchmarks/` is a tool run *against* it. `evaluation/` only ever runs
on the laptop, where the COCO data lives. Splitting them means the deployment
image copies `app/` alone — which matters on a 2 GB board.

---

## Technology Stack

| Tool | Role |
|------|------|
| **PyTorch** | Baseline inference and ONNX export |
| **ONNX / ONNX Runtime** | Interchange format, cross-runtime parity checks |
| **TensorRT 8.2** | Target inference engine — FP32 and FP16 |
| **CUDA 10.2 / cuDNN** | Pinned by JetPack 4.6; not independently upgradeable |
| **OpenCV 4.1.1 + GStreamer** | Capture and preprocessing (JetPack build, not pip) |
| **ROS2** | Detection publishing — *optional extension, not required for v1* |
| **Docker** | `l4t-base` runtime container, built on-device |
| **Prometheus / Grafana** | Metrics collection and dashboards |
| **systemd** | Process supervision and watchdog |
| **jetson-stats (`jtop`)** | Live thermal, power and utilisation monitoring |
| **pycocotools** | mAP evaluation |
| **GitHub Actions** | Lint, test, accuracy regression gate |

---

## Pipeline Stages

### 1. Export — `models/export_to_onnx.py`
- **Input:** PyTorch checkpoint
- **Output:** `model.onnx`
- Fixed input shape (dynamic shapes complicate TensorRT for no benefit here),
  opset 13 — TensorRT 8.2 supports up to roughly opset 13–14.
- Verifies ONNX Runtime output matches PyTorch within tolerance before proceeding.

### 2. Engine build — `models/build_trt_engine.py`
- **Input:** `model.onnx`
- **Output:** `model_fp32.engine`, `model_fp16.engine`
- **Must run on the Jetson.** TensorRT engines are specific to the GPU
  architecture and TensorRT version; an engine built elsewhere will not load.
- FP32 built and validated first, then FP16.

### 3. Inference — `app/infer_trt.py`
- **Input:** engine + video source
- **Output:** detections, metrics
- Device buffers allocated once outside the frame loop; explicit host↔device
  transfers. `app/postprocess.py` decodes the raw (1, 84, 8400) tensor and runs
  NMS in hand-written NumPy — Ultralytics cannot run on the Nano's Python 3.6.

### 4. Benchmark — `benchmarks/benchmark.py`
- **Output:** appended row in `results/speed.csv`
- Separates **cold-start** (model load, engine deserialisation) from
  **steady-state** latency, and reports **engine throughput** and **end-to-end
  throughput** as distinct numbers.
- Per-stage timings, p50, p95, sustained FPS over the final 60 seconds, peak
  memory, max temperature, and a throttling flag.

### 5. Evaluate — `evaluation/coco_eval.py`
- **Output:** appended row in `results/accuracy.csv`
- mAP@50, mAP@50-95, precision and recall on a **fixed** 500-image COCO subset
  held constant across every runtime, joined to speed results by config hash.
- Reports deltas against the FP32 baseline, not just absolute numbers.

---

## Getting Started

### Prerequisites

**Hardware**

| Item | Detail |
|---|---|
| Board | Jetson Nano 2GB Developer Kit (P3541) |
| Storage | 128 GB microSD, UHS-I U3 / V30 |
| Power | 5V / 3A USB-C — **not** a phone charger |
| Cooling | 40mm 5V fan on the stock heatsink |
| Camera | Raspberry Pi Camera Module v2 (IMX219) — optional; video file works |

**Software**

- JetPack 4.6.x (L4T 32.7.x) — the 2GB board cannot run JetPack 5 or 6
- Python 3.6.9 on device (pinned by JetPack), 3.10 on the development machine

### 1. Clone the repository

```bash
git clone https://github.com/RaviTejaNjr/edgevision.git
cd edgevision
```

### 2. Development machine setup

```bash
python -m venv venv
source venv/bin/activate          # Linux / macOS
# venv\Scripts\activate           # Windows
pip install -r requirements/laptop.txt
```

### 3. Jetson setup

Flash the **Jetson Nano 2GB** SD card image (distinct from the 4GB image), then:

```bash
# free ~300 MB by disabling the desktop
sudo systemctl set-default multi-user.target

# live monitoring
sudo pip3 install jetson-stats

# reduce SD card wear
sudo sed -i 's/errors=remount-ro/noatime,errors=remount-ro/' /etc/fstab
echo "SystemMaxUse=50M" | sudo tee -a /etc/systemd/journald.conf

# GPU access inside containers by default
sudo tee /etc/docker/daemon.json > /dev/null <<'EOF'
{ "default-runtime": "nvidia",
  "runtimes": { "nvidia": { "path": "nvidia-container-runtime", "runtimeArgs": [] } } }
EOF
sudo systemctl restart docker
sudo usermod -aG docker $USER

# guard against a bricking bootloader upgrade
sudo apt-mark hold nvidia-l4t-bootloader nvidia-l4t-init
```

Then install PyTorch from **NVIDIA's aarch64 wheel** for JetPack 4.6 — PyPI has no
Jetson build — and build torchvision from source against that exact version.

```bash
pip3 install -r requirements/jetson.txt
```

> **Back up the card now.** `sudo dd if=/dev/sdX of=jetson_base.img bs=4M status=progress`
> A working base image turns a corrupted card from a lost day into a 20-minute restore.

---

## Running the Pipeline

```bash
# 1. Export to ONNX (development machine)
python models/export_to_onnx.py --config configs/params.yaml

# 2. Copy artefacts to the Jetson
scp model.onnx data/test_video.mp4 user@<jetson-ip>:~/edgevision/

# 3. Build engines (on the Jetson)
python3 models/build_trt_engine.py --precision fp32
python3 models/build_trt_engine.py --precision fp16

# 4. Run
python3 app/infer_trt.py --engine models/model_fp16.engine --source data/test_video.mp4
```

### Containerised

```bash
# build on the Jetson — arm64 CUDA images cannot be built on x86
docker build -f docker/Dockerfile.jetson -t edgevision:latest .

docker run --runtime nvidia --rm \
  -v $(pwd)/models:/app/models \
  -v $(pwd)/data:/app/data \
  edgevision:latest
```

### As a service

```bash
sudo cp systemd/edgevision.service /etc/systemd/system/
sudo systemctl enable --now edgevision
journalctl -u edgevision -f
```

---

## Benchmarking & Evaluation

Lock the board state before measuring, or DVFS will make results drift:

```bash
sudo nvpmodel -m 0     # 10W mode (use -m 1 for the 5W comparison)
sudo jetson_clocks     # lock clocks
```

```bash
# single configuration
python3 benchmarks/benchmark.py \
  --model yolov5nu --runtime tensorrt --precision fp16 \
  --device nano --duration 600

# full sweep
bash benchmarks/run_matrix.sh

# regenerate README tables and plots from the CSVs
python benchmarks/make_report.py
```

### How results are stored

Two append-only CSVs, both committed to git, joined on `config_hash`:

- **`results/speed.csv`** — one row per run. Speed varies with thermal state, so a
  config may have several rows.
- **`results/accuracy.csv`** — one row per config. Accuracy is deterministic;
  evaluate once.

Every row records the **git commit SHA** and a **config hash**, so any number in
this README can be traced back to the exact code and configuration that produced
it. Per-frame latency arrays are stored under `results/raw/` as `.npy` for the
percentile and thermal plots.

Results are a deliverable, not a build artefact — `results/` is **not**
gitignored.

### Accuracy regression gate

`tests/test_regression.py` fails CI if the FP16 engine's mAP@50-95 drops more than
a configured threshold below the FP32 baseline. Optimisation that quietly costs
accuracy is a bug, and it is treated as one.

---

## Monitoring & Resilience

`monitoring/exporter.py` exposes inference latency, FPS, GPU and CPU temperature,
memory usage and the active power mode. Prometheus scrapes it; Grafana renders it.

```bash
docker compose -f docker/docker-compose.monitoring.yml up -d
# Grafana: http://<jetson-ip>:3000
```

### Failure recovery — tested, not assumed

| Scenario | Expected behaviour | Verified |
|---|---|---|
| Process killed (`SIGKILL`) | systemd restarts within _pending_ s | ☐ |
| Video source removed mid-run | Logged, graceful exit, restart with backoff | ☐ |
| One hour continuous run | No memory growth, thermal steady state reached | ☐ |
| Power mode dropped to 5W | Degraded FPS, no crash | ☐ |

---

## Configuration

All parameters live in `configs/params.yaml`. Nothing is hardcoded in the source.

```yaml
model:
  name: yolov5nu
  weights: models/yolov5nu.pt
  input_res: 640
  opset: 13

engine:
  precision: fp16          # fp32 | fp16
  workspace_mb: 256        # keep low — 2GB board

runtime:
  source: data/test_video.mp4
  conf_threshold: 0.25
  nms_iou: 0.45

benchmark:
  warmup_frames: 50
  duration_s: 600
  power_mode: 10W          # 10W | 5W
  clocks_locked: true
  fan: true

evaluation:
  coco_val_images: 500
  map_regression_threshold: 0.01   # CI fails beyond this drop
```

---

## Limitations

Stated plainly, because a benchmark without its constraints is not a result.

- **Batch size 1 only.** With ~1.4 GB usable shared memory there is no headroom
  for batching, and single-stream latency is the metric that matters for this
  class of device anyway.
- **No INT8 quantisation.** TensorRT's INT8 path requires compute capability 6.1
  or higher; this board is SM 5.3. `platform_has_fast_int8` returns False and
  INT8 calibrators fail at engine build rather than degrading gracefully. Among
  Jetson platforms, INT8 support begins with Xavier. FP16 is therefore the
  reduced-precision target — a hardware constraint identified and documented, not
  an experiment left undone.
- **Laptop GPU fan disconnected** for noise. Reconnected and temperature logged
  for any sustained GPU benchmark; the Nano is the primary measurement target
  regardless.
- **No tensor cores.** The Maxwell GPU supports native FP16 at 2× FP32 throughput,
  which is why FP16 helps — but none of the tensor-core acceleration that modern
  Jetson benchmarks assume.
- **JetPack 4.6 pins the stack.** CUDA 10.2, TensorRT 8.2, Python 3.6. Several
  modern libraries (FastAPI, current `transformers`) simply cannot run on-device.
- **TensorRT 8.2 predates native LayerNorm support** (added in 8.6), so
  transformer architectures decompose into elementwise operations and run slower
  than the architecture warrants.
- **Ultralytics cannot run on the Nano.** It requires Python 3.8+, while
  TensorRT's bindings on JetPack 4.6 are built for the system Python 3.6 only —
  mutually exclusive. Model export and COCO evaluation therefore run on the
  laptop; the Nano runs TensorRT, PyCUDA, NumPy and JetPack's OpenCV, with
  postprocessing written by hand.

---

## Roadmap

Planned extensions are tracked in [`PROJECT_PLAN.md`](PROJECT_PLAN.md) §10:
a cross-hardware model comparison matrix, a transformer feasibility study on
Maxwell, and on-device temporal action recognition.

---

## License

MIT

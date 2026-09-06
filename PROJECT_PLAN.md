# EdgeVision — Project Plan

Production edge AI deployment on an NVIDIA Jetson Nano 2GB: take a
pretrained object detector, optimise it for constrained hardware, deploy it
reproducibly, and measure what it actually costs.

**Status:** Parts 0–5 complete. TensorRT FP16 measured **1.84× faster** than
TorchScript FP32 for a **0.03% mAP loss** on the full COCO validation set.
Part 6 (Docker) is next, and applications go out after it.
**Owner:** Ravi Teja Gudupu
**Started:** August 2026

> Revised across several rounds of review. Where earlier drafts disagreed, the
> reasoning behind each resolution is recorded in §11 Decision log.

---

## 1. Core question

> How does optimising and deploying the same detector through different
> inference runtimes affect **accuracy, latency, throughput, memory, thermal
> behaviour and power** on resource-constrained edge hardware?

The optimisation path:

```
YOLOv5n (pretrained, COCO) → PyTorch → ONNX → TensorRT FP32 → TensorRT FP16
```

The strongest claim this project supports is not "I ran YOLO on a Jetson Nano."
It is: *I built a reproducible edge deployment pipeline and quantitatively
evaluated how runtime and precision affect accuracy, latency, throughput, memory,
thermal behaviour and power on a 2GB Jetson Nano.*

### Minimum viable v1 — non-negotiable core

If everything optional slips, this is still a complete, publishable project:

```
COCO subset → YOLOv5n → PyTorch → ONNX
    → TensorRT FP32 → TensorRT FP16
    → benchmarking → Docker → results published
```

Nothing outside that chain is allowed to block shipping. Flask, MLflow, Grafana,
ROS2 and the thermal study are all improvements layered on a project that is
already finished.

### Success criteria for v1.0

- [x] Environment documented (Part 0 report committed)
- [x] PyTorch baseline measured **on the Nano** — via TorchScript, since
      Ultralytics cannot run on Python 3.6
- [x] ONNX export with numerical parity verified — 6 tests passing
- [x] TensorRT FP32 and FP16 engines built on-device and measured
- [x] **Engine throughput and end-to-end throughput reported separately** —
      19.75 vs 12.60 FPS
- [x] Accuracy verified — full COCO val2017, 5,000 images, not a subset
- [x] **ΔmAP reported against the FP32 baseline** — −0.0001 mAP@50-95
- [ ] Bare-metal vs Docker overhead measured and reported ← **Part 6**
- [ ] Sustained (not peak) throughput measured under thermal load
- [ ] 5W vs 10W performance-per-watt comparison
- [x] Experiments tracked; every result traceable to a commit
- [ ] CI green, with an accuracy regression gate
- [x] README a stranger understands in 90 seconds

### Explicit non-goals for v1.0

No training. No INT8 (see §2). No multi-camera, tracking, SLAM, segmentation,
transformers, VLMs, RAG, agents or action recognition. No self-hosted CI runner
on the Nano. No MLOps infrastructure hosted on the Nano.

**Do fewer things, but measure them properly.**

---

## 2. Hardware and constraints

| Item | Detail |
|---|---|
| Board | Jetson Nano 2GB Developer Kit (P3541) |
| Compute | 128-core Maxwell, SM 5.3, **no tensor cores** |
| Memory | 2 GB shared LPDDR4 — roughly 1.4 GB usable |
| JetPack | 4.6.x (L4T 32.7.x) — the 2GB board cannot run JetPack 5 or 6 |
| Pinned by JetPack | CUDA 10.2, TensorRT 8.2, OpenCV 4.1.1, **Python 3.6.9**, Ubuntu 18.04 |
| Storage | SanDisk Extreme 128 GB microSDXC (U3 / V30) |
| Power | Raspberry Pi 4 official PSU, 5.1V / 3A USB-C |
| Cooling | 5V 40mm fan on the stock heatsink |
| Network | Ethernet to laptop via Windows ICS (192.168.137.x) |
| Camera | Raspberry Pi Camera Module v2 (IMX219) — **out of stock**; video file used |
| Dev machine | Laptop, RTX 3050 4 GB — GPU functional; fan disconnected because it is noisy, reconnect for sustained GPU work and monitor temperature |

### The Python environment split — read this before installing anything

This is the constraint that shapes the whole project.

**Ultralytics requires Python 3.8+. TensorRT's Python bindings on JetPack 4.6 are
built for the system Python 3.6 only.** These are mutually exclusive. Installing
Python 3.8 in a venv to get Ultralytics leaves you with `No module named
tensorrt`, and often `Illegal instruction (core dumped)` on torch import.

TensorRT is the point of the project, so the Nano stays on Python 3.6 and
Ultralytics never touches it.

| | Ultralytics | TensorRT bindings | NVIDIA torch |
|---|---|---|---|
| Nano, Python 3.6 (system) | ✗ | ✓ | ✓ |
| Nano, Python 3.8 (venv) | ✓ | ✗ | fragile |

**Work split by machine:**

| Laptop (Python 3.10) | Nano (Python 3.6) |
|---|---|
| YOLOv5 repo, model handling | TensorRT bindings |
| ONNX export | PyCUDA |
| COCO evaluation (`pycocotools`) | NumPy |
| MLflow server, Prometheus, Grafana | JetPack OpenCV 4.1.1 |
| Plot and report generation | torch 1.10 (NVIDIA wheel) + torchvision from source |

**Consequence: postprocessing is written by hand.** The TensorRT path does not go
through YOLOv5's Python code, so decoding the raw output tensor and running NMS
in NumPy is your job — roughly half a day. This is good work, not a workaround:
understanding the model's raw output layout is exactly what separates you from
someone who only ever calls `model.predict()`, and it is a strong interview
answer.

**Model choice: YOLOv5n, not YOLOv8n.** YOLOv5 runs on the Nano under Python 3.6
with NVIDIA's torch wheel; Ultralytics/YOLOv8 does not. Pin a **v6.0–v6.1 era
tag** — later tags moved the Python floor to 3.7. Check `requirements.txt` on the
tag before cloning, and expect to pin pandas, matplotlib, scipy and seaborn
versions by hand, since modern releases of those dropped 3.6.

### INT8 — excluded, with the reason stated precisely

INT8 is excluded from this project because TensorRT's INT8 optimisation path
requires compute capability 6.1 or higher, and this board's Maxwell GPU is
SM 5.3 with no tensor cores. `builder.platform_has_fast_int8` returns False, and
INT8 calibrators fail at engine build time rather than degrading gracefully.
Among Jetson platforms, INT8 support begins with Xavier. **FP16 is therefore the
primary reduced-precision target.**

State it this precisely in the README's limitations section — the
compute-capability threshold is specific and checkable, which makes it read as a
finding rather than an excuse. An embedded interviewer will recognise it
immediately.

Do not put INT8 on the CV for this project.

### Constraints that will actually bite

- **Python 3.6.9** is the single biggest source of friction. Check every
  dependency's version floor first. Ultralytics, FastAPI, modern `transformers`
  and much else require 3.7+ or 3.8+.
- **Do not install Python 3.8 on the Nano.** It breaks the TensorRT bindings and
  destabilises torch. See the environment split above.
- **PyTorch must come from NVIDIA's aarch64 wheel** for JetPack 4.6 (torch 1.10,
  Python 3.6). PyPI has no Jetson build.
- **torchvision must be built from source** against that exact torch version.
  This is the most painful install in the project: 1–2 hours on a 2 GB board with
  real OOM risk. Timeboxed — see Part 4.
- **Do not `pip install opencv-python`.** JetPack ships OpenCV 4.1.1 with
  GStreamer; the pip wheel has neither CUDA nor GStreamer and shadows it.
- **TensorRT engines are hardware- and version-specific.** Export ONNX anywhere;
  build the engine *on the Nano*.
- **Docker images must be built on the Nano.** arm64 CUDA base images cannot be
  built on an x86 laptop.
- **No unclean shutdowns.** `sudo shutdown -h now`, wait ten seconds, then pull
  power. The green LED is a power-present indicator and never goes out.
- **Build one Docker image at a time.** Layer extraction on 2 GB RAM is where OOM
  kills happen.
- **Build the deployment container TensorRT-only — no torch inside.** A container
  with torch is several GB and slow to build on-device. Excluding it keeps the
  image manageable and is more honest as a deployment artifact.
- **The engine build itself may OOM at 640×640.** Set `workspace_mb` low, stop
  everything else first, keep swap enabled. Fall back to 416×416 if needed and
  record the resolution in the results.

---

## 3. Architecture

Heavy infrastructure lives on the laptop. The Nano's job is inference.

```
              DEVELOPMENT LAPTOP
   ┌────────────────────────────────────┐
   │  model export · COCO evaluation    │
   │  MLflow tracking server            │
   │  Prometheus · Grafana              │
   │  GitHub Actions (hosted runners)   │
   └──────────────┬─────────────────────┘
                  │  git · scp · metrics scrape
                  ▼
        ┌─────────────────────┐
        │    JETSON NANO      │
        │                     │
        │  Docker container   │
        │   ├ capture         │
        │   ├ preprocess      │
        │   ├ TensorRT FP16   │
        │   ├ postprocess/NMS │
        │   └ metrics export  │
        │                     │
        │  systemd watchdog   │
        └──────────┬──────────┘
                   │
            video file / IMX219
```

### The inner loop, and where time actually goes

```
capture → preprocess → inference → postprocess (NMS) → sink
```

Preprocessing and NMS run on the Nano's weak ARM cores and can easily exceed
engine time. **Instrument every stage separately.** "Inference: 20 ms" is not
actionable; "preprocess 30 / inference 20 / NMS 15" tells you where to spend the
next hour.

This produces the project's two headline throughput numbers, which must always be
reported as a pair:

- **Engine throughput** — inference only. What TensorRT achieved.
- **End-to-end throughput** — capture, preprocess, inference, postprocess, NMS.
  What the application actually delivers.

A TensorRT engine can hit 45 FPS while the complete application runs at 18. Most
published edge benchmarks quietly report only the first number. Reporting both,
and explaining the gap, is one of the strongest signals in this project — it is
the mistake nearly every edge project makes, and catching it is a very good
interview answer.

---

## 4. Dataset and evaluation protocol

COCO, with a **fixed evaluation subset** (500 val images) held constant across
every runtime. No training — a pretrained **YOLOv5n** checkpoint throughout (see
§2 for why not YOLOv8n).

The same checkpoint, same subset, same input resolution for every configuration,
or the comparison is meaningless.

Metrics: mAP@50, mAP@50-95, precision, recall — via `pycocotools`.

A short self-recorded video (30–60 s, fixed resolution and frame rate) is the
input for latency and thermal runs. Own footage avoids licensing questions when
the demo GIF goes public.

**The camera is a demonstration, never a benchmark input.** Live capture FPS
varies with lighting, scene content and capture timing, which makes runtime
comparisons unfair and irreproducible. All quantitative results come from the
fixed COCO subset (accuracy) and the fixed video (latency, throughput, thermal).
The camera exists for the demo GIF and to prove the CSI path works.

---

## 5. Repository structure

```
edgevision/
├── app/
│   ├── capture.py
│   ├── preprocess.py
│   ├── infer_torch.py
│   ├── infer_onnx.py
│   ├── infer_trt.py
│   ├── postprocess.py
│   └── sink.py                  # stdout/JSONL now, ROS2 later
├── models/
│   ├── export_to_onnx.py
│   ├── check_onnx.py            # validate opset + shapes before transfer
│   ├── build_trt_engine.py      # run ON the Nano
│   └── README.md                # model, opset, rationale
├── benchmarks/
│   ├── benchmark.py             # written at Part 2, reused unchanged
│   ├── run_matrix.sh
│   └── make_report.py           # CSVs → README tables + plots
├── evaluation/
│   └── coco_eval.py
├── monitoring/
│   ├── exporter.py
│   ├── prometheus.yml
│   └── grafana_dashboard.json
├── systemd/
│   └── edgevision.service
├── docker/
│   ├── Dockerfile.jetson
│   └── docker-compose.monitoring.yml
├── tests/
│   ├── test_postprocess.py
│   ├── test_parity.py
│   └── test_regression.py       # accuracy gate
├── configs/
│   └── params.yaml
├── docs/
│   ├── environment_report.md    # Part 0 output
│   └── SETUP_LOG.md             # every command, its output, and why
├── results/                     # a deliverable — NOT gitignored
│   ├── speed.csv
│   ├── accuracy.csv
│   ├── raw/                     # per-frame latency arrays
│   └── plots/
├── data/                        # test video (gitignored)
├── assets/                      # demo GIF, diagrams
├── requirements/
│   ├── laptop.txt
│   └── jetson.txt               # Python 3.6 — diverges deliberately
├── .github/workflows/ci.yml
├── PROJECT_PLAN.md
├── NOTES.md
└── README.md
```

No directory exists to make the repo look bigger. Every one has a purpose.

**Why `app/`, `benchmarks/` and `evaluation/` are separate** rather than one
`src/`: they have different lifetimes. `app/` is what the Docker container
deploys. `benchmarks/` is a tool run *against* it. `evaluation/` only ever runs
on the laptop, where the COCO data lives. Splitting them means the deployment
image copies `app/` alone — which matters on a 2 GB board.

---

## 6. Build plan

Parts 0–3 need no hardware beyond what you have. Do not let camera or fan
shipping block anything.

### Part 0 — Environment validation report ✅
*~1 h · Nano · **done***

Record and commit to `docs/environment_report.md`: JetPack/L4T, CUDA, TensorRT,
Python, RAM, GPU, CPU, storage, idle temperature, power mode.

```bash
cat /etc/nv_tegra_release
nvcc --version
dpkg -l | grep -i tensorrt
python3 --version
free -h && df -h /
sudo nvpmodel -q
jtop
```

Hardening (done): desktop disabled, `noatime`, journald capped, `jetson-stats`
installed, Docker default runtime set to nvidia, bootloader packages held, base
card image backed up.

**Deliverable:** committed environment report. The Nano runs an old stack — this
document is what makes every later number interpretable.

---

### Part 0.5 — Day-zero spike: prove the TensorRT path exists
*~2 h · Nano · **do this before anything else***

Nothing but proving the critical path works. No repo, no structure, no polish.

1. `python3 -c "import tensorrt; print(tensorrt.__version__)"` in **system Python 3.6**
2. `sudo pip3 install pycuda` — confirm it imports
3. Download any pretrained ONNX detector on the laptop, `scp` it over
4. `/usr/src/tensorrt/bin/trtexec --onnx=model.onnx --fp16 --saveEngine=test.engine`
5. Load the engine in Python, run one inference, print the raw output shape

If all five work, the plan is sound and everything after is execution.

If step 1 or 4 fails, you have found the project-killing problem on day zero
instead of day five — and there is still time to change course.

**Deliverable:** five commands that worked, pasted into `NOTES.md`. Also note the
engine build time and peak memory; those are your first real data points.

---

### Part 1 — Repo scaffold and input source
*~2 h · laptop*

Repo, structure above, `configs/params.yaml`, split requirements files, test
video recorded and normalised. Commit `PROJECT_PLAN.md` and the README
skeleton. Start `NOTES.md`.

**Deliverable:** repo initialised (private for now), video plays via OpenCV.

---

### Part 2 — Baseline and benchmark harness
*~5 h · laptop*

Pretrained YOLO running on the video in PyTorch. Then write `benchmark.py`
**before optimising anything**.

The harness must separate:

- **Cold-start latency** — model load and engine deserialisation, measured once
- **Steady-state latency** — after N warmup frames, the number that goes in the table

and must instrument each pipeline stage, not just inference.

and must report **both throughput numbers**:

- `fps_engine` — inference only
- `fps_end_to_end` — full pipeline

Outputs per run: p50 / p95 / mean latency (engine and end-to-end), both FPS
figures, FPS sustained over the final 60 s, per-stage timings, peak RAM, CPU and
GPU utilisation, max temperature, throttled flag.

Every later stage reuses this harness unchanged. That is what makes the final
comparison trustworthy.

**Deliverable:** `results/speed.csv` with an FP32 baseline.
**Note:** the laptop baseline is an optional reference point, not a required
result — the Nano is the benchmark target. If you run laptop rows, reconnect the
GPU fan first and log temperature so the numbers are comparable.

---

### Part 3 — ONNX export and parity
*~3 h · laptop*

Fixed input shape, opset 13 (TensorRT 8.2 supports roughly 13–14). Verify ONNX
Runtime output matches PyTorch within tolerance before proceeding — if accuracy
drifts later you need to know whether export or engine build caused it.

Establish a clean intermediate representation; don't optimise aggressively yet.

**Deliverable:** `model.onnx`, `tests/test_parity.py` passing, ONNX row in results.

---

### Part 4 — Port the baseline to the Nano ✅
*budgeted 4 h, took ~10 min*

> **What actually happened.** The plan timeboxed the torch install at two hours
> with Option B as a fallback, because torchvision compiling from source on a 2 GB
> board was the single riskiest step. **torchvision turned out not to be needed at
> all** — YOLOv5n is pure convolutions, so the TorchScript graph contains no
> torchvision operators and `torch.jit.load` works with torch alone.
>
> The saving came from choosing **TorchScript over cloning YOLOv5**, which
> sidestepped the entire Python 3.6 dependency-pinning exercise. See §11.

**Option A (chosen): on-device PyTorch baseline with YOLOv5n.**

Install NVIDIA's torch 1.10 aarch64 wheel, then build torchvision from source
against it. Clone YOLOv5 at a v6.0–v6.1 tag and pin its dependencies by hand.
`scp` the video across, run the **same** `benchmark.py`.

This gives the headline story on a single device: PyTorch at a few FPS, TensorRT
FP16 at many, same board, same video, same harness.

> **Timebox: 2 hours for torch + torchvision + YOLOv5 dependencies.**
>
> If torchvision has not compiled, or the dependency chain walls out, stop and
> fall back to **Option B**: PyTorch baseline on the laptop only, TensorRT FP32
> becomes the on-device baseline. Record it in the limitations section —
> *"PyTorch baseline measured on the development machine; the Nano's Python 3.6
> environment made an on-device PyTorch baseline impractical."*
>
> That is a legitimate finding about the platform, not a failure. Do not spend a
> day on it.

Lock state before measuring and record it:

```bash
sudo nvpmodel -m 0     # 10W
sudo jetson_clocks     # lock clocks — DVFS otherwise makes numbers drift
```

**Deliverable:** the honest, slow baseline. This number is what makes Part 5
impressive.

---

### Part 5 — TensorRT FP32 then FP16
*~6 h · the core of the project*

Prove the path with `trtexec`, then `models/build_trt_engine.py`. Build on the
Nano. FP32 first and validate, then FP16.

In `infer_trt.py`: allocate device buffers once outside the frame loop, explicit
host↔device transfers.

**`postprocess.py` is written from scratch in NumPy** — box decoding from the raw
output tensor, confidence filtering, and NMS. YOLOv5's Python code is not
available on the Nano, so this is not optional. Budget half a day. Validate it by
comparing detections against the laptop's YOLOv5 output on the same frames before
trusting any accuracy number.

For each of the four configurations record mAP@50, mAP@50-95, precision, recall,
engine FPS, end-to-end FPS, p50, p95, RAM, GPU utilisation, temperature — and the
**deltas against the FP32 baseline** for all three accuracy metrics.

The goal is not maximum FPS. The goal is the trade-off sentence:

> FP16 increased end-to-end throughput by X% while mAP@50-95 changed by only
> Y points.

That single sentence is a stronger engineering result than any speed number
alone, and it is what the README leads with.

**Deliverable:** the four-row runtime table with deltas. This is the README
headline.

---

### Part 6 — Dockerised deployment
*~4 h*

`Dockerfile.jetson` on `nvcr.io/nvidia/l4t-base:r32.7.1`, built on the Nano.
Model and video mounted as volumes. Run with `--runtime nvidia`.

Then measure **bare metal vs container** on FPS, p50, p95, RAM, temperature, and
report the overhead — including if it is zero.

This is a result, not a sanity check. It shows containerisation was introduced
without assuming it was free.

**Deliverable:** container reproduces Part 5 numbers; overhead table.

> **Applications go out here.** Repo public, README carrying the runtime table
> and the Docker comparison. Parts 7–10 land while applications are in flight.

---

### Part 7 — Thermal and power benchmarking
*~5 h · a headline result, not an afterthought*

Sustained performance, not peak. Long runs (10 min minimum) monitoring
temperature, FPS, latency, GPU/CPU utilisation and clock behaviour.

Then the power-mode comparison:

| Mode | FPS | p50 | p95 | Temp | Perf/W |
|---|---|---|---|---|---|
| 5W (`nvpmodel -m 1`) | | | | | |
| 10W (`nvpmodel -m 0`) | | | | | |

**The key engineering question: what is the best performance-per-watt
configuration?** For an embedded role this is the question that matters most, and
almost no published Jetson benchmark answers it — most report a 30-second peak
and ignore throttling entirely.

Also run with and without the fan to quantify what active cooling buys.

**Deliverable:** FPS-vs-temperature plot, power-mode table, perf/W conclusion.

---

### Part 8 — Inference service
*~4 h*

A simple interface around inference. `app/sink.py` writing JSONL plus a
lightweight HTTP endpoint if the environment allows — **Flask, not FastAPI**,
which needs Python 3.7+.

**ROS2 is an optional extension, not a v1 requirement.** ROS2 Foxy targets Ubuntu
20.04 and JetPack 4.6 is 18.04; the workable route is a separate 20.04 arm64
container, which is real work on a 2 GB board. Do not let the interface layer
become the project. The core deliverable is the TensorRT inference service.

If implemented later: `/camera/image → /perception → /detections`.

**Deliverable:** detections leave the process in a consumable form.

---

### Part 9 — Experiment tracking and monitoring
*~5 h*

**MLflow on the laptop**, never on the Nano. Log over the Ethernet link
(`MLFLOW_TRACKING_URI=http://192.168.137.1:5000`). If the client is awkward on
Python 3.6, fall back to: Nano writes CSVs, laptop script ingests them.

Params: model, version, runtime, precision, input resolution, power mode,
docker/native. Metrics: mAP, precision, recall, FPS, p50, p95, RAM, temperature,
power.

**Prometheus and Grafana on the laptop**, scraping a lightweight exporter on the
Nano (`inference_fps`, `inference_latency`, `cpu_usage`, `gpu_usage`, `ram_usage`,
`temperature`).

**systemd** unit with `Restart=always`, plus tested failure paths:

| Scenario | Expected | Verified |
|---|---|---|
| Process killed | restarts within _pending_ s | ☐ |
| Video source removed | logged, graceful exit, backoff | ☐ |
| One hour continuous | no memory growth, thermal steady state | ☐ |
| Dropped to 5W | degraded FPS, no crash | ☐ |

**Deliverable:** Grafana screenshot, MLflow comparison view, documented recovery.

---

### Part 10 — CI/CD, README, ship
*~4 h*

GitHub Actions on **hosted runners** — lint, unit tests, integration tests,
Docker build for the x86 components. **No self-hosted runner on the 2GB Nano**;
its job is inference. An arm64 workflow can come later.

`tests/test_regression.py` fails CI if FP16 mAP drops more than a configured
threshold below FP32. Optimisation that quietly costs accuracy is a bug.

**This gate runs no inference.** It reads `results/accuracy.csv`, compares two
rows, and asserts a threshold — roughly ten lines, milliseconds to run, no GPU
and no Jetson required. It is not an attempt to reproduce TensorRT inference
inside GitHub Actions, which would indeed be disproportionate. Keeping it in v1
costs nothing once the CSVs exist, and gating on results rather than merely
recording them is a genuine differentiator.

README story: problem → objective → hardware → model → optimisation path →
deployment → results → **lessons learned** (which runtime won, how much accuracy
moved, Docker overhead, thermal behaviour, power trade-off, memory limits,
deployment obstacles).

Tag `v1.0`. Make the repo public. Record a short demo video.

---

## 7. Results storage and provenance

Two append-only CSVs in `results/`, both **committed to git**, joined on
`config_hash`:

**`speed.csv`** — one row per benchmark execution (speed varies with thermal
state, so a config may have several rows):

```
run_id, config_hash, timestamp, git_commit,
model, runtime, precision, device, input_res, batch_size,
power_mode, clocks_locked, fan,
warmup_frames, n_frames, duration_s,
cold_start_ms,
engine_p50_ms, engine_p95_ms, engine_mean_ms,
e2e_p50_ms, e2e_p95_ms, e2e_mean_ms,
preprocess_ms, inference_ms, postprocess_ms,
fps_engine, fps_end_to_end, fps_sustained_last_60s,
peak_mem_mb, cpu_util_pct, gpu_util_pct,
gpu_temp_max_c, throttled, notes
```

**`accuracy.csv`** — one row per config (accuracy is deterministic; evaluate once):

```
config_hash, model, runtime, precision, device, input_res,
n_images, mAP50, mAP50_95, precision, recall,
delta_mAP50, delta_mAP50_95, delta_precision, delta_recall
```

Per-frame latency arrays go to `results/raw/{run_id}.npy` for percentiles and
thermal plots.

**Every row carries the git commit SHA and a config hash.** That is what makes
any number in the README traceable to the exact code and configuration that
produced it — the cheapest reproducibility measure available, and the thing that
separates a results table from decoration.

`benchmark.py` appends; never edit the CSVs by hand. `make_report.py` regenerates
the README tables and plots from them, so they are never stale or
hand-transcribed wrong.

Graduate to MLflow at Part 9 — with this schema the ingest script is twenty lines.

---

## 8. Final benchmark tables

**Runtime comparison** — all rows on the Jetson Nano, 10W, clocks locked

| Runtime | Precision | Engine FPS | E2E FPS | p95 (e2e) | RAM | mAP@50-95 | ΔmAP | ΔP | ΔR |
|---|---|---|---|---|---|---|---|---|---|
| PyTorch | FP32 | | | | | | — | — | — |
| ONNX Runtime | FP32 | | | | | | | | |
| TensorRT | FP32 | | | | | | | | |
| TensorRT | FP16 | | | | | | | | |

**Pipeline breakdown** — where the time actually goes

| Stage | ms | % of end-to-end |
|---|---|---|
| Capture | | |
| Preprocess | | |
| Inference (engine) | | |
| Postprocess + NMS | | |

**Deployment overhead**

| Deployment | FPS | p95 | RAM |
|---|---|---|---|
| Bare metal | | | |
| Docker | | | |

**Power modes**

| Mode | FPS | Temp | Perf/W |
|---|---|---|---|
| 5W | | | |
| 10W | | | |

Plus the FPS-vs-time thermal curve. Cross-device rows (laptop CPU / laptop GPU)
are an optional reference, not a required result — the Nano is the benchmark
target.

---

## 9. Timeline

| Parts | Effort | Needs hardware |
|---|---|---|
| 0 | ~1 h | done |
| 0.5 (spike) | ~2 h | yes — do first |
| 1–3 | ~10 h | no — laptop only |
| 4–6 | ~16 h | yes |
| 7–10 | ~18 h | yes |
| **Total** | **~47 h** | ≈ 8–10 working days |

Both source plans underestimated this, and v3 still did. Ten days is achievable
only if Parts 1–3 happen on the laptop while hardware work proceeds in parallel,
Part 7 gets its own day, and the Part 4 timebox is respected.

**The failure mode is not bad work — it is a repo that stays private for two
months because it is never quite finished.** Ship at Part 6.

---

## 10. Deferred

Tracked, not acted on until v1.0 ships and applications are out.

- **Cross-hardware matrix** — three models × three devices × runtimes, with mAP
  alongside latency, Pareto frontier plotted
- **Object tracking** — ByteTrack or similar, giving stable object IDs
- **ROS2 perception pipeline** — the Part 8 optional route, properly built
- **Second edge device** — Orin Nano or Raspberry Pi for comparison
- **Transformer feasibility** — MobileViT-XXS as primary, RT-DETR-R18 at 320×320
  as a documented boundary test. Note TensorRT 8.2 predates native LayerNorm
  support (added in 8.6) and Maxwell has no tensor cores, so expect CNNs to win
  decisively — that is itself the finding
- **Action recognition** — MoViNet-A0 or X3D-XS on-device; or a cascade sending
  keyframes to a laptop VLM. No VLM runs on the Nano: ~3.7 GB at FP16 against
  ~1.4 GB usable, and modern `transformers` needs Python 3.8+
- **RAG / agents** — a separate repo, not an EdgeVision extension

**Not deferred — excluded.** INT8 (§2), self-hosted CI on the Nano, MLOps
infrastructure hosted on the Nano, multi-camera, SLAM, large segmentation models.

---

## 11. Decision log

| Decision | Reason |
|---|---|
| TorchScript for the on-device PyTorch baseline | Ultralytics needs Python 3.8+; TensorRT bindings are 3.6-only. TorchScript is self-contained, needs only torch, and is what production deployment looks like anyway |
| Full COCO val2017, not a 500-image subset | At 500 images, differences below ~1 mAP point are indistinguishable from sampling noise — and the expected FP16 difference was far smaller. Sample size must match the effect size |
| Evaluate at conf 0.001, not the runtime's 0.25 | mAP integrates the PR curve; high-recall points come only from low-confidence detections. 0.001 is the COCO convention every published figure uses |
| Build a TensorRT **FP32** engine as a control | Without it, 1.84× mixes runtime and precision. With it: 1.31× runtime × 1.40× precision |
| Score mAP on the laptop, not the Nano | `pycocotools` compiles a C extension; keeping it off the board avoids another build, and one implementation scores every runtime |
| Engine selected by precision key, with an explicit raise | A single hardcoded path silently loaded FP16 for a run labelled FP32. A missing engine must stop the run, not substitute a different one |
| Part 7 measures with the fan governor **active** | Manual `target_pwm` writes do not stick — a kernel-level governor engages around 50 °C. "Sustained performance under stock thermal management" is the more honest claim |
| YOLOv5n, not YOLOv8n | Ultralytics needs Python 3.8+; TensorRT bindings on JetPack 4.6 are 3.6-only. Mutually exclusive, and TensorRT wins. YOLOv5 runs under 3.6 with NVIDIA's torch wheel |
| Ultralytics stays on the laptop | Model handling, ONNX export and COCO evaluation don't need to run on the Nano |
| Postprocessing written by hand in NumPy | The TensorRT path bypasses YOLOv5's Python code. Not a workaround — decoding raw output is a genuine skill signal |
| Part 0.5 spike before any repo work | Five commands prove the TensorRT path exists. Finding a blocker on day zero beats finding it on day five |
| Torch install timeboxed to 2 hours | torchvision from source on 2 GB is the project's riskiest install. Option B fallback is a legitimate finding, not a failure |
| Deployment container excludes torch | Multi-GB image otherwise; TensorRT-only is smaller and more honest as an artifact |
| INT8 excluded, reason stated precisely | TensorRT INT8 needs CC ≥ 6.1; board is SM 5.3. Stated as a specific constraint rather than an absolute — precision reads as a finding, vagueness reads as an excuse |
| Engine FPS and end-to-end FPS both reported | An engine can hit 45 FPS while the application delivers 18. Reporting only the first is the standard omission in edge benchmarks |
| Nano mandatory, laptop optional | The project's claim is about constrained hardware. Laptop rows are context, not deliverables |
| Named minimum viable core | Protects the timeline — if MLflow, Grafana, Flask or ROS2 slips, the project is still complete and publishable |
| Δ on mAP, precision and recall | "X% faster for Y points of mAP" is a stronger result than a speed table and an accuracy table side by side |
| Camera is demo only, never benchmark input | Live capture FPS varies with scene and lighting; fixed inputs make comparisons fair |
| Accuracy gate stays in v1 | It reads two CSV rows and asserts a threshold — no GPU, no inference, no Jetson. Gating on results rather than only recording them is a real differentiator |
| Laptop GPU fan reconnectable | Disconnected for noise, not broken. Reconnect for sustained GPU work; laptop benchmarks are usable when needed |
| ROS2 demoted to optional | JetPack 4.6 is Ubuntu 18.04; Foxy needs 20.04. Container workaround is real work and shouldn't block v1 |
| Flask not FastAPI on-device | FastAPI requires Python 3.7+; Nano is pinned to 3.6.9 |
| MLflow included in v1 | The runtime sweep genuinely is an experiment matrix — legitimate use, not keyword decoration |
| MLflow / Prometheus / Grafana on laptop | The Nano's job is inference; hosting infra on 2 GB corrupts the measurements |
| CI on hosted runners only | A self-hosted runner on a 2 GB board competes with the workload it's testing |
| Video file as primary input | Camera out of stock, and identical frames make benchmarks reproducible |
| Ship at Part 6 | Repo public early beats repo perfect late |
| Results committed to git with SHA + config hash | Provenance is what makes the table defensible |
| Laptop-first sequencing for Parts 1–3 | Neither source plan did this; it removes the hardware dependency from the critical path |

---

## 12. Open questions

**Answered so far:**

- ~~Does torchvision compile on the Nano within the timebox?~~ **Not needed.**
- ~~Does the hand-written NumPy postprocessing reproduce YOLOv5's detections?~~
  **Yes** — 99.2% agreement, 0.546 px mean box error, and mAP within 0.9 points of
  the published figure.
- ~~How large is the gap between engine FPS and end-to-end FPS?~~ **36%** —
  19.75 vs 12.60 FPS. CPU stages are 35% of the frame.
- ~~Does preprocessing dominate on ARM?~~ **No** — the prediction failed. CPU
  stages became a *smaller* share (40% → 35%) because inference scaled 3.2× while
  they scaled 2.1–2.5×.

**Still open:**

- [ ] Does torchvision compile on the Nano within the 2-hour timebox, or does the
      project fall back to Option B?
- [ ] Does the hand-written NumPy postprocessing reproduce YOLOv5's detections on
      the same frames? (Validate before trusting any accuracy number.)
- [ ] How large is the gap between engine FPS and end-to-end FPS? (This is the
      most interesting number in the project. If it is large, the story becomes
      "optimising the model stopped mattering — here is what did.")
- [ ] Does preprocessing or NMS dominate engine time on the Nano? (Answered at
      Part 2 by stage-level instrumentation — likely yes, and it changes what to
      optimise next.)
- [ ] Measurable Docker overhead, or none?
- [ ] Does the fan change the sustained-FPS conclusion, or only peak temperature?
- [ ] Is 5W or 10W better on performance-per-watt?

---

## 13. Skills this demonstrates

Connected by one real system rather than listed as unrelated technologies.

**Computer vision** — object detection, COCO evaluation, mAP/precision/recall,
preprocessing pipelines
**Edge AI** — Jetson, CUDA, TensorRT, ONNX, FP16, latency and throughput
optimisation
**Embedded engineering** — memory constraints, power modes, thermal behaviour,
sustained inference, performance-per-watt
**Software engineering** — Python, git, testing, containerisation, service design
**MLOps** — experiment tracking, CI/CD, reproducibility, monitoring, regression
gating

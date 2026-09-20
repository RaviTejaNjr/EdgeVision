# Results

This directory contains the benchmark, accuracy and thermal results used in the main project README. Raw timing data is kept where it is small enough to be useful for reproducing plots and checking individual runs.

## Files

| File | Purpose |
|---|---|
| `speed.csv` | Main benchmark results. Includes `host_profile` and run metadata. |
| `speed_exploratory.csv` | Earlier laptop runs collected before `host_profile` was added. |
| `accuracy.csv` | COCO mAP results for each model/runtime configuration. |
| `raw/*.npz` | Per-frame timing arrays, keyed by `run_id`. |
| `plots/thermal_*.png` | Thermal plots generated from sustained runs. |
| `detections_*.json` | COCO detections generated during evaluation. Gitignored because each file is about 49 MB and can be regenerated. |

## Speed results

Use `speed.csv` for comparisons.

`speed_exploratory.csv` is kept because it records an issue found early in the project: identical code on the same laptop produced very different timings when the Windows power state changed. Those runs are useful as a record, but they are not clean baselines.

### Run metadata

The main fields used to identify a result are:

| Column | Meaning |
|---|---|
| `run_id` | Unique ID for one benchmark execution. Also used for the matching file in `raw/`. |
| `config_hash` | Stable hash of the model/runtime configuration. Used to join speed and accuracy results. |
| `git_commit` | Git SHA for the code that produced the run. `-dirty` means there were uncommitted changes. |
| `host_profile` | Recorded host power/profile state for that run. |
| `notes` | Free-form run notes. |

Several speed rows can share the same `config_hash`. That is expected. The hash identifies the model configuration, while `host_profile` and `notes` record the machine state used for a particular benchmark.

### Timing fields

Two FPS values are kept:

- `fps_engine`: inference only
- `fps_end_to_end`: capture + preprocess + inference + postprocess/NMS

`cold_start_ms` is measured once and is not included in steady-state FPS.

`fps_sustained_last_60s` is only populated for long runs. It is the throughput over the final 60 seconds and is used for the thermal tests.

p50 and p95 are kept alongside the mean so that latency spread is visible instead of being hidden by one average value.

### Memory field

`peak_mem_mb` is process RSS, not GPU memory.

For example, one laptop run reported 5115 MB RSS while `nvidia-smi` showed about 1212 MiB of GPU memory in use. Do not read `peak_mem_mb` as VRAM usage.

`throttled` is currently filled in manually after inspecting the run.

## Jetson runtime comparison

Headline Jetson benchmarks were run at 10 W (`nvpmodel -m 0`) with clocks locked using `jetson_clocks`, fan off, batch size 1 and 640x640 input. Each configuration was repeated three times.

| Runtime | Precision | Inference | End-to-end FPS | Peak RSS |
|---|---|---:|---:|---:|
| TorchScript | FP32 | 93.15 ms | 8.21 | - |
| TensorRT | FP32 | 70.89 ms | 10.04 | 1414 MB |
| TensorRT | FP16 | 50.64 ms | 12.60 | 1086 MB |

From these runs:

- TorchScript FP32 -> TensorRT FP32: 1.31x inference speedup
- TensorRT FP32 -> TensorRT FP16: 1.40x
- TorchScript FP32 -> TensorRT FP16: 1.84x inference speedup
- End-to-end throughput improved from 8.21 FPS to 12.60 FPS
- TensorRT FP16 used 328 MB less peak RSS than TensorRT FP32


## Accuracy

`accuracy.csv` stores one row per model/runtime configuration. Results are joined to `speed.csv` through `config_hash`.

### COCO evaluation setup

| Setting | Value |
|---|---|
| Dataset | COCO val2017, full 5,000 images / 36,781 annotations |
| Confidence threshold | 0.001 |
| NMS IoU | 0.45 |
| Input | square 640x640 letterbox |
| Scoring | `pycocotools` |

The low confidence threshold is used for evaluation so that low-confidence predictions are still available when the precision-recall curve is built. The runtime demo uses a higher threshold because it serves a different purpose.

The detection export does not apply an extra project-specific per-image cap before COCO scoring. The evaluator handles its own `maxDets` settings.

### COCO results

| Runtime | Precision | mAP@50-95 | mAP@50 | mAP@75 | Detections |
|---|---|---:|---:|---:|---:|
| TorchScript | FP32 | 0.3343 | 0.5005 | 0.3529 | 530,418 |
| TensorRT | FP32 | 0.3343 | 0.5005 | 0.3529 | 530,417 |
| TensorRT | FP16 | 0.3342 | 0.5003 | 0.3529 | 531,012 |

TorchScript FP32 and TensorRT FP32 match to four decimal places. TensorRT FP16 is 0.0001 lower on mAP@50-95 in this evaluation.

### Regenerating COCO results

Prepare the dataset on the laptop:

```bash
python evaluation/prepare_coco.py --n 5000
```

Generate detections on the Nano:

```bash
python3 evaluation/run_coco_detections.py \
    --runtime tensorrt \
    --precision fp16 \
    --subset 5000
```

Score them on the laptop:

```bash
python evaluation/coco_eval.py \
    --detections results/detections_tensorrt_fp16.json \
    --runtime tensorrt \
    --precision fp16 \
    --device nano \
    --subset 5000
```

Scoring stays on the laptop so the same `pycocotools` environment is used for all three runtimes.

## Docker comparison

The TensorRT FP16 pipeline was also benchmarked bare-metal and inside the L4T container using the same harness.

| Metric | Bare metal | Docker |
|---|---:|---:|
| Inference | 50.77 +/- 0.03 ms | 50.84 +/- 0.02 ms |
| End-to-end FPS | 12.57 +/- 0.01 | 12.49 +/- 0.02 |
| Peak RSS | 1065 MB | 1109 MB |
| Detections | identical | identical |

Measured differences:

- inference: +0.14% in the container
- end-to-end FPS: -0.66%
- preprocessing: +2.48%
- peak RSS: +44 MB

The container had very little effect on TensorRT inference time. Most of the small end-to-end difference came from preprocessing.

## Sustained thermal and power-mode runs

TensorRT FP16 was run continuously for 10 minutes at two Jetson operating points. These runs are recorded separately from the repeated headline benchmarks above.

| Metric | 10 W / MAXN | 5 W |
|---|---:|---:|
| Run ID | `6348178c575d` | `585939e8ae8b` |
| Clock lock | `true` | `false` |
| Fan policy | `auto` | `auto` |
| Frames | 7,512 | 5,118 |
| FPS first 60 s | 12.61 | 8.65 |
| FPS last 60 s | 12.49 | 8.46 |
| First-to-last change | -0.94% | -2.11% |
| GPU temperature | 33.0 -> 45.0 C | 34.0 -> 44.5 C |
| Maximum GPU temperature | 50.0 C | 44.5 C |
| Sustained FPS / configured W | 1.249 | 1.692 |

At 5 W the Nano sustained 8.46 FPS versus 12.49 FPS at the recorded 10 W operating point. That is 67.7% of the 10 W throughput.

Using the configured `nvpmodel` limits, the two runs work out to 1.692 versus 1.249 FPS per configured watt, a difference of 35.5%.

No direct electrical power measurement was taken. These numbers are therefore reported as throughput per configured power envelope, not measured energy efficiency.

The clock settings are also different between the two runs. The 10 W run used locked clocks and the 5 W run did not, so this is a comparison of the two recorded deployment operating points rather than an isolated `nvpmodel` experiment.

At 10 W the GPU reached 50 C and the stock thermal governor switched the fan on at PWM 80. The run continued at roughly the same throughput while the temperature fell back toward 45 C. The 5 W run peaked at 44.5 C and the fan stayed off.

Plots:

- `plots/thermal_6348178c575d.png`
- `plots/thermal_585939e8ae8b.png`

## Measurement protocol

For the repeated headline benchmarks:

1. Run from mains power.
2. Record the active host/power profile.
3. Run each configuration at least three times.
4. Run GPU tests before CPU tests on the laptop.
5. Leave roughly 30 seconds between repeated runs.
6. On the Nano, record `nvpmodel`, clock-lock state and fan configuration.

Long thermal runs use their own recorded settings and are not assumed to follow the same clock policy as the headline benchmark rows.

## Current timing variance

| Configuration | n | Inference | CV |
|---|---:|---:|---:|
| Nano TensorRT FP32 | 3 | 70.89 +/- 0.01 ms | 0.014% |
| Nano TensorRT FP16 | 3 | 50.64 +/- 0.12 ms | 0.23% |
| Nano TorchScript FP32 | 3 | 93.15 +/- 0.47 ms | 0.50% |
| Laptop GPU FP32 | 8 | 13.94 +/- 1.08 ms | 7.7% |
| Laptop GPU FP16 | 6 | 15.11 +/- 0.42 ms | 2.8% |
| Laptop CPU FP32 | 6 | ~47.5 ms | ~4% |

The Jetson runs have much lower relative timing variance than the laptop runs. For small differences, compare against the CV before treating the change as real.

## Reproducing a benchmark row

A benchmark can be rerun from the configuration recorded in `speed.csv`. Example:

```bash
python benchmarks/benchmark.py \
  --runtime pytorch \
  --device cuda \
  --precision fp32 \
  --frames 500 \
  --host-profile best-performance-mains \
  --notes "reproduction"
```

Check out the `git_commit` recorded in the row first. A row marked `-dirty` cannot be reproduced exactly because the working tree contained uncommitted changes.

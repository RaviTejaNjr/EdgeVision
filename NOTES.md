# EdgeVision — Build Notes

A running log of what actually happened, as opposed to what the plan said would
happen. One entry per session. Problems and their root causes are the valuable
part — most of the README's honest detail comes from here.

---

## 2026-08-31 — Part 0: flash and first boot

**Hardware:** Jetson Nano 2GB Developer Kit (P3541), bought used via
eBay Kleinanzeigen. The 802.11ac USB WiFi adapter listed on the box (SKU
945-13541-0000-000) was **missing from the package** — the seller kept it.
Working around this with Ethernet from the laptop.

- Flashed `sd-blob.img` (Jetson Nano **2GB** image, distinct from the 4GB one) to
  a SanDisk Extreme 128 GB microSDXC (U3/V30) using Balena Etcher.
- First boot with HDMI monitor, USB keyboard and mouse.
- Setup wizard: APP partition set to maximum, default swap accepted, **MAXN**
  power mode selected.
- Board reached the LXDE desktop without issues. Used hardware appears healthy.

**Networking:** no WiFi dongle and no router in reach, so internet is shared from
the laptop via **Windows Internet Connection Sharing** over Ethernet. Laptop
takes 192.168.137.1 and runs DHCP on that link; the Nano picks up a
192.168.137.x address.

---

## 2026-08-31 — Part 0.5: TensorRT path spike

Goal: prove the critical path exists before writing any project code.

### Confirmed working

| Check | Result |
|---|---|
| `python3 -c "import tensorrt; print(tensorrt.__version__)"` | **8.2.1.8** |
| `ls -lh /usr/src/tensorrt/bin/trtexec` | present, 390K, dated Nov 17 2021 |
| `nvcc --version` | **CUDA 10.2**, V10.2.300 |
| `python3 --version` | 3.6.9 (system) |
| `sudo pip3 --version` | pip 9.0.1 (stock, later upgraded) |

TensorRT Python bindings and `trtexec` both available in the **system Python
3.6**. This is the environment the project must live in — see the plan's §2 on
why Python 3.8 is not an option here.

### Problem: pip installs kept dying midway

**Symptom.** `sudo apt-get install python3-pip` broke partway through. Later,
`pip3 install --upgrade pip setuptools wheel` failed repeatedly with:

```
Failed to establish a new connection: [Errno 113] No route to host
Failed to establish a new connection: [Errno -2] Name or service not known
```

It looked like an intermittent connection drop, and the first instinct was to
blame the ICS link being flaky.

**Diagnosis.** It was not flaky. Two pings separated the cause cleanly:

```
ping -c 3 8.8.8.8      →  0% packet loss
ping -c 3 pypi.org     →  PING pypi.org(2a04:4e42:400::223) — 100% packet loss
```

DNS resolution worked fine. The problem was the *address it returned*:
`2a04:4e42:400::223` is **IPv6**. Windows ICS only routes IPv4, so every IPv6
connection attempt went nowhere and hung until it timed out. `8.8.8.8` worked
because it is an IPv4 literal and never needed resolving.

**Root cause.** The Nano resolves hostnames to IPv6 addresses by preference, but
has no working IPv6 route over the ICS link.

**Fix.** Force IPv4 preference in `/etc/gai.conf` by uncommenting:

```
precedence ::ffff:0:0/96  100
```

**Verification.**

```
ping -c 3 pypi.org  →  PING pypi.org (151.101.0.223) — 0% packet loss
```

**Why this matters.** The failure presented as an unreliable network and would
plausibly have been misattributed to the ICS setup, the Ethernet cable, or the
used board — potentially costing hours across multiple sessions. The two-ping
split (IP literal vs hostname) is the diagnostic that separates "no route" from
"wrong address family".

### Interrupted apt recovery

After the first broken install, `sudo dpkg --configure -a` returned nothing,
confirming no packages were left half-configured. apt handled the interruption
cleanly; no manual cleanup was needed.

### Other notes

- `pip3` was already present at 9.0.1 despite an earlier "command not found" —
  most likely a typo in the original command.
- Running `sudo pip3` without `-H` produces cache-permission warnings about
  `/home/raviteja/.cache/pip`. Harmless, but `sudo -H pip3` avoids them.
- pip must be pinned **below 21.0** — pip 21 dropped Python 3.6 support.
- Ignored apt's `autoremove` suggestion for ~90 leftover packages from the Ubuntu
  installer. Not worth the risk of removing something load-bearing on a working
  system.

### Environment quirk to remember

Working directly on the Nano with monitor, keyboard and mouse rather than over
SSH. Consequences: commands are retyped by hand rather than pasted, and the LXDE
desktop consumes roughly 300 MB of the 2 GB. The desktop should be disabled
(`sudo systemctl set-default multi-user.target`) before the TensorRT engine
build, which is the step most at risk of running out of memory.

Terminal shortcuts on the Nano desktop: `Ctrl+Alt+T` for a new terminal,
`Ctrl+Shift+C` / `Ctrl+Shift+V` for copy and paste.

---

### Spike results — all five checks passed

| Check | Result |
|---|---|
| TensorRT Python bindings | 8.2.1.8 ✅ |
| `trtexec` present | ✅ |
| PyCUDA → GPU | NVIDIA Tegra X1 ✅ |
| ONNX → FP16 engine | 654.8 s, `&&&& PASSED` ✅ |
| Engine loads from Python | bindings match ONNX ✅ |

**Model:** `yolov5nu` — Ultralytics' anchor-free retrofit of YOLOv5, *not* the
original 2020 architecture. 2,649,200 params, 7.7 GFLOPs. Exported at opset 13,
fixed 640×640, from torch 2.5.1 on the laptop.

### First real numbers

| Metric | Value |
|---|---|
| Engine build time | 654.8 s (~11 min) |
| Engine throughput | **21.7 FPS** |
| GPU compute | 45.28 ms mean |
| Host latency | 46.01 ms mean, p99 46.17 ms |
| Latency spread over 67 runs | 45.92 – 46.17 ms |
| H2D / D2H transfer | 0.46 / 0.27 ms |
| Enqueue time | 9.81 ms mean |
| Engine size | 12 MB |
| Peak GPU memory during build | ~1935 MiB of 1979 available |

**Caveat that matters:** this is *engine* throughput measured on random input
tensors. No decode, no letterbox resize, no NMS. End-to-end application FPS will
be lower, and the gap between the two is one of the headline results this project
is meant to produce.

### Findings worth keeping

**Fusion is visible in the build log.** Entries like
`PWN(PWN(/model.0/act/Sigmoid), /model.0/act/Mul)` show Sigmoid and Mul — the two
halves of SiLU — collapsed into a single kernel. `||` between two convolutions
means they run in parallel. Concrete evidence of the optimisation, worth
screenshotting for the README.

**Transfers are not the bottleneck.** 0.73 ms of 46 ms, about 1.6%. Compute
dominates. Worth knowing before spending time on data movement.

**The CPU is not starving the GPU.** Enqueue 9.81 ms against 45.28 ms compute.
That margin would shrink with heavy preprocessing on the same thread.

**Memory ceiling is real.** Build peaked at 1.7/1.9 GB. Left running deliberately
— with swap on, hitting the ceiling means slowdown rather than a crash, and
killing it would have discarded ~10 minutes of kernel timing. Worked.

**Output dtype is FLOAT, not HALF.** FP16 is internal to computation; TensorRT
returns fp32 at the boundary. `postprocess.py` reads float32 directly.

**Workspace was a constraint.** `Some tactics do not have sufficient workspace
memory to run` — some faster kernels needed more than 256 MB and were skipped.
`--workspace=512` is worth trying as its own benchmark row.

**No DLA.** The `Layers Running on DLA` section was empty, as expected. The Nano
has no DLA accelerator; those begin with Xavier. Same family of hardware limits
as the INT8 situation.

### Problem: CUDA context destroyed at exit

Benign, but it will recur in real code.

After printing everything correctly, `spike_test.py` hung and emitted
`Cuda Driver (context is destroyed)` errors; `Ctrl+C` then produced a
segmentation fault.

**Cause:** destruction order. `pycuda.autoinit` destroys the CUDA context at
interpreter exit, and TensorRT's engine object is collected afterwards — so it
frees GPU memory in a context that no longer exists.

**Impact:** none. All work completed before teardown; the engine file and driver
are unaffected.

**Fix for `infer_trt.py`:** manage the context explicitly instead of using
`pycuda.autoinit` — hold a reference and delete the engine before the context.

### Mistake worth remembering — scp direction

Ran `scp` on the Nano instead of the laptop, twice. The first time it copied the
engine to itself and reported success at 66.7 MB/s — suspiciously fast, because
it never left the board. The second attempt used a Windows path (`C:\Users\...`)
on Linux and hung at a `>` continuation prompt.

**The underlying model:** the Nano runs an SSH server; the laptop is the client.
Only the laptop can reach the Nano, so `scp` always runs from the laptop —
regardless of which direction the file travels. The `user@ip:` prefix just moves
to whichever side is remote.

**The habit to build:** glance at the prompt before pressing Enter.
`raviteja@raviteja-pc:~$` is the Nano; `PS C:\Users\gravi\...>` is Windows. This
will keep mattering through Parts 2–6, which involve pushing code to the board
constantly. (`rsync` over SSH would be faster for that than repeated `scp` — only
transfers what changed.)

### Memory after the spike — not a leak

`jtop` showed 537 MB used against a ~350 MB post-desktop baseline. `ps aux` found
no leftover `spike_test.py`; the memory was **jetson-stats itself** — three
daemon processes started at boot plus the interactive `jtop` window. The tool
measuring memory was a meaningful share of the memory measured.

~1.4 GB available, and the engine build peaked at 1.9 GB and still succeeded. Not
a constraint.

### Card backup — skipped deliberately

Working state now diverges from `jetson_base.img` (IPv6 fix, PyCUDA, upgraded
pip, built engine). Decided not to re-image: `SETUP_LOG.md` documents every
command, so recreating this state is ~30 minutes of following notes versus ~40
minutes and 128 GB for an image. The one expensive artifact — the 11-minute
engine build — is a 12 MB file that can just be copied off. Revisit after Part 6.

---

## 2026-09-02 — Part 1: repo scaffold

Repo created (`RaviTejaNjr/EdgeVision`, private until Part 6), structure
scaffolded, plan and docs committed, `configs/params.yaml` written.

**Test video.** Recorded 34.7 s, 1280×720, landscape, static camera, moderate
foot traffic. Transcoded with ffmpeg to force **constant** frame rate — phone and
editor exports often use variable frame rate, where the encoder drops or
duplicates frames during static scenes. "30 fps" is nominal in that case, and any
FPS measurement would inherit the variance.

```bash
ffmpeg -i Inference_Video.mp4 -vf scale=1280:720 -r 30 -c:v libx264 -crf 23 \
       -preset medium -an data/test_video.mp4
```

Verified through **OpenCV**, not ffmpeg — OpenCV uses different decoders, and the
whole pipeline reads through it:

| Check | Result |
|---|---|
| reported frames | 1040 |
| actually read | 1040 |
| fps | 30.0 |
| size | 1280 × 720 |

Reported and actual match, so no container metadata quirks.

**Incident:** edited `README.md` in GitHub's web editor while also working
locally, putting the local copy one commit behind. Resolved with
fetch → diff → pull. Written up in full in `docs/GIT_NOTES.md`. Habit adopted:
pull before starting work, and avoid the web editor entirely.

---

## 2026-09-04 — Part 2: benchmark harness and first measurements

### The harness

Four files: `app/preprocess.py`, `app/postprocess.py`, `app/backends.py`,
`benchmarks/benchmark.py`.

Written **before** any optimisation and reused unchanged from here on. All three
runtimes sit behind one `Backend` interface with `load()` and `infer()`, so
`benchmark.py` never knows which it is measuring — differences between results
are differences between runtimes, not between measurement methods.

`postprocess.py` was written now rather than at Part 5. It has to exist for
end-to-end timing anyway, and writing it on the laptop means Ultralytics is
available as a reference to validate against — which it will not be on the Nano.

Two details carried forward from earlier findings:

- **`synchronize()` calls are load-bearing.** CUDA work is queued
  asynchronously; without them the timer would stop at submission rather than
  completion, producing impossibly fast results.
- **The TensorRT backend manages its CUDA context explicitly**, releasing the
  engine before tearing down the context. That is the fix for the spike's
  teardown segfault, written in from the start rather than rediscovered.

### First run — the decoder works

200 frames, laptop, PyTorch FP32, CUDA. **7.5 mean detections per frame** — the
hand-written NumPy decode is correct. A wrong transpose or the classic-YOLOv5
objectness assumption would have produced 0 or nonsense.

### The headline gap, visible on day one

```
FPS engine     : 36.01
FPS end-to-end : 22.84
```

**A 37% drop.** The model runs at 36 FPS; the application delivers 22.8. This is
the number most published edge benchmarks omit, and it is one of the project's
intended results.

Per-stage breakdown (200-frame run):

| Stage | ms | Share |
|---|---|---|
| capture | 3.55 | 9% |
| preprocess | 9.47 | 23% |
| **inference** | **27.77** | **68%** |
| postprocess + NMS | 2.96 | 7% |

**Prediction to check at Part 4:** preprocessing already costs a third of what
inference costs, on an i9. On the Nano's four ARM Cortex-A57 cores that ratio
should get dramatically worse, and preprocessing may become the dominant term.

### Cold start is enormous and variable

10,295 ms on the first run, 3,811 ms on the second. The difference is Windows
filesystem caching — the weights were already in RAM the second time.

Against a 28 ms frame this is ~370× the per-frame cost, which is exactly why it
is measured separately rather than averaged in. It also means cold-start figures
depend on whether the OS has recently seen the file, and that caveat belongs
alongside any cold-start number reported.

---

## PROBLEM 4 — the laptop GPU is stuck at idle clocks

**This invalidates every laptop GPU measurement taken so far.**

### Symptom

At 200 frames, engine p95 was 29.90 ms against a p50 of 27.44 — tight and
healthy. At 500 frames, **p95 rose to 71.61 ms against a p50 of 27.07** — a 2.6×
tail. The progress log showed frame 500 taking 71.9 ms of engine time, roughly
triple the earlier frames.

### Two wrong hypotheses

**Thermal throttling** — plausible, since the laptop's GPU fan is disconnected
for noise.

**Memory pressure** — `peak_mem_mb` had grown 3048 → 5108 MB between runs, and
5 GB on a 4 GB card would mean spilling into shared system memory.

### The diagnostic

Ran `nvidia-smi` polling once per second in a second terminal, alongside the
benchmark:

```bash
nvidia-smi --query-gpu=timestamp,temperature.gpu,clocks.sm,utilization.gpu,memory.used \
           --format=csv -l 1
```

### What it actually showed

| Hypothesis | Evidence | Verdict |
|---|---|---|
| Thermal | 57 °C → 70 °C over the run. Ampere throttles around 87 °C+ | ❌ not thermal |
| GPU memory | flat at **1212 MiB** throughout | ❌ not memory |
| **Clock speed** | **pinned at 210 MHz for the entire run** | ✅ **this is it** |

**210 MHz is the idle clock.** An RTX 3050 mobile should boost to roughly
1400–1700 MHz under load. It was running at **~15% of its clock speed** while
reporting 85% GPU utilisation — genuinely busy, just crawling.

### Root cause

The disconnected GPU fan. Not throttling from measured heat — the driver or
firmware appears to be refusing to boost *at all*, presumably because it cannot
see a working fan and will not risk it. Some laptop BIOSes implement exactly this
as a safety fallback.

### Secondary finding — `peak_mem_mb` is ambiguous

The 5 GB figure is `psutil` reporting the **Python process's RSS** — interpreter,
torch, CUDA libraries and all — **not GPU memory**. GPU memory was flat at
1212 MiB. That column measures host memory, and the README should say so.

### Impact

**Laptop GPU rows measured so far are not RTX 3050 numbers.** They are "RTX 3050
held at idle clocks" numbers, and publishing them as a laptop baseline would be
misleading.

**Nothing else is blocked.** The Nano is the primary target and laptop rows are
an optional reference. The harness, decoder and backends all work.

### Action

Reconnect the fan, re-run the identical command, and compare. Same config hash,
so the CSV rows align directly. Expect a large jump — inference at full clocks
could plausibly be 5–10 ms rather than 27.

### Why this one is worth telling

Both intuitive explanations were wrong, and one measurement ruled out both while
identifying the real cause. It also demonstrates why a benchmark that does not
record the hardware's actual operating state is not reproducible — a
throughput figure without its clock state is an anecdote.

---

## PROBLEM 4 — continued: it was never the fan

The clock investigation pointed at the disconnected GPU fan. Reconnecting it made
things **worse**, uniformly:

| Stage | before fan | after fan |
|---|---|---|
| inference | 33.66 ms | 74.33 ms |
| preprocess | 5.67 ms | 22.72 ms |
| capture | 2.32 ms | 6.75 ms |
| postprocess | 2.19 ms | 7.86 ms |

**The tell: CPU-only stages got 3–4× slower too.** Preprocessing and NMS never
touch the GPU, so reconnecting a GPU fan cannot possibly slow them down. The
whole machine was slower, not the GPU.

### The real cause: Windows was in "Whisper" mode

A vendor quiet profile capping both CPU and GPU clocks. It had been active for
every run up to that point.

Progression across four runs of the **identical command**:

| Power state | GPU clock | Inference | Engine FPS | E2E FPS |
|---|---|---|---|---|
| Whisper | 210 MHz | 74.33 ms | 13.5 | 9.0 |
| Whisper (earlier) | 210 MHz | 33.66 ms | 29.7 | 22.8 |
| Balanced | ~550–1057 MHz | 18.16 ms | 55.1 | 31.4 |
| **Best performance** | ~950–1057 MHz | **14.05 ms** | **71.2** | **41.1** |

**5.3× faster inference from power settings alone**, on identical code and
identical hardware.

`nvidia-smi --query-gpu=clocks.max.sm` reports **2100 MHz**. The GPU had been
running at 210 MHz — exactly **10%** of capability.

### Still not at maximum

Even at Best performance the GPU peaks around 1057 MHz of 2100, and utilisation
tops out at ~42%. A vendor profile still caps it.

**Deliberately not chased further.** At 42% utilisation the GPU is idle more than
half the time — waiting on the CPU. Capture (2.4) + preprocess (5.99) +
postprocess (1.91) = **10.3 ms of CPU work** against 14.05 ms of GPU work. Raising
the GPU clock shrinks the 14 ms and leaves the 10.3 ms untouched.

The pipeline is already partly CPU-bound **on an i9**. On the Nano's four ARM
cores it will be decisively so.

### Correction to the earlier entry

The first diagnosis — "the driver refuses to boost without a fan" — was wrong.
The fan was irrelevant; the power profile was everything. Later measurements
(below) confirmed the fan has no effect on CPU inference at all.

---

## 2026-09-04 — Part 2 results: 28 runs, three configurations

### Method

Every configuration measured at least three times, on mains, at Best performance.
GPU runs first so CPU load would not heat the machine and skew them; ~30 s
between runs.

### Final laptop baseline

| Config | n | Inference | CV | E2E FPS |
|---|---|---|---|---|
| GPU FP32 | 8 | **13.94 ± 1.08 ms** | 7.7% | 42.69 ± 3.28 |
| GPU FP16 | 6 | **15.11 ± 0.42 ms** | 2.8% | 41.47 ± 1.92 |
| CPU FP32 | 6 | **~47.5 ms** | ~4% | ~17.3 |

### Findings

**Engine vs end-to-end: 73.6 → 42.7 FPS, a 42% drop.** Present in every run. This
is the number most published edge benchmarks omit.

**GPU over CPU: 3.38× on inference, 2.45× end-to-end.** The gap between those two
figures is the CPU stages diluting the GPU's advantage — Amdahl's law in the
project's own measurements.

**PyTorch FP16 is 8.4% *slower* than FP32 on inference, and equivalent
end-to-end** (−2.9%, within noise).

Counterintuitive, and worth reporting. Three reasons:

- PyTorch's `.half()` converts dtypes but keeps the same layer-by-layer
  execution, adding conversion overhead at boundaries. It does not change kernels.
- The GPU is only 42% utilised — already waiting on the CPU, so making the GPU
  part faster does not help.
- Ampere's tensor cores need specific shapes and cuDNN paths to deliver their
  theoretical 2×; naive `.half()` often does not trigger them.

**This sets up the contrast for Part 5:** *PyTorch FP16 gave no speedup. TensorRT
FP16 gave N×. The difference is that TensorRT changes the kernels rather than just
the dtype.*

**FP32 is 2.7× noisier than FP16** (7.7% vs 2.8% CV) — consistent with FP32
pushing the GPU harder and hitting the power ceiling more often, so the governor
intervenes more.

**The decoder is deterministic: 8.27 mean detections across all 28 runs** — GPU
and CPU, FP32 and FP16. Precision-independent, as it should be.

### A retraction worth recording

After two runs, FP16 looked measurably slower. After a third FP32 run landed at
16.17 ms — 15% from the first — that conclusion was withdrawn as noise.

With 8 and 6 samples it is back, and now defensible: FP32 spans 12.85–16.17,
FP16 spans 14.65–15.68. The 8.4% difference exceeds the overlap.

**The lesson: two runs cannot establish a difference under ~15%.** Three is the
minimum, and the CV should be reported alongside any comparison.

---

## PROBLEM 5 — battery mode costs 2.6× on CPU

Discovered while testing whether the GPU fan affects CPU runs.

| State | Inference | E2E FPS | Sustained 60 s |
|---|---|---|---|
| Battery | **132.95 ms** | 6.23 | **4.88** |
| Mains | **51.54 ms** | 15.54 | — |

**Windows caps CPU clocks hard on battery regardless of the "Best performance"
setting.**

The progress log shows the collapse happening live: 74.9 → 313.0 → 214.9 → 204.6
ms. It started near normal and degraded after roughly 100 frames — the CPU
boosting briefly, then hitting the battery power cap.

**The `fps_sustained_last_60s` metric caught it**: 4.88 against a mean of 6.23.
That column exists for exactly this, and this is the first time it earned its
place.

**Consequence:** every laptop measurement must be on mains. Added to the run
checklist.

---

## Does the GPU fan affect CPU inference? No.

The original question behind the battery discovery.

| Condition | n | Inference |
|---|---|---|
| Fan on, mains | 3 | 47.11 ± 1.72 ms |
| Fan off, mains | 3 | 48.80 ± 2.51 ms |

Overlapping error bars. **The GPU fan has no measurable effect on CPU inference**,
as expected — but now measured rather than assumed. Fan stays detached.

*(The first fan-off run read 51.54 ms and briefly looked like a real effect. Runs
2 and 3 came in at 47.11 and 47.76. The first run after plugging in was still
ramping clocks — another argument for three runs minimum.)*

---

## Harness change: `--host-profile`

Added after the fact. Two runs of identical code differed by **2.6×** with nothing
in the CSV to distinguish them, because power state was invisible to the harness.

On the Nano `nvpmodel -q` reports it automatically; on Windows nothing does. The
field is now part of the config hash and captures power profile, mains vs battery,
and fan state.

**On the hash collision in the older rows:** several GPU FP32 rows share
`e403c324` while differing 5×. This turns out **not** to matter for the join to
`accuracy.csv` — accuracy does not depend on power profile, so one accuracy row
per model/runtime/precision is correct. The `notes` column distinguishes the speed
rows. Older runs are kept in `results/speed_exploratory.csv`.

---

## 2026-09-04 — Part 3: export parity and decoder validation

Two separate questions, and the second matters more.

### 1. Did the ONNX export preserve the model?

`tests/test_parity.py` — five frames through both PyTorch and ONNX Runtime,
comparing the raw `(1, 84, 8400)` tensors.

```
6 passed in 3.24s
```

Tolerances are `atol/rtol = 1e-3` with mean absolute difference under `1e-4`.
FP32 arithmetic is not associative, so two runtimes that fuse or reorder
operations differently will not produce bit-identical output; the bounds allow
for that while still catching a genuinely broken export.

**`test_class_scores_are_probabilities` is the quietly useful one.** It asserts
the 80 columns after the box coordinates all lie in [0, 1]. If the head had an
objectness column, those columns would be offset by one and the last would hold
box data — unbounded, and the test would fail. It passing confirms the anchor-free
layout assumption.

### 2. Does the NumPy decoder match Ultralytics?

`evaluation/validate_decoder.py` — 10 frames, matched at IoU ≥ 0.9, same class
only.

| Metric | Result |
|---|---|
| Detections, ours | 125 |
| Detections, Ultralytics | 118 |
| Matched | 117 |
| **Agreement** | **99.2%** |
| Box coordinate error, mean | **0.546 px** |
| Box coordinate error, max | 8.485 px |
| Score error, mean | 0.018 |
| Score error, max | 0.162 |

**Mean box error of 0.546 px is the number that establishes correctness.**
Sub-pixel agreement means the letterbox-undo maths — subtract padding, divide by
scale, clip — is right.

### Where they disagree, and why

**Every unmatched detection is between conf 0.252 and 0.287**, against a 0.25
threshold. Borderline cases, as predicted.

Visual inspection of the overlays settled the rest:

**Frame 0** — ours found a person Ultralytics missed at conf 0.257. Checked
against the image: there really is a person there.

**Frame 808** — the interesting one. Ultralytics produced a *single enormous box*
labelled "kite 0.26" spanning the entire bunting line across the frame. Ours
produced three separate tight boxes on individual kites.

That is not a disagreement about detection; it is a disagreement about **NMS**.
Ultralytics appears to run class-agnostic suppression or a lower IoU threshold on
that class, merging overlapping kite detections into one box covering a third of
the sky. Per-class NMS at IoU 0.45 keeps them separate.

**Ours is the more sensible output here.** Three tight boxes on three kites is
correct; one box over the whole bunting line is not.

### The max errors: preprocessing, not decoding

8.485 px and a 0.162 score difference are too large to be rounding.

**Cause: letterbox geometry.** Ultralytics pads to a multiple of the model stride
(32) and often uses a *rectangular* letterbox, padding only as far as the next
stride multiple. `app/preprocess.py` always pads to a square 640×640.

Different padding means slightly different input pixels, so slightly different
predictions. That is a difference between two preprocessing conventions, not a
decoder bug — and square 640×640 is the correct choice here, because it is what
the ONNX export declares and what the TensorRT engine was built for.

### Tolerances set accordingly

Defaults changed from 2 px / 0.01 to **10 px / 0.2**, with the reasoning recorded
in the source. Mean error of 0.546 px is what proves correctness; the tolerance
exists to catch a broken decoder, not the known geometry difference.

### Conclusion

The decoder is validated. Every accuracy number from Part 5 onward rests on this,
and it is the only opportunity to check — Ultralytics cannot run on the Nano.

### One bug found and fixed

The first run crashed on frame 2 with:

```
RuntimeError: Input type (torch.FloatTensor) and weight type (torch.cuda.FloatTensor)
should be the same
```

`yolo.predict()` moves the underlying model to CUDA as a side effect, so the next
raw call fed a CPU tensor to a now-GPU model. Fixed by pinning the model, the
input tensor and `predict()` to CPU explicitly.

---

## 2026-09-05 — Part 4 begins: bringing the Nano back up

### The board needed two attempts to boot

First power-on: no SSH, and `ping 192.168.137.244` returned 100% packet loss.

Diagnosis was quick because the two-machine split makes it easy to isolate:

```
arp -a | findstr 192.168.137
→ Interface: 192.168.137.1 --- 0x10
  192.168.137.244  48-b0-2d-2f-64-83  static
```

Windows ICS was running (laptop had .1) and the Nano's MAC was in the ARP table,
so the network configuration was intact. The `static` rather than `dynamic` entry
was the hint — a cached record, not a live one.

**Resolution:** attached a monitor and keyboard. Initially nothing on screen, then
a restart brought it up normally to the login prompt.

**This is the second time the board has needed a restart to come up cleanly.**
Twice could be coincidence; a third time would be worth investigating. Noted in
case it recurs. Possible causes to check if it does: SD card seating, power supply
under-delivery at boot, or a filesystem check stalling.

### Housekeeping done while it was up

**CUDA paths made permanent.** They had been re-typed in three separate sessions:

```bash
echo 'export PATH=/usr/local/cuda/bin:$PATH' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
```

`>>` appends; a single `>` would overwrite the whole file.

**Back to headless.** Attaching the monitor had started the desktop, which showed
up immediately in the memory figures:

| State | Used | Available |
|---|---|---|
| Desktop running | 592 MB | 1.2 GB |
| **Text mode (`multi-user.target`)** | **271 MB** | **1.6 GB** |

**321 MB recovered**, and 1.6 GB available is better than the ~350 MB baseline
recorded earlier. That headroom matters: the torchvision compile ahead is the most
memory-hungry step in the project.

```bash
sudo systemctl set-default multi-user.target   # graphical.target to reverse
systemctl get-default                          # check which is active
sudo systemctl start graphical.target          # start desktop now, without changing boot
```

**Disk:** 96 GB free of 118 GB. No concern.

**IPv6 gone.** `hostname -I` now returns only `192.168.137.244` and Docker's
`172.17.0.1` — no `2003:d1:...` address this time. The `gai.conf` fix plus a clean
boot left it IPv4-only.

**HDMI is hot-pluggable** — the monitor can be pulled while the board runs, with
no desktop session to disturb in text mode.

### Ignoring the update notice

Login banner reports 329 available updates, 275 of them security. **Deliberately
not applied.** `nvidia-l4t-bootloader` and `nvidia-l4t-init` are held, and a broad
`apt upgrade` on JetPack 4.6 risks pulling package versions that conflict with the
pinned L4T stack. This is a development board on an isolated link, not a
production system.

---

## 2026-09-05 — Part 4: TensorRT running on the Nano

### Setup

Cloned the repo onto the board with a fine-grained GitHub personal access token
(Contents: read **and** write, so results can be pushed back from the Nano). The
alternative — `scp`ing files across — was rejected because the `git_commit` column
in `speed.csv` would read `unknown` on a machine with no repo, breaking the
provenance chain that makes the results table defensible.

The spike folder was renamed `~/spike_artifacts` before cloning, to avoid
`~/edgevision` and `~/EdgeVision` coexisting as separate directories.

The ONNX and engine files were moved into `models/` — gitignored, so the clone
could not bring them, but `configs/params.yaml` expects them there. The test video
went across by `scp`, being the one file that must.

**Dependencies:** `numpy 1.13.3` and `cv2 4.1.1` already present (JetPack builds,
CUDA and GStreamer enabled — do not replace). `yaml` already present. Only
`psutil` needed installing, and it arrived as a **prebuilt aarch64 wheel** for
cp36 — no compilation. Version 7.2.2, matching the laptop, so memory figures are
directly comparable.

### It worked first try

```bash
sudo nvpmodel -m 0        # MAXN / 10W
sudo jetson_clocks        # lock clocks
python3 benchmarks/benchmark.py --runtime tensorrt --precision fp16 \
    --frames 500 --host-profile jetson-10w-clocks-locked-fan-off \
    --clocks-locked true --fan false --notes "nano tensorrt fp16 run 1"
```

TensorRT inference on the Jetson, through the same harness written on the laptop,
with no code changes. The pluggable-backend design paid off exactly as intended.

### Results — three runs

| Metric | Mean | SD | CV |
|---|---|---|---|
| Inference | **51.76 ms** | 0.13 | **0.25%** |
| Engine FPS | **19.32** | 0.04 | 0.23% |
| End-to-end FPS | **12.51** | 0.06 | 0.46% |

**The Nano is roughly 30× more reproducible than the laptop.** Laptop CVs ranged
2.8–7.7%; this is 0.25%. Locked clocks, no competing desktop workload, no thermal
governor intervening. Engine p50/p95 within a single run: 51.65 / 51.92 ms — a
0.27 ms spread over 500 frames.

**Worth stating explicitly in the README:** a constrained embedded board is a
*better* measurement instrument than a general-purpose laptop, because there is
almost nothing else happening on it.

**8.32 mean detections**, against the laptop's 8.27 on the same 500 frames. The
decoder produces the same output on both machines — the hand-written NumPy
postprocessing validated on target hardware.

### The prediction was wrong, and that is the interesting part

**Predicted at Part 2:** CPU stages are ~40% of the frame on an i9; on four ARM
Cortex-A57 cores they should *dominate*.

**Measured:**

| Stage | Laptop | Nano | Ratio |
|---|---|---|---|
| capture | 2.32 ms | 5.70 ms | 2.5× |
| preprocess | 6.43 ms | 13.67 ms | 2.1× |
| inference | 16.17 ms | 51.76 ms | **3.2×** |
| postprocess + NMS | 2.16 ms | 8.77 ms | 4.1× |

| | Laptop | Nano |
|---|---|---|
| CPU stages | 10.91 ms (**40%**) | 28.14 ms (**35%**) |
| Inference | 16.17 ms (60%) | 51.76 ms (65%) |

**CPU stages became a *smaller* share, not a larger one.**

The reason: inference scaled by 3.2× while the CPU stages scaled by 2.1–2.5×.
Going from an RTX 3050 to a 128-core Maxwell GPU is a bigger step down than going
from an i9-11900H to four Cortex-A57 cores — at least for this workload, and
noting that the laptop GPU was itself capped near 1057 MHz of 2100.

Postprocess is the exception at 4.1×, the worst-scaling stage. NMS is
sort-and-compare on a single core with no vectorisation benefit, which is exactly
what ARM does worst.

**Recording this as a failed prediction is more valuable than being right.** It
shows the measurement was made to test something rather than to confirm it.

### A thermal signal, too small to matter but real

Across the three consecutive runs:

| Run | Inference | Max temp |
|---|---|---|
| 1 | 51.68 ms | 37.5 °C |
| 2 | 51.70 ms | 40.0 °C |
| 3 | 51.91 ms | 42.5 °C |

**A 0.4% slowdown over 5 °C.** Far too small to affect anything here, but it is
the same mechanism Part 7 will measure over ten-minute runs. Good to have caught
its beginning.

### Fan state

`cat /sys/devices/pwm-fan/target_pwm` → `0`. The fan is fitted and detected but
off. `jetson_clocks` did not start it.

**Deliberately left off** for the baseline. The fan-on/fan-off comparison is a
Part 7 experiment; changing conditions midway through baselining would invalidate
the comparison.

To control it manually:

```bash
sudo sh -c 'echo 255 > /sys/devices/pwm-fan/target_pwm'    # 0-255
```

### Two things to carry forward

**The `clocks_locked` field is not auto-detected.** The first run recorded `false`
despite `jetson_clocks` having been applied, because the harness takes whatever
the flag says. Passed explicitly from run 2 onward. **A field that can silently
disagree with reality is worse than no field** — worth revisiting whether it can
be read from the system.

**TensorRT warning on every run:**

```
[TRT] [W] Using an engine plan file across different models of devices is not
recommended and is likely to affect performance or even cause errors.
```

The engine was built during the day-zero spike, before several reboots. Probably
benign — the numbers are stable and the detections correct — but Part 5 rebuilds
it properly from `build_trt_engine.py`, which should clear it.

---

## 2026-09-05 — TorchScript: the PyTorch path for the Nano

### Why TorchScript rather than YOLOv5-on-device

The plan's Part 4 called for a PyTorch baseline on the Nano. But `backends.py`
loads models through Ultralytics, which **cannot run on Python 3.6**. Two options:

- Clone YOLOv5 at a v6.x tag on the Nano and load through its own code — which
  means pinning pandas, matplotlib, scipy and seaborn to versions that still
  support 3.6
- **Export TorchScript** — a self-contained file holding architecture and weights,
  loadable with `torch.jit.load()` and nothing else

Chose TorchScript, for three reasons:

**It removes the dependency problem entirely.** No YOLOv5 repo, no version
pinning exercise.

**It is what you would actually do in production.** Nobody deploys a training
repo to an edge device; freezing the graph is the standard way to ship a model
without shipping the framework around it.

**It isolates the runtime honestly.** YOLOv5's Python path would add its own
pre- and postprocessing into a measurement meant to compare inference engines.

### The manual trace failed — twice, for different reasons

**First attempt:**

```
RuntimeError: Tracer cannot infer type of (tensor(...), {'boxes': ..., 'scores': ...,
'feats': [...]})
Dictionary inputs to traced functions must have consistent type.
Found Tensor and List[Tensor]
```

The model returns `(predictions, extras)` where `extras` is a dict of mixed
tensors and lists. The tracer cannot infer a consistent type for that.

**Second attempt**, after wrapping the model to return only the prediction tensor:

```
ERROR: Tensor-valued Constant nodes differed in value across invocations.
This often indicates that the tracer has encountered untraceable code.
```

The graph diff showed `make_anchors` present in the first invocation and absent
in the second. **Ultralytics caches anchor points after the first forward pass**,
so the two trace runs produce different graphs and the checker rejects it.

**The answer was to stop hand-rolling it.** Ultralytics has a built-in TorchScript
export, same as the ONNX one already in use:

```python
model = YOLO(cfg["model"]["weights"])
model.export(format="torchscript", imgsz=cfg["model"]["input_res"])
```

It warms the model before tracing, so the anchor cache is populated and both
invocations match. **Lesson: when a library's own exporter exists, use it — it
knows about the library's internal state.**

**Verified it loads standalone**, importing only torch:

```
returns a tensor: (1, 84, 8400) torch.float32
```

A bare tensor, no unwrapping needed, straight into the NumPy decoder.

---

## PROBLEM 6 — the vendor fan profile governs GPU clocks

**Three TorchScript runs gave 67.85, 24.78 and 66.30 ms.** Wildly inconsistent —
itself the signature of an unstable clock rather than a real difference.

### The sequence of wrong conclusions

1. **"TorchScript is 78% slower than PyTorch."** Wrong — measured under a capped
   clock.
2. **"The GPU won't boost because the fan is detached."** Wrong — but only
   discovered later, because the test that seemed to disprove it was itself
   contaminated by Whisper mode being active simultaneously.
3. `powercfg /getactivescheme` returned **Balanced**, having silently reverted
   from Best performance at some point. Setting it back helped, but not enough.

### The actual cause

The **vendor utility's fan profile was set to Whisper**, and on this laptop that
profile caps GPU power and clocks regardless of what Windows' power mode says.

`nvidia-smi` during a run: **210 MHz for the entire duration** — 10% of the
2100 MHz ceiling, at 85% utilisation.

### The fix, and the confirmation

Setting the **CPU fan profile to Performance** released the GPU as well:

| Reading | Whisper | Performance |
|---|---|---|
| Power cap | 30 W | **56 W** |
| Perf state | P8 (idle) | **P0 (max)** |
| Clock under load | 210 MHz | **1057–1567 MHz** |

**The GPU fan being detached was never the cause.** It may cap the ceiling
somewhat — clocks top out at 1567 of 2100 — but Whisper mode was the 10× factor.

### Protocol change

**`powercfg` is not sufficient.** Windows can report "Best performance" while a
vendor profile silently overrides it. The only reliable check is the hardware's
own telemetry during an actual run:

```
nvidia-smi --query-gpu=clocks.sm,power.limit,pstate --format=csv
```

`P0` and a 56 W cap means the machine is in the right state. Anything else and
the measurement is invalid.

**This is the third occasion laptop measurements were silently invalidated by
power state.** Worth stating as a limitation of the laptop as a measurement
platform — and a sharp contrast with the Jetson, where `nvpmodel -q` and
`jetson_clocks` are explicit and honoured.

---

## TorchScript vs PyTorch — six clean runs

Alternated between runtimes so any drift would affect both equally. All at
Best performance, mains, CPU fan Performance, clocks verified above 900 MHz.

| Runtime | Inference | CV | Engine FPS | E2E FPS |
|---|---|---|---|---|
| PyTorch | 13.63 ± 0.45 ms | 3.3% | 73.45 | 41.73 |
| **TorchScript** | **8.33 ± 0.81 ms** | 9.7% | **120.79** | **57.47** |

**TorchScript is 38.9% faster on inference, 37.7% faster end-to-end.**

**Why:** TorchScript freezes the graph and executes it in C++, removing Python
interpreter dispatch from the forward pass. For a small model with many cheap
layers, that dispatch cost is a large share of the total.

**Detections identical at 8.27** across all six runs — the frozen graph produces
the same output as eager mode.

### The clock log explains the variance

Peak `clocks.sm` per run:

| Run | Peak |
|---|---|
| PyTorch 1 / 2 / 3 | 1057 / 1057 / 1080 MHz |
| TorchScript 1 / 2 / 3 | 1305 / **1567** / **1567** MHz |

**TorchScript drove the GPU to higher clocks than PyTorch ever reached.** Likely
because the workload is denser and sustained — the boost algorithm responds to
continuous demand rather than the stop-start pattern of Python dispatch.

So part of the 38.9% is the runtime itself, and part is the GPU being permitted
to work harder *because* of the runtime. Both are real consequences of the same
cause, and the honest framing is that they are not separable from these
measurements alone.

It also explains TorchScript's higher CV (9.7% vs 3.3%): its clocks ranged
1305–1567 across runs while PyTorch sat consistently near 1057.

**Utilisation still peaks around 43%**, so the pipeline remains partly CPU-bound
even at these clocks.

---

## 2026-09-05 — Part 4 complete: the headline result

### The torchvision build was never needed

The plan timeboxed the torch install at two hours, with Option B written in as a
fallback if torchvision would not compile. **It took about ten minutes**, and
torchvision turned out to be unnecessary.

```bash
sudo apt-get install -y libopenblas-base libopenmpi-dev libomp-dev
wget <nvidia box url> -O torch-1.10.0-cp36-cp36m-linux_aarch64.whl   # 308 MB, 33 s
pip3 install --user --no-deps torch-1.10.0-cp36-cp36m-linux_aarch64.whl
```

```
torch 1.10.0
cuda available: True
device: NVIDIA Tegra X1
```

Then the question the whole TorchScript detour was for:

```
python3 models/check_torchscript.py
→ returns a tensor: (1, 84, 8400) torch.float32
```

**`torch.jit.load` works with torch alone.** YOLOv5n is pure convolutions, so the
traced graph has no torchvision operators — no `nms`, no `roi_align`. The riskiest
step in the plan, a 1–2 hour source compile on a 2 GB board with real OOM risk,
simply evaporated.

**Why `--no-deps`:** the wheel declares `typing-extensions` and `dataclasses`,
which pip would resolve from PyPI and potentially pull versions that fail to build
on Python 3.6. Nothing was missing, so nothing needed adding.

**The apt packages matter.** The wheel is dynamically linked against OpenBLAS,
OpenMPI and OpenMP. Without them the install succeeds and `import torch` then
fails on a missing shared library — a confusing failure one step removed from its
cause.

---

## THE HEADLINE RESULT

Three runs each, same board, same model, same harness, same video, same 500
frames. 10 W, clocks locked, fan off.

| | TorchScript FP32 | TensorRT FP16 | Improvement |
|---|---|---|---|
| **Inference** | 93.15 ± 0.47 ms | **51.76 ± 0.13 ms** | **1.80×** |
| Engine FPS | 10.74 ± 0.06 | **19.32 ± 0.04** | 1.80× |
| End-to-end FPS | 8.21 ± 0.06 | **12.51 ± 0.06** | 1.52× |
| Cold start | 24.3 s | **4.1 s** | **5.9×** |
| Mean detections | 8.33 | 8.32 | **−0.1%** |
| CV | 0.50% | 0.25% | — |

### Four things make this defensible rather than merely impressive

**Detections agree to 0.1%.** 8.33 versus 8.32 on identical frames. The speedup
did not cost correctness — the FP32/FP16 numerical difference is barely
measurable. (Proper mAP comes at Part 5.)

**End-to-end is 1.52× while inference is 1.80×.** The ~28 ms of CPU work is
unchanged, so it dilutes the gain. Reporting both stops the claim being
overstated — and most published edge benchmarks report only the first.

**CVs of 0.25–0.50%.** Three runs each. Tighter than anything the laptop produced.

**Cold start improved 5.9×** — 24.3 s to 4.1 s. TorchScript deserialises and sets
up a graph; TensorRT loads a pre-compiled engine. A real operational difference
for a service that must come up quickly, and one nobody usually measures.

### The contrast the plan predicted

**PyTorch FP16 on the RTX 3050 gave no speedup over FP32** (8.4% *slower* on
inference, equivalent end-to-end).

**TensorRT FP16 on the Nano gave 1.80×.**

Same precision change, opposite outcomes. The reason: `.half()` converts dtypes
while keeping the same layer-by-layer execution; **TensorRT selects different
fused kernels and halves actual memory traffic.** On a bandwidth-bound board that
is the whole game.

This was written into the plan as the expected contrast, and it held.

### Thermal trend, now clearer

| Run | Inference | Max temp |
|---|---|---|
| smoke (200 frames) | 92.54 ms | 36.0 °C |
| run 1 | 92.61 ms | 42.5 °C |
| run 2 | 93.43 ms | 46.5 °C |
| run 3 | 93.41 ms | 49.5 °C |

**1% slowdown over 13.5 °C.** More pronounced than the TensorRT runs showed,
because TorchScript works the GPU harder for longer. Still negligible at this
scale, but the trend is now visible enough to predict what Part 7's ten-minute
runs will show.

`fps_sustained_last_60s` read 8.30, 8.19, 8.25 against means of 8.26, 8.15, 8.21
— slightly *higher* at the end of each run. No degradation within a single run at
this length.

---

## 2026-09-06 — Part 5: separating the runtime gain from the precision gain

### The session started with a networking failure — and the cause was physical

Third SSH failure in four cold boots. This time it was diagnosed properly rather
than worked around.

**Step 1 — the board was fine.** Monitor attached: `raviteja-pc login:` at 9.4 s.
Clean boot. The `tegra-i2c: no acknowledge from address 0x50` messages are the
kernel probing for a camera EEPROM that is not plugged in; `edid invalid` was the
monitor being attached five minutes after boot. Both harmless.

**Step 2 — the interface was up but had no address:**

```
3: eth0: <BROADCAST,MULTICAST,UP,LOWER_UP> ... state UP
    link/ether 48:b0:2d:2f:64:83
    inet6 fe80::2686:ced4:2bbb:29f2/64 scope link
```

**No `inet` line.** Only a self-assigned link-local IPv6 that routes nowhere.

**Step 3 — DHCP was asking, nothing was answering:**

```
DHCPDISCOVER on eth0 to 255.255.255.255 port 67 interval 3
DHCPDISCOVER ... interval 6
DHCPDISCOVER ... interval 8   (eight attempts, no DHCPOFFER)
```

A working exchange shows `DHCPOFFER` then `DHCPACK`. Only DISCOVER with growing
backoff means the request is going out and nothing is replying.

**Step 4 — the actual cause.** Windows reported **"Network cable unplugged"** on
its Ethernet adapter, while the Nano reported `LOWER_UP`. One end saw a link and
the other did not.

**The fix:** `ncpa.cpl` → right-click Ethernet → **Disable**, wait, **Enable**.
Windows had the adapter in a stuck state. Re-seating the cable at both ends did
nothing; the disable/enable cycle fixed it immediately.

**This explains all three earlier failures.** They were never a boot problem or a
DHCP race — the Windows adapter intermittently drops into a phantom
"unplugged" state, probably after sleep/wake.

**New first step when SSH fails:** disable/enable the Windows Ethernet adapter,
before touching the board.

### Rebuilding the engine — a 2.2% improvement

`models/build_trt_engine.py` replaces the day-zero `trtexec` invocation. It also
prints the platform capability flags, which is a stronger citation for the README
than documentation:

```
platform_has_fast_fp16 : True
platform_has_fast_int8 : False
```

**That is TensorRT's own API declaring INT8 unavailable on this hardware**, and it
is reproducible by anyone with the board.

| Engine | Build time | Inference | CV |
|---|---|---|---|
| Spike (`trtexec`, during setup) | 654.8 s | 51.76 ± 0.13 ms | 0.25% |
| **Rebuilt (script, idle board)** | **362.6 s** | **50.64 ± 0.12 ms** | 0.23% |

**2.17% faster, and five times the CV** — so a real difference, not noise. The
"different models of devices" warning also disappeared.

**Likely cause:** the rebuild ran headless on an idle board with more free memory,
letting the auto-tuner consider kernels it previously skipped for lack of
workspace. The near-halved build time is consistent with a less constrained
environment.

**A finding in its own right:** TensorRT's auto-tuning is sensitive to the state
of the machine it runs on. An engine built during setup, with a desktop running,
was measurably worse than the same conversion on an idle board.

---

## PROBLEM 7 — a config key that silently produced a wrong row

Running `--precision fp32` returned **50.74 ms** — identical to FP16.

`app/backends.py` read `cfg["model"]["engine"]`, a single hardcoded path pointing
at the FP16 file. The `--precision` flag reached the CSV row label but never
reached engine selection.

**So the run loaded the FP16 engine and recorded it as FP32.** The numbers were
real; the label was a lie. That is worse than a crash — a crash gets fixed, a
mislabelled row gets published.

### Fix

Two keys in `configs/params.yaml`:

```yaml
  engine_fp16: models/yolov5nu_fp16.engine
  engine_fp32: models/yolov5nu_fp32.engine
```

And selection by key, with an **explicit failure** rather than a fallback:

```python
key = "engine_%s" % precision
if key not in cfg["model"]:
    raise ValueError("no %s in config -- build it with "
                     "models/build_trt_engine.py --precision %s" % (key, precision))
```

The raise is the important part. A missing engine should stop the run, not
quietly substitute a different one.

The mislabelled row (`231d2542c8bd`) and its `.npz` were deleted.

**Same family as two earlier problems:** `clocks_locked` recording `false` while
`jetson_clocks` was applied, and `peak_mem_mb` measuring host RSS while looking
like it might mean GPU memory. **A field that can silently disagree with reality
is worse than no field**, because it looks like evidence.

---

## THE DECOMPOSITION

Three runs each, same board, same harness, 10 W, clocks locked, fan off.

| Configuration | Inference | E2E FPS | Peak memory | Detections | CV |
|---|---|---|---|---|---|
| TorchScript FP32 | 93.15 ± 0.47 ms | 8.21 | ~1150 MB | 8.33 | 0.50% |
| **TensorRT FP32** | **70.89 ± 0.01 ms** | 10.04 | **1414 MB** | 8.33 | **0.014%** |
| **TensorRT FP16** | **50.64 ± 0.12 ms** | 12.60 | **1086 MB** | 8.32 | 0.23% |

**The 1.84× splits almost evenly:**

| Change | Speedup | What it is |
|---|---|---|
| TorchScript → TensorRT (both FP32) | **1.31×** | the **runtime**: kernel fusion, auto-tuned kernel selection, static memory planning |
| TensorRT FP32 → FP16 | **1.40×** | the **precision**: halved memory traffic on a bandwidth-bound board |
| Combined | **1.84×** | 1.31 × 1.40 = 1.84 ✓ |

**This is why the FP32 engine was worth building.** Without it the headline is a
single number mixing two independent changes. With it, the contribution of each is
separable and the mechanism behind each is nameable.

### A second finding: FP16 saves 328 MB

| Precision | Engine on disk | Peak process memory |
|---|---|---|
| FP32 | 22.9 MB | 1414 MB |
| FP16 | 11.8 MB | 1086 MB |

**On a board with ~1.4 GB usable, FP32 at 1414 MB is effectively at the ceiling.**
FP16 buys headroom as well as speed — and on this hardware the headroom may matter
more, since it is what makes room for the rest of the pipeline.

### The tightest measurement in the project

**TensorRT FP32: 70.88, 70.89, 70.90 ms. CV 0.014%.**

An order of magnitude tighter than anything else measured, including the FP16
runs. Locked clocks, an idle headless board, and a deterministic engine.

### Accuracy, so far as detection counts show it

**8.33 for both TensorRT precisions**, against TorchScript's 8.33. The 1.40×
precision speedup cost nothing measurable at this level.

**That is not accuracy.** Detection count is insensitive to boxes shifting or
scores moving slightly. The COCO mAP evaluation is what turns this into a claim.

### Build times

| Engine | Build time |
|---|---|
| FP32 | 151.0 s |
| FP16 | 362.6 s |

**FP16 takes 2.4× longer to build** — the auto-tuner has both precisions available
per layer, so the search space is larger.

---

## 2026-09-06 — Part 5 complete: the accuracy cost, measured

### Full validation set, not a subset

The plan called for a fixed 500-image COCO subset. Revised to the **full 5,000
val2017 images** once the throughput was known.

The reasoning: with 500 images, confidence intervals on mAP are wide enough that
differences below roughly one point are indistinguishable from sampling noise —
and the expected FP16 difference was *far* smaller than that. Detection counts
already agreed to 0.1%, so a subset would have measured nothing.

Cost: ~9 minutes of Nano inference per configuration, three configurations, and a
777 MB one-off transfer. Worth it to remove "is this real or sampling noise" from
the answer entirely.

### Evaluation runs at conf 0.001, not 0.25

Deliberate, and worth being able to explain.

**mAP is the area under the precision–recall curve.** High-recall points come only
from low-confidence detections. Cutting at the runtime's 0.25 truncates the curve
and understates mAP — consistently across runtimes, but not comparably with any
published figure.

**0.001 with max_det=300 is the COCO convention** — Ultralytics, MMDetection,
Detectron2 and the original YOLO papers all use it.

| Purpose | Threshold |
|---|---|
| Deployment | 0.25 — clean output, not 106 boxes per frame |
| Evaluation | 0.001 — the whole PR curve |

That is why `params.yaml` says 0.25 while `run_coco_detections.py` overrides to
0.001. Two different jobs.

*(Detections averaged 106.2 per image, so the 300-detection cap was never hit.)*

### The category id trap

COCO category ids are **not** 0..79. The original 91-class set had 11 removed, so
the ids skip: 12, 26, 29, 30, 45, 66, 68, 69, 71, 83, 91.

Model index 11 maps to category **13**, not 12.

A naive `cls + 1` would score every detection against the wrong class and produce
a plausible-looking near-zero mAP rather than an error. The `COCO91` list in
`run_coco_detections.py` was verified for length, uniqueness, ordering and gap
positions before use.

---

## THE ACCURACY RESULT

Full COCO val2017, 5,000 images, 36,781 annotations.

| Runtime | Precision | mAP@50-95 | mAP@50 | mAP@75 | Detections | Inference | E2E FPS |
|---|---|---|---|---|---|---|---|
| TorchScript | FP32 | **0.3343** | 0.5005 | 0.3529 | 530,418 | 93.15 ms | 8.21 |
| TensorRT | FP32 | **0.3343** | 0.5005 | 0.3529 | 530,417 | 70.89 ms | 10.04 |
| TensorRT | FP16 | **0.3342** | 0.5003 | 0.3529 | 531,012 | 50.64 ms | 12.60 |

### The headline

> **TensorRT FP16 is 1.84× faster than TorchScript FP32 on a Jetson Nano 2GB,
> for a 0.03% mAP loss** — 0.3343 → 0.3342 mAP@50-95 over the full 5,000-image
> COCO validation set.

### Three things this establishes

**The TensorRT conversion is provably faithful.** TorchScript and TensorRT FP32
give **identical mAP to four decimal places** — 0.3343, 0.5005, 0.3529 across all
three metrics — and differ by **one detection out of 530,418**. Two completely
different inference engines, the same numbers. Not "close enough": identical.

**FP16 costs essentially nothing.** −0.0001 mAP@50-95, −0.0002 mAP@50, and
mAP@75 unchanged at 0.3529. For 1.40× speed and 328 MB less memory.

**The whole pipeline is validated end to end.** 0.3343 against Ultralytics'
published **0.343** for `yolov5nu` — 0.9 points apart, and the gap is explained:
Ultralytics evaluates with a rectangular stride-aligned letterbox and an explicit
`max_det=300`, while this project pads to a square 640×640 because that is what
the ONNX export declares and the engine was built for.

**That last point is the important one.** A subtly wrong decoder still produces
detections, which is why the 99.2% agreement check against Ultralytics was
necessary but not sufficient. mAP against real ground truth is the check that
cannot be fooled — a wrong category mapping, a broken letterbox-undo, or a
misread output layout would all show up here as a collapsed score.

### The shape of the numbers is right too

| Object size | mAP@50-95 |
|---|---|
| small | 0.153 |
| medium | 0.369 |
| large | 0.468 |

Small objects are three times harder than large — exactly the expected profile for
a nano-scale detector at 640×640.

---

## The fan has a thermal governor

Discovered by accident, and it changes how Part 7 has to be designed.

`echo 255 > /sys/devices/pwm-fan/target_pwm` took effect, then the value **reset
to 0 on its own**. Later, under sustained load, it **rose to 80 without
intervention** at somewhere around 50 °C.

`systemctl list-units | grep -i fan` returns only `ubuntu-fan.service` — which is
container networking (Fabric Area Network), not cooling. Unfortunate name
collision. So the control lives in the kernel `pwm-fan` driver or `nvpmodel`'s
thermal policy, not a userspace daemon that can simply be stopped.

**Consequences:**

**Earlier "fan off" runs were genuinely fan-off** — they peaked at 42–49.5 °C,
just under the trigger. So those results stand. But `--fan false` means "the
governor did not engage", not "the fan was disabled", and the notes should say so.

**Part 7 cannot compare fan-on against fan-off by writing to `target_pwm`** — a
ten-minute run will heat past the threshold and the governor will engage partway
through, producing a mixed condition.

**Decision: measure with the governor active.** "Sustained performance under stock
thermal management" is the more honest claim anyway — it is how the board actually
behaves in service. The FPS-versus-temperature curve with the fan engaging
partway is a *better* result than an artificially suppressed one.

Trigger temperature is approximately 50 °C, observed rather than measured
precisely. Worth pinning down during the Part 7 runs.

---

## 2026-09-06 — Part 6: containerisation costs nothing

### Two Part 0 steps had silently not applied

`docker images` returned a permission error. `groups` showed no `docker` group —
the `usermod -aG docker $USER` from the Part 0 hardening never took.

And `/etc/docker/daemon.json` defined the nvidia runtime but **omitted
`"default-runtime": "nvidia"`**, so every `docker run` had been using the standard
runtime with no GPU access.

**Neither failed loudly at the time.** Both surfaced only when something actually
needed them, weeks later.

*(Group membership applies to new login sessions only — after `usermod`, log out
and back in.)*

### The base image gives less than expected

```
docker run --rm --runtime nvidia nvcr.io/nvidia/l4t-base:r32.7.1 \
    python3 -c "import tensorrt; print(tensorrt.__version__)"
→ 8.2.1.8
```

**TensorRT works inside a base image that contains no TensorRT.** The nvidia
container runtime bind-mounts it from the host. That is why the image stays small
and why it must be built and run on the Jetson.

Everything else had to be installed:

| Dependency | Source |
|---|---|
| TensorRT 8.2.1.8 | **mounted from host** |
| CUDA runtime | in the base image |
| OpenCV, NumPy, PyYAML, PyCUDA | ✗ installed in the Dockerfile |

**OpenCV is a real difference worth stating.** `apt install python3-opencv` gives
the plain Debian build — no CUDA, no GStreamer, unlike JetPack's 4.1.1 on the
host. Adequate here, since the pipeline uses `imread`, `resize` and
`VideoCapture` on a file, none of which need the CUDA build. Mounting the host's
OpenCV would keep it, but would make the container depend on host paths and
defeat the point.

---

## Three failures, each one an implicit host dependency

This is the part worth keeping. **Every failure was something the host provided
ambiently that the container did not inherit** — which is exactly what
containerisation exists to surface.

### 1. `--no-build-isolation` unrecognised

```
no such option: --no-build-isolation
```

Ubuntu ships **pip 9.0.1**; the flag arrived in pip 10. The host had been upgraded
to 20.3.4 back in Part 0, so it worked there.

**Fix:** upgrade pip in the container first, capped below 21.0 which drops Python
3.6 entirely.

### 2. UnicodeDecodeError reading `params.yaml`

```
UnicodeDecodeError: 'ascii' codec can't decode byte 0xe2 in position 329
```

Byte `0xe2` is the first byte of an **em dash** — in a YAML *comment*, on the
`workspace_mb` line.

**The container has no locale.** Python 3.6 falls back to ASCII for file I/O and
fails on any non-ASCII character. The host has a UTF-8 locale and never noticed.

**Fix:** `ENV LANG=C.UTF-8` and `LC_ALL=C.UTF-8`. `C.UTF-8` needs no locale data
installed, unlike `en_US.UTF-8`.

### 3. `ModuleNotFoundError: No module named 'six'`

`pycuda.driver` imports `six` at runtime but **does not declare it as a
dependency**. The host had it from some other package.

**Fix:** install it explicitly alongside PyCUDA.

### Why this sequence is the point

Three separate implicit dependencies — a pip version, a locale, an undeclared
import — none of which were visible on the host, all of which would have been
deployment-time surprises.

**A container that builds and runs is a proof that the dependency list is
complete.** That is a stronger claim than "it works on my machine", and it is the
argument for containerising that people usually skip past.

---

## `.dockerignore` — a 15,000× reduction

The first builds reported:

```
Sending build context to Docker daemon  1.192GB
```

**Docker copies the entire project directory to the daemon before building** —
including `data/coco/`, 777 MB of images the container never uses.

After adding `.dockerignore`:

```
Sending build context to Docker daemon  76.29kB
```

Nothing about the image changed; every subsequent build just stopped wasting
minutes shipping data it would discard.

### Layer ordering matters

Adding the `ENV LANG` lines invalidated the cache for **every step below them**,
forcing apt and PyCUDA to rebuild — 10 minutes.

The Dockerfile is ordered so this rarely bites: `ENV` and dependency installs
first, `COPY app/` last. Reversing that would mean **every code change rebuilds
PyCUDA**.

---

## THE RESULT: zero container overhead

Three runs each, same board, same engine, same video, 10 W, clocks locked.

| | Bare metal | Container | Difference |
|---|---|---|---|
| Inference | 50.64 ± 0.12 ms | **50.50 ± 0.06 ms** | **−0.27%** |
| End-to-end FPS | 12.60 | **12.55** | −0.34% |
| p95 inference | 50.93 ms | 50.81 ms | — |
| CV | 0.23% | **0.13%** | — |
| Cold start (warm cache) | 4191 ms | 4351 ms | +4% |
| Objects per frame | 9 | 9 | identical |

**Both differences sit below the run-to-run CV.** Containerisation costs nothing
measurable.

### Why this is expected — and why measuring it still matters

Docker is **process isolation using kernel namespaces and cgroups, not
virtualisation**. There is no hypervisor between the code and the GPU; the nvidia
runtime bind-mounts the device nodes and libraries directly. Near-zero overhead is
what theory predicts.

But plenty of people assume containers cost 5–10% on GPU workloads. **"Theory
predicts" and "I measured it" are different claims**, and this project now has the
second one.

*(The container's CV is actually lower — 0.13% vs 0.23%. Probably chance with
three samples, though a container does run in a slightly more isolated process
environment.)*

### Cold start, and a caveat about it

First container run: **8713.9 ms**. Second: 4472.5. Third: 4229.5.

That is **page-cache warming**, not container overhead — the engine file had not
been read recently. Same effect seen on the laptop, where cold start varied
10,295 → 3,811 ms for identical code.

### A note on measurement provenance

`app/run.py` reports to stdout; it does **not** append to `results/speed.csv`.
Deliberate: it is a service, not the measurement harness, and `benchmarks/` is not
copied into the image.

Mixing its numbers into `speed.csv` would break the one-instrument rule the whole
results table rests on. The container rows are reported separately and labelled as
coming from `app/run.py`.

---

## Next
- [ ] Extra rows: `--workspace=512` on the Nano; a separate laptop-built engine
- [ ] Record a longer clip (60–90 s) before the Part 7 thermal runs
- [ ] Part 3: ONNX parity — validate the NumPy decoder against Ultralytics on
      identical frames before trusting any accuracy number
- [ ] Part 4: port to the Nano; check the preprocessing-share prediction above
- [ ] Extra rows: `--workspace=512` on the Nano; a separate laptop-built engine
- [ ] Record a longer clip (60–90 s) before the Part 7 thermal runs — 34.7 s
      loops ~17× in a 10-minute run, and the repeating pattern could look like an
      artifact in the FPS-over-time plot

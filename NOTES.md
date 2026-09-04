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

## Next

- [ ] Part 3: ONNX parity — validate the NumPy decoder against Ultralytics on
      identical frames before trusting any accuracy number
- [ ] Part 3: ONNX parity — validate the NumPy decoder against Ultralytics on
      identical frames before trusting any accuracy number
- [ ] Part 4: port to the Nano; check the preprocessing-share prediction above
- [ ] Extra rows: `--workspace=512` on the Nano; a separate laptop-built engine
- [ ] Record a longer clip (60–90 s) before the Part 7 thermal runs — 34.7 s
      loops ~17× in a 10-minute run, and the repeating pattern could look like an
      artifact in the FPS-over-time plot

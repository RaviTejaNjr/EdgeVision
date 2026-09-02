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

## Next

- [ ] Part 1: repo scaffold, `configs/params.yaml`, record and normalise test video
- [ ] Part 2: `benchmark.py` — cold-start vs steady-state, per-stage timing
- [ ] Extra rows: `--workspace=512` on the Nano; a separate laptop-built engine

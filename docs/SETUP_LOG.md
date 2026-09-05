# EdgeVision — Setup Command Reference

Every command run during setup, what it does, and what happened. Kept as a
reference so any step can be repeated or explained later.

Two machines are involved and they are **not** interchangeable:

| | Jetson Nano 2GB | Laptop (Windows) |
|---|---|---|
| OS | Ubuntu 18.04 (JetPack 4.6) | Windows |
| Python | 3.6.9 — system, no venv | 3.10 — conda env |
| Role | TensorRT inference | model export, evaluation, reporting |
| Package manager | `pip3 --user` | `pip` inside conda env |

**Always `python3` on the Nano. Never `python`.**

---

## Part 0.5 — Jetson Nano: prove the TensorRT path exists

### 1. Confirm TensorRT Python bindings

```bash
python3 -c "import tensorrt; print(tensorrt.__version__)"
```

Imports the TensorRT Python module and prints its version. This is the single
most important check in the whole setup — if TensorRT is not importable from the
system Python, the project has no path forward on this board.

**Output:** `8.2.1.8` ✅

---

### 2. Confirm the engine-building tool exists

```bash
ls -lh /usr/src/tensorrt/bin/trtexec
```

`trtexec` is NVIDIA's command-line tool for converting ONNX models into TensorRT
engines and benchmarking them. It ships with JetPack rather than pip. `-l` shows
permissions (need the `x` execute bits), `-h` shows a human-readable size.

**Output:** `-rwxr-xr-x 1 root root 390K Nov 17 2021 /usr/src/tensorrt/bin/trtexec` ✅

---

### 3. Put CUDA on the PATH

```bash
export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
```

`PATH` tells the shell where to find executables — this adds the CUDA toolkit's
`bin` so `nvcc` (the CUDA compiler) can be found. `LD_LIBRARY_PATH` does the same
for shared libraries at runtime.

**Both print nothing. Silence means success.**

⚠️ **Session-only.** These are lost on reboot. To make permanent:

```bash
echo 'export PATH=/usr/local/cuda/bin:$PATH' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
```

---

### 4. Verify the CUDA compiler

```bash
nvcc --version
```

**Output:** CUDA release **10.2**, V10.2.300 ✅

10.2 is pinned by JetPack 4.6 and cannot be upgraded independently on this board.

---

### 5. Install Python development headers

```bash
sudo apt-get install -y python3-dev
```

Provides the C header files (`Python.h` and friends) that any package compiling
a C extension needs — PyCUDA among them. `-y` auto-confirms prompts.

**Output:** completed with `Processing triggers for ...` lines ✅

*Note: the password prompt shows nothing as you type. Not even asterisks. That is
normal Linux behaviour.*

---

### 6. Recover from an interrupted apt install

```bash
sudo dpkg --configure -a
```

Finishes configuring any package left half-installed when apt was interrupted.
Safe to run even when nothing is broken — it simply prints nothing.

**Output:** nothing ✅ (no packages were left half-configured)

---

## PROBLEM 1 — pip installs failing intermittently

### Symptom

```
Failed to establish a new connection: [Errno 113] No route to host
Failed to establish a new connection: [Errno -2] Name or service not known
```

Looked like a flaky Ethernet/ICS link. It was not.

### Diagnosis — the two-ping test

```bash
ping -c 3 8.8.8.8      # IP literal: does a route out exist at all?
ping -c 3 pypi.org     # hostname: does name resolution give a usable address?
```

**Results:**

```
8.8.8.8   →  0% packet loss                          ✅
pypi.org  →  PING pypi.org(2a04:4e42:400::223)       ❌ 100% packet loss
```

DNS worked fine. The problem was the *kind* of address it returned:
`2a04:4e42:400::223` is **IPv6**. Windows Internet Connection Sharing only routes
IPv4, so every IPv6 attempt went nowhere and hung until timeout. `8.8.8.8`
worked because it is an IPv4 literal that never needed resolving.

### Root cause

The Nano prefers IPv6 addresses when resolving hostnames, but has no working
IPv6 route over the ICS link.

### Fix

```bash
sudo nano /etc/gai.conf
```

Uncomment this line by deleting the leading `#`:

```
precedence ::ffff:0:0/96  100
```

Save with `Ctrl+O`, `Enter`, exit with `Ctrl+X`.

This raises the priority of IPv4-mapped addresses in the system's address
selection policy, so IPv4 is tried first.

### Verification

```bash
ping -c 3 pypi.org
```

**Output:** `PING pypi.org (151.101.0.223)` — 0% packet loss ✅

### Why this matters

The failure presented as an unreliable network and could easily have been blamed
on the cable, the ICS setup, or the used board — costing hours across several
sessions. The two-ping split (IP literal vs hostname) is the diagnostic that
distinguishes "no route" from "wrong address family".

---

### 7. Check pip

```bash
sudo pip3 --version
```

**Output:** `pip 9.0.1 from /usr/lib/python3/dist-packages (python 3.6)`

Stock version, quite old. Works, but fails in confusing ways on packages that
compile.

*Note: an earlier "command not found" was a typo, not a missing pip.*

---

### 8. Upgrade pip, capped below 21

```bash
sudo -H pip3 install --upgrade "pip<21.0" setuptools wheel
```

- `-H` sets `HOME` to root's home, avoiding cache-permission warnings about
  `/home/raviteja/.cache/pip`
- **`"pip<21.0"` is not optional.** pip 21 dropped Python 3.6 support entirely.

**Output:** `Successfully installed pip-20.3.4 setuptools-59.6.0 wheel-0.37.1` ✅

The `Not uninstalling ... outside environment /usr` messages are normal — new
versions installed alongside system packages rather than replacing them.

---

## PROBLEM 2 — PyCUDA trying to compile numpy 1.12.1 from 2017

### Symptom

```bash
sudo -H pip3 install pycuda
```

```
Collecting numpy==1.12.1
  Building wheel for numpy (setup.py): still running...
numpy/core/src/multiarray/numpyos.c:18:10: fatal error: xlocale.h: No such file or directory
ERROR: Failed building wheel for numpy
```

### Root cause

PyCUDA's build metadata declares `numpy==1.12.1` for Python 3.6. pip obeys this
literally: it creates an **isolated build environment** and tries to compile that
exact 2017 numpy from source, ignoring the perfectly good numpy already
installed.

numpy 1.12.1 includes `<xlocale.h>`, a header removed from glibc years ago. It
**cannot** succeed. Retrying is pointless.

### Fix, in three parts

**a) Confirm a usable numpy already exists**

```bash
python3 -c "import numpy; print(numpy.__version__)"
```

**Output:** `1.13.3` ✅ (system numpy from Ubuntu)

**b) Do not use sudo**

`sudo` resets `PATH`, so the build would not find `nvcc`. Install to the user
directory instead:

```bash
export PATH=/usr/local/cuda/bin:$PATH
which nvcc
```

**Output:** `/usr/local/cuda/bin/nvcc` ✅

**c) Install with build isolation disabled**

```bash
pip3 install --user --no-build-isolation --timeout 60 --retries 10 "pycuda==2020.1"
```

| Flag | Why |
|---|---|
| `--user` | installs to `~/.local/`, leaves system packages untouched |
| `--no-build-isolation` | **the actual fix** — uses the existing numpy instead of building 1.12.1 |
| `--timeout 60` | ICS link is jittery (9–52 ms); default 15 s was too aggressive |
| `--retries 10` | rides out brief drops |
| `pycuda==2020.1` | era-appropriate version for Python 3.6 |
| *no* `sudo` | preserves the CUDA `PATH` |

**Output:**

```
Building wheel for pycuda (setup.py) ... done
Successfully installed appdirs-1.4.4 dataclasses-0.8 platformdirs-2.4.0
  pycuda-2020.1 pytools-2022.1.12 typing-extensions-4.1.1
```
✅ — and much faster than expected, because it reused the existing numpy.

*Ignore the "pip 21.3.1 is available" warning. Staying on 20.3.4 is deliberate.*

---

### 9. Verify PyCUDA can reach the GPU

```bash
python3 -c "import pycuda.driver as d; d.init(); print(d.Device(0).name())"
```

More than an import check — this initialises the CUDA driver and asks the GPU to
identify itself.

**Output:** `NVIDIA Tegra X1` ✅

Both halves of the inference stack are now confirmed: **TensorRT** for the
engine, **PyCUDA** for moving data to and from the GPU.

---

## Why no virtual environment on the Nano

Deliberate, not an oversight.

- **TensorRT's Python bindings are a system package**, installed by JetPack at
  `/usr/lib/python3.6/dist-packages/tensorrt/`. A fresh venv cannot see it, so
  `import tensorrt` would fail — and TensorRT is the entire point of the project.
- **JetPack's OpenCV 4.1.1** is built with CUDA and GStreamer support. Inside a
  venv you would lose it and end up pip-installing a CPU-only build that shadows
  the good one.

`pip3 install --user` provides the isolation instead: packages go to
`~/.local/lib/python3.6/site-packages/`, system packages stay clean, and the
whole lot can be wiped with `rm -rf ~/.local/lib/python3.6`.

If a venv were ever needed: `python3 -m venv --system-site-packages venv` would
work, but adds confusion for no gain on a single-project board.

**Proper isolation comes at Part 6** — a Docker container built on `l4t-base`.
That is the real answer to reproducibility here, and why containerisation is in
the plan rather than being optional polish.

---

## Part 1 — Laptop: development environment

### 10. Create the project folder

```
C:\Users\gravi\Portfolio\EdgeVision
```

---

### 11. Create a conda environment inside the project folder

```
conda create -p venv python=3.10 -y
conda activate venv/
```

`-p` (short for `--prefix`) creates the environment **inside the project folder**
rather than in conda's central store, so deleting the project removes the
environment with it.

Two syntax notes:

- **`python=3.10`, single equals.** Conda's `==` means *exactly* that string, and
  no build is literally "3.10" (they are 3.10.13, 3.10.14, …). The `==` habit
  comes from pip.
- **`conda activate venv/` needs the slash.** Without it, conda looks for a
  *named* environment called "venv" and fails. Anything with a path separator is
  treated as a path: `venv/`, `./venv`, `.\venv` all work.

Optional prompt tidy-up (the full path is ugly by default):

```
conda config --set env_prompt "({name})"
```

⚠️ Add `venv/` to `.gitignore`. The environment must never be committed.

---

### 12. Fix PowerShell blocking activation

**Symptom:**

```
venv\Scripts\activate : File ...Activate.ps1 cannot be loaded because running
scripts is disabled on this system.
```

**Fix:**

```powershell
Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned
```

Confirm with `Y`. Affects your user only, not the system. `RemoteSigned` allows
locally-created scripts while still requiring signatures on downloaded ones —
Microsoft's own recommended setting for development.

*(Not needed if using Anaconda Prompt rather than PowerShell.)*

---

### 13. Install Ultralytics

```
pip install ultralytics
```

`pip` not `conda` — Ultralytics is not in the main conda channels, and
conda-forge lags. General rule inside a conda env: conda for what conda packages
well (Python, numpy, scipy, CUDA toolkits), pip for everything else. Mixing is
fine; installing the *same* package with both is not.

**Output:** `Successfully installed ... torch-2.13.0 ultralytics-8.4.137
torchvision-0.28.0 ...` ✅

---

### 14. Swap CPU torch for the CUDA build

The default torch from PyPI is **CPU-only**. Laptop GPU benchmark rows need CUDA.

**Check what the driver supports:**

```
nvidia-smi
```

**Output:** CUDA Version **12.2**

**Remove the CPU build:**

```
pip uninstall torch torchvision -y
```

**Install the CUDA build:**

```
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

`cu121` is built against CUDA 12.1 and runs fine on a 12.2 driver — CUDA minor
versions are forward-compatible. ~2.4 GB, since the CUDA libraries are bundled.

**Output:** `Successfully installed sympy-1.13.1 torch-2.5.1+cu121
torchvision-0.20.1+cu121` ✅

Note the torch version went **down**, 2.13.0 → 2.5.1. The cu121 index does not
carry the newest releases. Not a problem — 2.5.1 is well within Ultralytics'
supported range. The `+cu121` suffix is the confirmation it is the CUDA build.

⚠️ Do this **before** writing any code against the CPU build.

---

### 15. Verify torch can reach the GPU

```
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

The package being installed does not guarantee the driver connection works. This
is the real test.

**Output:** `2.5.1+cu121 True NVIDIA GeForce RTX 3050 Laptop GPU` ✅

---

### 16. ONNX export dependencies

```
pip install onnx onnxruntime-gpu onnxslim
```

Ultralytics does not pull these in by default. **`onnxruntime-gpu`, not plain
`onnxruntime`** — the GPU build enables the ONNX Runtime CUDA benchmark rows,
which are a genuinely interesting middle point between PyTorch and TensorRT.

**Output:** `Successfully installed ... onnx-1.22.0 onnxruntime-gpu-1.23.2
onnxslim-0.1.96 ...` ✅

⚠️ To watch later: onnxruntime-gpu 1.23 is built against CUDA 12.x, which matches
the driver, but ORT sometimes needs cuDNN present separately on Windows. Will
surface when the ORT-CUDA rows are actually run — the export step does not need
the GPU.

---

### 17. Export YOLOv5n to ONNX

`export_to_onnx.py`:

```python
from ultralytics import YOLO

model = YOLO("yolov5nu.pt")
model.export(format="onnx", opset=13, imgsz=640, simplify=True, dynamic=False)
```

```
python export_to_onnx.py
```

Two arguments are load-bearing:

| Argument | Why |
|---|---|
| `opset=13` | TensorRT 8.2 supports roughly opset 13–14. A newer opset exports fine on the laptop, then fails at engine build with unsupported-node errors |
| `dynamic=False` | Fixed 640×640 input. Dynamic shapes complicate TensorRT and buy nothing here |

**Output:**

```
Downloading yolov5nu.pt: 5.3MB
YOLOv5n summary (fused): 84 layers, 2,649,200 parameters, 7.7 GFLOPs
PyTorch: starting from 'yolov5nu.pt' with input shape (1, 3, 640, 640) BCHW
         and output shape(s) (1, 84, 8400) (5.3 MB)
ONNX: starting export with onnx 1.22.0 opset 13...
ONNX: slimming with onnxslim 0.1.96...
ONNX: export success 8.5s, saved as 'yolov5nu.onnx' (10.3 MB)
```
✅

**Two numbers that matter later:**

- **Output shape `(1, 84, 8400)`** — this is the tensor `postprocess.py` must
  decode by hand. 8400 candidate boxes × 84 values = 4 box coordinates + 80 COCO
  class scores. **There is no separate objectness score**, which means this is
  the newer YOLOv8-style anchor-free head. Important when writing the decoder.
- **2,649,200 parameters, 7.7 GFLOPs** — the model-size baseline.

⚠️ **Naming accuracy for the README:** `yolov5nu` is Ultralytics' retrofit of
YOLOv5 with an anchor-free head. It is *not* the original 2020 YOLOv5. State this
correctly — an interviewer who knows the difference will notice.

---

### 18. Validate the ONNX file before shipping it

`check_onnx.py`:

```python
import onnx

m = onnx.load("yolov5nu.onnx")
onnx.checker.check_model(m)
print("valid")
print("opset:", m.opset_import[0].version)
for i in m.graph.input:
    print("input:", i.name, [d.dim_value for d in i.type.tensor_type.shape.dim])
for o in m.graph.output:
    print("output:", o.name, [d.dim_value for d in o.type.tensor_type.shape.dim])
```

```
python check_onnx.py
```

Confirms the opset really is 13 and the shapes are fixed rather than dynamic —
both would break the engine build on the Nano, and both are cheaper to catch here
than after the file has been transferred.

**Output:**

```
valid
opset: 13
input: images [1, 3, 640, 640]
output: output0 [1, 84, 8400]
```
✅

No zeros or dynamic dimensions. Tensor names — `images` in, `output0` out — are
what the inference code will bind to.

---

### 19. Transfer the ONNX file to the Nano

**Find the Nano's IP — run on the Nano:**

```bash
hostname -I
```

**Output:** `192.168.137.244  172.17.0.1  2003:d1:c705:3ffd:e9ba:a300:544b:5b2a`

| Address | What it is |
|---|---|
| `192.168.137.244` | the ICS link to the laptop — **this is the one to use** |
| `172.17.0.1` | Docker's internal bridge. Confirms Docker is installed and running (needed at Part 6) |
| `2003:d1:...` | an IPv6 address it picked up. Harmless — the `gai.conf` fix means it is not preferred outbound |

**Copy the file — run on the LAPTOP, in the folder containing the file:**

```
scp yolov5nu.onnx raviteja@192.168.137.244:~/
```

`~/` is the destination — shorthand for the home directory of the user you log in
as, so the file lands at `/home/raviteja/yolov5nu.onnx`.

⚠️ **Mistake made here:** ran `scp` on the *Nano* rather than the laptop, which
told the Nano to copy a file to itself. Result: `yolov5nu.onnx: No such file or
directory`.

**Telling the two machines apart at a glance:**

| Prompt looks like | You are on |
|---|---|
| `raviteja@raviteja-pc:~$` | the Nano |
| `C:\Users\gravi\...>` | Windows |

On first connection ssh asks to confirm the host fingerprint. Type `yes` in full
— a bare `y` is rejected.

**Organise it on the Nano:**

```bash
mkdir -p ~/edgevision/models
mv ~/yolov5nu.onnx ~/edgevision/models/
ls -lh ~/edgevision/models/
```

**Output:** `-rw-rw-r-- 1 raviteja raviteja 11M Sep 1 18:39 yolov5nu.onnx` ✅

---

### 20. Build the TensorRT FP16 engine — the real test

```bash
cd ~/edgevision/models
/usr/src/tensorrt/bin/trtexec --onnx=yolov5nu.onnx --fp16 --workspace=256 --saveEngine=yolov5nu_fp16.engine
```

| Flag | Purpose |
|---|---|
| full path to `trtexec` | it is not on `PATH` |
| `--onnx=` | input model |
| `--fp16` | allow half precision — the main optimisation |
| `--workspace=256` | cap build scratch memory at 256 MB |
| `--saveEngine=` | write the compiled engine to disk |

Run `jtop` in a second terminal to watch memory. Over SSH that means a second SSH
window, not `Ctrl+Alt+T`.

**Memory during the build peaked at 1.7/1.9 GB.** Left running deliberately —
with swap enabled, hitting the ceiling means slowdown, not a crash. Killing it
would have thrown away progress.

**Result:** `&&&& PASSED` ✅

#### Numbers

| Metric | Value |
|---|---|
| Engine build time | **654.8 s** (~11 min) |
| Throughput | **21.7 qps** |
| GPU compute time | 45.28 ms mean |
| Host latency | 46.01 ms mean, p99 **46.17 ms** |
| Latency spread | min 45.92 / max 46.17 — 0.25 ms across 67 runs |
| H2D transfer | 0.46 ms |
| D2H transfer | 0.27 ms |
| Enqueue time | 9.81 ms mean |
| Engine size on disk | 12 MB (ONNX was 11 MB) |
| Peak GPU memory during build | ~1935 MiB |

#### Reading the output

**Fusion, visible in the log.** Entries like:

```
PWN(PWN(/model.0/act/Sigmoid), /model.0/act/Mul)
/model.2/cv1/conv/Conv || /model.2/cv2/conv/Conv
```

`PWN(...)` is a **fused kernel** — Sigmoid and Mul (the two halves of SiLU)
collapsed into one operation. `||` means two convolutions running in parallel.
This is concrete evidence of what TensorRT actually did, not just a claim. Worth
screenshotting for the README.

**Transfers are negligible** — 0.73 ms combined out of 46 ms, about 1.6%. Data
movement is not the bottleneck; compute is. Useful to know *before* spending time
optimising transfers.

**Enqueue time (9.81 ms) is well under compute time (45.28 ms)**, so the CPU is
not starving the GPU. That would change with heavy preprocessing on the same
thread.

⚠️ **This is engine throughput, not application throughput.** Measured with
random input tensors: no image decode, no letterbox resize, no NMS. End-to-end
application FPS will be meaningfully lower. This is the first half of the
engine-vs-end-to-end comparison the plan makes a headline result.

#### Warnings seen — all benign

| Warning | Meaning |
|---|---|
| `Some tactics do not have sufficient workspace memory to run` | Some faster kernels need >256 MB scratch, so TensorRT skipped them. May cost a little performance. **Worth trying `--workspace=512` later as a legitimate benchmark row.** |
| `Your ONNX model has been generated with INT64 weights... casting down to INT32` | Standard for PyTorch-exported ONNX. Handled automatically. |
| `---------- Layers Running on DLA ----------` (empty) | Correct. The Nano has no DLA accelerator; those start with Xavier. Same family of hardware limits as the INT8 situation. |

---

### 21. Confirm the engine file exists

```bash
ls -lh ~/edgevision/models/
```

**Output:**

```
total 23M
-rw-rw-r-- 1 raviteja raviteja 12M Sep  1 18:57 yolov5nu_fp16.engine
-rw-rw-r-- 1 raviteja raviteja 11M Sep  1 18:39 yolov5nu.onnx
```
✅

The engine is slightly *larger* than the ONNX — the compiled kernel selections
and weight layouts add to the payload.

---

### 22. Load the engine from Python

Proves the Python path works, not just the command-line tool.

`~/edgevision/models/spike_test.py`:

```python
import tensorrt as trt
import pycuda.autoinit

logger = trt.Logger(trt.Logger.WARNING)
with open("yolov5nu_fp16.engine", "rb") as f:
    engine = trt.Runtime(logger).deserialize_cuda_engine(f.read())

print("engine loaded")
for i in range(engine.num_bindings):
    kind = "INPUT " if engine.binding_is_input(i) else "OUTPUT"
    print(kind, engine.get_binding_name(i), engine.get_binding_shape(i), engine.get_binding_dtype(i))
```

```bash
cd ~/edgevision/models
python3 spike_test.py
```

**Output:**

```
engine loaded
INPUT  images  (1, 3, 640, 640) DataType.FLOAT
OUTPUT output0 (1, 84, 8400)    DataType.FLOAT
```
✅

Bindings match the ONNX exactly.

**Important for `postprocess.py`:** output dtype is **FLOAT, not HALF**. FP16 is
used *internally* for computation, but TensorRT hands back fp32 at the boundary.
The NumPy decoder reads float32 — no conversion needed.

---

## PROBLEM 3 — CUDA context destroyed at exit (benign, but will recur)

### Symptom

After all output printed successfully, the script hung and emitted:

```
[TRT] [E] 1: [defaultAllocator.cpp::deallocate::35] Error Code 1: Cuda Runtime (invalid argument)
[TRT] [E] 1: [cudaDriverHelpers.cpp::operator()::29] Error Code 1: Cuda Driver (context is destroyed)
```

`Ctrl+C` then produced `Segmentation fault (core dumped)`.

### Root cause

**Destruction-order bug, not a code bug.** `pycuda.autoinit` registers a cleanup
that destroys the CUDA context when Python exits. TensorRT's engine object is
still alive and gets garbage-collected *afterwards*, so it tries to free GPU
memory in a context that no longer exists. `Ctrl+C` interrupted that cleanup
mid-flight, hence the segfault.

### Impact

**None.** The engine file is untouched, the GPU driver is fine, and all required
output printed before any of this. The script's actual work completed.

### Fix for real code

In `infer_trt.py`, manage the context explicitly rather than using
`pycuda.autoinit` — keep a reference to the context and delete the engine before
it, or use `pycuda.driver` manually. Not worth fixing in a throwaway test, but it
**will** recur in the real inference code.

---

## Part 0.5 — COMPLETE ✅

| Check | Result |
|---|---|
| TensorRT Python bindings | 8.2.1.8 ✅ |
| `trtexec` present and executable | ✅ |
| PyCUDA installed, GPU reachable | NVIDIA Tegra X1 ✅ |
| ONNX transferred, FP16 engine built | 654.8 s, `&&&& PASSED` ✅ |
| Engine loads from Python, bindings correct | ✅ |

**No blockers.** Everything from here is execution rather than discovery.

**Baseline established:** 21.7 FPS engine throughput, 45.28 ms GPU compute,
p99 46.17 ms, 12 MB engine.

---

## Working across two machines — scp direction

Tripped over twice. Worth internalising before Parts 2–6, which involve pushing
code to the Nano constantly.

### The client/server relationship

**The Nano runs an SSH server** (`sshd`, listening on port 22, running since
JetPack was flashed). **The laptop is the client** — it initiates connections;
nothing listens on it.

`scp` rides on top of SSH and inherits the same relationship: the laptop can
reach the Nano, not the other way around.

**Therefore: always run `scp` from the laptop**, regardless of which direction the
file is travelling. Not because scp is one-way, but because only one machine is
reachable.

### The syntax

Always `scp SOURCE DESTINATION` — source first, destination second, like
copy-paste. The `user@ip:` prefix marks whichever side is remote:

```
# laptop → Nano  (push)
scp yolov5nu.onnx raviteja@192.168.137.244:~/
    └── source ──┘ └────── destination ──────┘
    local file      remote folder

# Nano → laptop  (pull)
scp raviteja@192.168.137.244:~/edgevision/models/yolov5nu_fp16.engine .
    └─────────────── source ────────────────────┘ └ destination
     remote file                                    local folder
```

`.` means "the folder I am currently in". Both commands run **on the laptop**.

### Mistakes made

**Ran the pull command on the Nano.** It copied the file to itself, reported
success at 66.7 MB/s (suspiciously fast — it never left the board), and `ls`
showed no new file.

**Then tried a Windows path from the Nano:**

```
scp raviteja@192.168.137.244:~/.../yolov5nu_fp16.engine C:\Users\gravi\...\models\
>
```

Hung at a `>` continuation prompt. Linux does not understand `C:\Users\...`, and
the trailing `\` made bash treat the line as unfinished.

### Check the prompt before pressing Enter

| Prompt | Machine |
|---|---|
| `raviteja@raviteja-pc:~/...$` | **Nano** |
| `PS C:\Users\gravi\...>` | **Windows** |

This is the single cheapest habit to build while working across two terminals.

### Could scp run *from* the Nano?

Only with setup that is not worth doing: Windows ships OpenSSH **Server** as an
optional feature (Settings → Apps → Optional Features), which then needs the
`sshd` service started and a firewall port opened. Then the Nano could push to
`192.168.137.1` — the laptop's address on the ICS link, always `.1`.

Adding a permanent network service to the laptop to avoid switching terminal
windows is not a good trade. Skipped.

### Worth knowing for later

`rsync` works over SSH the same way and only transfers what changed. Much faster
than repeated `scp` when pushing code to the Nano during Parts 2–6.

---

## Memory check after the spike — not a leak

`jtop` showed 537 MB used, up from the ~350 MB baseline after the desktop was
disabled. Investigated:

```bash
ps aux | grep -i python
```

**Finding:** `spike_test.py` was gone — it exited cleanly despite the segfault.
The memory belonged to **jetson-stats itself**: three background daemon processes
(PIDs 6129, 6282, 6290) started at boot, plus the interactive `jtop` window
(PID 7007, 15.8% CPU).

The tool used to watch memory was a meaningful share of the memory being watched.

537 MB of 1.9 GB leaves ~1.4 GB available. The engine build peaked at 1.9 GB and
still succeeded, so this is not a constraint. Nothing to fix.

*Note: `free -h` is the honest view — read the **available** column, not "free".
Linux counts cache as used, but releases it instantly under pressure.*

---

## Card backup — deliberately skipped

The working state now diverges from `jetson_base.img`: the IPv6 fix, PyCUDA, the
upgraded pip, and the built engine are all missing from it.

**Decision: do not re-image yet.** This log documents every command with its
output, so recreating the state is roughly 30 minutes of following these notes,
versus ~40 minutes and 128 GB of disk to produce an image.

The one expensive artifact — the engine, 11 minutes to build — is 12 MB and can
simply be copied off with `scp`.

Revisit after Part 6, when a built container and much more slow-to-recreate state
exist.

---

## Quick reference — recurring commands

**Nano**

```bash
export PATH=/usr/local/cuda/bin:$PATH          # after every reboot unless in .bashrc
sudo shutdown -h now                           # then wait 10 s before pulling power
jtop                                           # live temp / memory / power monitor
hostname -I                                    # find the Nano's IP for SSH or scp
ping -c 3 8.8.8.8 && ping -c 3 pypi.org        # the two-ping network diagnostic
```

**Laptop**

```
conda activate venv/
nvidia-smi
```

**Terminal shortcuts on the Nano desktop**

| Action | Keys |
|---|---|
| New terminal | `Ctrl+Alt+T` |
| Copy | `Ctrl+Shift+C` |
| Paste | `Ctrl+Shift+V` |
| Cancel running command | `Ctrl+C` |

**Editing files with nano**

| Action | Keys |
|---|---|
| Save | `Ctrl+O` then `Enter` |
| Exit | `Ctrl+X` |
| Navigate | arrow keys only — the mouse does nothing |

---

## Part 1 — Laptop: repository and test data

### 23. Video toolchain

```
winget install Gyan.FFmpeg
```

⚠️ **Restart the shell afterwards.** winget modifies `PATH`, but the running
terminal keeps its old copy — `ffmpeg -version` fails until a new one is opened.
In VS Code, close the whole window, not just the terminal panel.

**Inspect before transcoding:**

```
ffprobe -v error -show_entries stream=width,height,r_frame_rate,codec_name \
        -show_entries format=duration -of default=noprint_wrappers=1 input.mp4
```

**Output:** h264, 1280×720, 30/1 fps, 34.67 s, plus an AAC audio stream.

**Transcode to a reproducible input:**

```
ffmpeg -i Inference_Video.mp4 -vf scale=1280:720 -r 30 -c:v libx264 -crf 23 \
       -preset medium -an data/test_video.mp4
```

| Flag | Why |
|---|---|
| `-vf scale=1280:720` | force exact resolution |
| `-r 30` | force **constant** frame rate — the load-bearing one |
| `-c:v libx264` | H.264; OpenCV reads it reliably |
| `-crf 23` | quality, 0–51, lower is better |
| `-an` | strip audio |

**Why `-r 30` matters even though the source is already 30 fps:** phone and editor
exports often use *variable* frame rate, dropping or duplicating frames during
static scenes. "30 fps" is then nominal, and any FPS measurement inherits the
variance.

**Output:** `frame= 1040 ... time=00:00:34.60` — 1040 frames at 30 fps is exactly
34.67 s, confirming constant frame rate.

---

### 24. Verify through OpenCV, not ffmpeg

OpenCV uses different decoders, and the entire pipeline reads through it. A file
ffmpeg writes happily can still fail to open in OpenCV or report a wrong frame
count.

```python
cap = cv2.VideoCapture("data/test_video.mp4")
print(cap.isOpened(), int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))
n = 0
while cap.read()[0]:
    n += 1
print("actually read:", n)
```

| Check | Result |
|---|---|
| opened | True |
| reported frames | 1040 |
| **actually read** | **1040** |
| fps | 30.0 |
| size | 1280 × 720 |

Reported and actual match — no container metadata quirks.

---

### 25. Git repository

```
git init
git branch -M main                    # GitHub's default; avoids a push mismatch
git remote add origin https://github.com/RaviTejaNjr/EdgeVision.git
git add .
git commit -m "..."
git push -u origin main               # -u links local main to origin/main
```

**Empty directories do not exist to git.** It tracks files, not folders, so every
otherwise-empty directory needs a placeholder:

```
type nul > app\.gitkeep        # Windows equivalent of `touch`
```

`app/` later got `__init__.py` instead, which serves the same purpose and makes
the directory an importable package.

**Chaining commands in cmd:** `&` runs each regardless of the previous result;
`&&` stops at the first failure.

See `docs/GIT_NOTES.md` for the remote-edit incident and the fetch/diff/pull
pattern.

---

### 26. `app/__init__.py` — required, not optional

```
del app\.gitkeep
type nul > app\__init__.py
```

`benchmark.py` does `from app import postprocess, preprocess`. **Python 3.6
requires `__init__.py` for that**; newer versions are more forgiving via implicit
namespace packages.

Without it the code runs on the laptop and fails on the Nano — the worst kind of
bug, because it appears only on the machine that matters.

---

### 27. Remaining laptop dependencies

```
pip install pyyaml psutil pytest
```

`pyyaml` is imported directly by `benchmark.py`. `psutil` provides peak memory —
the code falls back to Linux's `resource` module, but Windows has no fallback.
`pytest` runs `tests/test_parity.py` and will be what CI invokes at Part 10.

*(All three were already present as Ultralytics dependencies.)*

---

## PROBLEM 4 — the laptop GPU ran at 10% of its clock

**This invalidated every laptop measurement taken before it was found.**

### Symptom

Engine p95 rose from 29.90 ms (200 frames) to 71.61 ms (500 frames) against a p50
of 27.07 — a 2.6× tail that appeared only in longer runs.

### Wrong hypothesis 1 — thermal throttling

The laptop's GPU fan was disconnected for noise, so this looked obvious.

### Wrong hypothesis 2 — memory pressure

`peak_mem_mb` had grown 3048 → 5108 MB, and 5 GB on a 4 GB card would mean
spilling into shared system memory.

### The diagnostic

```
nvidia-smi --query-gpu=timestamp,temperature.gpu,clocks.sm,utilization.gpu,memory.used \
           --format=csv -l 1
```

Run in a second terminal alongside the benchmark. `-l 1` polls once per second.

| Hypothesis | Evidence | Verdict |
|---|---|---|
| Thermal | 57 → 70 °C; Ampere throttles ~87 °C+ | ❌ |
| GPU memory | flat at **1212 MiB** | ❌ |
| **Clock speed** | **pinned at 210 MHz throughout** | ✅ |

```
nvidia-smi --query-gpu=clocks.max.sm,power.limit --format=csv
→ 2100 MHz, [N/A]
```

**210 MHz against a 2100 MHz ceiling — exactly 10%**, while reporting 85%
utilisation. Genuinely busy, just crawling.

*(`power.limit` returning `[N/A]` is normal for laptop GPUs under WDDM; vendor
firmware manages it rather than exposing it.)*

### Reconnecting the fan made things WORSE

| Stage | fan off | fan on |
|---|---|---|
| inference | 33.66 ms | 74.33 ms |
| preprocess | 5.67 ms | 22.72 ms |
| capture | 2.32 ms | 6.75 ms |
| postprocess | 2.19 ms | 7.86 ms |

**The tell: CPU-only stages slowed 3–4× too.** Preprocessing and NMS never touch
the GPU. Reconnecting a *GPU* fan cannot slow down NumPy — so the whole machine
was slower, not the GPU.

### Root cause: Windows "Whisper" power mode

A vendor quiet profile capping both CPU and GPU clocks. Active for every run up to
that point.

```
powercfg /getactivescheme
```

**Fix:** Windows Settings → System → Power & battery → Power mode → **Best
performance**. On mains.

### Result

| Power state | GPU clock | Inference | Engine FPS | E2E FPS |
|---|---|---|---|---|
| Whisper | 210 MHz | 74.33 ms | 13.5 | 9.0 |
| Balanced | ~550–1057 MHz | 18.16 ms | 55.1 | 31.4 |
| **Best performance** | ~950–1057 MHz | **14.05 ms** | **71.2** | **41.1** |

**5.3× from power settings alone**, identical code and hardware.

### Deliberately not pursued further

Even at Best performance the GPU peaks near 1057 MHz of 2100 and utilisation tops
out at ~42% — a vendor profile still caps it.

Not chased, because at 42% utilisation the GPU is idle more than half the time,
waiting on the CPU. Capture + preprocess + postprocess = **10.3 ms of CPU work**
against 14.05 ms of GPU work. Raising the GPU clock shrinks the 14 ms and leaves
the 10.3 ms untouched.

### The fan was irrelevant

Measured afterwards, three runs each on mains:

| Condition | Inference |
|---|---|
| Fan on | 47.11 ± 1.72 ms |
| Fan off | 48.80 ± 2.51 ms |

Overlapping error bars. **No measurable effect on CPU inference.** Fan stays
detached.

---

## PROBLEM 5 — battery mode costs 2.6× on CPU

| State | Inference | E2E FPS | Sustained 60 s |
|---|---|---|---|
| Battery | **132.95 ms** | 6.23 | **4.88** |
| Mains | **51.54 ms** | 15.54 | — |

**Windows caps CPU clocks hard on battery regardless of the "Best performance"
setting.**

The progress log shows the collapse live: 74.9 → 313.0 → 214.9 → 204.6 ms. Normal
for ~100 frames, then the battery power cap engages.

`fps_sustained_last_60s` caught it — 4.88 against a mean of 6.23. First time that
column earned its place.

**Consequence: every laptop measurement must be on mains.**

---

## Measurement protocol

Adopted after Problems 4 and 5, and followed for every row in `results/speed.csv`:

1. **Mains power.** Battery costs 2.6× on CPU.
2. **Highest performance profile**, recorded via `--host-profile`.
3. **Three runs minimum per configuration.** Two cannot establish a difference
   below ~15% — an 8.4% FP16-vs-FP32 gap was observed, withdrawn as noise, then
   confirmed once 8 and 6 samples existed.
4. **GPU runs before CPU runs**, so CPU load does not heat the machine and skew
   the GPU measurements.
5. **~30 s between runs**, so each starts from a similar thermal state.
6. On the Jetson: `nvpmodel` mode set and `jetson_clocks` applied, both recorded.

---

### 28. Harness change: `--host-profile`

Added after the fact, because two runs of identical code differed by **5×** with
nothing in the CSV to explain it.

```
python benchmarks/benchmark.py --runtime pytorch --device cuda --precision fp32 \
    --frames 500 --host-profile best-performance-mains --notes "run 1"
```

On the Jetson `nvpmodel -q` reports power state automatically; **Windows exposes
nothing equivalent**, so it is supplied explicitly and folded into the config
hash.

Existing rows were moved to `results/speed_exploratory.csv` rather than deleted —
they are the evidence behind Problem 4. See `results/README.md`.

---

## Part 3 — Validation

### 29. ONNX export parity

```
pytest tests/test_parity.py -v
→ 6 passed in 3.24s
```

Five frames through both PyTorch and ONNX Runtime, comparing raw
`(1, 84, 8400)` tensors at `atol/rtol = 1e-3`.

**Why not exact equality:** FP32 arithmetic is not associative. Two runtimes that
fuse or reorder operations differently will not produce bit-identical output.

**`test_class_scores_are_probabilities`** asserts the 80 class columns lie in
[0, 1]. If the head had an objectness column the columns would be offset by one
and the last would hold unbounded box data — so this passing confirms the
anchor-free layout.

---

### 30. Decoder validation against Ultralytics

```
python evaluation/validate_decoder.py --frames 10 --save-overlay
```

| Metric | Result |
|---|---|
| Agreement (IoU ≥ 0.9, same class) | **99.2%** — 117 of 118 |
| Box coordinate error, mean | **0.546 px** |
| Box coordinate error, max | 8.485 px |
| Score error, mean | 0.018 |

**Sub-pixel mean error establishes the letterbox-undo maths is correct.**

**Every unmatched detection fell between conf 0.252 and 0.287**, against a 0.25
threshold — borderline cases, as predicted.

Visual inspection of the overlays showed ours found a person Ultralytics missed
(frame 0, conf 0.257), and produced three tight boxes on individual kites where
Ultralytics produced one enormous box across the whole bunting line (frame 808).
The latter is an NMS difference, and ours is the more sensible output.

**The max errors are preprocessing, not decoding.** Ultralytics pads to a stride
multiple with a rectangular letterbox; `app/preprocess.py` always pads to a square
640×640. Different input pixels, slightly different predictions. Square padding is
correct here — it is what the ONNX export declares and what the engine was built
for.

Tolerances set to 10 px / 0.2 accordingly, with the reasoning in the source.

**Bug found:** the first run crashed on frame 2 with
`Input type (torch.FloatTensor) and weight type (torch.cuda.FloatTensor) should be
the same`. **`yolo.predict()` moves the underlying model to CUDA as a side
effect**, so the next raw call fed a CPU tensor to a GPU model. Fixed by pinning
the model, input tensor and `predict()` to CPU explicitly.

---

## Part 4 — Nano: running the pipeline on target

### 31. Recovering an unresponsive board

First power-on after several days: no SSH, `ping` returned 100% loss.

```
arp -a | findstr 192.168.137
→ Interface: 192.168.137.1 --- 0x10
  192.168.137.244  48-b0-2d-2f-64-83  static
```

Laptop had `.1` so ICS was running; the Nano's MAC was cached. `static` rather
than `dynamic` is the hint — a cached record, not proof the board is alive.

**Resolution: attach a monitor, restart.** Second time it booted normally. This
has now happened twice; a third occurrence would warrant investigating SD card
seating or power delivery at boot.

### 32. Making the CUDA paths permanent

```bash
echo 'export PATH=/usr/local/cuda/bin:$PATH' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
tail -2 ~/.bashrc                                   # verify
```

⚠️ **`>>` appends; a single `>` truncates the file first.** One character between
adding a line and destroying a shell configuration.

### 33. Back to headless

Attaching the monitor had started the desktop.

```bash
sudo systemctl set-default multi-user.target
sudo reboot
```

| State | Used | Available |
|---|---|---|
| Desktop | 592 MB | 1.2 GB |
| **Text mode** | **271 MB** | **1.6 GB** |

**321 MB recovered** — 16% of the board's total.

```bash
systemctl get-default                       # which is active
sudo systemctl set-default graphical.target # to reverse
sudo systemctl start graphical.target       # desktop now, boot unchanged
```

HDMI is hot-pluggable; the monitor can be pulled while running.

### 34. Cloning the repo with a personal access token

**GitHub → Settings → Developer settings → Personal access tokens →
Fine-grained tokens.**

| Field | Value |
|---|---|
| Name | `edgevision-nano` |
| Repository access | Only select repositories → EdgeVision |
| Permissions → Contents | **Read and write** |

Write access matters: results are generated on the Nano and pushed from there.

```bash
mv ~/edgevision ~/spike_artifacts     # avoid ~/edgevision vs ~/EdgeVision
cd ~
git clone https://github.com/RaviTejaNjr/EdgeVision.git
```

Username `RaviTejaNjr`, token as the password. Nothing displays while pasting.

**Why clone rather than `scp`:** without a repo on the Nano, the `git_commit`
column in `speed.csv` reads `unknown`, breaking the provenance chain. Code also
iterates between machines through Parts 5–6.

### 35. Placing the artifacts

```bash
ls -R ~/spike_artifacts                              # -R recurses
mv ~/spike_artifacts/models/yolov5nu.onnx models/
mv ~/spike_artifacts/models/yolov5nu_fp16.engine models/
```

Both are gitignored, so the clone could not bring them, but
`configs/params.yaml` expects them in `models/`.

**From the laptop**, the one file that must be copied manually:

```
scp data\test_video.mp4 raviteja@192.168.137.244:~/EdgeVision/data/
```

### 36. Dependency check before installing anything

```bash
python3 -c "import numpy, cv2; print(numpy.__version__, cv2.__version__)"
→ 1.13.3 4.1.1
python3 -c "import yaml; print('ok')"      → ok
python3 -c "import psutil"                 → ModuleNotFoundError
```

**Check before installing.** numpy and OpenCV are JetPack builds with CUDA and
GStreamer support; pip versions would shadow them with inferior ones.

```bash
pip3 install --user psutil
```

Arrived as a **prebuilt aarch64 wheel for cp36** — no compilation. Version 7.2.2,
matching the laptop, so memory figures are comparable.

### 37. Locking the board state before measuring

```bash
sudo nvpmodel -m 0          # MAXN / 10W
sudo jetson_clocks          # lock clocks against DVFS
sudo nvpmodel -q            # verify
cat /sys/devices/pwm-fan/target_pwm    # 0 = off, 255 = full
```

`jetson_clocks` does **not** start the fan. To control it:

```bash
sudo sh -c 'echo 255 > /sys/devices/pwm-fan/target_pwm'
```

Fan deliberately left **off** for the baseline — the fan-on comparison is a Part 7
experiment, and changing conditions midway would invalidate it.

### 38. The benchmark run

```bash
cd ~/EdgeVision
python3 benchmarks/benchmark.py --runtime tensorrt --precision fp16 \
    --frames 500 --host-profile jetson-10w-clocks-locked-fan-off \
    --clocks-locked true --fan false --notes "nano tensorrt fp16 run 1"
```

**Worked first attempt, no code changes.** Three runs:

| Metric | Mean | CV |
|---|---|---|
| Inference | **51.76 ms** | **0.25%** |
| Engine FPS | 19.32 | 0.23% |
| End-to-end FPS | 12.51 | 0.46% |
| Mean detections | 8.32 | — |

⚠️ **`--clocks-locked` is not auto-detected.** The first run recorded `false`
despite `jetson_clocks` having been applied — the harness takes whatever the flag
says. Pass it explicitly.

⚠️ **TensorRT warning on every run:**

```
[TRT] [W] Using an engine plan file across different models of devices is not
recommended and is likely to affect performance or even cause errors.
```

The engine was built during the day-zero spike, before several reboots. Numbers
are stable and detections correct, so probably benign. Part 5 rebuilds it properly.

---

## Still to do

- [ ] **Part 4 remainder:** PyTorch baseline on the Nano — NVIDIA torch wheel, torchvision from source,
      YOLOv5 dependencies pinned for Python 3.6. **Timeboxed to 2 hours**; fall
      back to Option B if torchvision does not compile
- [ ] Check the preprocessing-share prediction: CPU stages are ~40% of the frame
      on an i9; on four ARM Cortex-A57 cores they should dominate
- [ ] Try `--workspace=512` on the Nano as an additional benchmark row
- [ ] Build a TensorRT engine on the **laptop** too — separate engine from the same
      ONNX, its own results row
- [ ] Fix the CUDA context teardown properly in `app/backends.py` — **already
      written in**, but verify it works on device (see Problem 3)
- [ ] Re-image the SD card after Part 6, not before

# Results

Measurements are a deliverable of this project, so this directory is committed
rather than gitignored.

## Files

| File | Rows | Schema | Contents |
|---|---|---|---|
| `speed.csv` | growing | 37 columns | The clean record. Every run carries a `host_profile` describing the machine's power state. |
| `speed_exploratory.csv` | 23 | 36 columns | Earlier laptop runs, kept for the findings they produced. **No `host_profile` column.** |
| `raw/*.npz` | one per run | — | Per-frame timing arrays, keyed by `run_id` |
| `accuracy.csv` | — | — | mAP per configuration. Added at Part 5. |

## Why there are two speed files

`host_profile` was added after the exploratory runs, in response to a problem they
exposed: **two runs of identical code on identical hardware differed by 5×**, and
nothing in the CSV explained why.

The cause was the laptop's Windows power profile. It had been in a vendor "Whisper"
mode, holding the GPU at 210 MHz against a 2100 MHz ceiling. Later, a run on
battery was 2.6× slower than the same run on mains. Neither state was visible to
the harness — `nvpmodel -q` reports it on the Jetson, but Windows exposes nothing
equivalent — so it is now supplied explicitly on the command line and folded into
the config hash.

Rows in `speed_exploratory.csv` were taken at three different power profiles and
can only be told apart by reading the `notes` field. They are kept because they
are the evidence behind that finding, not because they are usable baselines.

**Use `speed.csv` for anything comparative.**

## Reading `speed.csv`

### Provenance

| Column | Meaning |
|---|---|
| `run_id` | Unique per execution. Matches the filename in `raw/`. |
| `config_hash` | Stable SHA-1 of the configuration. Join key to `accuracy.csv`. |
| `git_commit` | Short SHA, suffixed **`-dirty`** when the working tree had uncommitted changes — so a row cannot claim to come from code that is not what actually ran. |
| `host_profile` | Power state: profile, mains vs battery, fan. See above. |

Several rows may share a `config_hash` while differing substantially in speed.
That is expected and correct: the hash identifies the *model configuration*, and
accuracy does not depend on clock speed, so one accuracy row per hash is right.
Speed rows are distinguished by `host_profile` and `notes`.

### Timing

Two throughput figures are reported, and they are not interchangeable:

- **`fps_engine`** — inference alone
- **`fps_end_to_end`** — capture + preprocess + inference + postprocess + NMS

On the laptop these are 73.6 and 42.7 FPS: a **42% drop**. Capture,
preprocessing and NMS all run on the CPU. Most published edge benchmarks report
only the first figure.

`cold_start_ms` covers model loading, engine deserialisation and buffer
allocation — measured once, excluded from steady state. It is roughly 250× the
cost of a frame, and it varies with filesystem caching: the same run measured
10,295 ms cold and 3,811 ms warm.

`fps_sustained_last_60s` is throughput over the final 60 seconds only, and is
blank for runs shorter than that. It exists to catch degradation that a mean
hides — on the battery run it read 4.88 against a mean of 6.23.

p50 and p95 are reported alongside means because tail latency is what matters for
a real-time system. A mean of 46 ms with a p99 of 200 ms means one frame in a
hundred arrives catastrophically late.

### Two columns that are easy to misread

**`peak_mem_mb` is host memory, not GPU memory.** It is the Python process's
resident set size — interpreter, torch, CUDA libraries and all. During a run where
this column read 5115 MB, `nvidia-smi` showed GPU memory flat at 1212 MiB.

**`throttled` is filled in by hand** after inspecting a run. It is not detected
automatically.

## Reproducing a row

Every row can be regenerated from its own fields:

```bash
python benchmarks/benchmark.py \
  --runtime pytorch --device cuda --precision fp32 \
  --frames 500 --host-profile best-performance-mains \
  --notes "reproduction"
```

Check out the `git_commit` from the row first. Rows marked `-dirty` cannot be
reproduced exactly, since the code that produced them was never committed.

## Measurement protocol

Established after the power-profile findings, and followed for every row in
`speed.csv`:

1. **Mains power.** Battery costs 2.6× on CPU inference regardless of the
   Windows power setting.
2. **Highest performance profile**, and recorded in `host_profile`.
3. **Three runs minimum per configuration.** Two runs cannot establish a
   difference below roughly 15% — an 8.4% FP16-vs-FP32 gap was first observed,
   then withdrawn as noise, then confirmed once eight and six samples existed.
4. **GPU runs before CPU runs**, so CPU load does not heat the machine and skew
   the GPU measurements.
5. **~30 s between runs**, so each starts from a similar thermal state.
6. On the Jetson: `nvpmodel` mode set and `jetson_clocks` applied, both recorded.

## Current variance

| Configuration | n | Inference | CV |
|---|---|---|---|
| Laptop GPU FP32 | 8 | 13.94 ± 1.08 ms | 7.7% |
| Laptop GPU FP16 | 6 | 15.11 ± 0.42 ms | 2.8% |
| Laptop CPU FP32 | 6 | ~47.5 ms | ~4% |

FP32 is roughly 2.7× noisier than FP16, consistent with it pushing the GPU harder
and hitting the power ceiling more often.

**Any comparison smaller than the relevant CV should not be treated as a
difference.**

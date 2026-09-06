# TODO — EdgeVision

Things deferred deliberately, so they are not lost and not allowed to block
shipping. Anything here is *after* the current part, not instead of it.

---

## Immediate — before Part 5

- [ ] **Rebuild the engine with `models/build_trt_engine.py`.** The current one
      came from `trtexec` during the day-zero spike, and every run logs
      *"Using an engine plan file across different models of devices is not
      recommended"*. A script also makes it reproducible.
- [ ] **Fix the two blank-line PEP 8 issues** in `app/backends.py` — one before
      `class TorchScriptBackend`, one before `class OnnxBackend`. Ruff will flag
      them at Part 10.
- [ ] **Make `--clocks-locked` auto-detected** rather than trusting the flag. The
      first Nano run recorded `false` despite `jetson_clocks` having been applied.
      A field that can silently disagree with reality is worse than no field.
- [ ] Update `PROJECT_PLAN.md` — Parts 0–4 are complete, and it still describes
      Part 4 as the risky torchvision build that turned out to be unnecessary.

---

## Deferred to Part 10 — README and repo presentation

- [ ] **Re-add the CI badge** to `README.md`, the same day
      `.github/workflows/ci.yml` first goes green. Removed for now because it
      renders as a broken image until the workflow exists, and a broken badge at
      the top of a README looks careless. Note it only works on public repos.
- [ ] **Fill in the accuracy table** once `evaluation/coco_eval.py` runs — the
      skeleton with mAP@50 / mAP@50-95 / Δ is already in place.
- [ ] **Screenshot the `PWN(...)` fusion lines** from the TensorRT build log.
      Concrete evidence of kernel fusion, better than a speedup number alone.
- [ ] **Add an architecture diagram** as an image rather than ASCII — the ASCII
      version is fine in a terminal but weak in a rendered README.
- [ ] Consider a **thermal curve plot** (FPS vs time vs temperature) once the
      ten-minute runs exist. Currently the strongest un-visualised result.

---

## Carried forward from the spike and Part 4

- [x] ~~Fix the CUDA context teardown~~ — done, written into
      `app/backends.py::TensorRTBackend.close()` from the start rather than
      rediscovered.
- [ ] Try `--workspace=512` on the Nano as an extra benchmark row. The build
      logged *"Some tactics do not have sufficient workspace memory to run"* at
      256 MB, so faster kernels were skipped and there may be performance left on
      the table.
- [ ] Build a TensorRT engine **on the laptop** too — separate engine from the
      same ONNX, its own results row. One export, three hardware targets, each
      requiring its own compilation. *(Needs the vendor fan profile set to
      Performance, or the numbers are meaningless.)*
- [ ] **Record a longer test clip** (60–90 s) before the Part 7 thermal runs. The
      current 34.7 s clip loops ~17× in a ten-minute run, and the repeating
      pattern could look like an artifact in the FPS-over-time plot.
- [ ] Re-image the SD card **after Part 6**, not before.

---

## Known measurement caveats to revisit

- [ ] **`peak_mem_mb` measures host RSS, not GPU memory.** Either rename the
      column or add a separate GPU-memory reading via `tegrastats` on the Nano.
- [ ] **`throttled` is filled in by hand**, not detected. Could be derived from
      the temperature trace in the `.npz` files.
- [ ] The **laptop GPU never exceeded ~1567 MHz of 2100**. Worth one attempt with
      the vendor profile at maximum to see whether the ceiling moves, though the
      laptop is only a reference platform.

---

## Git — learn properly once v1.0 has shipped

Interest noted, deliberately deferred. Most of this will not stick without having
hit it first, and git is currently a tool needed to work rather than a topic to
study. The four or five commands already in use cover almost everything this
project needs.

**Suggested approach:** create a throwaway repo — `git-sandbox` — and deliberately
break it. Make a conflict on purpose. Force-push and watch what happens. Clone it
into two folders and pretend to be two people. An hour of that teaches more than a
day of articles, and nothing is at risk.

### Solo, single branch

- [ ] Stale local copy — already hit, documented in `GIT_NOTES.md`
- [ ] The ahead / behind / diverged states — documented
- [ ] Accidentally committing something that should have been gitignored, and
      removing it from history
- [ ] Undoing a commit that has already been pushed (`revert` vs `reset`)
- [ ] Forgetting to commit before switching machines

### Solo, multiple branches

- [ ] Branching for a feature, then merging back
- [ ] `merge` vs `rebase` — when each is appropriate
- [ ] Cleaning up history before it becomes public (interactive rebase, squash)
- [ ] What `--force-with-lease` is actually for, and why plain `--force` is
      dangerous
- [ ] Stashing work in progress

### Multiple people

- [ ] The same conflicts, but with someone else's intent involved
- [ ] Pull requests, review, and branch protection
- [ ] Someone force-pushed and rewrote history you had already pulled
- [ ] Two branches that both refactored the same file
- [ ] `git blame` and `git bisect` for finding when something broke

---

## Repo presentation — at Part 6, when it goes public

- [ ] Flip the repo from private to public
- [ ] Add topics: `jetson-nano`, `tensorrt`, `edge-ai`, `computer-vision`, `onnx`
- [ ] Fill in the About description
- [ ] Pin the repo to the GitHub profile
- [ ] Confirm the CI badge renders (it only works on public repos)

---

## Phase 2 — after v1.0 ships

Tracked in full in `PROJECT_PLAN.md` §10. Summary:

- [ ] Cross-hardware model comparison matrix, with mAP alongside latency and a
      Pareto frontier
- [ ] Object tracking (ByteTrack or similar)
- [ ] ROS2 perception pipeline — the Part 8 optional route, properly built
- [ ] Transformer feasibility study: MobileViT-XXS primary, RT-DETR-R18 at
      320×320 as a documented boundary test
- [ ] Action recognition — MoViNet-A0 or X3D-XS on-device, or a cascade sending
      keyframes to a laptop VLM
- [ ] RAG / agents project — **a separate repo**, not an EdgeVision extension

**Excluded, not deferred:** INT8 (SM 5.3 < required CC 6.1), self-hosted CI on the
Nano, MLOps infrastructure hosted on the Nano, multi-camera, SLAM, large
segmentation models.

---

## Career — from the earlier planning

- [ ] Send applications after **Part 6**, not Part 10 — repo public with the
      benchmark table and Docker comparison in the README
- [ ] Fix `mAP@50-90` → `mAP@50-95` on the CV (appears twice, both in headline
      achievements)
- [ ] Prepare the defence for the 91% mAP@50-95 figure on the Studienarbeit — it
      is above most published 3D detection results and will be questioned
- [ ] Trim the CV skills list; remove the "Work Style" block
- [ ] Move Kubernetes and AWS to a "Familiar" sub-line until there is evidence
- [ ] German B1 — the biggest structural constraint for German roles

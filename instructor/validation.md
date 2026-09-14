# Validation and release gates

Two kinds of checks. `tools/validate.sh` is automatic and runs on an author
machine; it regenerates every `.ipynb` from its `.py` source and executes the
core notebooks headless in CPU-fallback mode and, if an NVIDIA GPU is present,
in GPU mode. The manual checks below run in Google Colab and cannot be
scripted, because Colab hardware, quotas and package images change.

## Automatic (author machine)

```bash
uv sync --project env --extra gpu       # once; drop --extra gpu without a GPU
GPU_EXTRA=1 tools/validate.sh            # regenerates notebooks, executes core set
uv run --project env python tools/dump_outputs.py validation/gpu/02_gpu.ipynb   # eyeball outputs
```

Passing means: every core notebook (00a, 00b and its solution, 01, 02, 03, 04)
runs to the end without an error in both modes, and `git status` shows no
unexpected `.ipynb` change (source and notebook in sync). Executed copies land
in `validation/`, which is ignored by git. The extensions are regenerated but
not executed; run them by hand with `EXTRA_NOTEBOOKS="extensions/ext_gil_and_task_pools.ipynb"`.

Last automatic run on the author machine: 13 September 2026, 14 of 14
executions passed (32-core CPU, RTX 4090, numpy 2.5.3, numba 0.67.0,
cupy 14.2.0, Python 3.12.14).

## Manual Colab checks

Do these with a **student-level Google account**, not the author's, from the
links in the README. Record date, runtime type, GPU model (if any) and the
package versions printed by each setup cell.

### Fresh CPU runtime

1. Open `00_colab_ready` from the README link. Save a copy. Run all. Last cell
   says `READY for the workshop`. Restart session, Run all: still READY.
   Disconnect and delete runtime, Run all: packages reinstall, still READY.
   Note how long the install took.
2. Open `00_stencil_practice`. Run all: the check prints `FAIL` with the hint
   (placeholder still present). Fix the slice, rerun: `PASS`.
3. Open `01_measure_and_multicore`. Run all. Note the thread counts tested and
   whether the recorded sweep was loaded. Every cell under 30 s. Total
   computation under 5 minutes.
4. Open `02_gpu` on the **CPU** runtime. Run all. Banner says CPU fallback;
   the `RECORDED GPU RUN` line shows the intended metadata (after gate 4, a
   Colab GPU); plot shows live CPU and recorded GPU curves.
5. Open `03_parallel_models` and `04_capstone`. Run all. Capstone GPU rows
   absent, everything else complete.

### GPU runtime

6. Change runtime type to GPU. Open `02_gpu`. Run all. Banner says `MODE: GPU`
   with the device name. CuPy imported without installing (note if pip ran).
   The sync/no-sync cell shows the expected difference. All cells under 30 s.
7. Open `04_capstone` on the same runtime. Run all. Four candidates, both
   CuPy scopes present.
8. Download `timings_01_cpu.csv` (from the CPU run) and `timings_02_gpu.csv`
   (from this run). Copy into `data/reference_timings/` as `01_threads.csv`
   and `02_gpu.csv`. Commit and push. Repeat check 4 and confirm the recorded
   metadata now names Colab and the GPU.
9. Optional extensions on the GPU runtime: `ext_cuda_kernel` (does
   `numba-cuda` install and run?), `ext_gil_and_task_pools` (CPU or GPU).
   Either failing means the README keeps calling them optional and the
   run-of-show does not mention them beyond "available".

### Time budget check (rehearsal, gate 5)

Run the full agenda with a clock, including the runtime switch at 01:20 and
the student exercise time. Record actual minute marks in this file. Confirm
both cuts are still enough to recover 10 minutes.

## Release gates

| Gate | When | Passes when |
|---|---|---|
| 1 Scope and structure | done | agenda sums to 180 min; notebooks named; README is the landing page |
| 2 Colab execution reliable | done on author machine; Colab pending | automatic run passes both modes; no active student path needs a cluster |
| 3 Learning sequence | done | primer, practice, self-check, checkpoints, capstone, solutions exist |
| 4 Validate preparation | T-7 days | manual checks 1 to 8 pass; reference CSVs replaced by Colab data; every Colab link opens with a student account |
| 5 Rehearse | T-2 days | timed rehearsal within budget; slides, notes and recording complete; reminder sent by organisers |
| 6 Final check | T-1 day | checks 1, 4 and 6 repeated; tested revision and versions recorded below |

Tested revision for the event: *(fill in: git commit, date, Colab runtime
versions, GPU model)*.

## Known gaps at the time of writing

- `data/reference_timings/*.csv` are from the author's workstation, not Colab
  (labelled as such in their metadata). Gate 4 replaces them.
- The GPU demonstration recording has not been produced.
- The extension notebooks have not been executed in Colab; `ext_cuda_kernel`
  depends on the `numba-cuda` package installing on the Colab image.
- Timings in the run-of-show ("typical Colab result") are expectations, to be
  corrected from the Colab runs.

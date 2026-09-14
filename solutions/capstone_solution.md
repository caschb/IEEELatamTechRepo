# Capstone: what a complete answer looks like

The grading criterion is not the speedup. A complete answer has, for each
workload, a passing correctness check, a timing table labelled with the runtime,
and a recommendation that quotes the student's own numbers. Below are the
outcomes seen in validation and the reasoning that should accompany them.
Numbers on a Colab runtime will differ; the *shape* of the argument should not.

## Workload A: 48 independent grids of side 128, 30 steps

Observed on the author's 32-core machine (numbers from `data/reference_timings/04_capstone.csv`):

| candidate | time |
|---|---:|
| numpy | ~44 ms |
| numba | ~5 ms |
| numba_par (32 threads) | ~103 ms |

Expected reasoning:

- `numba` beats `numpy` because one compiled loop replaces four intermediate
  arrays per step; on a 128x128 grid those intermediates dominate.
- `numba_par` is **slower** than serial Numba: each grid is tiny, so waking and
  synchronising a thread pool 30 times per grid costs more than the work.
  Threads were put *inside* the wrong loop.
- The right parallel structure for this workload is *across* grids (one task
  per grid), because the grids exchange zero bytes. On a cluster that is a task
  scheduler (Dask, a job array), not message passing. A student who adds a
  candidate that `prange`s over the 48 grids inside one `@njit(parallel=True)`
  function has understood this; on a 2-CPU Colab runtime it will show at most
  a modest gain, which is itself worth noting.

## Workload B: one grid of side 2048, 40 steps

Observed on the author's machine:

| candidate | scope | time |
|---|---|---:|
| numpy | compute | ~470 ms |
| numba_par | compute | ~30 ms |
| cupy | compute, data created on device | ~10 ms |
| cupy | transfer-inclusive (upload once, download once) | ~17 ms |

Expected reasoning:

- One large coupled grid is the case where threads inside the grid pay off and
  where the GPU has enough work per kernel launch.
- Transfers were roughly as expensive as the compute itself here (upload plus
  download about 7 ms for 32 MB each way). Whether that changes the decision
  depends on how often the result must come back: every 40 steps, the GPU still
  wins on this hardware; every step, it would not.
- Without a GPU, the honest recommendation is `numba_par`, with the recorded
  GPU table cited as evidence about *other* hardware.
- On a cluster this workload maps onto message passing with halo exchange (two
  rows per neighbour per step), not a task scheduler.

## "One thing more hardware would not fix"

Any of: the small-grid workload (launch and thread overhead, not compute, is the
limit); a loop that copies data to the GPU every step; a memory-bound kernel
that is already saturating bandwidth; a serial fraction that Amdahl's law caps.

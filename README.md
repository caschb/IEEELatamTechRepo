# Modern Parallel Programming in Python for HPC and AI

A three-hour, hands-on workshop on measuring and speeding up Python code on
multicore CPUs and GPUs, and on what changes when work spans several machines
(topics 3, 5, 7, 8 and 9 of the IEEE Latam workshop). **Everything runs in
Google Colab.** No cluster account, no SSH, no local installation.

## 1. Before the event: preparation (45 to 60 minutes, CPU runtime)

Do this at least two days before the workshop, on a normal computer with a
browser and a Google account. Start with [`prep/README.md`](prep/README.md);
it links everything below in order.

| Step | Time | Open |
|---|---:|---|
| Read the primer | 15 min | [`prep/primer.md`](prep/primer.md) |
| Is my Colab ready? | 10-15 min | [Open in Colab](https://colab.research.google.com/github/caschb/IEEELatamTechRepo/blob/main/notebooks/00_colab_ready.ipynb) |
| The running example | 10-15 min | [Open in Colab](https://colab.research.google.com/github/caschb/IEEELatamTechRepo/blob/main/notebooks/00_stencil_practice.ipynb) |
| Self-check | 5-10 min | [`prep/self_check.md`](prep/self_check.md) |

A completed **CPU** readiness check is all that is required. You do not need a
GPU before the event. If Python or NumPy feel unfamiliar, the optional
[refresher](prep/python_numpy_refresher.md) comes before the primer.

## 2. Live agenda (180 minutes)

| Time | Min | Block | Notebook |
|---|---:|---|---|
| 00:00 | 10 | Welcome, outcomes, readiness check, recap of the stencil | [`00_stencil_practice`](https://colab.research.google.com/github/caschb/IEEELatamTechRepo/blob/main/notebooks/00_stencil_practice.ipynb) (recap only) |
| 00:10 | 25 | Measurement: repeated timings, warm-up, profiling demo, NumPy baseline | [`01_measure_and_multicore`](https://colab.research.google.com/github/caschb/IEEELatamTechRepo/blob/main/notebooks/01_measure_and_multicore.ipynb) |
| 00:35 | 35 | Numba, `prange`, thread comparison, Amdahl | same notebook |
| 01:10 | 10 | Break (save your work) | |
| 01:20 | 10 | Switch to a GPU runtime, setup, device check | [`02_gpu`](https://colab.research.google.com/github/caschb/IEEELatamTechRepo/blob/main/notebooks/02_gpu.ipynb) |
| 01:30 | 40 | CuPy stencil, synchronisation, size sweep, transfers, precision | same notebook |
| 02:10 | 20 | Beyond one machine: partitions, halos, communication, MPI and Dask roles | [`03_parallel_models`](https://colab.research.google.com/github/caschb/IEEELatamTechRepo/blob/main/notebooks/03_parallel_models.ipynb) |
| 02:30 | 20 | Capstone: choose and justify an implementation | [`04_capstone`](https://colab.research.google.com/github/caschb/IEEELatamTechRepo/blob/main/notebooks/04_capstone.ipynb) |
| 02:50 | 10 | Debrief, questions, next steps | |

Every notebook is self-contained: its first code cell installs what is missing
and must be rerun after any runtime restart. Timings are compared **within one
runtime**; a number from another machine is a different experiment.

**No GPU?** Stay on the CPU runtime. Notebook 02 detects that, runs the CPU
cells live, and shows a recorded GPU table (labelled with the hardware it came
from) so you can do the same comparisons. The instructor also has a screen
recording of the GPU demonstration.

## 3. After the event

- Slides: [`instructor/slides.md`](instructor/slides.md)
- Optional extensions (not covered live, run at your own pace):
  [threads, processes and the GIL](https://colab.research.google.com/github/caschb/IEEELatamTechRepo/blob/main/extensions/ext_gil_and_task_pools.ipynb)
  and, on a GPU runtime, [write your own CUDA kernel](https://colab.research.google.com/github/caschb/IEEELatamTechRepo/blob/main/extensions/ext_cuda_kernel.ipynb)
- Reading guide for MPI and Dask beyond one machine: [`instructor/reading_guide_mpi_dask.md`](instructor/reading_guide_mpi_dask.md)
- Solutions: [`solutions/`](solutions/)

## Prerequisites

Basic Python (functions, loops, lists), a Google account, and a browser.
Everything else is installed by the notebooks.

## For instructors and authors

| Path | What |
|---|---|
| `instructor/run_of_show.md` | Minute marks, prompts, misconceptions, exercise answers, setup triage, the two cuts |
| `instructor/validation.md` | Release gates and the manual Colab checks that the script cannot do |
| `notebooks/*.py` | Notebook sources in Jupytext percent format; the `.ipynb` files are generated from them |
| `tools/validate.sh` | Regenerates every notebook and executes the core ones headless in CPU-fallback and GPU mode |
| `env/` | Authoring environment (`uv`) and the minimal Colab package list |
| `data/reference_timings/` | Recorded timing tables with runtime metadata, used by the CPU fallback |
| `archive/kabre/` | The earlier cluster edition (SLURM, MPI, OnDemand); historical, not maintained |

Authoring loop:

```bash
uv sync --project env            # add --extra gpu on a machine with an NVIDIA GPU
GPU_EXTRA=1 tools/validate.sh    # regenerate .ipynb from .py and execute the core notebooks
```

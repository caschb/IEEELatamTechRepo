# Modern Parallel Programming in Python for HPC and AI

Material for the sessions on multicore parallelism, GPU computing, multi-node
computing, the modern Python HPC toolset, and performance measurement
(topics 3, 5, 7, 8 and 9 of the workshop). Three hours total.

## Layout

| Path | What |
|---|---|
| `env/pyproject.toml` | The uv project every notebook runs in |
| `env/setup_kabre.sh` | Builds the venv on a Kabre GPU node and registers the `hpc-course` Jupyter kernel for OnDemand |
| `notebooks/*.py` | Notebook sources in jupytext percent format; the `.ipynb` files are generated from them |
| `notebooks/execute.sh` | Runs notebooks headless on an allocation, for validation |
| `mpi/` | Standalone MPI hello job used while testing the environment |

## Sessions

| Notebook | Minutes | Covers |
|---|---|---|
| `00_check_environment` | 10 | Kernel, GPU, MPI and SLURM all reachable |
| `01_measure_and_multicore` | 60 | timeit, line_profiler, NumPy, Numba, prange, Amdahl, GIL, joblib, Dask local |
| `02_gpu` | 50 | CuPy drop-in, sync-before-timing, size and dtype crossover, transfer cost, numba.cuda kernel |
| `03_multinode` | 45 | mpi4py hello and halo-exchange stencil via sbatch, dask-jobqueue SLURMCluster |

## Setup on Kabre

```bash
rsync -a env/ kabre:/data/casch/hpc-course/env/
rsync -a notebooks/ kabre:/data/casch/hpc-course/notebooks/
ssh kabre 'sbatch --wait --partition=nukwa-l40s --time=0-00:20:00 --cpus-per-task=8 /data/casch/hpc-course/env/setup_kabre.sh'
```

Then in OnDemand launch Jupyter on `nukwa-l40s` and pick the kernel
"Python (hpc-course)". The setup script must run through `sbatch`, not `srun`:
importing mpi4py inside an `srun` step aborts in `MPI_Init`.

Regenerate the notebooks after editing a source:

```bash
cd env && uv run jupytext --to ipynb ../notebooks/0*.py
```

Fallback without the cluster: notebooks 01 and 02 run on Google Colab with a
GPU runtime after `pip install cupy-cuda12x numba line_profiler`; 03 needs SLURM.

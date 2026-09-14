# Historical archive: Kabré cluster edition

Everything in this directory belonged to the first edition of the course,
which ran on the Kabré cluster at CeNAT (Jupyter through OnDemand on a
`nukwa-l40s` node, MPI jobs through SLURM). It is kept for reference only and
is **not part of the active course**. Nothing here is validated against the
current notebooks or the current `env/pyproject.toml`.

| Path | What it was |
|---|---|
| `env/setup_kabre.sh` | Built the uv venv against the cluster MPI and CUDA modules and registered a Jupyter kernel |
| `env/smoke.py`, `env/run_smoke.sh`, `env/run_smoke.sbatch` | Smoke test of that venv on a GPU node |
| `env/uv.lock` | Lockfile of the cluster environment (mpi4py, dask-jobqueue, CUDA 12 bindings) |
| `mpi/` | Standalone two-node mpi4py hello job and a nested-sbatch diagnostic |
| `notebooks/check_env.py`, `notebooks/00_check_environment.ipynb` | Kernel, GPU, MPI and SLURM reachability check |
| `notebooks/03_multinode.py`, `.ipynb` | mpi4py halo-exchange stencil and dask-jobqueue `SLURMCluster`, submitted from the notebook |
| `notebooks/execute.sh` | Headless execution of notebooks on an allocation |

The active course runs entirely in Google Colab; see the repository README.
The cluster-specific procedures (module names, `mpirun` flags, the
`orte_keep_fqdn_hostnames` workaround, submitting from a scrubbed environment)
are preserved in the files above and in git history.

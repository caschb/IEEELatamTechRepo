# %% [markdown]
# # 3. Beyond one node
#
# Topic 7. One node has tens of cores, one GPU and a few hundred GB. When that is
# not enough, the cluster has more nodes, and nothing about Python changes
# except one fact: **nodes do not share memory**. Every byte that another node
# needs has to be sent over the network.
#
# Two ways to program that:
#
# - **MPI** (`mpi4py`): explicit messages. Fastest, most control, most code.
#   The lingua franca of HPC; every simulation code speaks it.
# - **Dask**: a scheduler farms tasks to workers on other nodes. Little code,
#   great for embarrassingly parallel work and for data larger than one node.
#
# Both need SLURM to give us the nodes. This notebook runs on one node; the
# jobs it submits run on others.

# %%
import os, subprocess, time, textwrap
from pathlib import Path

WORK = Path.home() / "hpc-course-jobs"
WORK.mkdir(exist_ok=True)
VENV_PYTHON = Path(os.environ["VIRTUAL_ENV"] if "VIRTUAL_ENV" in os.environ else
                   Path(__import__("sys").executable).parent.parent) / "bin/python"
print("jobs go in", WORK, "\nranks will run", VENV_PYTHON)

# This kernel is itself a SLURM job. sbatch passes SLURM_* variables on to the
# jobs we submit from here (even with --export=NONE), and the nukwa QOS sets
# SLURM_MEM_PER_CPU, which the srun that mpirun uses to start its daemons then
# requests on kura nodes and fails. So submit from a scrubbed environment.
CLEAN_ENV = {k: v for k, v in os.environ.items() if not k.startswith("SLURM_")}

def submit(script, *args, **sbatch_opts):
    """sbatch a script from this notebook, wait for it, and return its output."""
    opts = [f"--{k.replace('_', '-')}={v}" for k, v in sbatch_opts.items()]
    cmd = ["sbatch", "--parsable", "--wait", *opts, str(script), *map(str, args)]
    out = subprocess.run(cmd, capture_output=True, text=True, cwd=WORK, env=CLEAN_ENV)
    if out.returncode:
        raise RuntimeError(out.stderr)
    return (WORK / f"slurm-{out.stdout.strip()}.out").read_text()

# %% [markdown]
# ## 3.1 MPI in ten lines
#
# Every rank runs the same script. `comm.rank` tells it who it is. Collective
# operations (`reduce`, `bcast`, `scatter`, `gather`) do the communication.

# %%
_ = (WORK / "hello.py").write_text(textwrap.dedent('''
    from mpi4py import MPI
    import numpy as np

    comm = MPI.COMM_WORLD
    rank, size = comm.rank, comm.size
    local = np.random.default_rng(rank).random(1_000_000)
    total = comm.reduce(local.sum(), op=MPI.SUM, root=0)
    print(f"rank {rank}/{size} on {MPI.Get_processor_name()} local mean {local.mean():.4f}", flush=True)
    if rank == 0:
        print(f"global mean {total / (size * local.size):.4f}", flush=True)
'''))

_ = (WORK / "mpi.sbatch").write_text(textwrap.dedent(f'''
    #!/bin/bash
    #SBATCH --job-name=mpi-course
    #SBATCH --partition=kura-wide
    #SBATCH --time=0-00:05:00
    #SBATCH --cpus-per-task=1
    # nodes and tasks come from the command line: sbatch --nodes=N --ntasks-per-node=M
    module load gcc/12.4.0 openmpi/4.0.5
    export NUMBA_NUM_THREADS=${{SLURM_CPUS_PER_TASK:-1}}   # one Numba pool per rank, sized to the rank
    mpirun --mca orte_keep_fqdn_hostnames 1 {VENV_PYTHON} "$@" 2>&1 | grep -v openib
''').lstrip())

# %%
print(submit("mpi.sbatch", "hello.py", nodes=2, ntasks_per_node=4))

# %% [markdown]
# Eight ranks, two hostnames. The `sbatch` line is the whole "how do I use more
# nodes" story: `--nodes` and `--ntasks-per-node` decide, `mpirun` reads the
# allocation, the script is unchanged.
#
# Rules from the cluster notes worth repeating: launch with `mpirun` and the
# `orte_keep_fqdn_hostnames` flag on Kabré, not `srun --mpi=pmi2`; ranks come
# from `--ntasks`, threads from `--cpus-per-task`.
#
# ## 3.2 The stencil across nodes: domain decomposition
#
# Split the grid into horizontal strips, one per rank. Each step, a strip needs
# the row just above and just below it, which belong to its neighbours. Those are
# **halo** (ghost) rows, exchanged every iteration. Everything else is the Numba
# stencil from session 1, unchanged.

# %%
_ = (WORK / "stencil_mpi.py").write_text(textwrap.dedent('''
    import sys, time
    import numpy as np
    from mpi4py import MPI
    from numba import njit, prange

    @njit(parallel=True)
    def step(u, unew):
        n, m = u.shape
        for i in prange(1, n - 1):
            for j in range(1, m - 1):
                unew[i, j] = 0.25 * (u[i-1, j] + u[i+1, j] + u[i, j-1] + u[i, j+1])

    comm = MPI.COMM_WORLD
    rank, size = comm.rank, comm.size
    n, iters = int(sys.argv[1]), int(sys.argv[2])
    rows = n // size                                   # interior rows owned by this rank
    up = rank - 1 if rank > 0 else MPI.PROC_NULL      # PROC_NULL: sends and receives become no-ops
    down = rank + 1 if rank < size - 1 else MPI.PROC_NULL

    u = np.zeros((rows + 2, n))                        # +2 halo rows
    if rank == 0:
        u[1, :] = 100.0                                # hot top edge lives in rank 0's first real row
    unew = u.copy()
    step(u, unew)                                      # compile before timing

    comm.Barrier(); t0 = MPI.Wtime()
    for _ in range(iters):
        # send my first real row up, receive neighbour's last real row into my top halo (and vice versa)
        comm.Sendrecv(u[1], dest=up, recvbuf=u[0], source=up)
        comm.Sendrecv(u[rows], dest=down, recvbuf=u[rows + 1], source=down)
        if rank == 0: u[0, :] = u[1, :]                # top boundary: keep the hot edge
        step(u, unew)
        u, unew = unew, u
    elapsed = MPI.Wtime() - t0

    total_heat = comm.reduce(u[1:-1].sum(), op=MPI.SUM, root=0)
    if rank == 0:
        cells = n * n * iters
        print(f"{size:3d} ranks on {n}x{n}, {iters} steps: {elapsed:6.2f} s "
              f"({cells/elapsed/1e9:5.2f} Gcell-updates/s)  total heat {total_heat:.1f}", flush=True)
'''))

# %%
N, ITERS = 12000, 100
for nodes, per_node in ((1, 1), (1, 4), (2, 4), (4, 4)):
    out = submit("mpi.sbatch", "stencil_mpi.py", N, ITERS,
                 nodes=nodes, ntasks_per_node=per_node, cpus_per_task=5, hint="nomultithread")
    print(f"{nodes} node(s) x {per_node} ranks x 5 threads: {out.strip()}")

# %% [markdown]
# Hybrid parallelism: MPI between nodes, Numba threads within a rank. `total heat`
# is the correctness check: it must not change with the rank count.
#
# Why not perfect scaling? Each rank exchanges two rows per step. That is tiny
# compared with the strip, so this problem scales well. Problems that need
# all-to-all communication (FFTs, dense linear algebra) do not, and there the
# network decides. Measure the fraction of time in communication before buying
# more nodes.
#
# ## 3.3 Dask on many nodes, from the notebook
#
# For independent tasks, MPI is overkill. `dask-jobqueue` asks SLURM for worker
# jobs and connects them to a scheduler running here, in the kernel. The code
# you wrote against `LocalCluster` in session 1 runs unchanged.

# %%
from dask.distributed import Client
from dask_jobqueue import SLURMCluster

for k in list(os.environ):                 # dask-jobqueue calls sbatch too: same scrub as above
    if k.startswith("SLURM_"):
        del os.environ[k]

cluster = SLURMCluster(
    queue="kura", cores=20, processes=4, memory="40GB", walltime="00:10:00",
    job_extra_directives=["--exclusive"],                       # one node per job, so two jobs = two hosts
    python=str(VENV_PYTHON), log_directory=str(WORK),
    scheduler_options={"dashboard_address": ":0"},
)

print(cluster.job_script())

# %%
cluster.scale(jobs=2)                      # two SLURM jobs = two nodes, 4 workers each
client = Client(cluster)
client.wait_for_workers(8, timeout=300)
print(client)

# %%
import numpy as np

def simulate(alpha, n=1000, iters=200):
    """One parameter of a sweep: run the stencil with diffusion coefficient alpha, return mean temperature."""
    u = np.zeros((n, n)); u[0, :] = 100.0
    for _ in range(iters):
        u[1:-1, 1:-1] += alpha * (u[:-2, 1:-1] + u[2:, 1:-1] + u[1:-1, :-2] + u[1:-1, 2:] - 4 * u[1:-1, 1:-1])
    return alpha, float(u.mean())

alphas = np.linspace(0.05, 0.25, 32)
t0 = time.perf_counter()
results = client.gather(client.map(simulate, alphas))
print(f"{len(alphas)} simulations on {len(client.scheduler_info()['workers'])} workers "
      f"across nodes: {time.perf_counter() - t0:.1f} s")
print({w["host"] for w in client.scheduler_info()["workers"].values()})
results[:3]

# %%
client.close(); cluster.close()

# %% [markdown]
# Adaptive mode (`cluster.adapt(minimum_jobs=0, maximum_jobs=8)`) lets Dask
# request and release nodes as the task graph demands. That is the cluster
# equivalent of "scale to zero".
#
# ## 3.4 Which one?
#
# | You have | Use |
# |---|---|
# | Independent tasks (sweeps, files, folds) | Dask (`dask-jobqueue`) or a SLURM job array |
# | One big array or dataframe that does not fit in RAM | `dask.array` / `dask.dataframe` on `SLURMCluster` |
# | Tightly coupled simulation, halo exchange, custom communication | `mpi4py` + Numba |
# | Deep learning across GPUs | PyTorch DDP, which uses NCCL, the GPU cousin of MPI |
#
# ## Takeaways for the whole afternoon
#
# 1. Measure first. Profile, then optimise the line that matters.
# 2. Vectorise, then compile (Numba). Ten to a hundred times, no parallelism yet.
# 3. Threads for compiled or I/O code, processes for Python code, `prange` for loops.
# 4. GPU: same NumPy code with CuPy; batch the work; move data once; synchronise
#    before timing; know your float64 rate.
# 5. Multi-node: MPI for coupled work, Dask for independent work. SLURM hands out the
#    nodes either way.
# 6. Keep the environment reproducible (`uv`, a lockfile), so the number you got
#    today is the number you get next month.

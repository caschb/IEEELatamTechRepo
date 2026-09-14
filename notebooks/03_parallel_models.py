# %% [markdown]
# # 3. Beyond one machine: partitions, halos and communication
#
# **Live session, block 7 (02:10 to 02:30).** Any runtime; nothing here needs a
# GPU, a second machine or a cluster.
#
# One machine has a few cores, maybe one GPU and a bounded amount of memory. When
# that is not enough, you use several machines, and one fact changes everything:
# **separate machines do not share memory.** Every byte another worker needs has
# to be sent to it.
#
# > **What this notebook is.** A *model* of multi-worker execution, built with
# > NumPy arrays inside one Colab runtime. The partitions below are slices of one
# > array and the "messages" are array copies. It demonstrates *what* has to be
# > communicated and *why*; it makes **no claim about multi-node performance**,
# > because nothing here runs on more than one machine.
#
# Two ways to program real separate-memory workers:
#
# - **MPI** (`mpi4py`): every process runs the same script, and you write the
#   messages explicitly. Fastest, most control, most code. The lingua franca of
#   simulation codes.
# - **Dask**: a scheduler farms tasks to workers. Little code; ideal when tasks are
#   independent or data is larger than one machine.
#
# This notebook is self-contained.

# %%
# --- Setup: rerun after every runtime restart ------------------------------
import importlib, importlib.util, subprocess, sys

def ensure(module, package=None):
    if importlib.util.find_spec(module) is None:
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", package or module], check=True)
    return importlib.import_module(module)

np = ensure("numpy")
ensure("matplotlib"); import matplotlib.pyplot as plt
print("numpy", np.__version__)

# %% [markdown]
# ## 3.1 The serial stencil, once more
#
# Same function as notebooks 1 and 2. It is the reference every partitioned
# version must reproduce exactly.

# %%
def init_grid(n):
    u = np.zeros((n, n)); u[0, :] = 100.0
    return u

def step(u, unew):
    unew[1:-1, 1:-1] = 0.25 * (u[:-2, 1:-1] + u[2:, 1:-1] + u[1:-1, :-2] + u[1:-1, 2:])
    return unew

def run_serial(n, iters):
    u = init_grid(n); unew = u.copy()
    for _ in range(iters):
        step(u, unew); u, unew = unew, u
    return u

N, ITERS = 256, 30
ref = run_serial(N, ITERS)
print("reference computed:", ref.shape, "interior mean", ref[1:-1, 1:-1].mean().round(4))

# %% [markdown]
# ## 3.2 Domain decomposition: strips and halos
#
# Split the grid into `P` horizontal strips, one per worker. To update its top
# interior row, a strip needs the row just above it, which belongs to the
# neighbour. Each strip therefore keeps one extra row above and one below: the
# **halo** (ghost) rows. Every step, neighbours exchange their boundary rows to
# refresh each other's halos. Everything else is the serial stencil, untouched.
#
# ```
#         worker 0            worker 1            worker 2
#      +-----------+       +-----------+       +-----------+
#      | halo (top)|       | halo      | <---- | last real |
#      | real rows |       | real rows |       | real rows |
#      | real rows | ----> | halo      |       | ...       |
#      | halo (bot)| <---- | first real|       |           |
#      +-----------+       +-----------+       +-----------+
#      arrows = one row copied per neighbour per step
# ```
#
# **Exercise.** Before running the next cells: with `P` strips of an `n x n`
# grid, how many rows cross a boundary between two neighbours in one step, and
# how many bytes is that in float64? Write your answer down.

# %%
def partition(u, P):
    """Split the interior rows of u into P strips; each strip keeps one halo row above and below.

    The global top and bottom rows are physical boundaries: they sit in the halos of the
    first and last strip, are never updated, and never need to be communicated."""
    n = u.shape[0]
    bounds = [(rows[0], rows[-1] + 1) for rows in np.array_split(np.arange(1, n - 1), P)]
    return [u[lo - 1:hi + 1].copy() for lo, hi in bounds]      # rows lo-1 and hi are the halos

def exchange_halos(strips):
    """What the network would carry: each strip sends its boundary rows to its neighbours."""
    P = len(strips)
    nbytes = 0
    for r in range(P):
        if r > 0:                        # my top halo <- neighbour above's last real row
            strips[r][0] = strips[r-1][-2]; nbytes += strips[r][0].nbytes
        if r < P - 1:                    # my bottom halo <- neighbour below's first real row
            strips[r][-1] = strips[r+1][1]; nbytes += strips[r][-1].nbytes
    return nbytes

def reconstruct(strips):
    return np.vstack([strips[0][:1]] + [s[1:-1] for s in strips] + [strips[-1][-1:]])

def run_partitioned(n, iters, P):
    strips = partition(init_grid(n), P)
    news = [s.copy() for s in strips]
    bytes_per_step = 0
    for _ in range(iters):
        bytes_per_step = exchange_halos(strips)
        for s, snew in zip(strips, news):
            step(s, snew)                # each worker updates only its own real rows
        strips, news = news, strips
    return reconstruct(strips), bytes_per_step

for P in (1, 2, 4, 8):
    u_p, nbytes = run_partitioned(N, ITERS, P)
    assert u_p.shape == ref.shape and np.array_equal(u_p, ref), f"P={P} does not match the serial result"
    print(f"P={P}: identical to the serial result; {nbytes/1024:6.1f} KB exchanged per step")

# %% [markdown]
# Two things to notice in the cell above.
#
# 1. The **global top and bottom edges** never need a message: they are physical
#    boundaries, not neighbours. Only interior boundaries exchange rows.
# 2. The reconstructed grid is `array_equal` to the serial one, not merely close.
#    Decomposition does not change the arithmetic, only who performs it.
#
# **Check your exercise answer.** Each interior boundary carries two rows per
# step (one each way), so `2 * (P-1)` rows of `n` float64 values.
#
# ## 3.3 Is communication cheap or expensive here?
#
# Per step, each worker computes `n * (n/P)` cells and exchanges at most `2n`
# values. The ratio of communication to computation is `2P/n`: it shrinks with a
# bigger grid and grows with more workers. Any real network adds a fixed
# **latency** per message on top of the bytes.

# %%
n_values = np.array([256, 1024, 4096, 16384])
fig, ax = plt.subplots(figsize=(5.5, 3.5))
for P in (2, 8, 32, 128):
    ax.plot(n_values, 2 * P / n_values, "o-", label=f"P={P}")
ax.set(xlabel="grid side n", ylabel="values sent / cells computed, per worker per step",
       xscale="log", yscale="log", title="halo exchange relative to work (model, not a measurement)")
ax.legend(); fig.tight_layout()

# %% [markdown]
# **Predict and explain.** Suppose one message costs 10 microseconds of latency
# plus the bytes at 10 GB/s, and one cell update costs 1 nanosecond. For `n=256`
# and `P=128` (two rows per worker), is a worker mostly computing or mostly
# waiting for messages? What about `n=16384`? This is the whole reason small
# problems do not scale across machines, whatever the library.
#
# ## 3.4 The same idea in MPI (reading, not running)
#
# With `mpi4py`, every rank runs this script. `comm.rank` says who it is;
# `Sendrecv` does one exchange in both directions. `PROC_NULL` turns the edge
# ranks' missing neighbours into no-ops. Compare line by line with
# `exchange_halos` above: same rows, same direction.
#
# ```python
# from mpi4py import MPI
# comm = MPI.COMM_WORLD
# rank, size = comm.rank, comm.size
# up   = rank - 1 if rank > 0        else MPI.PROC_NULL
# down = rank + 1 if rank < size - 1 else MPI.PROC_NULL
#
# s = my_strip_with_halos()                       # shape (h + 2, n), as in partition()
# for _ in range(iters):
#     comm.Sendrecv(s[1],  dest=up,   recvbuf=s[0],  source=up)     # first real row up, top halo in
#     comm.Sendrecv(s[-2], dest=down, recvbuf=s[-1], source=down)   # last real row down, bottom halo in
#     step(s, snew); s, snew = snew, s
# total = comm.reduce(s[1:-1].sum(), op=MPI.SUM, root=0)            # a global check, like our array_equal
# ```
#
# Launching it is a cluster matter (`mpirun -np 8 python stencil_mpi.py` inside a
# job), not a Python one. The Python does not change between 2 ranks on a laptop
# and 2000 on a supercomputer.
#
# ## 3.5 Independent tasks: the other kind of parallelism
#
# A parameter sweep has no halos at all: each simulation is complete on its own,
# and the only communication is sending the parameter in and the result out. That
# is where task schedulers such as Dask shine. Same code on your laptop's cores
# today and on forty machines tomorrow:
#
# ```python
# from dask.distributed import Client
# client = Client()                               # local workers now; a cluster later, same code
#
# def simulate(alpha):                            # one complete, independent job
#     u = init_grid(512); ...; return alpha, float(u.mean())
#
# futures = client.map(simulate, np.linspace(0.05, 0.25, 32))
# results = client.gather(futures)
# ```
#
# Running such workers is outside what a free Colab runtime is meant for, so
# treat this as reading material; the instructor reading guide lists where to try
# it. The concept is what matters for the capstone: *independent* work needs a
# task scheduler, *coupled* work needs message passing.

# %%
# The independent-task pattern with no scheduler at all, to make the contrast concrete:
def simulate(alpha, n=128, iters=50):
    u = init_grid(n)
    for _ in range(iters):
        u[1:-1, 1:-1] += alpha * (u[:-2, 1:-1] + u[2:, 1:-1] + u[1:-1, :-2] + u[1:-1, 2:] - 4 * u[1:-1, 1:-1])
    return alpha, float(u.mean())

results = list(map(simulate, np.linspace(0.05, 0.25, 8)))   # replace map with client.map on a Dask cluster
print("bytes each task needs from any other task: 0")
print("\n".join(f"alpha={a:.3f} mean={m:.3f}" for a, m in results))

# %% [markdown]
# ## 3.6 Which one?
#
# | You have | Use |
# |---|---|
# | Independent tasks (sweeps, files, folds) | a task scheduler: Dask, or a job array on a cluster |
# | One array or table larger than one machine's RAM | `dask.array` / `dask.dataframe` |
# | Tightly coupled simulation with halos or custom communication | `mpi4py` (+ Numba or CuPy inside each rank) |
# | Deep learning across GPUs | PyTorch DDP, which uses NCCL, the GPU cousin of MPI |
#
# **Checkpoint.** Write one sentence for each of the two capstone workloads: what
# data, if any, must move between workers, and which row of the table applies.

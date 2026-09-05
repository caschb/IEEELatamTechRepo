# %% [markdown]
# # 1. Measure first, then go multicore
#
# Topics 9 (measurement) and 3 (multicore parallelism), with the tools of topic 8
# appearing as we need them.
#
# Three questions to ask before touching any code:
#
# 1. **Where does the time go?** A profiler answers this. Intuition is usually wrong.
# 2. **What is the ceiling?** Amdahl's law: if a fraction `s` of the work is serial,
#    no number of cores gets you past `1/s`.
# 3. **Did it help?** Measure again, on the same machine, the same way.
#
# Our running example is a 2D heat-diffusion stencil (Jacobi iteration). Each step
# replaces every interior cell with the average of its four neighbours. It is
# small, memory-bound, and it will follow us through the multicore, GPU and
# multi-node sessions.

# %%
import os, time
import numpy as np
import matplotlib.pyplot as plt

print("cores available to this kernel:", len(os.sched_getaffinity(0)))

# %% [markdown]
# ## 1.1 Pure Python: the baseline
#
# Every performance story starts with a number. Here is the stencil written the
# way a scientist writes it on day one.

# %%
def step_python(u, unew):
    n, m = len(u), len(u[0])
    for i in range(1, n - 1):
        for j in range(1, m - 1):
            unew[i][j] = 0.25 * (u[i-1][j] + u[i+1][j] + u[i][j-1] + u[i][j+1])
    return unew

def init_grid(n):
    u = np.zeros((n, n))
    u[0, :] = 100.0      # hot top edge
    return u

def run_python(n, iters):
    u = init_grid(n).tolist()
    unew = init_grid(n).tolist()
    for _ in range(iters):
        step_python(u, unew)
        u, unew = unew, u
    return np.array(u)

# %%
# %timeit -r 3 -n 1 run_python(200, 20)

# %% [markdown]
# `%timeit` runs the code several times and reports the best runs. Prefer it to a
# single `time.time()` pair: the first call pays for imports, caches, page faults
# and (later) JIT compilation.
#
# ## 1.2 Where does the time go? `line_profiler`
#
# Before optimising, look. `%lprun` shows time per line of one function.

# %%
# %load_ext line_profiler
# %lprun -f step_python run_python(100, 5)

# %% [markdown]
# Unsurprising here: all the time is the inner loop body. In real code the answer
# is often a surprise (a `print`, a `pandas` lookup inside a loop, a conversion).
# For whole-program pictures, `python -m cProfile -o out.prof script.py` plus
# `snakeviz`, or the sampling profiler `py-spy top --pid <pid>` on a running job.
#
# ## 1.3 NumPy: let compiled code do the loop
#
# The first and biggest win is usually not parallelism. It is moving the loop out
# of the interpreter.

# %%
def step_numpy(u, unew):
    unew[1:-1, 1:-1] = 0.25 * (u[:-2, 1:-1] + u[2:, 1:-1] + u[1:-1, :-2] + u[1:-1, 2:])
    return unew

def run(step, n, iters, dtype=np.float64):
    u = init_grid(n).astype(dtype)
    unew = u.copy()
    for _ in range(iters):
        step(u, unew)
        u, unew = unew, u
    return u

ref = run_python(200, 20)
assert np.allclose(run(step_numpy, 200, 20), ref)

# %%
t_py = %timeit -o -r 3 -n 1 run_python(200, 20)
t_np = %timeit -o -r 3 -n 5 run(step_numpy, 200, 20)
print(f"NumPy is {t_py.best / t_np.best:.0f}x faster than pure Python, with zero parallelism")

# %% [markdown]
# ## 1.4 Numba: compile your own loop
#
# NumPy slicing creates temporaries (four of them per step here) and walks memory
# several times. Numba compiles the explicit loop to machine code, so we keep the
# readable loop and lose the temporaries.

# %%
from numba import njit, prange

@njit
def step_numba(u, unew):
    n, m = u.shape
    for i in range(1, n - 1):
        for j in range(1, m - 1):
            unew[i, j] = 0.25 * (u[i-1, j] + u[i+1, j] + u[i, j-1] + u[i, j+1])
    return unew

@njit(parallel=True)
def step_numba_par(u, unew):
    n, m = u.shape
    for i in prange(1, n - 1):
        for j in range(1, m - 1):
            unew[i, j] = 0.25 * (u[i-1, j] + u[i+1, j] + u[i, j-1] + u[i, j+1])
    return unew

assert np.allclose(run(step_numba, 200, 20), ref)
assert np.allclose(run(step_numba_par, 200, 20), ref)   # first call also compiles

# %%
N, ITERS = 4000, 50
t_np = %timeit -o -r 3 -n 1 run(step_numpy, N, ITERS)
t_nb = %timeit -o -r 3 -n 1 run(step_numba, N, ITERS)
t_par = %timeit -o -r 3 -n 1 run(step_numba_par, N, ITERS)

# %% [markdown]
# ## 1.5 Scaling and Amdahl's law
#
# `prange` splits the outer loop over threads. How well does it scale with the
# number of threads? Measure, do not assume.

# %%
import numba

def best_of(fn, repeat=3):
    best = float("inf")
    for _ in range(repeat):
        t0 = time.perf_counter(); fn(); best = min(best, time.perf_counter() - t0)
    return best

max_threads = len(os.sched_getaffinity(0))
threads = [t for t in (1, 2, 4, 8, 16, 32) if t <= max_threads]
times = []
for t in threads:
    numba.set_num_threads(t)
    run(step_numba_par, N, 5)                    # warm the thread pool
    times.append(best_of(lambda: run(step_numba_par, N, ITERS)))
    print(f"{t:3d} threads: {times[-1]*1e3:7.1f} ms  speedup {times[0]/times[-1]:5.2f}x")
numba.set_num_threads(max_threads)

# %%
speedup = np.array(times[0]) / np.array(times)
fig, ax = plt.subplots(figsize=(5, 3.5))
ax.plot(threads, threads, "--", color="0.6", label="ideal")
for s in (0.02, 0.05, 0.10):
    ax.plot(threads, [1 / (s + (1 - s) / t) for t in threads], ":", label=f"Amdahl, {s:.0%} serial")
ax.plot(threads, speedup, "o-", label="measured")
ax.set(xlabel="threads", ylabel="speedup", xscale="log", yscale="log")
ax.set_xticks(threads); ax.set_xticklabels(threads)
ax.legend(); fig.tight_layout()

# %% [markdown]
# The stencil is memory-bound: past a few threads the memory bus, not the cores,
# is the limit. That is normal. The right response is to do more work per byte
# (fuse loops, use float32 when precision allows) rather than add threads.
#
# **Rule:** `NUMBA_NUM_THREADS` must match the cores SLURM gave you. Numba
# otherwise sizes its pool from the whole node and several jobs on one node
# fight for cores. The result is not an error, only silence and slowness.
#
# ## 1.6 Threads, processes and the GIL
#
# Numba's `prange` works because compiled code does not hold the Global
# Interpreter Lock. Pure Python code does, so Python threads do not run Python
# code in parallel. Watch the same parameter sweep three ways.

# %%
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor

def slow_python_task(seed, n=300_000):
    rng = np.random.default_rng(seed)
    acc = 0.0
    for _ in range(n):                      # pure-Python loop: holds the GIL
        acc += rng.random() ** 2
    return acc

seeds = list(range(8))
t0 = time.perf_counter(); serial = [slow_python_task(s) for s in seeds]; t_serial = time.perf_counter() - t0
with ThreadPoolExecutor(8) as ex:
    t0 = time.perf_counter(); list(ex.map(slow_python_task, seeds)); t_threads = time.perf_counter() - t0
with ProcessPoolExecutor(8) as ex:
    t0 = time.perf_counter(); list(ex.map(slow_python_task, seeds)); t_procs = time.perf_counter() - t0
print(f"serial {t_serial:.2f}s   8 threads {t_threads:.2f}s   8 processes {t_procs:.2f}s")

# %% [markdown]
# Threads were *slower* than serial: eight of them fighting for one lock. Processes
# delivered. Processes cost more to start and must
# pickle their arguments and results, so they suit coarse tasks (one per file,
# one per parameter set). Compiled code that releases the GIL (`@njit(nogil=True)`,
# most of NumPy, I/O) is happy with threads.
#
# ## 1.7 The modern one-liners: joblib and Dask
#
# `concurrent.futures` is the standard library. Two ecosystem tools wrap the same
# idea with nicer ergonomics. `joblib` is what scikit-learn uses internally:

# %%
from joblib import Parallel, delayed

t0 = time.perf_counter()
res = Parallel(n_jobs=8)(delayed(slow_python_task)(s) for s in seeds)
print(f"joblib, 8 workers: {time.perf_counter() - t0:.2f}s")

# %% [markdown]
# Dask gives the same API but the "cluster" can be this node today and forty
# nodes tomorrow, with no code change. We will use that in session 3.

# %%
from dask.distributed import Client, LocalCluster

cluster = LocalCluster(n_workers=8, threads_per_worker=1, dashboard_address=":0")
client = Client(cluster)
print("dashboard:", client.dashboard_link)
t0 = time.perf_counter()
res = client.gather(client.map(slow_python_task, seeds))
print(f"dask LocalCluster, 8 workers: {time.perf_counter() - t0:.2f}s")
client.close(); cluster.close()

# %% [markdown]
# ## Takeaways
#
# | Situation | Reach for |
# |---|---|
# | Loop over array elements | NumPy vectorisation first, then Numba `@njit` |
# | Same loop, many cores | Numba `parallel=True` + `prange` |
# | Many independent Python tasks | `ProcessPoolExecutor`, `joblib`, or Dask |
# | Many independent compiled or I/O tasks | `ThreadPoolExecutor` |
# | Anything | Measure before and after, on the same allocation |

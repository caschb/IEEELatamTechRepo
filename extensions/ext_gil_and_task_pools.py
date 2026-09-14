# %% [markdown]
# # Extension A: threads, processes and the GIL (optional, not covered live)
#
# Numba's `prange` works because compiled code does not hold Python's Global
# Interpreter Lock (GIL). Pure Python code does, so Python *threads* do not run
# Python code in parallel. This extension watches one independent-task sweep run
# three ways. CPU runtime; the number of workers adapts to the runtime.
#
# *Status: optional extension, validated on an author machine, not part of the
# 180-minute session. If it does not run in your Colab, skip it.*

# %%
import os, time
import numpy as np
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor

WORKERS = min(4, os.cpu_count() or 1)
print("workers:", WORKERS, "(a 1- or 2-CPU runtime makes the difference small; read the explanation anyway)")

def slow_python_task(seed, n=200_000):
    rng = np.random.default_rng(seed)
    acc = 0.0
    for _ in range(n):                      # pure-Python loop: holds the GIL
        acc += rng.random() ** 2
    return acc

seeds = list(range(WORKERS * 2))

def timed(label, fn):
    t0 = time.perf_counter(); fn(); dt = time.perf_counter() - t0
    print(f"{label:28s} {dt:6.2f} s"); return dt

t_serial = timed("serial", lambda: [slow_python_task(s) for s in seeds])
with ThreadPoolExecutor(WORKERS) as ex:
    t_threads = timed(f"{WORKERS} threads", lambda: list(ex.map(slow_python_task, seeds)))
with ProcessPoolExecutor(WORKERS) as ex:
    t_procs = timed(f"{WORKERS} processes", lambda: list(ex.map(slow_python_task, seeds)))

# %% [markdown]
# Typical outcome on a machine with several cores: threads are no faster than
# serial (they take turns holding one lock), processes are faster (each has its
# own interpreter and lock) but pay to start and to pickle arguments and results.
# On a 1-CPU runtime none of the three can be faster than serial. Whatever you
# saw, the rule is: processes for pure-Python tasks that are coarse (one per
# file, one per parameter set); threads for compiled or I/O code that releases
# the GIL (`@njit(nogil=True)`, most of NumPy, file and network I/O).
#
# ## joblib: the same idea with nicer ergonomics
#
# `joblib` is what scikit-learn uses internally. Dask offers the same `map`
# shape and can later move to many machines without code changes.

# %%
import importlib.util, subprocess, sys
if importlib.util.find_spec("joblib") is None:
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "joblib"], check=True)
from joblib import Parallel, delayed

t_joblib = timed(f"joblib, {WORKERS} workers", lambda: Parallel(n_jobs=WORKERS)(delayed(slow_python_task)(s) for s in seeds))

# %% [markdown]
# ## A compiled task that releases the GIL

# %%
from numba import njit

@njit(nogil=True)
def compiled_task(seed, n=20_000_000):
    acc = 0.0; x = float(seed)
    for _ in range(n):
        x = (1664525.0 * x + 1013904223.0) % 4294967296.0
        acc += (x / 4294967296.0) ** 2
    return acc

compiled_task(0)                                   # compile before timing
t_c_serial = timed("compiled, serial", lambda: [compiled_task(s) for s in seeds])
with ThreadPoolExecutor(WORKERS) as ex:
    t_c_threads = timed(f"compiled, {WORKERS} threads", lambda: list(ex.map(compiled_task, seeds)))

# %% [markdown]
# With `nogil=True`, threads can run the compiled function at the same time, so
# on a multi-core runtime the threaded version wins with no pickling and no
# process start-up. This is the mechanism `prange` uses internally.

# %% [markdown]
# # 2. GPU computing from Python
#
# Topic 5. A GPU is thousands of slow cores sharing very fast memory. It wins when
# you have millions of identical, independent operations, and loses when you
# talk to it too often or move data back and forth.
#
# Two levels of access from Python:
#
# - **CuPy**: NumPy's API, arrays live on the GPU. Zero new concepts.
# - **numba.cuda**: write the kernel yourself. One new concept (the thread grid).
#
# Same stencil as session 1.

# %%
import time
import numpy as np
import cupy as cp
import matplotlib.pyplot as plt

props = cp.cuda.runtime.getDeviceProperties(0)
print(props["name"].decode(), f"{props['totalGlobalMem']/2**30:.0f} GB")

# %% [markdown]
# ## 2.1 CuPy: the drop-in
#
# Write the function once against an array module `xp`, then hand it NumPy or
# CuPy arrays. This is the standard pattern (the Array API) and it is how
# scikit-learn, SciPy and xarray grow GPU support without rewriting.

# %%
def step(u, unew):
    unew[1:-1, 1:-1] = 0.25 * (u[:-2, 1:-1] + u[2:, 1:-1] + u[1:-1, :-2] + u[1:-1, 2:])
    return unew

def run(xp, n, iters, dtype=np.float64):
    u = xp.zeros((n, n), dtype=dtype)
    u[0, :] = 100.0
    unew = u.copy()
    for _ in range(iters):
        step(u, unew)
        u, unew = unew, u
    return u

u_cpu = run(np, 1000, 50)
u_gpu = run(cp, 1000, 50)
print(type(u_gpu), u_gpu.device)
assert np.allclose(cp.asnumpy(u_gpu), u_cpu)

# %% [markdown]
# ## 2.2 Timing GPU code correctly
#
# GPU calls are **asynchronous**: `step()` returns before the GPU has finished.
# Without a `synchronize()` you time the launch, not the work. This is the single
# most common measurement mistake on GPUs.

# %%
def best_of(fn, repeat=3, sync=lambda: None):
    fn(); sync()                                     # warm-up: JIT, allocation, cache
    best = float("inf")
    for _ in range(repeat):
        t0 = time.perf_counter(); fn(); sync(); best = min(best, time.perf_counter() - t0)
    return best

sync = cp.cuda.Device().synchronize
n = 4000
print(f"without sync: {best_of(lambda: run(cp, n, 50))*1e3:7.2f} ms  <- wrong")
print(f"with sync:    {best_of(lambda: run(cp, n, 50), sync=sync)*1e3:7.2f} ms")

# %% [markdown]
# ## 2.3 When is the GPU worth it? Size and precision
#
# Sweep the grid size for CPU and GPU, then repeat on the GPU with `float32`.
# Watch two things: how the GPU time behaves for small grids, and what `float32`
# buys.

# %%
sizes = [250, 500, 1000, 2000, 4000, 8000]
rows = []
for n in sizes:
    t_cpu = best_of(lambda: run(np, n, 20))
    t_gpu64 = best_of(lambda: run(cp, n, 20), sync=sync)
    t_gpu32 = best_of(lambda: run(cp, n, 20, np.float32), sync=sync)
    rows.append((n, t_cpu, t_gpu64, t_gpu32))
    print(f"n={n:5d}  cpu {t_cpu*1e3:8.2f} ms   gpu f64 {t_gpu64*1e3:7.2f} ms ({t_cpu/t_gpu64:5.1f}x)"
          f"   gpu f32 {t_gpu32*1e3:7.2f} ms ({t_cpu/t_gpu32:5.1f}x)")

# %%
rows = np.array(rows)
fig, ax = plt.subplots(figsize=(5, 3.5))
ax.plot(rows[:, 0], rows[:, 1], "o-", label="NumPy, 1 core")
ax.plot(rows[:, 0], rows[:, 2], "s-", label="CuPy float64")
ax.plot(rows[:, 0], rows[:, 3], "^-", label="CuPy float32")
ax.set(xlabel="grid side n", ylabel="time for 20 steps [s]", xscale="log", yscale="log")
ax.legend(); fig.tight_layout()

# %% [markdown]
# Small grids: the GPU time is flat. Up to about a thousand by a thousand the
# time does not depend on the size at all, because each step launches a handful
# of kernels and a launch costs microseconds regardless of the work. There the
# GPU is idle most of the time, and a loop that launches tiny kernels can lose
# to NumPy. The GPU needs enough work per launch to amortise the launch.
#
# `float32` is about twice as fast as `float64` at large sizes. That is not the
# arithmetic rate: this stencil is memory-bound, and halving the bytes halves the
# time. A compute-bound kernel would show a much larger gap on this card, since
# workstation GPUs like the L40S run `float64` at a small fraction of their
# `float32` rate while datacentre parts (V100, A100, H100) do not. Two lessons:
# know which resource you are bound by, and know your hardware before picking a dtype.
#
# ## 2.4 The transfer tax
#
# PCIe moves roughly 10 to 25 GB/s. GPU memory moves hundreds of GB/s. A round trip
# of the array costs more than dozens of stencil steps on it.

# %%
n = 8000
a = np.random.rand(n, n)
t_up = best_of(lambda: cp.asarray(a), sync=sync)
a_gpu = cp.asarray(a)
t_down = best_of(lambda: a_gpu.get(), sync=sync)
t_step = best_of(lambda: step(a_gpu, a_gpu.copy()), sync=sync)
gb = a.nbytes / 2**30
print(f"{gb:.2f} GB   host->gpu {t_up*1e3:6.1f} ms ({gb/t_up:.1f} GB/s)   gpu->host {t_down*1e3:6.1f} ms"
      f"   one stencil step on the GPU {t_step*1e3:6.2f} ms")
print(f"one transfer up = {t_up/t_step:.0f} stencil steps")

# %% [markdown]
# **Rule:** move data to the GPU once, do all the work there, move results back
# once. Code that does `cp.asarray()` inside a loop is slower than NumPy.
#
# ## 2.5 numba.cuda: write the kernel yourself
#
# CuPy hides the kernel. Sometimes you need your own: an operation that is not a
# composition of array ops, or one where CuPy's temporaries (four per step here)
# cost too much. The mental model is simple: one Python function runs once per
# **thread**, and each thread asks "which element am I?"

# %%
from numba import cuda

@cuda.jit
def step_kernel(u, unew):
    i, j = cuda.grid(2)                      # this thread's (row, col)
    n, m = u.shape
    if 1 <= i < n - 1 and 1 <= j < m - 1:
        unew[i, j] = 0.25 * (u[i-1, j] + u[i+1, j] + u[i, j-1] + u[i, j+1])

def run_kernel(n, iters, dtype=np.float64):
    u = cp.zeros((n, n), dtype=dtype); u[0, :] = 100.0
    unew = u.copy()
    block = (16, 16)                                                # threads per block
    grid = ((n + block[0] - 1) // block[0], (n + block[1] - 1) // block[1])
    for _ in range(iters):
        step_kernel[grid, block](u, unew)
        u, unew = unew, u
    return u

assert np.allclose(cp.asnumpy(run_kernel(1000, 50)), u_cpu)
n = 8000
for dtype in (np.float64, np.float32):
    t_cupy = best_of(lambda: run(cp, n, 20, dtype), sync=sync)
    t_kern = best_of(lambda: run_kernel(n, 20, dtype), sync=sync)
    print(f"{dtype.__name__}: CuPy {t_cupy*1e3:6.2f} ms   numba.cuda kernel {t_kern*1e3:6.2f} ms")

# %% [markdown]
# The hand-written kernel reads each input once and writes once; CuPy's version
# materialises four shifted views plus intermediates. Same idea as NumPy vs Numba
# on the CPU. `numba.cuda` arrays and CuPy arrays interoperate through the
# `__cuda_array_interface__`, so you mix them freely.
#
# ## 2.6 Things that bite
#
# - **Memory**: CuPy caches freed blocks in a pool. `cp.get_default_memory_pool().used_bytes()`
#   tells you what is really in use; `nvidia-smi` shows the pool, not your data.
# - **Shared GPU**: several people on one node share the GPU. Timings will vary;
#   check `nvidia-smi` before reporting a number.
# - **Reductions to Python scalars** (`float(x.sum())`, `if x.max() > 1:`) force a
#   synchronisation and a transfer. Keep the control flow on the device when you can.
# - **Random numbers** on the GPU: `cp.random` is fine; do not generate on the CPU and copy.
#
# ## Where this leads (topic 6)
#
# PyTorch and JAX are CuPy's cousins: an array on a device, operations dispatched
# to kernels, plus autodiff and a compiler. Everything above about transfers,
# synchronisation, precision and launch overhead applies to them unchanged.

# %%
pool = cp.get_default_memory_pool()
print(f"pool used {pool.used_bytes()/2**30:.2f} GB, held {pool.total_bytes()/2**30:.2f} GB")
del a_gpu, u_gpu; pool.free_all_blocks()
print(f"after free: held {pool.total_bytes()/2**30:.2f} GB")

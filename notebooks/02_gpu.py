# %% [markdown]
# # 2. GPU computing from Python
#
# **Live session, blocks 5 and 6 (01:20 to 02:10).** Switch to a GPU runtime
# first: *Runtime > Change runtime type > T4 GPU* (or whatever GPU Colab offers),
# then run the setup cell. If no GPU is available, **stay on the CPU runtime and
# keep going**: the notebook detects that, skips the device cells, and shows a
# recorded GPU table so you can do the same comparisons.
#
# A GPU is thousands of slow cores sharing very fast memory. It wins when you
# have millions of identical, independent operations, and loses when you talk to
# it too often or move data back and forth. From Python:
#
# - **CuPy**: NumPy's API, arrays live on the GPU. Zero new concepts. (This notebook.)
# - **numba.cuda**: write the kernel yourself. One new concept, the thread grid.
#   (Optional extension `extensions/ext_cuda_kernel`.)
#
# Same stencil as notebook 1. This notebook is self-contained.

# %%
# --- Setup: rerun after every runtime restart ------------------------------
import csv, importlib, importlib.util, os, platform, shutil, subprocess, sys, time

def ensure(module, package=None):
    """Import `module`, installing `package` with pip only if the import fails."""
    if importlib.util.find_spec(module) is None:
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", package or module], check=True)
    return importlib.import_module(module)

np = ensure("numpy")
psutil = ensure("psutil")
ensure("matplotlib"); import matplotlib.pyplot as plt
IN_COLAB = "COLAB_RELEASE_TAG" in os.environ or "google.colab" in sys.modules

def gpu_name():
    if shutil.which("nvidia-smi") is None:
        return None
    r = subprocess.run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"], capture_output=True, text=True)
    return r.stdout.strip() or None if r.returncode == 0 else None

GPU_NAME = gpu_name()
HAVE_GPU, cp = False, None
if GPU_NAME and not os.environ.get("HPC_COURSE_FORCE_CPU"):
    try:
        cp = ensure("cupy", "cupy-cuda12x")   # Colab GPU runtimes ship CuPy; install only if the import fails
        assert cp.cuda.runtime.getDeviceCount() > 0
        (cp.arange(4) ** 2).sum().item()      # a real kernel launch: catches driver/library mismatches early
        HAVE_GPU = True
    except Exception as e:
        print("CuPy could not use the GPU:", repr(e))

def runtime_info():
    return {
        "runtime": "Google Colab" if IN_COLAB else f"local machine {platform.node()}",
        "python": platform.python_version(), "numpy": np.__version__,
        "cupy": cp.__version__ if HAVE_GPU else "unavailable",
        "cpus": os.cpu_count(), "ram_gb": round(psutil.virtual_memory().total / 2**30, 1),
        "gpu": GPU_NAME if HAVE_GPU else "none",
    }

def save_timings(name, header, rows):
    with open(name, "w", newline="") as f:
        for k, v in runtime_info().items():
            f.write(f"# {k}: {v}\n")
        w = csv.writer(f, lineterminator="\n"); w.writerow(header); w.writerows(rows)
    print("saved", name)

INFO = runtime_info(); print(INFO)
if HAVE_GPU:
    print(f"\nMODE: GPU. Device cells run live on {GPU_NAME}.")
else:
    print("\nMODE: CPU fallback. Device cells are skipped; GPU numbers come from the recorded table"
          " and are labelled as such. CPU cells still run live here.")

# %% [markdown]
# ## 2.1 CuPy: the drop-in
#
# Write the function once against an *array module* `xp`, then hand it NumPy or
# CuPy arrays. This is the standard pattern (the Array API) and it is how SciPy,
# scikit-learn and xarray grow GPU support without rewriting.

# %%
def step(u, unew):
    unew[1:-1, 1:-1] = 0.25 * (u[:-2, 1:-1] + u[2:, 1:-1] + u[1:-1, :-2] + u[1:-1, 2:])
    return unew

def run(xp, n, iters, dtype=np.float64):
    """`iters` stencil steps on a fresh n x n grid held in `xp` (numpy or cupy) arrays."""
    u = xp.zeros((n, n), dtype=dtype)
    u[0, :] = 100.0
    unew = u.copy()
    for _ in range(iters):
        step(u, unew)
        u, unew = unew, u
    return u

def to_numpy(a):
    return a.get() if HAVE_GPU and isinstance(a, cp.ndarray) else np.asarray(a)

def check(a, b, dtype):
    """Shape, boundaries and values, with a tolerance that matches the dtype."""
    a, b = to_numpy(a), to_numpy(b)
    assert a.shape == b.shape, (a.shape, b.shape)
    assert np.all(a[0] == 100.0) and np.all(a[-1] == 0.0) and np.all(a[1:, 0] == 0.0) and np.all(a[1:, -1] == 0.0)
    rtol = 1e-5 if dtype == np.float32 else 1e-10
    assert np.allclose(a, b, rtol=rtol, atol=rtol * 100), f"max abs diff {np.abs(a - b).max()}"

u_cpu = run(np, 512, 20)
if HAVE_GPU:
    u_gpu = run(cp, 512, 20)
    print(type(u_gpu), "on device", u_gpu.device)
    check(u_gpu, u_cpu, np.float64)
    print("GPU result matches the CPU result")
else:
    check(u_cpu, u_cpu, np.float64); print("CPU fallback: correctness check exercised on NumPy only")

# %% [markdown]
# **Checkpoint 1.** The same `step` function ran on a different kind of memory
# and produced the same numbers. Nothing about the algorithm changed.
#
# ## 2.2 Timing GPU code correctly
#
# GPU calls are **asynchronous**: `step()` returns before the GPU has finished.
# Without a `synchronize()` before you stop the clock you time the *launch*, not
# the work. Synchronise before starting and before stopping. This is the single
# most common measurement mistake on GPUs.

# %%
def best_of(fn, repeat=5, sync=lambda: None):
    """Warm up once, then (min, median) seconds over `repeat` calls, synchronising around each."""
    fn(); sync()
    ts = []
    for _ in range(repeat):
        sync(); t0 = time.perf_counter(); fn(); sync(); ts.append(time.perf_counter() - t0)
    return min(ts), float(np.median(ts))

def no_sync_timing(fn, repeat=5):
    fn()
    ts = []
    for _ in range(repeat):
        t0 = time.perf_counter(); fn(); ts.append(time.perf_counter() - t0)
    return min(ts)

if HAVE_GPU:
    sync = cp.cuda.Device().synchronize
    n = 1024
    print(f"n={n}, 20 steps   without sync: {no_sync_timing(lambda: run(cp, n, 20))*1e3:7.2f} ms  <- wrong")
    print(f"                  with sync:    {best_of(lambda: run(cp, n, 20), sync=sync)[0]*1e3:7.2f} ms")
else:
    sync = lambda: None
    print("CPU fallback: no asynchronous device to synchronise. The recorded table below has the GPU numbers.")

# %% [markdown]
# The "without sync" figure can even be *smaller* than the time of a single
# memory copy: the CPU queued the work and walked away. Every GPU timing in the
# rest of this notebook goes through `best_of(..., sync=sync)`.
#
# ## 2.3 When is the GPU worth it? Like-for-like size sweep
#
# Same workload, same dtype, same number of steps, timed the same way, on the CPU
# and on the GPU. **Predict** before running: at which grid side, if any, will the
# GPU overtake the CPU on this runtime?

# %%
SIZES, ITERS = (128, 512, 1024, 2048), 20
timings = []                                    # (impl, n, dtype, scope, min_s, median_s)
for n in SIZES:
    tc = best_of(lambda: run(np, n, ITERS)); timings.append(("numpy", n, "float64", "compute", *tc))
    line = f"n={n:5d}  cpu f64 {tc[0]*1e3:8.2f} ms"
    if HAVE_GPU:
        tg = best_of(lambda: run(cp, n, ITERS), sync=sync); timings.append(("cupy", n, "float64", "compute", *tg))
        line += f"   gpu f64 {tg[0]*1e3:8.2f} ms   cpu/gpu {tc[0]/tg[0]:5.1f}x"
    print(line)

# %%
import urllib.request
REF_URL = "https://raw.githubusercontent.com/caschb/IEEELatamTechRepo/main/data/reference_timings/02_gpu.csv"
LOCAL_REF = os.path.join("..", "data", "reference_timings", "02_gpu.csv")

def load_reference(url, local):
    try:
        text = open(local).read() if os.path.exists(local) else urllib.request.urlopen(url, timeout=10).read().decode()
    except Exception as e:
        print("no recorded data available:", repr(e)); return [], {}
    meta = dict(l[2:].split(": ", 1) for l in text.splitlines() if l.startswith("# "))
    return list(csv.DictReader(l for l in text.splitlines() if not l.startswith("#"))), meta

recorded, rec_meta = ([], {}) if HAVE_GPU else load_reference(REF_URL, LOCAL_REF)
if recorded:
    print("RECORDED GPU RUN, not this runtime:", {k: rec_meta.get(k) for k in ("runtime", "gpu", "cupy", "cpus")})
    print(f"{'impl':6s} {'n':>6s} {'dtype':8s} {'scope':18s} {'min ms':>10s}")
    for r in recorded:
        print(f"{r['impl']:6s} {int(r['n']):6d} {r['dtype']:8s} {r['scope']:18s} {float(r['min_s'])*1e3:10.2f}")

# %%
def series(rows, impl, dtype, scope="compute"):
    pts = sorted((int(r[1]), float(r[4])) for r in rows if r[0] == impl and r[2] == dtype and r[3] == scope)
    return np.array(pts) if pts else None

rec_rows = [(r["impl"], r["n"], r["dtype"], r["scope"], r["min_s"], r["median_s"]) for r in recorded]
fig, ax = plt.subplots(figsize=(5.5, 3.8))
for rows, label, style in ((timings, "this runtime", "-"), (rec_rows, "recorded", "--")):
    for impl, marker in (("numpy", "o"), ("cupy", "s")):
        s = series(rows, impl, "float64")
        if s is not None:
            ax.plot(s[:, 0], s[:, 1] * 1e3, marker + style, label=f"{impl} f64, {label}")
ax.set(xlabel="grid side n", ylabel=f"time for {ITERS} steps [ms]", xscale="log", yscale="log")
ax.set_xticks(SIZES); ax.set_xticklabels(SIZES); ax.legend(fontsize=8); fig.tight_layout()

# %% [markdown]
# **Explain what you see.** Two questions to answer from *your* table (or the
# recorded one, if you are in CPU mode):
#
# 1. For the smallest grids, does the GPU time change much with `n`? Each step
#    launches a handful of kernels, and a launch costs microseconds regardless of
#    the work. Below some size the GPU is mostly idle and can lose to NumPy.
# 2. Where, if anywhere, do the two curves cross? That crossover depends on the
#    CPU, the GPU and what else shares them. Do not memorise a number; memorise
#    the shape.
#
# ## 2.4 The transfer tax: compute-only versus transfer-inclusive
#
# GPU memory moves hundreds of GB/s; the PCIe link between host and device moves
# tens. A workload that copies an array up, does a little work, and copies it
# back can spend most of its time on the copies. Report the two scopes
# separately, and say which one you mean.
#
# **Predict:** for `n=2048` (32 MB in float64), is one upload cheaper or more
# expensive than one stencil step on the device?

# %%
n = 2048
a = np.zeros((n, n)); a[0, :] = 100.0

def cpu_only():                                          # the equivalent CPU workload, for scale
    u, unew = a.copy(), a.copy()
    for _ in range(ITERS): step(u, unew); u, unew = unew, u
    return u

def gpu_compute_only(u, unew):                           # data already on the device
    for _ in range(ITERS): step(u, unew); u, unew = unew, u
    return u

def gpu_transfer_inclusive():                            # upload, compute, download
    u = cp.asarray(a); unew = u.copy()
    return gpu_compute_only(u, unew).get()

tc = best_of(cpu_only); timings.append(("numpy", n, "float64", "transfer_inclusive", *tc))
print(f"CPU, {ITERS} steps:                     {tc[0]*1e3:8.2f} ms")
if HAVE_GPU:
    ug, ung = cp.asarray(a), cp.asarray(a)
    t_up = best_of(lambda: cp.asarray(a), sync=sync)[0]
    t_down = best_of(lambda: ug.get(), sync=sync)[0]
    t_step = best_of(lambda: step(ug, ung), sync=sync)[0]
    tk = best_of(lambda: gpu_compute_only(ug, ung), sync=sync); timings.append(("cupy", n, "float64", "compute", *tk))
    ti = best_of(gpu_transfer_inclusive, sync=sync); timings.append(("cupy", n, "float64", "transfer_inclusive", *ti))
    gb = a.nbytes / 2**30
    print(f"host->gpu {t_up*1e3:6.2f} ms ({gb/t_up:.1f} GB/s)   gpu->host {t_down*1e3:6.2f} ms   one step on device {t_step*1e3:6.2f} ms")
    print(f"one upload = {t_up/t_step:.1f} stencil steps")
    print(f"GPU compute-only, {ITERS} steps:          {tk[0]*1e3:8.2f} ms")
    print(f"GPU transfer-inclusive, {ITERS} steps:    {ti[0]*1e3:8.2f} ms   (transfers are {(ti[0]-tk[0])/ti[0]:.0%} of it)")
    del ug, ung
else:
    for r in recorded:
        if r["impl"] == "cupy" and int(r["n"]) == n:
            print(f"[recorded] GPU {r['scope']:18s} {float(r['min_s'])*1e3:8.2f} ms")

# %% [markdown]
# **Rule:** move data to the GPU once, do all the work there, move results back
# once. Code that does `cp.asarray()` inside a loop is usually slower than NumPy.
# When you quote a GPU speedup, say whether the transfers are inside the clock.
#
# ## 2.5 Precision: float32 (optional, second cut if the session runs late)
#
# `float32` halves the bytes per cell. For a **memory-bound** kernel like this one
# that can approach a 2x change on either device; the arithmetic rate is not the
# limit. Cards differ enormously in how fast they do `float64` arithmetic, so a
# **compute-bound** kernel can show a much larger gap on one GPU and almost none
# on another. Two lessons: know which resource you are bound by, and check your
# result still agrees within a tolerance that suits the dtype.

# %%
n = 1024
u64 = run(np, n, ITERS)
u32 = run(np, n, ITERS, np.float32); check(u32, u64, np.float32)
tc32 = best_of(lambda: run(np, n, ITERS, np.float32)); timings.append(("numpy", n, "float32", "compute", *tc32))
tc64 = next(r[4] for r in timings if r[:4] == ("numpy", n, "float64", "compute"))
print(f"CPU n={n}: f64 {tc64*1e3:8.2f} ms   f32 {tc32[0]*1e3:8.2f} ms   f64/f32 {tc64/tc32[0]:.2f}x")
if HAVE_GPU:
    g32 = run(cp, n, ITERS, np.float32); check(g32, u64, np.float32)
    tg32 = best_of(lambda: run(cp, n, ITERS, np.float32), sync=sync); timings.append(("cupy", n, "float32", "compute", *tg32))
    tg64 = next(r[4] for r in timings if r[:4] == ("cupy", n, "float64", "compute"))
    print(f"GPU n={n}: f64 {tg64*1e3:8.2f} ms   f32 {tg32[0]*1e3:8.2f} ms   f64/f32 {tg64/tg32[0]:.2f}x")

# %% [markdown]
# ## 2.6 Things that bite
#
# - **Memory pool.** CuPy caches freed blocks. `cp.get_default_memory_pool().used_bytes()`
#   is what your arrays occupy; `nvidia-smi` shows the pool, not your data.
# - **Shared hardware.** Colab GPUs are shared and time-limited; a T4 today may be
#   another card tomorrow. Always record which device produced a number.
# - **Reductions to Python scalars** (`float(x.sum())`, `if x.max() > 1:`) force a
#   synchronisation and a transfer. Keep control flow on the device when you can.
# - **Random numbers.** `cp.random` generates on the device; do not generate on the
#   CPU and copy.
#
# **Where this leads.** PyTorch and JAX are CuPy's cousins: an array on a device,
# operations dispatched to kernels, plus autodiff and a compiler. Everything above
# about transfers, synchronisation, precision and launch overhead applies unchanged.
#
# ## Checkpoint 2: your timing table

# %%
save_timings("timings_02_gpu.csv", ["impl", "n", "dtype", "scope", "min_s", "median_s"], timings)
print(f"{'impl':6s} {'n':>6s} {'dtype':8s} {'scope':18s} {'min ms':>10s}")
for impl, n, dtype, scope, tmin, _ in timings:
    print(f"{impl:6s} {n:6d} {dtype:8s} {scope:18s} {tmin*1e3:10.2f}")
if HAVE_GPU:
    pool = cp.get_default_memory_pool()
    print(f"\npool used {pool.used_bytes()/2**20:.0f} MB, held {pool.total_bytes()/2**20:.0f} MB")
    del u_gpu; pool.free_all_blocks()
    print(f"after free: held {pool.total_bytes()/2**20:.0f} MB")

# %% [markdown]
# **Exit question for this block.** A colleague reports "the GPU is 40x faster"
# for a kernel that runs for 2 ms. What three things do you ask before believing it?
# (Synchronised? Transfers included? Same dtype and workload on both sides?)

# %% [markdown]
# # 1. Measure first, then go multicore
#
# **Live session, blocks 2 and 3 (00:10 to 01:10).** Standard CPU runtime; no GPU needed.
#
# Three questions to ask before touching any code:
#
# 1. **Where does the time go?** Measure. Intuition is usually wrong.
# 2. **What is the ceiling?** Amdahl's law: if a fraction `s` of the work stays serial,
#    no number of cores gets you past a speedup of `1/s`.
# 3. **Did it help?** Measure again, on the same machine, the same way.
#
# The running example is the 2D heat-diffusion stencil from the preparation
# notebook. Each step replaces every interior cell with the average of its four
# neighbours. It is small, memory-bound, and it follows us into the GPU and
# multi-partition sessions.
#
# This notebook is self-contained: run the setup cell after every runtime restart.

# %%
# --- Setup: rerun after every runtime restart ------------------------------
import csv, importlib, importlib.util, os, platform, subprocess, sys, time

def ensure(module, package=None):
    """Import `module`, installing `package` with pip only if the import fails."""
    if importlib.util.find_spec(module) is None:
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", package or module], check=True)
    return importlib.import_module(module)

np = ensure("numpy")
numba = ensure("numba")
psutil = ensure("psutil")
ensure("matplotlib"); import matplotlib.pyplot as plt
IN_COLAB = "COLAB_RELEASE_TAG" in os.environ or "google.colab" in sys.modules

def runtime_info():
    smi = subprocess.run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                         capture_output=True, text=True) if importlib.util.find_spec("cupy") else None
    return {
        "runtime": "Google Colab" if IN_COLAB else f"local machine {platform.node()}",
        "python": platform.python_version(), "numpy": np.__version__, "numba": numba.__version__,
        "cpus": os.cpu_count(), "numba_max_threads": numba.config.NUMBA_NUM_THREADS,
        "ram_gb": round(psutil.virtual_memory().total / 2**30, 1),
        "gpu": smi.stdout.strip() if smi and smi.returncode == 0 and smi.stdout.strip() else "none",
    }

def save_timings(name, header, rows):
    """Write a CSV of timings with the runtime description as '#' comment lines."""
    with open(name, "w", newline="") as f:
        for k, v in runtime_info().items():
            f.write(f"# {k}: {v}\n")
        w = csv.writer(f, lineterminator="\n"); w.writerow(header); w.writerows(rows)
    print("saved", name, "(download it from the Files panel if you want to keep it)")

INFO = runtime_info()
print(INFO)

# %% [markdown]
# ## 1.1 The running example and its reference result
#
# `init_grid` builds an `n x n` grid with a hot top edge. `step_python` is the
# stencil written the way a scientist writes it on day one. It is the
# **reference**: slow, obviously correct, and the thing every faster version
# must agree with.

# %%
def init_grid(n, dtype=np.float64):
    u = np.zeros((n, n), dtype=dtype)
    u[0, :] = 100.0                        # hot top edge; the other three edges stay at 0
    return u

def step_python(u, unew):
    n, m = len(u), len(u[0])
    for i in range(1, n - 1):
        for j in range(1, m - 1):
            unew[i][j] = 0.25 * (u[i-1][j] + u[i+1][j] + u[i][j-1] + u[i][j+1])
    return unew

def run_python(n, iters):
    u, unew = init_grid(n).tolist(), init_grid(n).tolist()
    for _ in range(iters):
        step_python(u, unew)
        u, unew = unew, u
    return np.array(u)

N_REF, ITERS_REF = 64, 10
ref = run_python(N_REF, ITERS_REF)
print("shape", ref.shape, " top edge", ref[0, :3], " mean interior", ref[1:-1, 1:-1].mean().round(4))
assert ref.shape == (N_REF, N_REF) and np.all(ref[0] == 100.0) and np.all(ref[-1] == 0.0)

# %% [markdown]
# **Checkpoint 1.** The cell above ran without an assertion error. Keep `ref` and
# the `N_REF, ITERS_REF` case: every implementation below is checked against it.
#
# ## 1.2 Measuring fairly
#
# One `time.perf_counter()` pair gives one number. Run it again and you get a
# different one: the first call pays for imports, caches and (later) JIT
# compilation; later calls share the machine with whatever else is running.
# Two habits fix most of that:
#
# - **Warm up** once, then time several **repeats** and keep the minimum (the run
#   with the least interference) or the median (when you care about typical time).
# - Compare implementations **within one runtime**, on the **same workload and
#   dtype**. A number from another machine is a different experiment.

# %%
def best_of(fn, repeat=5):
    """Warm up once, then return (min, median) wall-clock seconds over `repeat` calls."""
    fn()
    ts = []
    for _ in range(repeat):
        t0 = time.perf_counter(); fn(); ts.append(time.perf_counter() - t0)
    return min(ts), float(np.median(ts))

t_min, t_med = best_of(lambda: run_python(128, 10), repeat=3)
print(f"pure Python, n=128, 10 steps: min {t_min*1e3:.0f} ms   median {t_med*1e3:.0f} ms")

# %% [markdown]
# IPython's `%timeit` does the same job. Read its output carefully: it reports the
# **mean and standard deviation across `-r` runs**, and each run is the time for
# `-n` loops divided by `n`. With `-o` you get the object back and `.best` is the
# fastest run.

# %%
t = %timeit -o -r 3 -n 1 run_python(128, 10)
print(f"best run {t.best*1e3:.0f} ms; %timeit's headline number is the mean of {t.repeat} runs")

# %% [markdown]
# **Predict, run, explain.** `n=256` has four times as many cells as `n=128`.
# Write down your prediction for the time ratio, then run the next cell.

# %%
t_128, _ = best_of(lambda: run_python(128, 10), repeat=3)
t_256, _ = best_of(lambda: run_python(256, 10), repeat=3)
print(f"n=128: {t_128*1e3:6.0f} ms   n=256: {t_256*1e3:6.0f} ms   ratio {t_256/t_128:.1f}x")

# %% [markdown]
# A ratio near 4 means the time is proportional to the number of cells: the
# interpreter overhead per cell dominates. Keep this number; it is the baseline
# everything else is measured against.
#
# ## 1.3 Where does the time go? (optional demonstration)
#
# `line_profiler` shows time per line of one function. Here the answer is
# unsurprising, all the time is the inner loop body, but in real code the
# expensive line is often a surprise: a `print`, a lookup inside a loop, a
# conversion. For whole programs use `python -m cProfile` or the sampling
# profiler `py-spy`. *This cell is the first one cut if the session runs late.*

# %%
try:
    ensure("line_profiler")
    get_ipython().run_line_magic("load_ext", "line_profiler")
    get_ipython().run_line_magic("lprun", "-f step_python run_python(64, 3)")
except Exception as e:
    print("profiler demonstration skipped:", repr(e))

# %% [markdown]
# ## 1.4 NumPy: let compiled code do the loop
#
# The first and biggest win is usually not parallelism. It is moving the loop out
# of the interpreter. Each slice below (`u[:-2, 1:-1]` and friends) is a **view**:
# no data is copied. The arithmetic between them is different: every `+` and the
# `0.25 *` allocates a full-size **intermediate array**, four per step here, and
# each one is written and read back through memory.

# %%
def step_numpy(u, unew):
    unew[1:-1, 1:-1] = 0.25 * (u[:-2, 1:-1] + u[2:, 1:-1] + u[1:-1, :-2] + u[1:-1, 2:])
    return unew

def run(step, n, iters, dtype=np.float64):
    """Run `iters` steps of `step` on a fresh grid and return the final grid."""
    u = init_grid(n, dtype)
    unew = u.copy()
    for _ in range(iters):
        step(u, unew)
        u, unew = unew, u
    return u

assert np.allclose(run(step_numpy, N_REF, ITERS_REF), ref)
t_np, _ = best_of(lambda: run(step_numpy, 128, 10))
print(f"NumPy n=128, 10 steps: {t_np*1e3:.2f} ms   ({t_128/t_np:.0f}x faster than pure Python, no parallelism yet)")

# %% [markdown]
# From here on the workloads grow, because NumPy at `n=128` finishes before the
# clock can resolve it. The three grid sides used throughout the live session are
# 128, 512 and 1024, with 20 steps each.

# %%
SIZES, ITERS = (128, 512, 1024), 20
timings = []                       # (implementation, n, threads, min_s, median_s)
for n in SIZES:
    tmin, tmed = best_of(lambda: run(step_numpy, n, ITERS))
    timings.append(("numpy", n, 1, tmin, tmed))
    print(f"NumPy n={n:5d}, {ITERS} steps: min {tmin*1e3:8.2f} ms   median {tmed*1e3:8.2f} ms")

# %% [markdown]
# **Checkpoint 2.** You have a NumPy baseline for three sizes. Which resource do
# you think limits it: arithmetic, or reading and writing the intermediates
# through memory? Section 1.5 tests that idea.
#
# ## 1.5 Numba: compile your own loop
#
# Numba compiles the explicit loop to machine code. We keep the readable loop and
# lose the intermediates: each output cell reads four inputs and writes once.
# The **first call compiles** (a second or two); never include it in a timing.

# %%
from numba import njit, prange

@njit
def step_numba(u, unew):
    n, m = u.shape
    for i in range(1, n - 1):
        for j in range(1, m - 1):
            unew[i, j] = 0.25 * (u[i-1, j] + u[i+1, j] + u[i, j-1] + u[i, j+1])
    return unew

t0 = time.perf_counter(); run(step_numba, N_REF, ITERS_REF); t_compile = time.perf_counter() - t0
assert np.allclose(run(step_numba, N_REF, ITERS_REF), ref)
print(f"first call, including compilation: {t_compile:.2f} s")

for n in SIZES:
    tmin, tmed = best_of(lambda: run(step_numba, n, ITERS))
    timings.append(("numba", n, 1, tmin, tmed))
    t_base = next(r[3] for r in timings if r[0] == "numpy" and r[1] == n)
    print(f"Numba n={n:5d}: min {tmin*1e3:8.2f} ms   vs NumPy {t_base/tmin:4.1f}x")

# %% [markdown]
# Whatever ratio you saw is the ratio **on this runtime**. On a machine with fast
# memory the gap between NumPy and Numba shrinks; on a slow, shared one it grows.
# The mechanism, fewer passes over memory, is the same everywhere.
#
# ## 1.6 Threads with `prange`
#
# `prange` splits the outer loop over threads. How many threads can this runtime
# actually use? `os.cpu_count()` reports what the operating system shows, and
# Numba caps its pool at `numba.config.NUMBA_NUM_THREADS`. A standard Colab
# runtime typically exposes two logical CPUs. Ask before assuming.

# %%
@njit(parallel=True)
def step_numba_par(u, unew):
    n, m = u.shape
    for i in prange(1, n - 1):
        for j in range(1, m - 1):
            unew[i, j] = 0.25 * (u[i-1, j] + u[i+1, j] + u[i, j-1] + u[i, j+1])
    return unew

assert np.allclose(run(step_numba_par, N_REF, ITERS_REF), ref)   # first call also compiles

MAX_THREADS = min(os.cpu_count() or 1, numba.config.NUMBA_NUM_THREADS)
THREADS = sorted({1, 2, MAX_THREADS // 2, MAX_THREADS} & set(range(1, MAX_THREADS + 1)))
print(f"cpu_count {os.cpu_count()}, Numba pool limit {numba.config.NUMBA_NUM_THREADS} -> thread counts to test: {THREADS}")

# %% [markdown]
# **Predict, run, explain.** With two threads, will the `n=1024` time halve?
# Write your prediction down before running.

# %%
N_SCALE = 1024
scaling = []
for t in THREADS:
    numba.set_num_threads(t)
    run(step_numba_par, N_SCALE, 2)                     # warm the thread pool at this size
    tmin, tmed = best_of(lambda: run(step_numba_par, N_SCALE, ITERS))
    scaling.append((t, tmin)); timings.append(("numba_prange", N_SCALE, t, tmin, tmed))
    print(f"{t:3d} thread(s): {tmin*1e3:8.2f} ms   speedup vs 1 thread {scaling[0][1]/tmin:4.2f}x")
numba.set_num_threads(MAX_THREADS)

# %% [markdown]
# If this runtime has a single usable core, the table has one row and cannot show
# scaling. The next cell loads a **recorded** thread sweep from the course
# repository so the interpretation exercise still works. The recording is labelled
# with the hardware it came from; it is evidence about *that* machine, not this one.

# %%
import io, urllib.request
REF_URL = "https://raw.githubusercontent.com/caschb/IEEELatamTechRepo/main/data/reference_timings/01_threads.csv"
LOCAL_REF = os.path.join("..", "data", "reference_timings", "01_threads.csv")

def load_reference(url, local):
    try:
        text = open(local).read() if os.path.exists(local) else urllib.request.urlopen(url, timeout=10).read().decode()
    except Exception as e:
        print("no recorded data available:", repr(e)); return None, {}
    meta = dict(l[2:].split(": ", 1) for l in text.splitlines() if l.startswith("# "))
    rows = list(csv.DictReader(l for l in text.splitlines() if not l.startswith("#")))
    return rows, meta

if len(scaling) < 2:
    rows, meta = load_reference(REF_URL, LOCAL_REF)
    if rows:
        print("recorded on:", {k: meta[k] for k in ("runtime", "cpus", "numba", "gpu") if k in meta})
        scaling = [(int(r["threads"]), float(r["min_s"])) for r in rows if r["impl"] == "numba_prange"]
        for t, s in scaling: print(f"{t:3d} thread(s): {s*1e3:8.2f} ms   speedup {scaling[0][1]/s:4.2f}x   [recorded]")

# %%
if len(scaling) >= 2:
    th = np.array([t for t, _ in scaling]); sp = scaling[0][1] / np.array([s for _, s in scaling])
    fig, ax = plt.subplots(figsize=(5, 3.5))
    ax.plot(th, th, "--", color="0.6", label="ideal")
    for s in (0.05, 0.20, 0.50):
        ax.plot(th, [1 / (s + (1 - s) / t) for t in th], ":", label=f"Amdahl, {s:.0%} serial")
    ax.plot(th, sp, "o-", label="measured")
    ax.set(xlabel="threads", ylabel="speedup", title=f"n={N_SCALE}, {ITERS} steps")
    ax.set_xticks(th); ax.legend(); fig.tight_layout()

# %% [markdown]
# **Reading the plot.** The Amdahl curves are *models*: pick the one closest to the
# measured points and you have an estimate of the "serial fraction". For this
# stencil there is no serial Python left in the loop, so a bad fit is not Amdahl
# at all. The usual culprits are:
#
# - **Memory bandwidth.** The stencil does four reads and one write per cell and
#   almost no arithmetic. Once a couple of threads saturate the memory bus, more
#   threads only wait in line.
# - **Shared or virtual cores.** Two logical CPUs on a shared host may be one
#   physical core. Then two threads compete for one unit and can be slower than one.
# - **Thread start-up.** For small grids, waking the pool costs more than the work.
#
# **Runtime-aware limit.** Never ask for more threads than the runtime gives you:
# `numba.set_num_threads(min(wanted, os.cpu_count()))`. Over-subscription is not an
# error, only silence and slowness.
#
# ## Checkpoint 3: your timing table
#
# Save the table with the runtime description. You will compare it with the GPU
# table in the next notebook, and reuse the functions in the capstone.

# %%
save_timings("timings_01_cpu.csv", ["impl", "n", "threads", "min_s", "median_s"], timings)
print(f"{'impl':14s} {'n':>6s} {'thr':>4s} {'min ms':>10s}")
for impl, n, thr, tmin, _ in timings:
    print(f"{impl:14s} {n:6d} {thr:4d} {tmin*1e3:10.2f}")

# %% [markdown]
# ## Takeaways
#
# | Situation | Reach for |
# |---|---|
# | Loop over array elements | NumPy vectorisation first, then Numba `@njit` |
# | Same loop, several cores | Numba `parallel=True` + `prange`, threads <= cores |
# | Many independent Python tasks | processes (`concurrent.futures`, `joblib`), or Dask; see the extension notebook |
# | Any performance claim | Measure before and after, same runtime, same workload, same dtype |
#
# **Optional extensions (not covered live):** `extensions/ext_gil_and_task_pools`
# shows why Python threads do not speed up pure-Python loops and compares
# `ProcessPoolExecutor` and `joblib` on an independent-task sweep.

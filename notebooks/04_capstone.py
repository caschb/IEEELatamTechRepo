# %% [markdown]
# # 4. Capstone: choose and justify an implementation
#
# **Live session, block 8 (02:30 to 02:50).** Works on a CPU runtime; the GPU
# rows appear only if a GPU is available. Self-contained: all functions from
# notebooks 1 and 2 are defined again below.
#
# You have two workloads. For **each one**, produce three things:
#
# 1. a correctness check against the reference,
# 2. a timing table on this runtime, with the runtime described,
# 3. a two-sentence recommendation in the final markdown cell, tied to *your*
#    numbers, not to the slides.
#
# There is no target speedup. You succeed by producing correct results and
# explaining what you measured.

# %%
# --- Setup: rerun after every runtime restart ------------------------------
import csv, importlib, importlib.util, os, platform, shutil, subprocess, sys, time

def ensure(module, package=None):
    if importlib.util.find_spec(module) is None:
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", package or module], check=True)
    return importlib.import_module(module)

np = ensure("numpy"); numba = ensure("numba"); psutil = ensure("psutil")
from numba import njit, prange
IN_COLAB = "COLAB_RELEASE_TAG" in os.environ or "google.colab" in sys.modules

HAVE_GPU, cp = False, None
if shutil.which("nvidia-smi") and not os.environ.get("HPC_COURSE_FORCE_CPU"):
    try:
        cp = ensure("cupy", "cupy-cuda12x"); assert cp.cuda.runtime.getDeviceCount() > 0
        (cp.arange(4) ** 2).sum().item(); HAVE_GPU = True
    except Exception as e:
        print("no usable GPU:", repr(e))

def runtime_info():
    return {"runtime": "Google Colab" if IN_COLAB else f"local machine {platform.node()}", "python": platform.python_version(),
            "numpy": np.__version__, "numba": numba.__version__, "cupy": cp.__version__ if HAVE_GPU else "unavailable",
            "cpus": os.cpu_count(), "ram_gb": round(psutil.virtual_memory().total / 2**30, 1),
            "gpu": subprocess.run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"], capture_output=True,
                                  text=True).stdout.strip() if HAVE_GPU else "none"}

def save_timings(name, header, rows):
    with open(name, "w", newline="") as f:
        for k, v in runtime_info().items(): f.write(f"# {k}: {v}\n")
        w = csv.writer(f, lineterminator="\n"); w.writerow(header); w.writerows(rows)
    print("saved", name)

def best_of(fn, repeat=3, sync=lambda: None):
    fn(); sync(); ts = []
    for _ in range(repeat):
        sync(); t0 = time.perf_counter(); fn(); sync(); ts.append(time.perf_counter() - t0)
    return min(ts)

MAX_THREADS = min(os.cpu_count() or 1, numba.config.NUMBA_NUM_THREADS)
numba.set_num_threads(MAX_THREADS)
sync = cp.cuda.Device().synchronize if HAVE_GPU else (lambda: None)
print(runtime_info(), "\nGPU rows:", "live" if HAVE_GPU else "skipped (no GPU on this runtime)")

# %% [markdown]
# ## The toolbox (from notebooks 1 and 2)

# %%
def init_grid(n, xp=np, dtype=np.float64, top=100.0):
    u = xp.zeros((n, n), dtype=dtype); u[0, :] = top
    return u

def step_numpy(u, unew):
    unew[1:-1, 1:-1] = 0.25 * (u[:-2, 1:-1] + u[2:, 1:-1] + u[1:-1, :-2] + u[1:-1, 2:])
    return unew

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

def run(step, n, iters, xp=np, dtype=np.float64, top=100.0):
    u = init_grid(n, xp, dtype, top); unew = u.copy()
    for _ in range(iters):
        step(u, unew); u, unew = unew, u
    return u

def to_numpy(a):
    return a.get() if HAVE_GPU and isinstance(a, cp.ndarray) else np.asarray(a)

CANDIDATES = {"numpy": (step_numpy, np), "numba": (step_numba, np), "numba_par": (step_numba_par, np)}
if HAVE_GPU:
    CANDIDATES["cupy"] = (step_numpy, cp)         # same slicing code, device arrays
for name, (step, xp) in CANDIDATES.items():       # warm up / compile everything once
    run(step, 64, 2, xp)
print("candidates:", list(CANDIDATES))

# %% [markdown]
# ## Workload A (CPU): many small independent simulations
#
# 48 grids of side 128, 30 steps each, each with a different top-edge
# temperature. Nothing is shared between grids. The reference is NumPy.
#
# **Decide:** which candidate, and why? Consider that the grids are small, and
# that `numba_par` spends its threads *inside* one grid. Fill in `CHOICE_A`, then
# run the cells. You may add a candidate of your own to `CANDIDATES` (for
# example, a plain loop over grids inside one `@njit(parallel=True)` function).

# %%
TEMPS = np.linspace(50, 150, 48)
N_A, ITERS_A = 128, 30

def sweep(step, xp=np):
    return [to_numpy(run(step, N_A, ITERS_A, xp, top=t))[1:-1, 1:-1].mean() for t in TEMPS]

ref_A = sweep(step_numpy)
timings_A = []
for name in ["numpy", "numba", "numba_par"]:            # <- CHOICE_A: edit this list to what you want to compare
    step, xp = CANDIDATES[name]
    assert np.allclose(sweep(step, xp), ref_A), f"{name} disagrees with the reference"
    t = best_of(lambda: sweep(step, xp), sync=sync if xp is not np else (lambda: None))
    timings_A.append(("A_sweep", name, N_A, ITERS_A, len(TEMPS), t))
    print(f"workload A  {name:10s} {t*1e3:9.1f} ms  (correct)")

# %% [markdown]
# ## Workload B: one large coupled simulation
#
# One grid of side 2048, 40 steps. On a GPU runtime the CuPy candidate is timed
# both **compute-only** and **transfer-inclusive** (upload once, compute, download
# once). Without a GPU, compare the CPU candidates and use the recorded GPU
# table from notebook 2 for the discussion.
#
# **Decide:** which candidate for this workload, and does the answer change if
# the result must come back to the CPU after every 40 steps?

# %%
N_B, ITERS_B = 2048, 40
ref_B = run(step_numpy, N_B, ITERS_B)
timings_B = []
for name in ["numpy", "numba_par"] + (["cupy"] if HAVE_GPU else []):     # <- CHOICE_B
    step, xp = CANDIDATES[name]
    out = run(step, N_B, ITERS_B, xp)
    assert np.allclose(to_numpy(out), ref_B), f"{name} disagrees with the reference"
    t = best_of(lambda: run(step, N_B, ITERS_B, xp), sync=sync if xp is not np else (lambda: None))
    scope = "compute" if xp is np else "compute (data created on device)"
    timings_B.append(("B_single", name, N_B, ITERS_B, 1, t))
    print(f"workload B  {name:10s} {t*1e3:9.1f} ms  {scope}  (correct)")

if HAVE_GPU:
    host = init_grid(N_B)
    def transfer_inclusive():
        u = cp.asarray(host); unew = u.copy()
        for _ in range(ITERS_B): step_numpy(u, unew); u, unew = unew, u
        return u.get()
    t = best_of(transfer_inclusive, sync=sync)
    timings_B.append(("B_single", "cupy_transfer_inclusive", N_B, ITERS_B, 1, t))
    print(f"workload B  {'cupy':10s} {t*1e3:9.1f} ms  transfer-inclusive")

# %%
save_timings("timings_04_capstone.csv", ["workload", "impl", "n", "iters", "tasks", "min_s"], timings_A + timings_B)

# %% [markdown]
# ## Your recommendation (edit this cell)
#
# **Runtime:** *(paste the runtime description printed by the setup cell)*
#
# **Workload A (48 small independent grids):** I would use ... because on this
# runtime it took ... versus ... . Threads inside one small grid did / did not
# help because ... . If I had a cluster, this workload would map onto ... (task
# scheduler / message passing) because the grids exchange ... bytes.
#
# **Workload B (one 2048x2048 grid):** I would use ... because ... . The
# transfer-inclusive time was ... of the compute-only time, so moving the result
# back every 40 steps would / would not change the decision.
#
# **One thing more hardware would not fix:** ...

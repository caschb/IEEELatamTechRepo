# %% [markdown]
# # 4. Capstone: elige y justifica una implementación
#
# **Sesión en vivo, bloque 8 (02:30 a 02:50).** Trabaja en un entorno de ejecución CPU; las filas de GPU aparecen solo si está disponible. Autocontenidos: se definen de nuevo todas las funciones de los notebooks 1 y 2.
#
# Tienes dos cargas de trabajo. Para **cada una**, produce tres cosas:
#
# 1. un chequeo de correctitud contra la referencia,
# 2. una tabla de tiempos en este entorno de ejecución, con la descripción del entorno de ejecución,
# 3. dos frases de recomendación en la última celda de markdown, ligadas a tus números, no a las diapositivas.
#
# No hay objetivo de aceleración. Has logrado si produciste resultados correctos y explicas lo que midiste.

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
# ## El toolbox (desde los notebooks 1 y 2)

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
# ## Workload A (CPU): muchas simulaciones pequeñas e independientes
#
# 48 grids de lado 128, 30 pasos cada uno, cada uno con una temperatura de borde superior diferente. Nothing es compartido entre los grids. El referente es NumPy.
#
# **Decide:** cual candidato, y por qué? Considera que los grids son pequeños, y que `numba_par` gasta sus hilos *dentro* de uno solo de los grids. Llena en `CHOICE_A`, luego ejecuta las celdas. Puedes añadir un candidato propio a `CANDIDATES` (por ejemplo, un bucle plano sobre los grids dentro de una función `@njit(parallel=True)`).

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
# ## Trabajo de carga B: una gran simulación unificada
#
# Una cuadrícula de lado 2048, 40 pasos. En un entorno de ejecución con GPU, el candidato de CuPy se mide tanto **solo de cálculo** como **inclusivo de transferencia** (subida una vez, cálculo, bajada una vez). Sin GPU, compara los candidatos de CPU y utiliza la tabla de GPU registrada en el cuaderno 2 para la discusión.
#
# **Decide:** qué candidato para este trabajo de carga, y si el resultado debe regresar al CPU después de cada 40 pasos, ¿cambia la respuesta?

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
# ## Recomendación (edite esta celda)
#
# **Runtime:** *(paste the runtime description printed by the setup cell)*
#
# **Carga de Trabajo A (48 pequeños grids independientes):** Usaría ... porque en este
# runtime tomó ... versus ... . Los hilos dentro de un solo pequeño grid ayudaron / no ayudaron porque ... . Si tuviera un cluster, esta carga de trabajo se asignaría a ... (manejador de tareas / comunicación por mensajería) porque los grids intercambian ... bytes.
#
# **Carga de Trabajo B (un grid de 2048x2048):** Usaría ... porque ... . El tiempo de transferencia incluida fue ... del tiempo de cálculo solo, por lo que mover el resultado de vuelta cada 40 pasos cambiaría / no cambiaría la decisión.
#
# **Una cosa más que el hardware no arreglaría:** ...

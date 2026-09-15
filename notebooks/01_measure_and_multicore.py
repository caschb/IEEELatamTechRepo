# %% [markdown]
# # 1. Medir primero, luego pasar a múltiples hilos
#
# **Sesión en vivo, bloques 2 y 3 (00:10 a 01:10).** Tiempo de ejecución estándar CPU; no es necesario GPU.
#
# Tres preguntas para hacer antes de tocar ningún código:
#
# 1. **¿Dónde va el tiempo?** Medir. La intuición suele estar equivocada.
# 2. **¿Cuál es el techo?** Ley de Amdahl: si una fracción `s` del trabajo se mantiene en secuencial,
#    ningún número de hilos te permite superar un aceleramiento de `1/s`.
# 3. **¿Fue útil?** Medir de nuevo, en la misma máquina, de la misma manera.
#
# El ejemplo de ejecución es el stencil de stencil de calor-difusión en dos dimensiones
# del cuaderno de preparación. Cada paso reemplaza cada celda interior con el promedio de sus cuatro vecinos.
# Es pequeño, limitado en memoria, y sigue a los pasos en GPU y sesiones múltiples-particionadas.
#
# Este cuaderno es autónomo: ejecutar la celda de configuración después de cada reinicio de tiempo de ejecución.

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
# ## 1.1 El ejemplo de ejecución y su resultado de referencia
#
# `init_grid` construye un grid `n x n` con un borde caliente en la parte superior. `step_python` es el
# stencil escrito de la manera en que un científico lo escribe el primer día. Es el
# **referencia**: es más lento, pero permite comprobar cada operación. Todas las
# versiones optimizadas deben producir el mismo resultado.

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
# **Checkpoint 1.** La celda superior se ejecutó sin error de afirmación. Mantén `ref` y
# la `N_REF, ITERS_REF` caso: cada implementación a seguirá siendo comprobada contra él.
#
# ## 1.2 Medir de manera equitativa
#
# Una `time.perf_counter()` par da un número. Se lo ejecutas de nuevo y obtienes un
# número diferente: la primera llamada paga por las importaciones, los cachés y (más tarde)
# la compilación JIT; las llamadas posteriores comparten la máquina con lo que esté
# corriendo en paralelo. Dos hábitos arreglan la mayoría de eso:
#
# - **Calienta** una vez, luego tiempo varias **repeticiones** y mantén el mínimo (la
#   ejecución con menos interferencia) o el mediano (cuando te importa el tiempo típico).
# - Compárate implementaciones **dentro de una misma ejecución**, en el **mismo trabajo
#   de carga y tipo de dato**. Un número de otro ordenador es un experimento diferente.

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
# IPython's `%timeit` hace lo mismo. Lee atentamente su salida: reporta el **promedio y la desviación estándar** a lo largo de `-r` ejecuciones, y cada ejecución es el tiempo para `-n` vueltas dividido por `n`. Con `-o` obtienes el objeto y `.best` es la ejecución más rápida.

# %%
# t = %timeit -o -r 3 -n 1 run_python(128, 10)
print(f"best run {t.best*1e3:.0f} ms; %timeit's headline number is the mean of {t.repeat} runs")

# %% [markdown]
# **Predicción, ejecución, explicación.** `n=256` tiene cuatro veces más células que `n=128`.
#
# Predicción para la razón de tiempo: **4**.
#
# Ejecutando el siguiente celda...

# %%
t_128, _ = best_of(lambda: run_python(128, 10), repeat=3)
t_256, _ = best_of(lambda: run_python(256, 10), repeat=3)
print(f"n=128: {t_128*1e3:6.0f} ms   n=256: {t_256*1e3:6.0f} ms   ratio {t_256/t_128:.1f}x")

# %% [markdown]
# ### 1.3 ¿Dónde va el tiempo? (demonstración opcional)
#
# `line_profiler` muestra el tiempo por línea de una función. Aquí la respuesta es previsible, todo el tiempo está en el cuerpo del bucle interno, pero en código real la línea costosa a menudo es una sorpresa: una `print`, una búsqueda dentro de un bucle, una conversión. Para programas enteros se utiliza `python -m cProfile` o el profiler de muestreo `py-spy`. *Esta celda es la primera que se corta si el sesión termina tarde.*

# %%
try:
    ensure("line_profiler")
    get_ipython().run_line_magic("load_ext", "line_profiler")
    get_ipython().run_line_magic("lprun", "-f step_python run_python(64, 3)")
except Exception as e:
    print("profiler demonstration skipped:", repr(e))

# %% [markdown]
# ## 1.4 NumPy: deja que el código compilado haga el bucle
#
# El primer y mayor ganador suele no ser la paralelización. Es mover el bucle fuera
# del intérprete. Cada vista debajo (`u[:-2, 1:-1]` y amigos) es una **vista**:
# no se copia ningún dato. El cálculo entre ellos es diferente: cada `+` y el
# `0.25 *` asigna un tamaño completo de **array intermedio**, cuatro por paso aquí, y
# cada uno se escribe y se lee de vuelta a través de la memoria.

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
# Desde aquí, los cargos de trabajo crecen, porque NumPy termina antes de `n=128` antes de que el reloj pueda resolverse. Los tres lados del cuadrícula utilizados durante toda la sesión en vivo son 128, 512 y 1024, con 20 pasos cada uno.

# %%
SIZES, ITERS = (128, 512, 1024), 20
timings = []                       # (implementation, n, threads, min_s, median_s)
for n in SIZES:
    tmin, tmed = best_of(lambda: run(step_numpy, n, ITERS))
    timings.append(("numpy", n, 1, tmin, tmed))
    print(f"NumPy n={n:5d}, {ITERS} steps: min {tmin*1e3:8.2f} ms   median {tmed*1e3:8.2f} ms")

# %% [markdown]
# **Checkpoint 2.** Tienes una base de NumPy para tres tamaños. ¿Qué recurso
# pensas que limita a la base: las operaciones aritméticas, o la lectura y escritura
# de los intermediarios a través de la memoria? El apartado 1.5 prueba esa idea.
#
# ## 1.5 Numba: compila tu propio bucle
#
# Numba compila el bucle explícito a código máquina. Mantenemos el bucle legible y
# perdemos los intermediarios: cada celda de salida lee cuatro entradas y escribe una vez.
# La **primera llamada se compila** (segundo o dos minutos); nunca incluyasla en los tiempos de ejecución.

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
# La relación observada corresponde **a este runtime**. En una máquina con memoria
# rápida, la diferencia entre NumPy y Numba se reduce; en una máquina más lenta o
# compartida, aumenta. El mecanismo es el mismo: se realizan menos recorridos por memoria.
#
# ## 1.6 Hilos con `prange`
#
# `prange` distribuye el bucle exterior entre hilos. `os.cpu_count()` informa lo
# que muestra el sistema operativo, y Numba limita su conjunto de hilos mediante
# `numba.config.NUMBA_NUM_THREADS`. Un runtime estándar de Colab suele exponer dos
# CPU lógicas, por lo que conviene consultar el runtime en vez de asumir una cantidad.

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
# **Predicción, ejecución, explicación.** Con dos hilos, la ejecución del tiempo disminuirá a la mitad para `n=1024`?
# Escribe tu predicción antes de ejecutar.

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
# Si este runtime tiene solo un núcleo usable, la tabla tiene una fila y no puede mostrar
# escalado. El siguiente celda carga un **grabado** de recorrido desde el repositorio del curso
# para que la actividad de interpretación aún funcione. El grabado está etiquetado con el hardware
# de donde provino; es evidencia sobre *ese* máquina, no sobre esta.

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
# **Leer el plot.** Las curvas de Amdahl son *modelos*: elige la más cercana a los puntos
# medidos y tienes una estimación del "fracaso secuencial". Para este stencil no queda
# Python secuencial en el bucle, así que una mala ajuste no es Amdahl en absoluto. Los culpables
# típicos son:
#
# - **Velocidad de banda de memoria.** El stencil hace cuatro lecturas y una escritura por celda y casi
#   ninguna aritmética. Una vez que unas pocas hilos saturan la bus de memoria, más hilos solo esperan.
# - **Cores compartidos o virtuales.** Dos CPUs lógicos en un host compartido pueden ser un solo
#   núcleo físico. Entonces dos hilos compiten por un solo unidad y pueden ser más lentos que uno.
# - **Iniciación de hilos.** Para grids pequeños, despertar la piscina cuesta más que el trabajo.
#
# **Límite de conocimiento del tiempo de ejecución.** Nunca pida más hilos que el tiempo de ejecución te da:
# `numba.set_num_threads(min(wanted, os.cpu_count()))`. La sobrecarga no es un error, solo silencio y lentitud.
#
# ## Punto de control 3: tu tabla de tiempos
#
# Guarda la tabla con la descripción del tiempo de ejecución. La compararás con la tabla del GPU
# en el siguiente cuaderno, y reutilizarás las funciones en el epílogo.

# %%
save_timings("timings_01_cpu.csv", ["impl", "n", "threads", "min_s", "median_s"], timings)
print(f"{'impl':14s} {'n':>6s} {'thr':>4s} {'min ms':>10s}")
for impl, n, thr, tmin, _ in timings:
    print(f"{impl:14s} {n:6d} {thr:4d} {tmin*1e3:10.2f}")

# %% [markdown]
# ## Síntesis
#
# | Situación | Relevancia |
# |---|---|
# | Iterar sobre elementos de un array | Vectorización con NumPy primero, luego `@njit` con Numba |
# | Mismo bucle, varios núcleos | `parallel=True` y `prange` con Numba, threads <= núcleos |
# | Tareas independientes independientes en muchos | Procesos (`concurrent.futures`, `joblib`), o Dask; ve el notebook de extensión |
# | Cualquier afirmación de rendimiento | Medir antes y después, mismo tiempo de ejecución, misma carga de trabajo, mismo tipo de datos |
#
# **Extensión opcional (no cubierto en vivo):** `ext_gil_and_task_pools`
# muestra por qué las hilos de Python no aceleran bucles puramente Python y compara
# `ProcessPoolExecutor` y `joblib` en una evaluación de tareas independientes.

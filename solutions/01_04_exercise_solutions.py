# %% [markdown]
# # Soluciones de los ejercicios, notebooks 01 a 04
#
# Este notebook reúne, en orden, las respuestas a los ejercicios, las predicciones y las
# preguntas de explicación de `01_measure_and_multicore`, `02_gpu`,
# `03_parallel_models` y `04_capstone`. La solución del notebook 00 está en
# `00_stencil_practice_solution` y el razonamiento esperado del proyecto final, con
# tiempos de referencia, en `capstone_solution.md`.
#
# Cada respuesta tiene dos partes: el **razonamiento**, que no depende de la máquina, y
# una **celda que lo comprueba** en el entorno de ejecución actual. Los números
# cambiarán de una máquina a otra; la forma del argumento no debería cambiar.
#
# Funciona en un entorno de ejecución de CPU. Si hay GPU, las celdas de la sección 2
# miden en vivo; si no, usan la tabla registrada e indican de qué hardware proviene.
# Este notebook es autónomo: ejecute la celda de configuración después de cada reinicio.

# %%
# --- Setup: rerun after every runtime restart ------------------------------
import csv, importlib, importlib.util, os, platform, shutil, subprocess, sys, time, urllib.request

def ensure(module, package=None):
    if importlib.util.find_spec(module) is None:
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", package or module], check=True)
    return importlib.import_module(module)

np = ensure("numpy"); numba = ensure("numba"); psutil = ensure("psutil")
ensure("matplotlib"); import matplotlib.pyplot as plt
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

def best_of(fn, repeat=5, sync=lambda: None):
    """Warm up once, then (min, median) seconds over `repeat` calls, synchronising around each."""
    fn(); sync(); ts = []
    for _ in range(repeat):
        sync(); t0 = time.perf_counter(); fn(); sync(); ts.append(time.perf_counter() - t0)
    return min(ts), float(np.median(ts))

def load_reference(name):
    """Recorded timing table from the course repository, with its runtime metadata."""
    local = os.path.join("..", "data", "reference_timings", name)
    url = f"https://raw.githubusercontent.com/caschb/IEEELatamTechRepo/main/data/reference_timings/{name}"
    try:
        text = open(local).read() if os.path.exists(local) else urllib.request.urlopen(url, timeout=10).read().decode()
    except Exception as e:
        print("no recorded data available:", repr(e)); return [], {}
    meta = dict(l[2:].split(": ", 1) for l in text.splitlines() if l.startswith("# "))
    return list(csv.DictReader(l for l in text.splitlines() if not l.startswith("#"))), meta

MAX_THREADS = min(os.cpu_count() or 1, numba.config.NUMBA_NUM_THREADS)
numba.set_num_threads(MAX_THREADS)
sync = cp.cuda.Device().synchronize if HAVE_GPU else (lambda: None)
print(runtime_info())

# %% [markdown]
# ## Funciones comunes (las mismas de los notebooks 1 a 4)

# %%
def init_grid(n, xp=np, dtype=np.float64, top=100.0):
    u = xp.zeros((n, n), dtype=dtype); u[0, :] = top
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
        step_python(u, unew); u, unew = unew, u
    return np.array(u)

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

ref = run_python(64, 10)
for step in (step_numpy, step_numba, step_numba_par):     # also compiles the Numba versions
    assert np.allclose(run(step, 64, 10), ref)
print("all implementations match the pure-Python reference")

# %% [markdown]
# # 1. Notebook 01: medir primero, luego pasar a múltiples hilos
#
# ## 1.1 ¿Por qué una sola llamada a `time.perf_counter()` no es una medición?
#
# Porque una sola ejecución mezcla el trabajo con efectos que no son el trabajo: la primera
# llamada paga importaciones, cachés fríos y, con Numba, la compilación; las siguientes
# comparten la máquina con otros procesos; y una duración muy corta queda cerca de la
# resolución del reloj. Por eso se calienta una vez, se repite y se informa el mínimo o la
# mediana. En `%timeit`, el número principal es el **promedio +- desviación estándar**
# de las `-r` ejecuciones, no el mejor tiempo; el mejor está en `.best` si se usa `-o`.
#
# ## 1.2 Predicción: razón de tiempos entre `n=256` y `n=128`
#
# **Respuesta: cerca de 4.** El trabajo por paso es proporcional al número de celdas
# interiores, `(n-2)^2`, y en Python puro cada celda cuesta lo mismo sin importar el
# tamaño. La razón exacta de celdas es `254^2 / 126^2 = 4.06`. Una desviación moderada
# (3.6 a 4.5) se debe a ruido y a cachés, no a otro orden de complejidad.

# %%
t_128, _ = best_of(lambda: run_python(128, 5), repeat=3)
t_256, _ = best_of(lambda: run_python(256, 5), repeat=3)
print(f"n=128: {t_128*1e3:6.0f} ms   n=256: {t_256*1e3:6.0f} ms   "
      f"measured ratio {t_256/t_128:.2f}x   predicted {(254/126)**2:.2f}x")

# %% [markdown]
# ## 1.3 ¿Cuántos arrays asigna la línea de NumPy y qué limita a la base?
#
# La línea `0.25 * (a + b + c + d)` asigna **cuatro** arrays intermedios del tamaño del
# interior: tres por las sumas y uno por el producto. Los slices son vistas y no asignan
# nada. La asignación final a `unew[1:-1, 1:-1]` copia el último intermedio.
#
# **Punto de control 2: el límite es el tráfico de memoria, no la aritmética.** Por celda
# se hacen 4 operaciones de punto flotante (3 sumas y 1 producto), pero se leen y
# escriben varios arrays completos: cada operación recorre la memoria de nuevo. La
# intensidad aritmética es muy baja, aun en el caso ideal de leer cada vecino una sola
# vez y escribir una vez (4 flops por 40 bytes, 0.1 flop/byte).
#
# La celda siguiente lo comprueba con tres versiones que hacen **exactamente la misma
# aritmética**:
#
# - `step_numpy`: la original, con cuatro intermedios nuevos por paso.
# - `step_numpy_out`: la misma secuencia con `out=`, reutilizando un único buffer. No
#   asigna, pero sigue haciendo cuatro recorridos por memoria.
# - `step_numba`: un solo recorrido; cada celda lee cuatro valores y escribe uno.
#
# Si el límite fuera la aritmética, las tres tardarían lo mismo. Lo esperado es que
# `step_numpy_out` quede muy cerca de `step_numpy` (evitar la asignación ayuda poco) y
# que `step_numba` sea varias veces más rápida: lo que cuesta es el número de
# recorridos por memoria, no la aritmética ni la asignación.

# %%
def make_step_numpy_out(n):
    tmp = np.empty((n - 2, n - 2))
    def step_numpy_out(u, unew):
        np.add(u[:-2, 1:-1], u[2:, 1:-1], out=tmp)
        np.add(tmp, u[1:-1, :-2], out=tmp)
        np.add(tmp, u[1:-1, 2:], out=tmp)
        np.multiply(tmp, 0.25, out=unew[1:-1, 1:-1])
        return unew
    return step_numpy_out

assert np.allclose(run(make_step_numpy_out(64), 64, 10), ref)
N_MEM, ITERS = 1024, 20
for name, step in (("numpy (4 new temporaries)", step_numpy),
                   ("numpy with out= (1 reused buffer)", make_step_numpy_out(N_MEM)),
                   ("numba (single pass)", step_numba)):
    tmin, _ = best_of(lambda: run(step, N_MEM, ITERS))
    print(f"{name:36s} n={N_MEM}, {ITERS} steps: {tmin*1e3:8.2f} ms")

# %% [markdown]
# Numba no gana por ser paralelo (`step_numba` usa un solo hilo) sino porque recorre la
# memoria una vez en lugar de varias. Este es el error común que conviene corregir en la
# sesión: "Numba es más rápido porque es paralelo".
#
# ## 1.4 Predicción: ¿dos hilos reducen el tiempo a la mitad para `n=1024`?
#
# **Respuesta: casi nunca exactamente.** En un Colab estándar la aceleración con dos
# hilos suele quedar entre 1.0x y 1.6x, y a veces es menor que 1. Tres causas:
#
# 1. **Ancho de banda de memoria.** El cálculo por vecindad está limitado por memoria
#    (apartado 1.3). Si un hilo ya consume buena parte del ancho de banda, un segundo hilo
#    espera al mismo bus.
# 2. **Núcleos virtuales.** Las dos CPU lógicas de Colab pueden ser dos hilos de
#    hardware de un mismo núcleo físico, que comparten unidades de cálculo y caché.
# 3. **Costo de iniciar y sincronizar hilos** en cada paso. Con `n=1024` es pequeño,
#    pero con cuadrículas pequeñas domina (véase el proyecto final, carga A).
#
# ## 1.5 Ejercicio de Amdahl: ¿qué fracción secuencial implica la medición?
#
# De la ley de Amdahl, `S(p) = 1 / (s + (1 - s)/p)`, se despeja la fracción secuencial
# que explicaría una aceleración medida `S` con `p` hilos (la métrica de Karp-Flatt):
#
# `s = (1/S - 1/p) / (1 - 1/p)`
#
# Si `s` crece al aumentar `p`, el modelo de Amdahl no describe bien lo que ocurre: no hay
# Python secuencial dentro del bucle, así que la "fracción secuencial" aparente es en
# realidad saturación de memoria y costo de sincronización. Esa es la lectura correcta
# del gráfico del notebook 01.

# %%
def serial_fraction(speedup, p):
    return (1 / speedup - 1 / p) / (1 - 1 / p)

N_SCALE = 1024
threads = sorted({1, 2, MAX_THREADS // 2, MAX_THREADS} & set(range(1, MAX_THREADS + 1)))
scaling = []
for t in threads:
    numba.set_num_threads(t)
    run(step_numba_par, N_SCALE, 2)
    scaling.append((t, best_of(lambda: run(step_numba_par, N_SCALE, ITERS))[0]))
numba.set_num_threads(MAX_THREADS)
source = "this runtime"

if len(scaling) < 2:
    rows, meta = load_reference("01_threads.csv")
    scaling = [(int(r["threads"]), float(r["min_s"])) for r in rows if r["impl"] == "numba_prange"]
    source = f"RECORDED on {meta.get('runtime')} ({meta.get('cpus')} cpus), not this runtime"

print(f"n={N_SCALE}, {ITERS} steps, {source}")
t1 = scaling[0][1]
for p, tp in scaling:
    S = t1 / tp
    s_txt = f"implied serial fraction {serial_fraction(S, p):6.1%}" if p > 1 else ""
    print(f"{p:3d} thread(s): {tp*1e3:8.2f} ms   speedup {S:4.2f}x   {s_txt}")

# %% [markdown]
# Con la tabla registrada (32 CPU lógicas), la aceleración sube hasta 16 hilos y baja con
# 32: la fracción secuencial aparente pasa de cerca de 30 % con 2 hilos a más de 50 % con 32. Un
# programa que obedeciera Amdahl mantendría `s` constante. La meseta temprana es la firma
# de un kernel limitado por memoria; los hilos extra solo esperan.
#
# **Límite del entorno.** `numba.set_num_threads(min(wanted, os.cpu_count()))`. Pedir más
# hilos que CPU no produce un error: solo hace el programa más lento sin avisar.
#
# # 2. Notebook 02: computación con GPU
#
# ## 2.1 Sincronización: ¿qué midió la versión sin `synchronize()`?
#
# Solo el tiempo que tarda la CPU en **encolar** los kernels. Las llamadas de CuPy son
# asíncronas: `step()` retorna antes de que la GPU termine. Por eso el tiempo sin
# sincronizar puede ser menor que una sola copia de memoria. La medición correcta
# sincroniza antes de iniciar el cronómetro y antes de detenerlo, como `best_of` arriba.
#
# ## 2.2 Predicción: ¿a partir de qué tamaño gana la GPU?
#
# **Respuesta: depende del par CPU/GPU, pero la forma es fija.** En cuadrículas pequeñas
# el tiempo de la GPU casi no cambia con `n`: está dominado por el costo fijo de lanzar
# cada kernel (microsegundos por lanzamiento, cinco lanzamientos por paso) y la GPU pasa
# la mayor parte del tiempo inactiva. El tiempo de NumPy crece con `n^2`. Las curvas se
# cruzan donde el trabajo por paso supera el costo de lanzamiento. En la tabla
# registrada (RTX 4090 frente a 32 CPU) el cruce ocurre cerca de `n=128`; en una T4 de
# Colab frente a 2 CPU suele estar entre 128 y 512. No se memoriza el número, se
# memoriza la forma.

# %%
SIZES = (128, 512, 1024, 2048)
if HAVE_GPU:
    gpu_rows = []
    for n in SIZES:
        tc = best_of(lambda: run(step_numpy, n, ITERS), repeat=3)[0]
        tg = best_of(lambda: run(step_numpy, n, ITERS, cp), repeat=3, sync=sync)[0]
        gpu_rows.append((n, tc, tg))
    source = f"this runtime ({runtime_info()['gpu']})"
else:
    rows, meta = load_reference("02_gpu.csv")
    pick = lambda impl, n: next(float(r["min_s"]) for r in rows
                                if r["impl"] == impl and int(r["n"]) == n and r["dtype"] == "float64" and r["scope"] == "compute")
    gpu_rows = [(n, pick("numpy", n), pick("cupy", n)) for n in SIZES] if rows else []
    source = f"RECORDED on {meta.get('gpu')}, not this runtime"

print(f"float64, {ITERS} steps, {source}")
for n, tc, tg in gpu_rows:
    print(f"n={n:5d}  cpu {tc*1e3:8.2f} ms   gpu {tg*1e3:8.2f} ms   cpu/gpu {tc/tg:6.1f}x   "
          f"{'GPU wins' if tg < tc else 'CPU wins'}")
if len(gpu_rows) >= 2:
    g = np.array([r[2] for r in gpu_rows])
    print(f"GPU time grows {g[1]/g[0]:.1f}x from n=128 to n=512 while the work grows {(510/126)**2:.1f}x: "
          "launch overhead dominates small grids")

# %% [markdown]
# ## 2.3 Predicción: ¿una transferencia de 32 MB cuesta más o menos que un paso?
#
# **Respuesta: bastante más; una subida cuesta el equivalente a diez pasos o más.**
# Estimación a mano para `n=2048` en float64:
#
# - Un paso en el dispositivo lee y escribe del orden de `5 x 32 MB = 160 MB` de la
#   memoria de la GPU (T4: unos 300 GB/s). Resultado: alrededor de 0.5 ms.
# - Una subida mueve 32 MB por PCIe desde memoria paginable del host (en la práctica
#   entre 3 y 12 GB/s). Resultado: entre 3 y 10 ms.
#
# Por lo tanto, **la regla** es subir los datos una vez, hacer todo el trabajo en la GPU
# y bajar el resultado una vez. Un `cp.asarray()` dentro del bucle suele ser más lento
# que NumPy. Al citar una aceleración de GPU, se indica si incluye transferencias.

# %%
n = 2048
a = init_grid(n)
if HAVE_GPU:
    ug, ung = cp.asarray(a), cp.asarray(a)
    t_up = best_of(lambda: cp.asarray(a), sync=sync)[0]
    t_step = best_of(lambda: step_numpy(ug, ung), sync=sync)[0]
    print(f"this runtime: upload {t_up*1e3:.2f} ms ({a.nbytes/2**30/t_up:.1f} GB/s), "
          f"one step on device {t_step*1e3:.3f} ms -> one upload = {t_up/t_step:.0f} steps")
    del ug, ung
else:
    rows, meta = load_reference("02_gpu.csv")
    get = lambda scope: min(float(r["min_s"]) for r in rows
                            if r["impl"] == "cupy" and int(r["n"]) == n and r["scope"] == scope)
    if rows:
        t_comp, t_incl = get("compute"), get("transfer_inclusive")
        t_step, t_xfer = t_comp / 20, (t_incl - t_comp) / 2        # recorded with 20 steps; one upload + one download
        print(f"RECORDED on {meta.get('gpu')}: {20} steps {t_comp*1e3:.2f} ms compute-only, "
              f"{t_incl*1e3:.2f} ms transfer-inclusive")
        print(f"one step ~{t_step*1e3:.3f} ms, one transfer ~{t_xfer*1e3:.2f} ms -> one transfer = {t_xfer/t_step:.0f} steps")

# %% [markdown]
# ## 2.4 float32: ¿cuánto se gana y por qué?
#
# **Respuesta: cerca de 2x en un kernel limitado por memoria**, porque se mueven la
# mitad de los bytes por celda. La razón f64/f32 es una propiedad del recurso que limita
# la ejecución, no de "la GPU". En tarjetas de consumo como la T4 la aritmética en
# float64 es mucho más lenta que en float32, pero este cálculo casi no hace aritmética,
# así que lo que se nota es el ancho de banda. El resultado se compara con una
# tolerancia adecuada para float32 (`rtol=1e-5`), no con la de float64.

# %%
u64 = run(step_numpy, 1024, ITERS)
u32 = run(step_numpy, 1024, ITERS, dtype=np.float32)
print("max abs difference f32 vs f64:", float(np.abs(u32 - u64).max()),
      "-> allclose(rtol=1e-5):", np.allclose(u32, u64, rtol=1e-5, atol=1e-3),
      "  allclose(rtol=1e-10):", np.allclose(u32, u64, rtol=1e-10, atol=1e-8))
t64 = best_of(lambda: run(step_numpy, 1024, ITERS))[0]
t32 = best_of(lambda: run(step_numpy, 1024, ITERS, dtype=np.float32))[0]
print(f"CPU n=1024: f64 {t64*1e3:.2f} ms   f32 {t32*1e3:.2f} ms   f64/f32 {t64/t32:.2f}x")

# %% [markdown]
# ## 2.5 Tres preguntas antes de creer "la GPU es 40 veces más rápida" (kernel de 2 ms)
#
# 1. **¿Se sincronizó antes de detener el cronómetro?** Si no, se midió el encolado.
# 2. **¿Están las transferencias dentro del tiempo?** Con un kernel de 2 ms, una subida
#    de unos pocos milisegundos cambia por completo la conclusión.
# 3. **¿Es el mismo trabajo, con el mismo tipo de dato, en ambos lados?** float32 en la
#    GPU contra float64 en la CPU, o NumPy de un hilo contra una GPU completa, no es una
#    comparación entre iguales. También conviene preguntar si hubo calentamiento
#    (compilación y asignación fuera del reloj).
#
# # 3. Notebook 03: particiones, halos y comunicación
#
# ## 3.1 Ejercicio: filas y bytes que cruzan fronteras en un paso
#
# **Respuesta.** Con `P` estratos hay `P - 1` fronteras internas. Por cada una viajan
# **dos filas** por paso, una en cada dirección, cada una de `n` valores float64:
#
# - total por paso: `2 (P - 1)` filas, es decir `2 (P - 1) n 8` bytes;
# - un trabajador del medio envía `2n` valores (`16 n` bytes) y los de los extremos, `n`.
#
# Los bordes globales superior e inferior no se comunican: son condiciones de frontera,
# no vecinos. La celda compara la fórmula con lo que cuenta la implementación del
# notebook 03 y confirma que el resultado reconstruido es idéntico, bit a bit, al serial.

# %%
def partition(u, P):
    n = u.shape[0]
    bounds = [(rows[0], rows[-1] + 1) for rows in np.array_split(np.arange(1, n - 1), P)]
    return [u[lo - 1:hi + 1].copy() for lo, hi in bounds]

def exchange_halos(strips):
    nbytes = 0
    for r in range(len(strips)):
        if r > 0:
            strips[r][0] = strips[r-1][-2]; nbytes += strips[r][0].nbytes
        if r < len(strips) - 1:
            strips[r][-1] = strips[r+1][1]; nbytes += strips[r][-1].nbytes
    return nbytes

def run_partitioned(n, iters, P):
    strips = partition(init_grid(n), P); news = [s.copy() for s in strips]
    for _ in range(iters):
        nbytes = exchange_halos(strips)
        for s, snew in zip(strips, news): step_numpy(s, snew)
        strips, news = news, strips
    return np.vstack([strips[0][:1]] + [s[1:-1] for s in strips] + [strips[-1][-1:]]), nbytes

N3, ITERS3 = 256, 30
ref3 = run(step_numpy, N3, ITERS3)
for P in (2, 4, 8, 16):
    u_p, measured = run_partitioned(N3, ITERS3, P)
    predicted = 2 * (P - 1) * N3 * 8
    assert np.array_equal(u_p, ref3) and measured == predicted
    print(f"P={P:2d}: predicted {predicted:6d} B   measured {measured:6d} B   result identical to serial")

# %% [markdown]
# ## 3.2 Predicción: ¿el trabajador calcula o espera?
#
# Modelo del enunciado: 10 microsegundos de latencia por mensaje, 10 GB/s de ancho de
# banda y 1 ns por actualización de celda. Un trabajador del medio calcula `n * n / P`
# celdas y envía dos mensajes de `n` valores float64 por paso.
#
# - **`n=256`, `P=128`:** calcula 512 celdas (unos 0.5 microsegundos) y espera por dos
#   mensajes (unos 20 microsegundos). Pasa más del 95 % del tiempo **esperando**.
# - **`n=16384`, `P=128`:** calcula unos 2.1 millones de celdas (unos 2.1 ms) y los
#   mensajes suman unos 46 microsegundos. Pasa casi todo el tiempo **calculando**.
#
# Por eso los problemas pequeños no escalan entre máquinas, sin importar la biblioteca:
# la latencia es un costo fijo por mensaje que no disminuye al dividir el trabajo.

# %%
LATENCY, BANDWIDTH, T_CELL = 10e-6, 10e9, 1e-9

def worker_step_model(n, P):
    compute = n * n / P * T_CELL
    comm = 2 * (LATENCY + n * 8 / BANDWIDTH)
    return compute, comm

for n in (256, 1024, 4096, 16384):
    compute, comm = worker_step_model(n, 128)
    print(f"n={n:6d}, P=128: compute {compute*1e6:9.1f} us   communication {comm*1e6:6.1f} us   "
          f"waiting {comm/(compute+comm):6.1%}")

# %% [markdown]
# ## 3.3 Punto de control: ¿qué modelo corresponde a cada carga del proyecto final?
#
# - **Carga A (48 cuadrículas independientes):** no se mueve ningún dato entre
#   cuadrículas, así que corresponde a **tareas independientes**: un planificador de
#   tareas (Dask, un array de trabajos de Slurm), sin mensajes.
# - **Carga B (una cuadrícula grande):** cada paso necesita las filas de frontera de los
#   vecinos, así que corresponde a una **simulación acoplada con intercambio de halos**
#   (MPI), con dos filas por frontera interna por paso.
#
# # 4. Notebook 04: proyecto final
#
# El razonamiento completo y los tiempos de la máquina del autor están en
# `capstone_solution.md`. Aquí se ejecuta la solución, incluido el candidato que
# ese documento recomienda añadir para la carga A.
#
# ## 4.1 Carga A: 48 cuadrículas de lado 128, 30 pasos
#
# **Elección:** `numba` serial por cuadrícula, o mejor aún, **paralelizar sobre las
# cuadrículas**. `numba_par` pone los hilos *dentro* de una cuadrícula de 128x128, donde
# cada paso tiene muy poco trabajo: despertar y sincronizar el grupo de hilos 30 veces por
# cuadrícula cuesta más que el cálculo. Como las cuadrículas no comparten nada, el
# paralelismo correcto va *alrededor* de ellas: un `prange` sobre las 48 temperaturas,
# con el bucle serial dentro. Es la misma idea que un planificador de tareas en un
# cluster, aplicada dentro de un proceso.

# %%
TEMPS = np.linspace(50, 150, 48)
N_A, ITERS_A = 128, 30

def sweep(step, xp=np):
    return np.array([to_numpy(run(step, N_A, ITERS_A, xp, top=t))[1:-1, 1:-1].mean() for t in TEMPS])

@njit
def _run_one(n, iters, top):
    u = np.zeros((n, n)); u[0, :] = top; unew = u.copy()
    for _ in range(iters):
        step_numba(u, unew); u, unew = unew, u
    return u[1:-1, 1:-1].mean()

@njit(parallel=True)
def sweep_par_over_grids(temps, n, iters):
    out = np.empty(temps.size)
    for k in prange(temps.size):           # one whole, independent simulation per iteration
        out[k] = _run_one(n, iters, temps[k])
    return out

CANDIDATES_A = {
    "numpy": lambda: sweep(step_numpy),
    "numba": lambda: sweep(step_numba),
    "numba_par (threads inside a grid)": lambda: sweep(step_numba_par),
    "numba prange over grids": lambda: sweep_par_over_grids(TEMPS, N_A, ITERS_A),
}
ref_A = sweep(step_numpy)
timings_A = {}
for name, fn in CANDIDATES_A.items():
    assert np.allclose(fn(), ref_A), f"{name} disagrees with the reference"
    timings_A[name] = best_of(fn, repeat=3)[0]
    print(f"workload A  {name:34s} {timings_A[name]*1e3:9.1f} ms  (correct)")

# %% [markdown]
# ## 4.2 Carga B: una cuadrícula de lado 2048, 40 pasos
#
# **Elección:** aquí sí conviene poner los hilos dentro de la cuadrícula (`numba_par`) o
# usar la GPU (`cupy`): cada paso tiene millones de celdas, suficiente trabajo para
# amortizar el arranque de hilos y el lanzamiento de kernels. Si el resultado debe
# volver a la CPU cada 40 pasos, se mide la versión que incluye una subida y una bajada.
# Con los números registrados (unos 10 ms solo cálculo contra unos 17 ms con
# transferencias) la GPU sigue ganando; si hubiera que bajar el resultado **en cada
# paso**, cada paso pagaría una transferencia de varios pasos de cálculo y la ventaja
# desaparecería. Sin GPU, la recomendación honesta es `numba_par`, citando la tabla
# registrada como evidencia sobre *otro* hardware.

# %%
N_B, ITERS_B = 2048, 40
ref_B = run(step_numpy, N_B, ITERS_B)
timings_B = {}
for name, step, xp in [("numpy", step_numpy, np), ("numba_par", step_numba_par, np)] + \
                      ([("cupy compute", step_numpy, cp)] if HAVE_GPU else []):
    assert np.allclose(to_numpy(run(step, N_B, ITERS_B, xp)), ref_B), f"{name} disagrees with the reference"
    timings_B[name] = best_of(lambda: run(step, N_B, ITERS_B, xp), repeat=3, sync=sync if xp is not np else (lambda: None))[0]
    print(f"workload B  {name:26s} {timings_B[name]*1e3:9.1f} ms  (correct)")

if HAVE_GPU:
    host = init_grid(N_B)
    def transfer_inclusive():
        u = cp.asarray(host); unew = u.copy()
        for _ in range(ITERS_B): step_numpy(u, unew); u, unew = unew, u
        return u.get()
    assert np.allclose(transfer_inclusive(), ref_B)
    timings_B["cupy transfer-inclusive"] = best_of(transfer_inclusive, repeat=3, sync=sync)[0]
    print(f"workload B  {'cupy transfer-inclusive':26s} {timings_B['cupy transfer-inclusive']*1e3:9.1f} ms  (correct)")

# %% [markdown]
# ## 4.3 Recomendación redactada a partir de los números
#
# La celda siguiente completa la plantilla de recomendación del notebook 04 con los
# tiempos medidos arriba. En una respuesta de estudiante, lo importante es que cada frase
# cite un número de su propia tabla y el entorno donde se obtuvo.

# %%
info = runtime_info()
best_A = min(timings_A, key=timings_A.get)
inside, outside = timings_A["numba_par (threads inside a grid)"], timings_A["numba"]
best_B = min((k for k in timings_B if k != "cupy transfer-inclusive"), key=timings_B.get)

print(f"Entorno de ejecución: {info}\n")
print(f"Carga A (48 cuadrículas pequeñas e independientes): usaría '{best_A}' porque en este entorno tardó "
      f"{timings_A[best_A]*1e3:.1f} ms frente a {timings_A['numpy']*1e3:.1f} ms con NumPy. Los hilos dentro de una "
      f"sola cuadrícula pequeña {'no ayudaron' if inside >= outside else 'ayudaron'} ({inside*1e3:.1f} ms frente a "
      f"{outside*1e3:.1f} ms con Numba serial) porque cada paso tiene muy poco trabajo para pagar el arranque de los "
      f"hilos. En un cluster, esta carga se asignaría a un planificador de tareas, porque las cuadrículas "
      f"intercambian 0 bytes.\n")

line_B = (f"Carga B (una cuadrícula de 2048x2048): usaría '{best_B}' porque tardó {timings_B[best_B]*1e3:.1f} ms "
          f"frente a {timings_B['numpy']*1e3:.1f} ms con NumPy.")
if HAVE_GPU:
    ratio = timings_B["cupy transfer-inclusive"] / timings_B["cupy compute"]
    gpu_still_wins = timings_B["cupy transfer-inclusive"] < timings_B["numba_par"]
    line_B += (f" El tiempo con transferencias fue {ratio:.1f} veces el de solo cálculo, así que devolver el "
               f"resultado cada 40 pasos {'no cambia' if gpu_still_wins else 'sí cambia'} la decisión; "
               f"devolverlo en cada paso sí la cambiaría.")
else:
    line_B += (" Este entorno no tiene GPU; la tabla de GPU registrada es evidencia sobre otro hardware, "
               "no sobre este.")
print(line_B + " En un cluster corresponde a intercambio de mensajes, con dos filas de halo por frontera "
      "interna en cada paso.\n")
print("Algo que más hardware no arreglaría: cuadrículas pequeñas donde arrancar hilos o lanzar kernels cuesta más "
      "que el trabajo, un kernel limitado por memoria que ya satura el ancho de banda, o copiar a la GPU dentro "
      "del bucle.")

# %% [markdown]
# ## Pregunta de cierre: ¿cuándo no ayuda disponer de más hardware?
#
# - Cuando la carga es pequeña y el costo de iniciar hilos o lanzar kernels supera el
#   trabajo (carga A con `numba_par`, cuadrículas pequeñas en la GPU).
# - Cuando hay transferencias entre CPU y GPU dentro del bucle.
# - Cuando el kernel está limitado por memoria y ya satura el ancho de banda: más
#   núcleos esperan al mismo bus (meseta del apartado 1.5).
# - Cuando queda una fracción secuencial real (ley de Amdahl).
# - Cuando la comunicación entre máquinas domina el cálculo por trabajador
#   (apartado 3.2).

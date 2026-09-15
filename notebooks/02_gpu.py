# %% [markdown]
# # 2. Computación con GPU desde Python
#
# **Sesión en vivo, bloques 5 y 6 (01:20 a 02:10).** Cambie a un entorno de ejecución con GPU primero: *Entorno de ejecución > Tipo de entorno de ejecución > T4 GPU* (o lo que ofrezca Colab con GPU),
# luego ejecute la celda de configuración. Si no hay GPU disponible, **quédese en el entorno de ejecución con CPU y continúe**: el cuaderno detecta que no hay GPU, omite las celdas de dispositivo y muestra una tabla de GPU grabada para que pueda hacer las mismas comparaciones.
#
# Una GPU contiene miles de núcleos que comparten memoria de alta velocidad. Es adecuada para millones de operaciones idénticas e independientes, pero pierde eficiencia cuando hay comunicaciones frecuentes o transferencias reiteradas de datos.
#
# - **CuPy**: API de NumPy, los arreglos viven en el GPU. No hay nuevos conceptos. (Este cuaderno.)
# - **numba.cuda**: permite escribir el kernel directamente y requiere definir la cuadrícula de hilos.
#   (Extensión opcional `extensions/ext_cuda_kernel`.)
#
# El cálculo por vecindad es el mismo que en el cuaderno 1. Este cuaderno es autónomo.

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
# ## 2.1 CuPy: el sustituto sin cambios
#
# Escriba la función una vez contra un *módulo de arrays* `xp`, luego házselo pasar a arrays de NumPy o
# a arrays de CuPy. Este es el patrón estándar (el Array API) y es cómo SciPy, scikit-learn y xarray obtienen soporte para GPU sin tener que escribir código nuevo.

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
# **Punto de control 1.** El mismo función `step` se ejecutó en un tipo diferente de memoria y produjo los mismos números. Nada del algoritmo cambió.
#
# ## 2.2 Tiempo correcto del código GPU
#
# Las llamadas a la GPU son asincrónicas: `step()` retorna antes de que termine el
# cálculo. Si no se llama a `synchronize()` antes de detener el cronómetro, se mide
# el inicio de la operación, no el trabajo completo. Sincronice antes de iniciar y
# antes de detener la medición.

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
# La figura "sin sincronización" puede incluso ser *menor* que el tiempo de una sola
# copia de memoria: el CPU ordenó el trabajo y se fue. Cada timing del GPU en el resto
# de esta notebook pasa por `best_of(..., sync=sync)`.
#
# ## 2.3 ¿Cuándo vale la pena el GPU? Revisión de tamaño por igual
#
# El mismo trabajo, el mismo tipo de dato, el mismo número de pasos, medidos de la misma
# forma, en CPU y en GPU. **Predice** antes de ejecutar: en qué lado de la rejilla, si es que
# algún, el GPU superará al CPU en este tiempo de ejecución?

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
# **Explain what you see.**
#
# 1. ¿Se mueve mucho el tiempo del GPU con `n` para las cuadrículas más pequeñas? Cada
#    paso lanza una cantidad limitada de hilos, y un lanzamiento cuesta microsegundos
#    independientemente del trabajo. A partir de cierto tamaño, el GPU está
#    principalmente inactivo y puede perderle a NumPy.
# 2. ¿Dónde, si acaso, se cruzan las dos curvas? La cruzada depende del CPU, del
#    GPU y de lo que comparten con ellos. No memoriza un número; memoriza la forma.
#
# ## 2.4 El impuesto a la transferencia: solo computación versus incluyente de transferencia
#
# La memoria de la GPU transfiere cientos de GB/s; la conexión PCIe entre el equipo y el
# dispositivo transfiere decenas. Un trabajo que copia un array a la GPU, hace un
# poco de trabajo y lo copia de vuelta puede pasar la mayor parte de su tiempo en las
# copias. Informe ambos tiempos por separado e indique a cuál se refiere.
#
# **Predicción:** Para `n=2048` (32 MB en float64), ¿es una transferencia más barata o más
# costosa que un paso de cálculo por vecindad en el dispositivo?

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
# **Regla:** Mueva los datos a la GPU una vez, realice allí todo el trabajo y devuelva
# los resultados una sola vez. El uso de `cp.asarray()` dentro de un bucle suele ser más lento que NumPy.
#
# Cuando hablemos de un aceleramiento del GPU, mencionaremos si las transferencias están dentro del reloj.
#
# ## 2.5 Precisión: float32 (opcional, segunda versión si la sesión se retrasa)
#
# `float32` reduce a la mitad los bytes por celda. En un kernel limitado por el acceso
# a memoria, esto puede acercarse a una mejora de 2x. El rendimiento con `float64`
# varía entre GPU. Identifique el recurso que limita la ejecución y compruebe el
# resultado con una tolerancia adecuada para el dtype.

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
# ## 2.6 Cosas que morden
#
# - **Grupo de memoria.** CuPy guarda bloques liberados. `cp.get_default_memory_pool().used_bytes()` muestra lo que ocupa sus arrays; `nvidia-smi` muestra el grupo, no los datos.
# - **Hardware compartido.** Los GPUs de Colab están compartidos y tienen tiempo límite; un T4 hoy puede ser otra tarjeta mañana. Al siempre anotar el dispositivo que produjo un número.
# - **Reducción a escalares Python** (`float(x.sum())`, `if x.max() > 1:`) fuerza una sincronización y una transferencia. Mantenga el flujo de control en el dispositivo cuando sea posible.
# - **Números aleatorios.** `cp.random` genera en el dispositivo; no genera en el CPU y copia.
#
# **Donde esto lleva.** PyTorch y JAX son primos de CuPy: un array en un dispositivo, operaciones enviadas a kernels, más autodiferenciación y un compilador. Todo lo anterior sobre transferencias, sincronización, precisión y costo adicional de lanzamiento se aplica sin cambios.
#
# ## Punto de control 2: su tabla de tiempos

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
# ¿Qué tres cosas pregunta antes de creer la afirmación de que el GPU es 40 veces más rápido para un kernel que tarda 2 ms? ¿Es la sincronización sincronizada? ¿Incluyen los transferes de datos? ¿Tienen los datos el mismo tipo de dato y el mismo trabajo en ambos lados?

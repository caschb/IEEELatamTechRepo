# %% [markdown]
# # 2. Computación con GPU desde Python
#
# **Sesión en vivo, bloques 5 y 6 (01:20 a 02:10).** Primero cambie a un entorno de ejecución con GPU: *Entorno de ejecución > Cambiar tipo de entorno de ejecución > T4 GPU* (o la GPU que ofrezca Colab),
# luego ejecute la celda de configuración. Si no hay GPU disponible, **quédese en el entorno de ejecución con CPU y continúe**: el cuaderno detecta que no hay GPU, omite las celdas de dispositivo y muestra una tabla de GPU registrada para que pueda hacer las mismas comparaciones.
#
# Una GPU contiene miles de núcleos que comparten memoria de alta velocidad. Es adecuada para millones de operaciones idénticas e independientes, pero pierde eficiencia cuando hay comunicaciones frecuentes o transferencias reiteradas de datos.
#
# - **CuPy**: la API de NumPy con arreglos que viven en la GPU. No introduce conceptos nuevos. (Este cuaderno.)
# - **numba.cuda**: permite escribir el kernel directamente y requiere definir la cuadrícula de hilos.
#   (Extensión opcional `extensions/ext_cuda_kernel`.)
#
# El cálculo por vecindad es el mismo que en el cuaderno 1. Este cuaderno es autónomo.
#
# **Estructura.** Las secciones 2.1 a 2.6 forman la sesión en vivo y terminan en el punto de control 2.
# Las secciones 2.7 a 2.9 (*Para profundizar*) son opcionales, para después del taller: latencia y ancho de banda de PCIe,
# fusión de kernels y el modelo *roofline*.

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
from matplotlib.colors import LogNorm
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

# One colour per device in every figure: CPU blue, GPU orange, a third series aqua.
C_CPU, C_GPU, C_ALT = "#2a78d6", "#eb6834", "#1baf7a"
plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
                     "grid.alpha": 0.3, "figure.dpi": 110, "legend.frameon": False})

INFO = runtime_info(); print(INFO)
if HAVE_GPU:
    print(f"\nMODE: GPU. Device cells run live on {GPU_NAME}.")
else:
    print("\nMODE: CPU fallback. Device cells are skipped; GPU numbers come from the recorded table"
          " and are labelled as such. CPU cells still run live here.")

# %% [markdown]
# ## 2.1 CuPy: el sustituto directo de NumPy
#
# La función se escribe una vez contra un *módulo de arreglos* `xp` y después se le pasan arreglos de NumPy o
# de CuPy. Este es el patrón estándar (la Array API) y es la forma en que SciPy, scikit-learn y xarray obtienen soporte para GPU sin escribir código nuevo.

# %%
def step(u, unew):
    unew[1:-1, 1:-1] = 0.25 * (u[:-2, 1:-1] + u[2:, 1:-1] + u[1:-1, :-2] + u[1:-1, 2:])
    return unew

def run(xp, n, iters, dtype=np.float64, step_fn=step):
    """`iters` stencil steps on a fresh n x n grid held in `xp` (numpy or cupy) arrays."""
    u = xp.zeros((n, n), dtype=dtype)
    u[0, :] = 100.0
    unew = u.copy()
    for _ in range(iters):
        step_fn(u, unew)
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
    print("GPU result matches the CPU result; max |gpu - cpu| =", float(np.abs(u_gpu.get() - u_cpu).max()))
else:
    check(u_cpu, u_cpu, np.float64); print("CPU fallback: correctness check exercised on NumPy only")

# %% [markdown]
# Para ver qué calcula realmente el programa: el borde superior está a 100 grados y los otros tres a 0.
# Cada paso difunde un poco el calor hacia el interior. La misma función `run` produce las cuatro imágenes;
# con GPU se calculan en el dispositivo y solo se copian al host para dibujarlas.

# %%
XP = cp if HAVE_GPU else np
snap_steps = (0, 100, 1000, 10000)
fig, axs = plt.subplots(1, len(snap_steps), figsize=(11, 3.0), constrained_layout=True)
for ax, k in zip(axs, snap_steps):
    im = ax.imshow(to_numpy(run(XP, 128, k)), cmap="inferno", vmin=0, vmax=100)
    ax.set_title(f"{k} steps"); ax.set_axis_off(); ax.grid(False)
fig.colorbar(im, ax=axs, shrink=0.85, label="temperature")
fig.suptitle(f"128 x 128 grid, computed with {'CuPy on ' + GPU_NAME if HAVE_GPU else 'NumPy (CPU)'}");

# %% [markdown]
# **Punto de control 1.** La misma función `step` se ejecutó sobre otro tipo de memoria y produjo los mismos números. Nada del algoritmo cambió.
#
# Observe también cuántos pasos hacen falta para que el calor llegue al fondo: miles, para una cuadrícula
# pequeña. Las simulaciones reales hacen muchos pasos sobre cuadrículas grandes, y por eso interesa el tiempo por paso.
#
# ## 2.2 Medición correcta del código de GPU
#
# Las llamadas a la GPU son asíncronas: `step()` retorna antes de que termine el
# cálculo. Si no se llama a `synchronize()` antes de detener el cronómetro, se mide
# el encolado de la operación, no el trabajo completo. Sincronice antes de iniciar y
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
# La cifra "sin sincronización" puede ser incluso *menor* que el tiempo de una sola
# copia de memoria: la CPU encargó el trabajo y siguió adelante. La siguiente celda lo hace visible: para `k` pasos
# mide cuándo Python recupera el control (encolado) y cuándo la GPU termina de verdad.
# La distancia entre las dos curvas es trabajo que un cronómetro sin sincronizar no ve.

# %%
if HAVE_GPU:
    n, ks = 2048, (1, 2, 5, 10, 20, 50, 100)
    ug, ung = cp.zeros((n, n)), cp.zeros((n, n))
    enqueue, finish = [], []
    for k in ks:
        best_q, best_f = np.inf, np.inf
        for _ in range(3):
            sync(); t0 = time.perf_counter()
            for _ in range(k):
                step(ug, ung)
            t1 = time.perf_counter(); sync(); t2 = time.perf_counter()
            best_q, best_f = min(best_q, t1 - t0), min(best_f, t2 - t0)
        enqueue.append(best_q); finish.append(best_f)
    del ug, ung
    fig, ax = plt.subplots(figsize=(5.5, 3.6))
    ax.plot(ks, np.array(finish) * 1e3, "o-", color=C_GPU, label="GPU finished (after synchronize)")
    ax.plot(ks, np.array(enqueue) * 1e3, "s--", color=C_CPU, label="Python got control back (enqueued)")
    ax.fill_between(ks, np.array(enqueue) * 1e3, np.array(finish) * 1e3, color=C_GPU, alpha=0.12)
    ax.set(xscale="log", yscale="log", xlabel="stencil steps issued", ylabel="elapsed [ms]",
           title=f"Asynchronous launches, n={n}, {GPU_NAME}")
    ax.legend(fontsize=8); fig.tight_layout()
    print(f"per step: enqueue {enqueue[-1]/ks[-1]*1e6:.0f} us, device work {finish[-1]/ks[-1]*1e6:.0f} us")
else:
    print("CPU fallback: this plot needs a GPU.")

# %% [markdown]
# CuPy incluye una herramienta que hace esta medición por usted: `cupyx.profiler.benchmark` informa por separado el
# tiempo de CPU (encolado) y el tiempo de GPU medido con eventos CUDA en el propio dispositivo.
# Cada medición de GPU en el resto de este cuaderno pasa por `best_of(..., sync=sync)`.

# %%
if HAVE_GPU:
    from cupyx.profiler import benchmark
    print(benchmark(run, (cp, 1024, 20), n_repeat=10, n_warmup=1))

# %% [markdown]
# ## 2.3 ¿Cuándo vale la pena la GPU? Barrido de tamaños en igualdad de condiciones
#
# El mismo trabajo, el mismo tipo de dato, el mismo número de pasos, medidos de la misma
# forma, en CPU y en GPU. **Prediga** antes de ejecutar: ¿a partir de qué tamaño, si es que en
# alguno, la GPU superará a la CPU en este entorno de ejecución?

# %%
SIZES, ITERS = (64, 128, 256, 512, 1024, 2048, 4096), 20
timings = []                                    # (impl, n, dtype, scope, min_s, median_s)
for n in SIZES:
    tc = best_of(lambda: run(np, n, ITERS), repeat=3); timings.append(("numpy", n, "float64", "compute", *tc))
    line = f"n={n:5d}  cpu f64 {tc[0]*1e3:8.2f} ms"
    if HAVE_GPU:
        tg = best_of(lambda: run(cp, n, ITERS), repeat=3, sync=sync); timings.append(("cupy", n, "float64", "compute", *tg))
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

# %% [markdown]
# Dos vistas de los mismos números. A la izquierda, el tiempo total. A la derecha, el **rendimiento**:
# miles de millones de celdas actualizadas por segundo. Una línea horizontal en la derecha significa
# que el dispositivo ya trabaja a su ritmo máximo; una línea que sube significa que todavía le sobra capacidad.

# %%
def series(rows, impl, dtype, scope="compute"):
    best = {}
    for r in rows:
        if r[0] == impl and r[2] == dtype and r[3] == scope:
            best[int(r[1])] = min(best.get(int(r[1]), np.inf), float(r[4]))
    return np.array(sorted(best.items())) if best else None

rec_rows = [(r["impl"], r["n"], r["dtype"], r["scope"], r["min_s"], r["median_s"]) for r in recorded]
fig, (ax_t, ax_r) = plt.subplots(1, 2, figsize=(11, 3.9))
for rows, label, style in ((timings, "this runtime", "-"), (rec_rows, "recorded", "--")):
    for impl, marker, color in (("numpy", "o", C_CPU), ("cupy", "s", C_GPU)):
        s = series(rows, impl, "float64")
        if s is None:
            continue
        cells = (s[:, 0] - 2) ** 2 * ITERS
        ax_t.plot(s[:, 0], s[:, 1] * 1e3, marker + style, color=color, label=f"{impl} f64, {label}")
        ax_r.plot(s[:, 0], cells / s[:, 1] / 1e9, marker + style, color=color, label=f"{impl} f64, {label}")
ax_t.set(xlabel="grid side n", ylabel=f"time for {ITERS} steps [ms]", xscale="log", yscale="log", title="Time")
ax_r.set(xlabel="grid side n", ylabel="cell updates per second [10^9]", xscale="log", yscale="log", title="Throughput")
for ax in (ax_t, ax_r):
    ax.set_xticks(SIZES); ax.set_xticklabels(SIZES); ax.minorticks_off()
ax_r.legend(fontsize=8); fig.tight_layout()

# %% [markdown]
# **Explique lo que ve.**
#
# 1. ¿Cambia mucho el tiempo de la GPU con `n` en las cuadrículas pequeñas? Cada paso
#    lanza cinco kernels, y cada lanzamiento cuesta microsegundos sin importar cuánto
#    trabajo haga. Con pocos datos la GPU pasa casi todo el tiempo inactiva y puede perder contra NumPy.
#    En la gráfica de rendimiento esto aparece como una recta que sube: más datos, mismo costo fijo.
# 2. ¿Dónde, si acaso, se cruzan las dos curvas de tiempo? El cruce depende de la CPU, de la
#    GPU y de lo que compartan con otros usuarios. No memorice un número; memorice la forma.
# 3. En la curva de rendimiento de la CPU suele verse un escalón hacia abajo cuando la cuadrícula deja de caber
#    en la memoria caché (unos pocos MB). La GPU tiene su propio escalón, más a la derecha.
#
# ## 2.4 El costo de transferir: solo cálculo frente a cálculo con transferencias
#
# La memoria de la GPU transfiere cientos de GB/s; la conexión PCIe entre el host y el
# dispositivo transfiere decenas. Un trabajo que copia un arreglo a la GPU, hace un
# poco de trabajo y lo copia de vuelta puede pasar la mayor parte de su tiempo en las
# copias. Informe ambos tiempos por separado e indique a cuál se refiere.
#
# **Predicción:** Para `n=2048` (32 MB en float64), ¿una transferencia es más barata o más
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
    cpu_step, gpu_step, xfer_note = tc[0] / ITERS, tk[0] / ITERS, ""
    del ug, ung
else:
    for r in recorded:
        if r["impl"] == "cupy" and int(r["n"]) == n:
            print(f"[recorded] GPU {r['scope']:18s} {float(r['min_s'])*1e3:8.2f} ms")
    pick = lambda impl, scope: min((float(r["min_s"]) for r in recorded
                                    if r["impl"] == impl and int(r["n"]) == n and r["dtype"] == "float64"
                                    and r["scope"] == scope), default=None)
    rec_cpu, rec_comp, rec_incl = pick("numpy", "compute"), pick("cupy", "compute"), pick("cupy", "transfer_inclusive")
    if None not in (rec_cpu, rec_comp, rec_incl):
        cpu_step, gpu_step = rec_cpu / ITERS, rec_comp / ITERS
        t_up = t_down = (rec_incl - rec_comp) / 2        # the table stores only the total, so split it evenly
        xfer_note = f" [RECORDED on {rec_meta.get('gpu')}; upload/download split evenly]"

# %% [markdown]
# La misma información, dibujada. A la izquierda, en qué se va el tiempo de una ejecución de 20 pasos.
# A la derecha, un modelo sencillo construido con los tiempos medidos: el tiempo total en función de
# cuántos pasos se hacen por cada viaje de ida y vuelta a la GPU. Las transferencias son un costo fijo;
# cuantos más pasos se hagan con los datos ya en el dispositivo, menos pesan.

# %%
if "gpu_step" in globals():
    fig, (ax_b, ax_m) = plt.subplots(1, 2, figsize=(11, 3.6), gridspec_kw={"width_ratios": (1, 1.2)})
    segments = (("upload", t_up, C_ALT), (f"{ITERS} steps", gpu_step * ITERS, C_GPU), ("download", t_down, C_ALT))
    left = 0.0
    for label, width, color in segments:
        ax_b.barh(0, width * 1e3, left=left * 1e3, height=0.5, color=color, edgecolor="white", linewidth=2)
        ax_b.text((left + width / 2) * 1e3, 0.3, f"{label}\n{width*1e3:.2f} ms", ha="center", va="bottom", fontsize=8)
        left += width
    ax_b.set(xlabel="time [ms]", ylim=(-0.5, 0.9), yticks=[],
             title=f"One GPU run, n={n}, float64: {left*1e3:.1f} ms\n(the same {ITERS} steps on the CPU: {cpu_step*ITERS*1e3:.1f} ms)")
    ax_b.grid(axis="y", visible=False); ax_b.spines["left"].set_visible(False)

    k = np.logspace(0, 4, 200)
    ax_m.plot(k, k * cpu_step * 1e3, color=C_CPU, label="CPU")
    ax_m.plot(k, (t_up + t_down + k * gpu_step) * 1e3, color=C_GPU, label="GPU incl. transfers")
    ax_m.plot(k, k * gpu_step * 1e3, "--", color=C_GPU, alpha=0.6, label="GPU compute only")
    k_even = (t_up + t_down) / (cpu_step - gpu_step) if cpu_step > gpu_step else np.inf
    if 1 <= k_even <= k[-1]:
        ax_m.axvline(k_even, color="gray", lw=1, ls=":")
        ax_m.annotate(f"break-even\n~{k_even:.0f} steps", (k_even, k_even * cpu_step * 1e3), xytext=(8, 20),
                      textcoords="offset points", fontsize=8)
    else:
        ax_m.text(0.03, 0.97, "GPU wins from the first step" if k_even < 1 else "the GPU never catches up",
                  transform=ax_m.transAxes, va="top", fontsize=8)
    ax_m.set(xscale="log", yscale="log", xlabel="steps per round trip to the GPU", ylabel="total time [ms]",
             title="Transfers are a fixed cost")
    ax_m.legend(fontsize=8); fig.suptitle("Where the time goes" + xfer_note, fontsize=10); fig.tight_layout()
else:
    print("No GPU and no recorded data: nothing to draw.")

# %% [markdown]
# **Regla:** mueva los datos a la GPU una vez, realice allí todo el trabajo y devuelva
# los resultados una sola vez. Usar `cp.asarray()` dentro de un bucle suele ser más lento que NumPy.
# En la gráfica de la derecha, eso equivale a quedarse en el tramo plano de la curva naranja continua,
# donde casi todo el tiempo son transferencias.
#
# Cuando se hable de una aceleración de la GPU, se indicará si las transferencias están dentro del cronómetro.
#
# ## 2.5 Precisión: float32 (opcional, segundo recorte si la sesión se retrasa)
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
# ¿Qué se pierde con `float32`? La siguiente celda ejecuta las dos precisiones lado a lado y dibuja el error
# absoluto de `float32` respecto de `float64`: dónde aparece y cómo crece con los pasos. El error no es uniforme:
# se concentra donde el campo cambia, y se acumula lentamente con cada paso.

# %%
n_err, K_ERR, every = 128, 5000, 100
v64, w64 = XP.zeros((n_err, n_err)), XP.zeros((n_err, n_err)); v64[0, :] = w64[0, :] = 100.0
v32, w32 = v64.astype(np.float32), w64.astype(np.float32)
err_steps, err_max, err_mean = [], [], []
for k in range(1, K_ERR + 1):
    step(v64, w64); v64, w64 = w64, v64
    step(v32, w32); v32, w32 = w32, v32
    if k % every == 0:
        e = abs(v32.astype(np.float64) - v64)
        err_steps.append(k); err_max.append(float(e.max())); err_mean.append(float(e.mean()))
err_map = to_numpy(abs(v32.astype(np.float64) - v64))

fig, (ax_f, ax_e, ax_g) = plt.subplots(1, 3, figsize=(12, 3.5), gridspec_kw={"width_ratios": (1, 1, 1.3)})
ax_f.imshow(to_numpy(v64), cmap="inferno", vmin=0, vmax=100); ax_f.set_title(f"float64 field, {K_ERR} steps")
im = ax_e.imshow(np.where(err_map > 0, err_map, np.nan), cmap="Oranges",
                 norm=LogNorm(vmin=max(err_map[err_map > 0].min(), 1e-9), vmax=err_map.max()))
ax_e.set_title("|float32 - float64|"); fig.colorbar(im, ax=ax_e, shrink=0.85)
for ax in (ax_f, ax_e):
    ax.set_axis_off(); ax.grid(False)
ax_g.plot(err_steps, err_max, color=C_GPU, alpha=0.5, label="maximum over the grid")
ax_g.plot(err_steps, err_mean, color=C_CPU, label="mean over the grid")
ax_g.set(xlabel="steps", ylabel="abs error [degrees]", yscale="log", title="float32 error accumulates")
ax_g.legend(fontsize=8)
fig.tight_layout()
print(f"after {K_ERR} steps: max error {err_max[-1]:.2e} degrees on a 0-100 scale "
      f"(float32 carries about 7 significant digits)")

# %% [markdown]
# ## 2.6 Cosas que muerden
#
# - **Grupo de memoria.** CuPy conserva los bloques liberados para reutilizarlos. `cp.get_default_memory_pool().used_bytes()` muestra lo que ocupan sus arreglos; `nvidia-smi` muestra el grupo completo, no los datos.
# - **Hardware compartido.** Las GPU de Colab son compartidas y tienen tiempo límite; una T4 hoy puede ser otra tarjeta mañana. Anote siempre el dispositivo que produjo un número.
# - **Reducción a escalares de Python** (`float(x.sum())`, `if x.max() > 1:`) fuerza una sincronización y una transferencia. Mantenga el flujo de control en el dispositivo cuando sea posible.
# - **Números aleatorios.** `cp.random` genera en el dispositivo; no genere en la CPU para después copiar.
#
# **Hacia dónde lleva esto.** PyTorch y JAX son parientes de CuPy: un arreglo en un dispositivo, operaciones enviadas a kernels, más diferenciación automática y un compilador. Todo lo anterior sobre transferencias, sincronización, precisión y costo de lanzamiento se aplica sin cambios.
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
# ¿Qué tres cosas preguntaría antes de creer la afirmación de que la GPU es 40 veces más rápida en un kernel que tarda 2 ms? ¿Se sincronizó antes de detener el cronómetro? ¿Se incluyen las transferencias de datos? ¿Se hizo el mismo trabajo con el mismo tipo de dato en ambos lados?
#
# ---
#
# # Para profundizar (opcional, no se cubre en vivo)
#
# Las tres secciones siguientes responden preguntas que suelen aparecer al final de la sesión: ¿por qué una
# transferencia pequeña es tan cara?, ¿se puede hacer más rápido el paso de CuPy? y ¿cómo saber cuánto
# más rápido *podría* ir un kernel? Requieren GPU; en modo alternativo de CPU se ejecutan solo las partes de CPU.
#
# ## 2.7 PCIe: latencia más tamaño dividido entre ancho de banda
#
# Una copia entre host y dispositivo cuesta aproximadamente `t = alfa + bytes / beta`: una **latencia** fija
# `alfa` más el tamaño dividido entre el **ancho de banda** `beta`. Es el mismo modelo de comunicación del cuaderno 3,
# ahora dentro de una sola máquina. La celda mide copias de 4 KB a 256 MB desde dos tipos de memoria del host:
#
# - **paginable** (un arreglo normal de NumPy): el controlador primero la copia a un búfer intermedio;
# - **fijada** (*pinned*, `cupyx.empty_pinned`): la GPU la lee directamente por DMA.

# %%
if HAVE_GPU:
    import cupyx
    sizes_b = 2 ** np.arange(12, 29, 2)                 # 4 KiB .. 256 MiB
    t_pageable, t_pinned = [], []
    for nb in sizes_b:
        h = np.ones(nb // 8)
        hp = cupyx.empty_pinned(h.shape, h.dtype); hp[...] = h
        d = cp.empty(h.shape, h.dtype)
        t_pageable.append(best_of(lambda: d.set(h), sync=sync)[0])
        t_pinned.append(best_of(lambda: d.set(hp), sync=sync)[0])
        del h, hp, d
    t_pageable, t_pinned = np.array(t_pageable), np.array(t_pinned)

    # t = alpha + bytes / beta. A least-squares fit would be dominated by the largest copies and can return a
    # negative alpha, so take alpha from the smallest copy and beta from the largest.
    alpha = t_pinned[0]
    beta_inv = (t_pinned[-1] - alpha) / sizes_b[-1]
    fig, (ax_t, ax_b) = plt.subplots(1, 2, figsize=(11, 3.7))
    for t, label, color in ((t_pageable, "pageable (NumPy)", C_CPU), (t_pinned, "pinned", C_GPU)):
        ax_t.plot(sizes_b, t * 1e6, "o-", color=color, label=label)
        ax_b.plot(sizes_b, sizes_b / t / 1e9, "o-", color=color, label=label)
    ax_t.plot(sizes_b, (alpha + sizes_b * beta_inv) * 1e6, ":", color="gray",
              label=f"model: {alpha*1e6:.0f} us + bytes / {1/beta_inv/1e9:.1f} GB/s")
    ax_b.axhline(1 / beta_inv / 1e9, color="gray", ls=":", lw=1)
    ax_t.set(xscale="log", yscale="log", xlabel="bytes copied host -> device", ylabel="time [us]", title="Copy time")
    ax_b.set(xscale="log", xlabel="bytes copied host -> device", ylabel="effective bandwidth [GB/s]",
             title="Small copies never reach full bandwidth")
    ax_t.legend(fontsize=8); fig.tight_layout()
    half = alpha / beta_inv
    print(f"latency ~{alpha*1e6:.0f} us, bandwidth ~{1/beta_inv/1e9:.1f} GB/s (pinned). "
          f"Half of the peak bandwidth needs copies of ~{half/2**20:.2f} MiB.")
else:
    print("CPU fallback: this experiment needs a GPU.")

# %% [markdown]
# **Lectura.** Por debajo de unas decenas de KB, el tiempo casi no depende del tamaño: se paga la latencia.
# Por eso copiar muchos arreglos pequeños es mucho más caro que copiar uno grande con los mismos bytes.
# La memoria fijada suele alcanzar un ancho de banda mayor, pero es un recurso escaso del sistema operativo:
# se reserva para búferes que se reutilizan (por ejemplo, los `DataLoader(pin_memory=True)` de PyTorch).
#
# ## 2.8 Fusión: un kernel en lugar de cinco
#
# La línea de `step` parece una sola operación, pero CuPy la ejecuta como cinco kernels: tres sumas, una
# multiplicación y la copia a `unew`. Cada uno lee sus entradas de la memoria de la GPU y escribe un arreglo
# temporal. Por celda se mueven unos 13 valores, cuando bastaría con leer uno (los vecinos
# quedan en caché) y escribir otro.
#
# Un kernel propio fusiona todo en una pasada. `cp.RawKernel` compila unas líneas de CUDA C en tiempo de ejecución;
# cada hilo calcula una celda `(i, j)`. (La extensión `ext_cuda_kernel` hace lo mismo con `numba.cuda`, en Python.)

# %%
if HAVE_GPU:
    stencil_kernel = cp.RawKernel(r"""
    extern "C" __global__
    void stencil(const double* u, double* unew, int n) {
        int j = blockIdx.x * blockDim.x + threadIdx.x;
        int i = blockIdx.y * blockDim.y + threadIdx.y;
        if (i > 0 && i < n - 1 && j > 0 && j < n - 1)
            unew[i * n + j] = 0.25 * (u[(i - 1) * n + j] + u[(i + 1) * n + j] + u[i * n + j - 1] + u[i * n + j + 1]);
    }
    """, "stencil")

    def step_fused(u, unew):
        n = u.shape[0]
        stencil_kernel(((n + 31) // 32, (n + 7) // 8), (32, 8), (u, unew, np.int32(n)))
        return unew

    check(run(cp, 512, 20, step_fn=step_fused), u_cpu, np.float64)
    print("fused kernel matches the CPU result")

    big, big_dst = cp.ones(2**25), cp.zeros(2**25)       # 256 MiB; a device-to-device copy reads and writes it once
    copy_bw = 2 * big.nbytes / best_of(lambda: cp.copyto(big_dst, big), sync=sync)[0]
    del big, big_dst
    print(f"device copy bandwidth (practical ceiling): {copy_bw/1e9:.0f} GB/s")

    FUSE_SIZES, fused_rows = (512, 1024, 2048, 4096), []
    for n in FUSE_SIZES:
        t_sl = best_of(lambda: run(cp, n, ITERS), repeat=3, sync=sync)[0]
        t_fu = best_of(lambda: run(cp, n, ITERS, step_fn=step_fused), repeat=3, sync=sync)[0]
        fused_rows.append((n, t_sl, t_fu))
        print(f"n={n:5d}  CuPy slicing {t_sl*1e3:8.2f} ms   fused kernel {t_fu*1e3:8.2f} ms   {t_sl/t_fu:4.1f}x")

    # Minimum traffic per cell update: read one value, write one value.
    eff_bw = lambda n, t: 16 * (n - 2) ** 2 * ITERS / t / 1e9
    x = np.arange(len(FUSE_SIZES)); w = 0.38
    fig, ax = plt.subplots(figsize=(6.5, 3.7))
    ax.bar(x - w / 2, [eff_bw(n, t) for n, t, _ in fused_rows], w, color=C_GPU, alpha=0.55, label="CuPy slicing (5 kernels)")
    ax.bar(x + w / 2, [eff_bw(n, t) for n, _, t in fused_rows], w, color=C_GPU, label="fused RawKernel (1 kernel)")
    ax.axhline(copy_bw / 1e9, color="gray", ls=":", lw=1.2)
    ax.text(len(x) - 0.5, copy_bw / 1e9, "device copy bandwidth", ha="right", va="bottom", fontsize=8)
    ax.set_xticks(x); ax.set_xticklabels(FUSE_SIZES); ax.grid(axis="x", visible=False)
    ax.set(xlabel="grid side n", ylabel="useful bandwidth [GB/s]",
           title=f"Stencil, float64, {GPU_NAME}\n(16 bytes of necessary traffic per cell)")
    ax.legend(fontsize=8, loc="upper left"); fig.tight_layout()
else:
    print("CPU fallback: this experiment needs a GPU.")

# %% [markdown]
# **Lectura.** El ancho de banda *útil* cuenta solo los bytes imprescindibles. El kernel fusionado se acerca
# al techo que marca una copia simple; la versión con slicing queda muy por debajo porque mueve varias veces
# más bytes de los necesarios. Si una barra supera la línea, los dos arreglos cupieron en la caché L2 de la GPU
# (decenas de MB en tarjetas recientes, 4 MB en una T4) y la copia de 256 MB no es el techo correcto para ese tamaño. Esta es la optimización que hacen automáticamente `torch.compile`, `jax.jit`
# y `cupy.fuse` sobre operaciones elemento a elemento: menos kernels y menos arreglos temporales.
#
# ## 2.9 ¿Qué limita a un kernel? El modelo *roofline*
#
# Todo kernel necesita dos recursos: **operaciones** (FLOP/s) y **bytes** de memoria (GB/s). La
# **intensidad aritmética** es el cociente entre ambos: FLOP realizados por byte movido.
#
# - El cálculo por vecindad hace 4 FLOP por celda y mueve al menos 16 bytes: intensidad 0.25. Está limitado por memoria.
# - Una multiplicación de matrices `n x n` hace `2 n^3` FLOP sobre `3 n^2` valores: intensidad de cientos.
#   Está limitada por cálculo. Es la operación central del aprendizaje profundo.
#
# El *roofline* dibuja el rendimiento máximo alcanzable, `min(pico de FLOP/s, ancho de banda x intensidad)`,
# como un techo con una rampa y una parte plana. Aquí los techos son **prácticos**: el ancho de banda se mide
# con una copia y el pico con una multiplicación de matrices grande. Después se ubican los kernels debajo.

# %%
def matmul_flops(xp, n, dtype, sync_fn):
    A = xp.ones((n, n), dtype=dtype); B = xp.ones((n, n), dtype=dtype)
    t = best_of(lambda: A @ B, repeat=3, sync=sync_fn)[0]
    return 2 * n ** 3 / t

def copy_bandwidth(xp, sync_fn, n_values=2**25):
    src, dst = xp.ones(n_values), xp.zeros(n_values)    # preallocated: a fresh copy() would also time page faults
    return 2 * src.nbytes / best_of(lambda: xp.copyto(dst, src), repeat=3, sync=sync_fn)[0]

roof = {"CPU": {"bw": copy_bandwidth(np, lambda: None), "color": C_CPU,
                "peak": {dt: matmul_flops(np, 2048, dt, lambda: None) for dt in ("float64", "float32")}}}
points = []                                              # (label, device, intensity, flop/s)
stencil_ai = 4 / 16
t_np = series(timings, "numpy", "float64")
points.append(("stencil NumPy", "CPU", stencil_ai, 4 * (t_np[-1, 0] - 2) ** 2 * ITERS / t_np[-1, 1]))
if HAVE_GPU:
    roof["GPU"] = {"bw": copy_bw, "color": C_GPU,
                   "peak": {dt: matmul_flops(cp, 4096, dt, sync) for dt in ("float64", "float32", "float16")}}
    n_f, t_sl, t_fu = fused_rows[-1]
    points += [("stencil CuPy", "GPU", stencil_ai, 4 * (n_f - 2) ** 2 * ITERS / t_sl),
               ("stencil fused", "GPU", stencil_ai, 4 * (n_f - 2) ** 2 * ITERS / t_fu)]
    for dt, nbytes in (("float64", 8), ("float32", 4), ("float16", 2)):
        points.append((f"matmul {dt}", "GPU", 2 * 4096 / (3 * nbytes), roof["GPU"]["peak"][dt]))
for dt, nbytes in (("float64", 8), ("float32", 4)):
    points.append((f"matmul {dt}", "CPU", 2 * 2048 / (3 * nbytes), roof["CPU"]["peak"][dt]))

for dev, r in roof.items():
    print(f"{dev}: bandwidth {r['bw']/1e9:7.1f} GB/s   " +
          "   ".join(f"{dt} {p/1e12:7.2f} TFLOP/s" for dt, p in r["peak"].items()))

# %%
ai = np.logspace(-2, 3.5, 300)
fig, ax = plt.subplots(figsize=(8, 5))
for dev, r in roof.items():
    for dt, ls in (("float64", "-"), ("float32", "--"), ("float16", ":")):
        if dt in r["peak"]:
            ax.plot(ai, np.minimum(r["peak"][dt], r["bw"] * ai) / 1e9, ls, color=r["color"], lw=1.5,
                    label=f"{dev} roof, {dt}")
markers = {"stencil": "o", "matmul": "s"}
for label, dev, x, y in points:
    ax.plot(x, y / 1e9, markers[label.split()[0]], color=roof[dev]["color"], ms=8,
            mec="white", mew=1.5, alpha=0.6 if "CuPy" in label else 1)
    below = dev == "CPU" or "CuPy" in label
    left = label == "matmul float64" and dev == "CPU"    # sits right next to the CPU float32 point
    ax.annotate(f"{label} ({dev})", (x, y / 1e9), xytext=(-6 if left else 6, -12 if below else 4),
                textcoords="offset points", fontsize=7, ha="right" if left else "left")
ax.set(xscale="log", yscale="log", xlabel="arithmetic intensity [FLOP / byte]", ylabel="performance [GFLOP/s]",
       title=f"Roofline, measured on {'this runtime: ' + GPU_NAME if HAVE_GPU else 'this CPU runtime'}")
ax.legend(fontsize=7, loc="lower right"); fig.tight_layout()

# %% [markdown]
# **Lectura del gráfico.**
#
# 1. Los puntos del cálculo por vecindad están en la **rampa**: el ancho de banda los limita. Más núcleos o más
#    FLOP/s no los aceleran; solo mover menos bytes (fusión, `float32`) o tener memoria más rápida.
#    La distancia vertical entre "stencil CuPy" y la rampa es el costo de los arreglos temporales de la sección 2.8.
#    "stencil NumPy" queda aún más lejos de la rampa de la CPU: usa un solo hilo y también crea temporales.
#    La versión de Numba con `prange` del cuaderno 1 corrige ambas cosas.
# 2. Las multiplicaciones de matrices están en la **parte plana**: las limita la aritmética. Ahí sí importa el
#    tipo de dato. En muchas GPU de consumo y de Colab (T4, L4) `float64` es decenas de veces más lento que
#    `float32`, y `float16` usa los *tensor cores*. Por eso el entrenamiento de redes neuronales usa precisión reducida.
# 3. Compare la razón entre GPU y CPU en la rampa (la razón de anchos de banda) con la razón en la parte plana
#    (la razón de FLOP/s). La aceleración que ofrece una GPU depende de en qué parte del techo vive su kernel.

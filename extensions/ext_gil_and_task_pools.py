# %% [markdown]
# # Extensión A: hilos, procesos y el GIL (opcional, no cubierto en vivo)
#
# La función `prange` de Numba funciona porque el código compilado no sosten
# el Lock Global Interpreter (GIL) de Python. El código Python sí lo hace, por lo que
# las *threads* de Python no ejecutan código Python en paralelo. Esta extensión observa
# una tarea independiente que se ejecuta de tres formas diferentes. El tiempo de ejecución en CPU; el número de trabajadores se adapta al tiempo de ejecución.
#
# *Estado: extensión opcional, validada en una máquina del autor, no parte del
# sesión de 180 minutos. Si no se ejecuta en tu Colab, déjala de lado.*

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
# Resultado típico en una máquina con varios núcleos: las hilos no son más rápidos que
# serial (toman turnos para obtener una licencia), los procesos son más rápidos (cada uno tiene su propio intérprete y licencia) pero pagan por iniciar y para serializar los argumentos y los resultados.
# En un entorno de ejecución con solo CPU, ninguno de los tres puede ser más rápido que serial. Lo que veas, la regla es: los procesos para tareas puramente Python que son gruesas (uno por archivo, uno por conjunto de parámetros); los hilos para código compilado o con liberación del GIL (como `@njit(nogil=True)`, la mayoría de NumPy, lectura y escritura en archivos y red).
#
# ## joblib: la misma idea con una ergonomía mejorada

# %%
import importlib.util, subprocess, sys
if importlib.util.find_spec("joblib") is None:
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "joblib"], check=True)
from joblib import Parallel, delayed

t_joblib = timed(f"joblib, {WORKERS} workers", lambda: Parallel(n_jobs=WORKERS)(delayed(slow_python_task)(s) for s in seeds))

# %% [markdown]
# ## Tarea compilada que libera el GIL

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
# Con `nogil=True`, las hilas pueden ejecutar la función compilada al mismo tiempo, por lo que
# en un entorno de múltiples núcleos, la versión en hilas gana sin serialización y sin inicio de proceso. Este es el mecanismo que `prange` utiliza internamente.

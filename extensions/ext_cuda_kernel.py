# %% [markdown]
# # Extensión B: escribe tu propio kernel con numba.cuda (opcional)
#
# **Requiere un entorno de ejecución con GPU.** CuPy oculta el kernel. A veces
# es necesario tu propio: una operación que no es una composición de operaciones
# de array, o una donde los intermediarios de CuPy (cuatro por paso de stencil)
# cuestan demasiado. El modelo mental: una función de Python se ejecuta una vez
# por **hilo**, y cada hilo pregunta "¿Qué elemento soy?".
#
# *Estado: extensión opcional. El objetivo de Numba CUDA se movió a la
# separada paquete `numba-cuda`; este cuaderno instala el paquete si es necesario.
# La compatibilidad de ese paquete con Colab debe ser comprobada separadamente
# antes de recomendar este cuaderno a los estudiantes.*

# %%
import importlib, importlib.util, subprocess, sys, time
import numpy as np

def ensure(module, package=None):
    if importlib.util.find_spec(module) is None:
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", package or module], check=True)
    return importlib.import_module(module)

cp = ensure("cupy", "cupy-cuda12x")
try:
    from numba import cuda
    assert cuda.is_available()
except Exception:
    ensure("numba_cuda", "numba-cuda[cu12]")
    from numba import cuda
print("device:", cuda.get_current_device().name.decode() if isinstance(cuda.get_current_device().name, bytes)
      else cuda.get_current_device().name)

# %%
def step_cupy(u, unew):
    unew[1:-1, 1:-1] = 0.25 * (u[:-2, 1:-1] + u[2:, 1:-1] + u[1:-1, :-2] + u[1:-1, 2:])
    return unew

def run(step, n, iters, dtype=np.float64):
    u = cp.zeros((n, n), dtype=dtype); u[0, :] = 100.0; unew = u.copy()
    for _ in range(iters):
        step(u, unew); u, unew = unew, u
    return u

@cuda.jit
def step_kernel(u, unew):
    i, j = cuda.grid(2)                      # this thread's (row, col)
    n, m = u.shape
    if 1 <= i < n - 1 and 1 <= j < m - 1:
        unew[i, j] = 0.25 * (u[i-1, j] + u[i+1, j] + u[i, j-1] + u[i, j+1])

def step_numba_cuda(u, unew):
    block = (16, 16)
    grid = ((u.shape[0] + 15) // 16, (u.shape[1] + 15) // 16)
    step_kernel[grid, block](u, unew)        # CuPy arrays are accepted through __cuda_array_interface__
    return unew

ref = run(step_cupy, 512, 20).get()
assert np.allclose(run(step_numba_cuda, 512, 20).get(), ref)
print("kernel result matches CuPy")

# %%
sync = cp.cuda.Device().synchronize
def best_of(fn, repeat=5):
    fn(); sync(); ts = []
    for _ in range(repeat):
        sync(); t0 = time.perf_counter(); fn(); sync(); ts.append(time.perf_counter() - t0)
    return min(ts)

for n in (512, 1024, 2048):
    for dtype in (np.float64, np.float32):
        t_cupy = best_of(lambda: run(step_cupy, n, 20, dtype))
        t_kern = best_of(lambda: run(step_numba_cuda, n, 20, dtype))
        print(f"n={n:5d} {dtype.__name__}: CuPy {t_cupy*1e3:7.2f} ms   numba.cuda kernel {t_kern*1e3:7.2f} ms")

# %% [markdown]
# La kernel en mano lee cada entrada una vez y escribe una vez; la versión de CuPy materializa los intermedios. Dependiendo de la tamaño y el cartón: revisa la relación que midiste, en lugar de asumir. La misma historia de NumPy vs. Numba desde la sesión de CPU, un nivel más abajo.

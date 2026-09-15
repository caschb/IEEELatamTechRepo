# %% [markdown]
# # 0a. ¿Está mi Colab listo? (preparación, 10 a 15 minutos)
#
# Objetivo: Antes del evento, compruebe que puede abrir, ejecutar, editar y guardar
# un cuaderno en Google Colab, y que las paquetes necesarias para el curso instalen en
# un **entorno de ejecución estándar CPU**. No es necesario tener una GPU para esto. Un chequeo completo de
# la CPU es todo lo que se requiere para la preparación.
#
# ## Paso 1: haga su propia copia
#
# El Colab abrió este cuaderno de forma sololectiva desde GitHub. Haga clic en el cuadro
# abajo y presione **Archivo > Guardar una copia en Drive** para que sus ediciones y
# salidas se mantengan. Trabaje con la copia desde ahora.
#
# ## Paso 2: ejecute un cuadro
#
# Haga clic en el cuadro debajo y presione **Shift+Enter** (o el botón de reproducción).

# %%
print("Hello from Colab. This cell ran.")

# %% [markdown]
# ## Paso 3: editar una celda
#
# Cambie el número en la siguiente celda a su número favorito, luego ejecute la celda. El
# check en la línea segunda debería decir `PASS`.

# %%
favourite = 7            # <- change this
print("PASS: you edited and ran a cell" if favourite != 7 else "not yet: change the number and run again")

# %% [markdown]
# ## Paso 4: instalar lo que necesita el curso
#
# Cada cuaderno de notebook de curso comienza con una celda de configuración como esta. Importa cada
# paquete y lo instala con pip **sólo si la importación falla**, así que en una
# entorno de ejecución estándar de Colab, generalmente es rápido. Deberás ejecutarla de nuevo cada vez que la
# entorno de ejecución se reinicie, ya que una entorno de ejecución nueva no tiene ninguna de sus instalaciones.

# %%
import importlib, importlib.util, os, platform, subprocess, sys

def ensure(module, package=None):
    """Import `module`, installing `package` with pip only if the import fails."""
    if importlib.util.find_spec(module) is None:
        print("installing", package or module)
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", package or module], check=True)
    return importlib.import_module(module)

np = ensure("numpy")
numba = ensure("numba")
psutil = ensure("psutil")
ensure("matplotlib"); import matplotlib.pyplot as plt
ensure("line_profiler")
print("all packages import")

# %% [markdown]
# ## Paso 5: revisar la máquina que le dieron
#
# Colab asigna diferentes máquinas virtuales en momentos diferentes. Registrar la máquina asignada permite interpretar correctamente las mediciones.

# %%
IN_COLAB = "COLAB_RELEASE_TAG" in os.environ or "google.colab" in sys.modules
print("running in Colab:", IN_COLAB)
print("python           ", platform.python_version())
print("numpy / numba    ", np.__version__, "/", numba.__version__)
print("logical CPUs     ", os.cpu_count())
print("Numba thread cap ", numba.config.NUMBA_NUM_THREADS)
print("RAM              ", round(psutil.virtual_memory().total / 2**30, 1), "GB")

# %% [markdown]
# ## Paso 6: Una función compilada muy pequeña
#
# Este verifica que Numba puede compilar en este entorno de ejecución. La primera llamada tarda unos segundos (compilación); la segunda es rápida.

# %%
import time
from numba import njit

@njit
def sum_of_squares(a):
    s = 0.0
    for i in range(a.shape[0]):
        s += a[i] * a[i]
    return s

x = np.random.default_rng(0).random(2_000_000)
t0 = time.perf_counter(); r1 = sum_of_squares(x); t_first = time.perf_counter() - t0
t0 = time.perf_counter(); r2 = sum_of_squares(x); t_second = time.perf_counter() - t0
assert np.isclose(r1, (x * x).sum())
print(f"first call (compiles): {t_first*1e3:7.1f} ms   second call: {t_second*1e3:6.1f} ms   PASS")

# %% [markdown]
# ## Paso 7: Reiniciar y volver a ejecutar
#
# Elija **Entorno de ejecución > Reiniciar sesión**, luego **Entorno de ejecución > Ejecutar todo**. Todo
# lo que está arriba debería pasar de nuevo sin que hagas nada más. Eso es exactamente
# lo que deberá hacer durante el evento si se desconecta el entorno de ejecución.
#
# Si en cambio elija **Entorno de ejecución > Desconectar y eliminar entorno de ejecución**, la próxima ejecución
# tendrá que instalar de nuevo los paquetes. Eso es esperado.
#
# ## Paso 8: Informe de estado
#
# Ejecute la celda siguiente y conserve la salida. Si algo falla, péguela donde el README de preparación
# indique cómo informar problemas.

# %%
report = {
    "colab": IN_COLAB, "python": platform.python_version(),
    "numpy": np.__version__, "numba": numba.__version__,
    "cpus": os.cpu_count(), "ram_gb": round(psutil.virtual_memory().total / 2**30, 1),
    "numba_compiles": bool(np.isclose(r1, r2)),
}
print("STATUS " + " ".join(f"{k}={v}" for k, v in report.items()))
print("READY for IEEE Latam Tech" if report["numba_compiles"] else "NOT READY: see the error above")

# %% [markdown]
# **Listo.** Guarde el notebook (Ctrl+S). Continúe con `00_stencil_practice`.

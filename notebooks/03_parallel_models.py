# %% [markdown]
# # 3. Más allá de un solo máquina: particiones, halos y comunicación
#
# **Sesión en vivo, bloque 7 (02:10 a 02:30).** Cualquier entorno de ejecución; nada aquí necesita
# un GPU, una segunda máquina o un cluster.
#
# Un solo máquina tiene algunas núcleos, posiblemente un GPU y un límite de memoria. Cuando
# eso no es suficiente, se usa varias máquinas, y un hecho cambia todo: las máquinas separadas no comparten memoria. Cada byte que necesita otro hilo
# tiene que ser enviado a él.
#
# > **Qué es este cuaderno.** Un **proyecto final** de ejecución multi-hilos, construido con
# > arrays de NumPy dentro de un entorno de ejecución Colab. Las particiones a continuación son
# > cortes de un solo array y los "mensajes" son copias de arrays. Demostración de
# > **qué** tiene que ser comunicado y **por qué**; no hace **ninguna afirmación sobre el rendimiento multi-nodo**, porque nada aquí corre en más de una máquina.
#
# Dos formas de programar hilos separados reales:
#
# - **MPI** (`mpi4py`): cada proceso ejecute el mismo script, y escribís los mensajes explícitamente. Más rápido, más control, más código. La lengua franca de los códigos de simulación.
# - **Dask**: un agente de trabajo distribuye tareas. Poca codificación; ideal cuando las tareas son independientes o el datos es más grande que una máquina.
#
# Este cuaderno es autónomo.

# %%
# --- Setup: rerun after every runtime restart ------------------------------
import importlib, importlib.util, subprocess, sys

def ensure(module, package=None):
    if importlib.util.find_spec(module) is None:
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", package or module], check=True)
    return importlib.import_module(module)

np = ensure("numpy")
ensure("matplotlib"); import matplotlib.pyplot as plt
print("numpy", np.__version__)

# %% [markdown]
# ## 3.1 El cálculo por vecindad serial, una vez más
#
# Esta función es la misma que en los cuadernos 1 y 2. Es la referencia que cada versión particionada debe reproducir exactamente.

# %%
def init_grid(n):
    u = np.zeros((n, n)); u[0, :] = 100.0
    return u

def step(u, unew):
    unew[1:-1, 1:-1] = 0.25 * (u[:-2, 1:-1] + u[2:, 1:-1] + u[1:-1, :-2] + u[1:-1, 2:])
    return unew

def run_serial(n, iters):
    u = init_grid(n); unew = u.copy()
    for _ in range(iters):
        step(u, unew); u, unew = unew, u
    return u

N, ITERS = 256, 30
ref = run_serial(N, ITERS)
print("reference computed:", ref.shape, "interior mean", ref[1:-1, 1:-1].mean().round(4))

# %% [markdown]
# ## 3.2 Decomposición del dominio: strips y halos
#
# Divide la cuadrícula en `P` estratos horizontales, uno por trabajador. Para actualizar su fila interior superior, un estrato necesita la fila justo encima, que pertenece al vecino. Cada estrato, por lo tanto, conserva una fila extra arriba y abajo: las filas de **halo** (fantasma). Cada paso, los vecinos intercambian sus filas de frontera para actualizar los halos del otro. Todo lo demás es el cálculo por vecindad serial, sin tocar.
#
# ```
#         trabajador 0            trabajador 1            trabajador 2
#      +-----------+       +-----------+       +-----------+
#      | halo (top)|       | halo      | <---- | última fila real |
#      | fila real |       | fila real |       | fila real   |
#      | fila real | ----> | halo      |       | ...         |
#      | halo (bot)| <---- | primera fila real|       |           |
#      +-----------+       +-----------+       +-----------+
#      flechas = una fila copiada por vecino por paso
# ```
#
# **Ejercicio.** Antes de ejecutar las siguientes celdas: con `P` estratos de una cuadrícula `n x n`, ¿cuántas filas cruzan una frontera entre dos vecinos en un paso, y cuántos bytes es eso en float64? Escriba su respuesta.

# %%
def partition(u, P):
    """Split the interior rows of u into P strips; each strip keeps one halo row above and below.

    The global top and bottom rows are physical boundaries: they sit in the halos of the
    first and last strip, are never updated, and never need to be communicated."""
    n = u.shape[0]
    bounds = [(rows[0], rows[-1] + 1) for rows in np.array_split(np.arange(1, n - 1), P)]
    return [u[lo - 1:hi + 1].copy() for lo, hi in bounds]      # rows lo-1 and hi are the halos

def exchange_halos(strips):
    """What the network would carry: each strip sends its boundary rows to its neighbours."""
    P = len(strips)
    nbytes = 0
    for r in range(P):
        if r > 0:                        # my top halo <- neighbour above's last real row
            strips[r][0] = strips[r-1][-2]; nbytes += strips[r][0].nbytes
        if r < P - 1:                    # my bottom halo <- neighbour below's first real row
            strips[r][-1] = strips[r+1][1]; nbytes += strips[r][-1].nbytes
    return nbytes

def reconstruct(strips):
    return np.vstack([strips[0][:1]] + [s[1:-1] for s in strips] + [strips[-1][-1:]])

def run_partitioned(n, iters, P):
    strips = partition(init_grid(n), P)
    news = [s.copy() for s in strips]
    bytes_per_step = 0
    for _ in range(iters):
        bytes_per_step = exchange_halos(strips)
        for s, snew in zip(strips, news):
            step(s, snew)                # each worker updates only its own real rows
        strips, news = news, strips
    return reconstruct(strips), bytes_per_step

for P in (1, 2, 4, 8):
    u_p, nbytes = run_partitioned(N, ITERS, P)
    assert u_p.shape == ref.shape and np.array_equal(u_p, ref), f"P={P} does not match the serial result"
    print(f"P={P}: identical to the serial result; {nbytes/1024:6.1f} KB exchanged per step")

# %% [markdown]
# Dos cosas para notar en la celda anterior:
#
# 1. Las **límites superiores e inferiores globales** nunca necesitan un mensaje: son límites físicos, no vecinos. Solo las límites interiores intercambian filas.
# 2. La rediseñada es igual a la serial en términos de `array_equal`, no es solo aproximada. La descomposición no cambia el cálculo, solo quien lo realiza.
#
# **Verifica su respuesta al ejercicio.** Cada límite interno lleva dos filas por paso (una en cada dirección), por lo que `2 * (P-1)` filas de `n` valores `float64`.
#
# ## 3.3 ¿Es la comunicación cara o barata aquí?
#
# Por paso, cada trabajador computa `n * (n/P)` celdas y intercambia al menos `2n` valores. La relación de comunicación con cálculo es `2P/n`: disminuye con un tamaño de cuadrícula más grande y aumenta con un número de trabajadores mayor. Cualquier red real añade un **latencia** fija por mensaje más que los bytes.

# %%
n_values = np.array([256, 1024, 4096, 16384])
fig, ax = plt.subplots(figsize=(5.5, 3.5))
for P in (2, 8, 32, 128):
    ax.plot(n_values, 2 * P / n_values, "o-", label=f"P={P}")
ax.set(xlabel="grid side n", ylabel="values sent / cells computed, per worker per step",
       xscale="log", yscale="log", title="halo exchange relative to work (model, not a measurement)")
ax.legend(); fig.tight_layout()

# %% [markdown]
# **Predicción y explicación.** Supón que un mensaje cuesta 10 microsegundos de latencia más los bytes a 10 GB/s, y una actualización de celda cuesta 1 nanosegundo. Para `n=256` y `P=128` (dos filas por trabajador), ¿es un trabajador que está principalmente realizando cálculos o que está principalmente esperando por mensajes? ¿Y para `n=16384`? Esto es la razón por la cual los problemas pequeños no escalan a través de máquinas, sin importar la biblioteca.
#
# ## 3.4 El mismo concepto en MPI (lectura, no ejecución)
#
# Con `mpi4py`, cada rango ejecute este script. `comm.rank` dice quién es; `Sendrecv` hace una intercambio en ambas direcciones. `PROC_NULL` convierte los vecinos faltantes de los rango de bordes en operaciones nulas. Compare línea por línea con `exchange_halos` arriba: las mismas filas, misma dirección.
#
# ```python
# from mpi4py import MPI
# comm = MPI.COMM_WORLD
# rank, size = comm.rank, comm.size
# up   = rank - 1 if rank > 0        else MPI.PROC_NULL
# down = rank + 1 if rank < size - 1 else MPI.PROC_NULL
#
# s = my_strip_with_halos()                       # forma (h + 2, n), como en la partición()
# for _ in range(iters):
#     comm.Sendrecv(s[1],  dest=up,   recvbuf=s[0],  source=up)     # primera fila real hacia arriba, halo superior en
#     comm.Sendrecv(s[-2], dest=down, recvbuf=s[-1], source=down)   # última fila real hacia abajo, halo inferior en
#     step(s, snew); s, snew = snew, s
# total = comm.reduce(s[1:-1].sum(), op=MPI.SUM, root=0)            # un chequeo global, como nuestro array_equal
# ```
#
# El inicio depende del clúster (`mpirun -np 8 python stencil_mpi.py` dentro de una
# tarea). El código Python es el mismo con 2 procesos en una computadora o con 2000 en un supercomputador.
#
# ## 3.5 Tareas independientes: el otro tipo de paralelismo
#
# Un barrido de parámetros no necesita halos: cada simulación es independiente. La
# comunicación se limita al envío del parámetro y la recepción del resultado. Este
# patrón se adapta a planificadores de tareas como Dask:
#
# ```python
# from dask.distributed import Client
# client = Client()                               # trabajadores locales ahora; un cluster más tarde, mismo código
#
# def simulate(alpha):                            # una tarea completa e independiente
#     u = init_grid(512); ...; return alpha, float(u.mean())
#
# futures = client.map(simulate, np.linspace(0.05, 0.25, 32))
# results = client.gather(futures)
# ```
#
# Un entorno gratuito de Colab no está diseñado para ejecutar varios procesos de
# trabajo distribuidos. Use este apartado como material de lectura y consulte la guía
# del instructor para probarlo. Las tareas independientes se asignan a un planificador;
# las tareas acopladas requieren intercambio de mensajes.

# %%
# Patrón de tareas independientes sin un planificador, para mostrar la diferencia:
def simulate(alpha, n=128, iters=50):
    u = init_grid(n)
    for _ in range(iters):
        u[1:-1, 1:-1] += alpha * (u[:-2, 1:-1] + u[2:, 1:-1] + u[1:-1, :-2] + u[1:-1, 2:] - 4 * u[1:-1, 1:-1])
    return alpha, float(u.mean())

results = list(map(simulate, np.linspace(0.05, 0.25, 8)))   # replace map with client.map on a Dask cluster
print("bytes each task needs from any other task: 0")
print("\n".join(f"alpha={a:.3f} mean={m:.3f}" for a, m in results))

# %% [markdown]
# **Punto de control.** Para el primer trabajo de consola, no se debe mover ningún dato entre trabajadores; la fila adecuada en la tabla es **Tareas Independientes** (sweep, archivo, rama). Para el segundo trabajo de consola, se debe mover el dato entre trabajadores; la fila adecuada en la tabla es **Simulación estrechamente acoplada con halos o comunicación personalizada**.

# %% [markdown]
# # 0b. Ejemplo de ejecución: un cálculo por vecindad de difusión de calor (preparación, 10 a 15 minutos)
#
# Cada sesión del taller utiliza un pequeño programa, para que podamos dedicar
# el tiempo en vivo a su medición y aceleración, en lugar de explicarlo. Este
# notebook muestra lo que hace. No se asume ningún conocimiento de física o métodos numéricos.
# Se utiliza un entorno de ejecución estándar de CPU.
#
# Guarde una copia en Drive primero (**File > Guardar una copia en Drive**).

# %%
import numpy as np
import matplotlib.pyplot as plt

def init_grid(n):
    u = np.zeros((n, n))
    u[0, :] = 100.0          # the top edge is held at 100 degrees; everything else starts at 0
    return u

u = init_grid(8)
print(u)

# %% [markdown]
# ¿Qué hace una sola etapa?
#
# Imagina una placa metálica cuadrada. La parte superior está caliente. En cada etapa, cada celda interior toma el promedio de sus cuatro vecinos (arriba, abajo, izquierda, derecha). La calor se extiende hacia abajo desde la parte caliente, una fila por etapa al principio. Las celdas de la borda nunca cambian: son las condiciones de contorno.
#
# Esta es la misma regla escrita con bucles directos. Aunque su ejecución es lenta, permite comprobar cada operación y por eso se usa como referencia.

# %%
def step_python(u, unew):
    n, m = len(u), len(u[0])
    for i in range(1, n - 1):
        for j in range(1, m - 1):
            unew[i][j] = 0.25 * (u[i-1][j] + u[i+1][j] + u[i][j-1] + u[i][j+1])
    return unew

u = init_grid(8)
after_one = step_python(u.tolist(), u.tolist())
print(np.array(after_one))

# %% [markdown]
# ## Visualizarlo

# %%
def run_python(n, iters):
    u, unew = init_grid(n).tolist(), init_grid(n).tolist()
    for _ in range(iters):
        step_python(u, unew)
        u, unew = unew, u
    return np.array(u)

fig, axes = plt.subplots(1, 3, figsize=(10, 3.2))
for ax, iters in zip(axes, (0, 10, 100)):
    im = ax.imshow(run_python(40, iters), vmin=0, vmax=100, cmap="inferno")
    ax.set(title=f"after {iters} steps", xticks=[], yticks=[])
fig.colorbar(im, ax=axes, label="temperature"); plt.show()

# %% [markdown]
# ## Ejercicio: el mismo paso con slices de NumPy
#
# Los loops en Python son lentos. NumPy nos permite actualizar todas las celdas interiores de una vez con
# **slices**. `u[1:-1, 1:-1]` son todas las celdas interiores. Su vecino *arriba* es
# `u[:-2, 1:-1]` (las filas desplazadas hacia arriba por una unidad), su vecino *abajo* es `u[2:, 1:-1]`,
# y su vecino *izquierda* es `u[1:-1, :-2]`.
#
# **Completar el cuarto término**: el vecino de la *derecha*. Se debe sustituir
# el marcador y ejecutar la celda. (Pista: desplazar las columnas, no las filas.)

# %%
def step_numpy(u, unew):
    right = u[1:-1, 1:-1]          # TODO: replace with the slice of the right-hand neighbours
    unew[1:-1, 1:-1] = 0.25 * (u[:-2, 1:-1] + u[2:, 1:-1] + u[1:-1, :-2] + right)
    return unew

# %% [markdown]
# ## Comprobación con la referencia
#
# Se aplican dos comprobaciones. La primera verifica que la suma interior después de un
# paso en una cuadrícula de 8x8 sea `150.0` (seis celdas interiores de la fila 1,
# con un valor de 25 cada una). Esta prueba no detecta todos los errores porque casi
# todas las celdas siguen en cero. La segunda compara la cuadrícula completa después de
# 20 pasos con la versión basada en bucles.

# %%
u = init_grid(8)
one = step_numpy(u, u.copy())
print("interior sum after one step:", one[1:-1, 1:-1].sum(), "(reference: 150.0)")

def run_numpy(n, iters):
    u = init_grid(n); unew = u.copy()
    for _ in range(iters):
        step_numpy(u, unew)
        u, unew = unew, u
    return u

mine, reference = run_numpy(32, 20), run_python(32, 20)
if np.allclose(mine, reference):
    print("PASS: your NumPy step matches the reference after 20 steps")
else:
    bad = np.argwhere(~np.isclose(mine, reference))
    print(f"FAIL: {len(bad)} cells differ, first at (row, col) = {tuple(int(v) for v in bad[0])}. "
          "Check which neighbour your fourth slice really selects.")

# %% [markdown]
# ## Qué recordar para la sesión
#
# - `init_grid(n)`: Crea una cuadrícula `n x n`, con un borde caliente en la parte superior y ceros en el resto.
# - `step_*(u, unew)`: Lee el contenido de `u`, escribe en `unew`; el llamador intercambia estos en cada paso.
# - La versión en Python puro es el punto de referencia. Cada versión más rápida debe coincidir con ella,
#   y la comprobaremos antes de medir cualquier cosa.
#
# Guarde el notebook. Luego lea `prep/self_check.md`.

# Revisión opcional: el taller de Python y NumPy que se utiliza

Puede omitir este material si programa habitualmente en Python. De lo contrario,
conviene leerlo antes del material introductorio.
Sólo cubre lo que usan los cuadernos del taller. Intenta cada snippet en una celda de Colab.

## Python

```python
def average_of_four(a, b, c, d):  # una función con cuatro argumentos
    return 0.25 * (a + b + c + d)

for i in range(1, 5):  # 1, 2, 3, 4 (el final está excluido)
    print(i, average_of_four(i, i, i, i))

cuadrícula = [[0.0] * 4 for _ in range(3)]  # una lista de 3 listas de 4 ceros
cuadrícula[0][1] = 100.0  # fila 0, columna 1
print(len(cuadrícula), len(cuadrícula[0]))  # 3 4

a, b = b, a  # intercambia dos nombres; los cuadernos lo hacen cada paso
```

`import time; t0 = time.perf_counter(); ...; elapsed = time.perf_counter() - t0`
es cómo los cuadernos miden los segundos de reloj.

## NumPy

```python
import numpy as np
u = np.zeros((4, 6))  # 4 filas, 6 columnas, float64
u[0, :] = 100.0  # toda la primera fila
print(u.shape, u.dtype, u.nbytes)  # (4, 6) float64 192

v = u[1:-1, 1:-1]  # una vista de la interior: filas 1..2, columnas 1..4
v[:] = 5.0  # escriba en u también
print(u)

w = u.copy()  # una copia independiente
w[:] = 0  # u no se ve afectada

a = np.arange(6).reshape(2, 3)  # [[0 1 2], [3 4 5]]
print(a[:, 1:], a[:, :-1])  # elimina la primera columna / elimina la última columna
print(a + a, a * 0.5, a.sum(), a.mean())
print(np.allclose(a * 0.5 * 2, a))  # True: compare floats con una tolerancia
```

Cosas que notar:

- El índice es `[fila, columna]`, ambos comenzando en 0; los índices negativos cuentan desde el final.
- Una vista `inicio:fin` excluye `fin`; `:-1` significa "todos menos el último".
- La aritmética entre arrays de la misma forma es elemento por elemento y produce un nuevo array.
- `astype(np.float32)` convierte el tipo de dato; `float32` usa la mitad de bytes de `float64`.

## Jupyter y Colab

- **Shift+Enter** ejecuta una celda y mueve a la siguiente.
- Una celda que comienza con `%timeit` tiembla varias veces la línea que sigue.
- Las variables definidas en una celda se disponen en las celdas posteriores, hasta que reinicie el
  entorno de ejecución. Después de una reinicialización, ejecute desde el principio.

---
marp: true
paginate: true
---

# Programación Paralela Moderna en Python para HPC y AI

Medida. Vectorización. Compilación. Luego, y solo entonces, paralelización.

Tres horas, un ejemplo en ejecución, todo en Colab.

---

# Resultados

1. Establece un resultado de referencia correcto y mide repetidas ejecuciones de manera justa.
2. Compara Python, NumPy y compilado Numba; explica por qué más hilos no ayudarían.
3. Ejecuta una operación de array con CuPy, comprueba que funciona y separa el tiempo de cálculo del tiempo incluido en transferencia.
4. Explica cuando los hilos separados deben comunicarse y elige un administrador de tareas o comunicación de mensajes según sea necesario.

No se busca un aceleramiento específico. Resultados correctos, explicados.

---

# Ejemplo de ejecución: un stencil de difusión de calor

- Una cuadrícula `n x n`, la parte superior fija en 100, las demás en 0.
- Cada paso: cada celda interior se convierte en la media de sus cuatro vecinos.
- Las bordes nunca cambian: son las condiciones de frontera.
- El bucle de Python estándar es el **referencia**. Cualquier versión más rápida debe coincidir con él.

```python
unew[1:-1, 1:-1] = 0.25 * (u[:-2, 1:-1] + u[2:, 1:-1] + u[1:-1, :-2] + u[1:-1, 2:])
```

Las cortesías son vistas. Cada `+` asigna un array intermedio.

---

# ¿Qué contiene un tiempo de ejecución

- **Calentamiento** primero: compilación, importaciones, cachés, asignaciones.
- **Repetir**; reporta el mínimo (menos interferencia) o el mediano (típico). Decile qué.
- `%timeit` imprime **promedio y desviación estándar** entre ejecuciones; `.best` a petición.
- **Mismo experimento**: mismo tiempo de ejecución, carga de trabajo, tipo de dato, pasos.
- En un GPU: **sincronizar** antes de comenzar y antes de detener el cronómetro.

Un número de otra máquina es un experimento diferente.

---

# Ley de Amdahl

    speedup(p) = 1 / (s + (1 - s) / p)

10% serial, 4 núcleos: 3.1x. 16 núcleos: 6.4x. Sin límite: 10x.

La fracción serial establece el techo. En la práctica, el techo llega antes:
la banda ancha de memoria, la coordinación, los núcleos compartidos.

Para nuestro stencil, una mala ajuste a Amdahl suele ser **la banda ancha de memoria**, no
el código serial.

---

# Memoria del host y memoria del dispositivo

```
   CPU  <-- RAM -->              PCIe (tens of GB/s)            <-- VRAM --> GPU
                    cp.asarray(x)  ------------------------->
                    x.get()        <-------------------------
```

- Mover datos una vez, hacer mucho cálculo, mover de vuelta una vez.
- `cp.asarray` dentro de un bucle suele ser más lento que NumPy.
- Reporta **tiempo de cálculo solo** y **incluido transferencia** separadamente.

---

# Escala de tiempo, en una sola diapositiva

| escala | incluye |
|---|---|
| tiempo de cálculo solo | cálculos en datos ya en el dispositivo, sincronización en ambos extremos |
| incluido transferencia | subida, cálculos, bajada, sincronización en ambos extremos |
| equivocado | lanzamiento solo, no sincronización |

"40x más rápido": sincronización? transferencias dentro? mismo carga de trabajo y tipo de dato?

---

# Particiones y halos

```
   trabajador 0            trabajador 1            trabajador 2
+-----------+       +-----------+       +-----------+
| halo (arriba)|       | halo      | <---- | último real |
| real filas |       | real filas |       | real filas |
| real filas | ----> | halo      |       | ...       |
| halo (abajo)| <---- | primer real|       |           |
+-----------+       +-----------+       +-----------+
```

- Cada paso: dos filas por cada borde interno, una en cada dirección.
- Comunicación / cálculo por trabajador: `2P / n`. Aumenta con P, disminuye con n.
- Además, una latencia fija por mensaje. Problemas pequeños no escalan entre máquinas.

---

# Dos modelos de programación

- **Comunicación por mensajes (MPI, `mpi4py`)**: cada rango ejecuta el mismo script; escribiste las mensajería. Simulaciones enlazadas, halos, comunicación personalizada.
- **Administración de tareas (Dask, arrays de tareas)**: tareas independientes, resultados recopilados.
  Sweep de parámetros, archivos, faldas; datos más grandes que una máquina.

El código de Python es el mismo para 2 trabajadores y 2000.

---

# Tabla de decisiones

| Tienes | Llama a |
|---|---|
| Recorres los elementos de un array | NumPy primero, luego Numba `@njit` |
| El mismo bucle, varios núcleos | Numba `parallel=True` + `prange`, hilos <= núcleos |
| Millones de operaciones de elementos idénticos e independientes | CuPy, datos mantenidos en el dispositivo |
| Muchas tareas independientes Python | procesos, `joblib`, Dask |
| Tareas independientes entre máquinas | administrador de tareas (Dask, arrays de tareas) |
| Simulación enlazada entre máquinas | `mpi4py` (+ Numba o CuPy por rango) |
| Cualquier afirmación de rendimiento | medir antes y después, mismo tiempo de ejecución |

---

# Cuando más hardware no ayuda

- El trabajo de carga es demasiado pequeño: el lanzamiento o el inicio de hilos domina.
- Los datos cruzan la link de PCIe en cada iteración.
- El kernel está limitado por la memoria y la pista ya está saturada.
- Una fracción serial establece el aceleramiento (Amdahl).
- La descomposición está limitada por la comunicación (`2P/n` grande, la latencia domina).

Encuentra qué uno **antes** de pedir más máquinas.

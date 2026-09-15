# Capstone: qué debe tener una respuesta completa

La criterización no es el acelerado. Una respuesta completa tiene, para cada
carga de trabajo, una comprobación de correctitud aprobada, una tabla de tiempos
etiquetada con el tiempo de ejecución, y una recomendación que cita los números del
estudiante. A continuación se ven los resultados observados en la validación y la
razonamiento que debería acompañarlos.
Los números de un tiempo de ejecución Colab variarán; la *forma* del argumento no
debe.

## Carga de trabajo A: 48 grids independientes de lado 128, 30 pasos

Se observó en el máquina del autor (números de `data/reference_timings/04_capstone.csv`):

| candidato | tiempo |
|---|---:|
| numpy | ~44 ms |
| numba | ~5 ms |
| numba_par (32 hilos) | ~103 ms |

Razónamiento esperado:

- `numba` supera a `numpy` porque una sola función compilada reemplaza cuatro
  arrays intermedios por paso; en un grid de 128x128, esos intermedios dominan.
- `numba_par` es **más lento** que el Numba serial: cada grid es muy pequeña, así
  que despertar y sincronizar un pool de hilos 30 veces por grid cuesta más que el
  trabajo. Los hilos fueron puesto *dentro* del bucle incorrecto.
- El mejor estructura paralela para este carga de trabajo es *alrededor* de grids
  (una tarea por grid), porque los grids intercambian cero bytes. En un cluster
  que es una tarea scheduler (Dask, un array de tareas), no es el intercambio de
  mensajes. Un estudiante que añade una candidata que `prange` sobre los 48 grids
  dentro de una función `@njit(parallel=True)` tiene entendido esto; en un
  Colab de 2 CPUs mostrará un aumento modesto, que es propio de notar.

## Carga de trabajo B: una grid de lado 2048, 40 pasos

Se observó en la máquina del autor:

| candidato | alcance | tiempo |
|---|---|---:|
| numpy | cálculo | ~470 ms |
| numba_par | cálculo | ~30 ms |
| cupy | cálculo, datos creados en el dispositivo | ~10 ms |
| cupy | incluye transferencia (subida una vez, bajada una vez) | ~17 ms |

Razónamiento esperado:

- Un grid grande y unido es el caso donde los hilos dentro del grid pagan y donde
  el GPU tiene suficiente trabajo por cada lanzamiento de kernel.
- Las transferencias fueron aproximadamente tan costosas como el cálculo mismo
  aquí (subida y bajada de 7 ms para 32 MB en cada dirección). Si eso cambia la
  decisión depende de cuántas veces el resultado debe regresar: cada 40 pasos, el
  GPU aún gana en este hardware; cada paso, no lo haría.
- Sin un GPU, la recomendación honesta es `numba_par`, con la tabla de GPU
  registrada como evidencia sobre *otro* hardware.
- En un cluster, este carga de trabajo se mapea a intercambio de mensajes con
  el halo intercambio (dos filas por vecino por paso), no a un scheduler de tareas.

## "Una cosa más: el hardware no lo arregla"

Cualquiera de: la carga de trabajo pequeña de grid (el overhead de lanzamiento y
hilos no es el límite del cálculo); un bucle que copia datos al GPU cada paso; un
kernel que es límite de memoria que ya está saturando la banda ancha; una fracción
serial que Amdahl's law cuelga.

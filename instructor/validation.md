# Validación y pasos de lanzamiento

Hay dos tipos de comprobaciones. `tools/validate.sh` es automático y se ejecuta en una
máquina del autor; regenera cada `.ipynb` a partir de su fuente `.py` y ejecuta los
carnes de caja en modo CPU y, si está presente una tarjeta gráfica NVIDIA, en modo GPU. Las
comprobaciones manuales siguientes se ejecutan en Google Colab y no pueden ser scripteadas,
porque el hardware, las quotas y las imágenes de paquetes de Colab cambian.

## Comprobaciones automáticas (máquina del autor)

```bash
uv sync --project env --extra gpu       # una vez; sin --extra gpu sin una tarjeta GPU
GPU_EXTRA=1 tools/validate.sh            # regenera y ejecuta los notebooks principales
uv run --project env python tools/dump_outputs.py validation/gpu/02_gpu.ipynb   # revise los resultados
```

El paso significa: cada caja de caja crítica (00a, 00b y su solución, 01, 02, 03, 04)
termina sin error en ambos modos y `git status` muestra ningún cambio inesperado en
`.ipynb` (fuente y caja de caja sincronizados). Las extensiones se regeneran pero no se
ejecutan; ejecútalas por mano con `EXTRA_NOTEBOOKS="extensions/ext_gil_and_task_pools.ipynb"`.

La última comprobación automática en la máquina del autor: 13 de septiembre de 2026, 14 de 14
comprobaciones pasaron (CPU de 32 núcleos, RTX 4090, numpy 2.5.3, numba 0.67.0, cupy 14.2.0,
Python 3.12.14).

## Comprobaciones manuales en Google Colab

Realiza estas con una **cuenta de estudiante de Google**, no la del autor, desde
las enlaces en el README. Registra la fecha, el tipo de ejecución de la máquina, el
modelo de tarjeta gráfica (si hay) y las versiones de paquete impresas por cada
celda de configuración.

### Entorno de ejecución de CPU fresco

1. Abra `00_colab_ready` desde el enlace del README. Guarde una copia. Ejecute todo.
   La última celda dice `READY for IEEE Latam Tech`. Reinicie la sesión y ejecute todo:
   sigue `READY`. Desconecte y borre la ejecución, ejecute todo: las paquetes se
   reinstalan, sigue `READY`. Nota el tiempo que tardó la instalación.
2. Abra `00_stencil_practice`. Ejecute todo: la celda de comprobación imprime `FAIL`
   con el consejo (el placeholder aún presente). Corrige el corte, vuelva a ejecutar:
   `PASS`.
3. Abra `01_measure_and_multicore`. Ejecute todo. Nota los contadores de hilos
   probados y si el recorrido registrado se cargó. Cada celda bajo los 30 segundos.
   Total de cálculo bajo los 5 minutos.
4. Abra `02_gpu` en el **entorno de ejecución de CPU**. Ejecute todo. El banner dice
   `CPU fallback`; la línea `RECORDED GPU RUN` muestra el metadata intencionado (después
   de la puerta 4, una tarjeta GPU de Colab); la gráfica muestra las curvas CPU y
   recorridas en vivo.
5. Abra `03_parallel_models` y `04_capstone`. Ejecute todo. Las filas de GPU de
   `capstone` están ausentes, todo lo demás está completo.

### Entorno de ejecución de GPU

6. Cambie el tipo de ejecución a GPU. Abra `02_gpu`. Ejecute todo. El banner dice
   `MODE: GPU` con el nombre del dispositivo. CuPy importa sin instalar (nota si pip
   corrió). La celda de sincronización/no-sincronización muestra la diferencia esperada.
   Todas las celdas bajo los 30 segundos.
7. Abra `04_capstone` en la misma ejecución. Ejecute todo. Cuatro candidatos, ambas
   celdas de CuPy están presentes.
8. Descargue `timings_01_cpu.csv` (del entorno de ejecución de CPU) y `timings_02_gpu.csv`
   (de esta ejecución). Copie en `data/reference_timings/` como `01_threads.csv` y
   `02_gpu.csv`. Comitee y pushe. Repita la comprobación 4 y confirme que el metadata
   recorrido ahora nombró Colab y la tarjeta GPU.
9. Opcionalmente las extensiones en el entorno de ejecución de GPU: `ext_cuda_kernel`
   (instala y ejecuta `numba-cuda`?), `ext_gil_and_task_pools` (CPU o GPU). Si alguna
   falla, el README sigue llamándolas opcionales y el show de show no menciona nada más que
   "disponible".

### Comprobación de presupuesto de tiempo (rehechura, puerta 5)

Ejecute el programa completo con un reloj, incluyendo el cambio de entorno a 01:20 y el
tiempo del ejercicio del estudiante. Registre los marcos de tiempo reales en este archivo.
Confirme que ambos cortes aún son suficientes para recuperar 10 minutos.

## Pasos de lanzamiento

| Puerta | Cuando | Pasa cuando |
|---|---|---|
| 1 Especie y estructura | hecho | la agenda suma 180 minutos; los cajones están nombrados; el README es la página de llegada |
| 2 Ejecución de Colab fiable | hecho en la máquina del autor; Colab pendiente | la comprobación automática pasa en ambos modos; no hay ruta estudiantil activa que necesite un cluster |
| 3 Secuencia de aprendizaje | hecho | el primer plano, la práctica, el auto-comprobación, los puntos de control, el Proyecto final y las soluciones existen |
| 4 Preparación de la comprobación | T-7 días | las comprobaciones 1 a 8 pasan; los archivos CSV de referencia se reemplazan por los datos de Colab; cada enlace de Colab abra con una cuenta estudiante |
| 5 Rehechura | T-2 días | la comprobación de rehechura dentro del presupuesto; las diapositivas, las notas y la grabación están completas; se envía un recordatorio por parte de los organizadores |
| 6 Comprobación final | T-1 día | se repiten las comprobaciones 1, 4 y 6; se prueba la revisión y se registran las versiones |

Revisión probada para el evento: *(reemplace: git commit, fecha, versiones de la ejecución de Colab, modelo de tarjeta GPU)*.

## Gaps conocidos al momento de escribir

- `data/reference_timings/*.csv` provienen de la máquina del autor, no de Colab
  (se les etiquetaron como tales en su metadata). La puerta 4 los reemplace.
- No se ha producido la grabación de la demostración de GPU.
- Las celdas de extensión no se ejecutaron en Colab; `ext_cuda_kernel` depende de que
  el paquete `numba-cuda` se instale en la imagen de Colab.
- Las mediciones en el show de show ("resultado típico de Colab") son expectativas que
  deben corregirse a partir de las mediciones de Colab.

# Programación paralela moderna en Python para HPC e IA

Taller práctico de tres horas sobre cómo medir y acelerar código Python en CPU
multinúcleo y GPU, y sobre los cambios necesarios cuando el trabajo abarca varias
máquinas (temas 3, 5, 7, 8 y 9 del taller IEEE Latam). **Todo se ejecuta en
Google Colab.** No se requieren una cuenta de clúster, SSH ni instalación local.

## 1. Antes del evento: preparación (45 a 60 minutos, tiempo de ejecución en CPU)

Se recomienda completar esta preparación al menos dos días antes del taller, en
una computadora con navegador y una cuenta de Google. El punto de partida es
[`prep/README.md`](prep/README.md), que enlaza los materiales en orden.

| Paso | Tiempo | Abrir |
|---|---:|---|
| Leer el material introductorio | 15 min | [`prep/primer.md`](prep/primer.md) |
| ¿Está mi Colab listo? | 10-15 min | [Abrir en Colab](https://colab.research.google.com/github/caschb/IEEELatamTechRepo/blob/main/notebooks/00_colab_ready.ipynb) |
| El ejemplo de práctica | 10-15 min | [Abrir en Colab](https://colab.research.google.com/github/caschb/IEEELatamTechRepo/blob/main/notebooks/00_stencil_practice.ipynb) |
| Autocomprobación | 5-10 min | [`prep/self_check.md`](prep/self_check.md) |

Solo se requiere completar la verificación de **CPU**. No es necesario disponer
de una GPU antes del evento. Si se necesita repasar Python o NumPy, conviene
consultar primero el [repaso opcional](prep/python_numpy_refresher.md).

## 2. Agenda en vivo (180 minutos)

| Tiempo | Min | Bloque | Notebook |
|---|---:|---|---|
| 00:00 | 10 | Bienvenida, objetivos, chequeo de preparación, recapitulación del stencil | [`00_stencil_practice`](https://colab.research.google.com/github/caschb/IEEELatamTechRepo/blob/main/notebooks/00_stencil_practice.ipynb) (solo recapitulación) |
| 00:10 | 25 | Medición: tiempos repetidos, calentamiento, demostración de perfilado, base de NumPy | [`01_measure_and_multicore`](https://colab.research.google.com/github/caschb/IEEELatamTechRepo/blob/main/notebooks/01_measure_and_multicore.ipynb) |
| 00:35 | 35 | Numba, `prange`, comparación de hilos, Amdahl | mismo notebook |
| 01:10 | 10 | Descanso (guardar el trabajo) | |
| 01:20 | 10 | Cambio a un runtime de GPU, configuración, verificación del dispositivo | [`02_gpu`](https://colab.research.google.com/github/caschb/IEEELatamTechRepo/blob/main/notebooks/02_gpu.ipynb) |
| 01:30 | 40 | Stencil con CuPy, sincronización, barrido de tamaños, transferencias, precisión | mismo notebook |
| 02:10 | 20 | Más allá de una máquina: particiones, halos, comunicación, roles de MPI y Dask | [`03_parallel_models`](https://colab.research.google.com/github/caschb/IEEELatamTechRepo/blob/main/notebooks/03_parallel_models.ipynb) |
| 02:30 | 20 | Proyecto final: elegir y justificar una implementación | [`04_capstone`](https://colab.research.google.com/github/caschb/IEEELatamTechRepo/blob/main/notebooks/04_capstone.ipynb) |
| 02:50 | 10 | Revisión, preguntas, pasos siguientes | |

Cada notebook es autónomo: su primera celda de código instala lo que falta y debe
ejecutarse de nuevo después de reiniciar el runtime. Los tiempos se comparan
**dentro de un mismo runtime**; un valor de otra máquina corresponde a otro experimento.

**¿No hay GPU disponible?** Se puede continuar en el runtime de CPU. El notebook
02 lo detecta, ejecuta las celdas de CPU y muestra una tabla registrada en GPU,
identificada con el hardware correspondiente. También hay una grabación de la
demostración en GPU.

## 3. Después del evento

- Diapositivas: [`instructor/slides.md`](instructor/slides.md)
- Extensiones opcionales (no se cubren en vivo):
  [hilos, procesos y el GIL](https://colab.research.google.com/github/caschb/IEEELatamTechRepo/blob/main/extensions/ext_gil_and_task_pools.ipynb) y, en un runtime de GPU, [escribir un kernel CUDA](https://colab.research.google.com/github/caschb/IEEELatamTechRepo/blob/main/extensions/ext_cuda_kernel.ipynb)
- Guía de lectura para MPI y Dask más allá de una máquina: [`instructor/reading_guide_mpi_dask.md`](instructor/reading_guide_mpi_dask.md)
- Soluciones: [`solutions/`](solutions/)

## Requisitos previos

Python básico (funciones, bucles, listas), una cuenta de Google y un navegador. Todo lo demás se instala por los notebooks.

## Para instructores y autores

| Ruta | Qué |
|---|---|
| `instructor/run_of_show.md` | Marcas de tiempo, preguntas, errores frecuentes, respuestas, triaje de configuración y dos recortes previstos |
| `instructor/validation.md` | Criterios de publicación y verificaciones manuales de Colab que el script no realiza |
| `notebooks/*.py` | Fuentes de los notebooks en formato porcentual de Jupytext; a partir de ellas se generan los archivos `.ipynb` |
| `tools/validate.sh` | Regenera cada notebook y ejecuta los principales en modo fallback de CPU y en modo GPU |
| `env/` | Entorno de autoría (`uv`) y lista mínima de paquetes para Colab |
| `data/reference_timings/` | Tablas de tiempos registradas con metadatos del runtime, utilizadas por el fallback de CPU |
| `archive/kabre/` | Edición anterior para clúster (SLURM, MPI, OnDemand); es histórica y no recibe mantenimiento |

Ciclo de autoría:

```bash
uv sync --project env            # agregar --extra gpu en una máquina con GPU NVIDIA
GPU_EXTRA=1 tools/validate.sh    # regenerar los .ipynb desde .py y ejecutar los notebooks principales
```

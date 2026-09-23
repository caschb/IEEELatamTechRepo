# Programación paralela moderna en Python para HPC e IA

Taller práctico de tres horas sobre cómo medir y acelerar código Python en CPU
multinúcleo y GPU, y sobre los cambios necesarios cuando el trabajo abarca varias
máquinas (temas 3, 5, 7, 8 y 9 de IEEE Latam Tech). Todo se ejecuta en
Google Colab. No se requieren una cuenta de clúster, SSH ni instalación local.

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
| 00:00 | 10 | Bienvenida, objetivos, chequeo de preparación, recapitulación del cálculo por vecindad | [`00_stencil_practice`](https://colab.research.google.com/github/caschb/IEEELatamTechRepo/blob/main/notebooks/00_stencil_practice.ipynb) (solo recapitulación) |
| 00:10 | 25 | Medición: tiempos repetidos, calentamiento, demostración de perfilado, base de NumPy | [`01_measure_and_multicore`](https://colab.research.google.com/github/caschb/IEEELatamTechRepo/blob/main/notebooks/01_measure_and_multicore.ipynb) |
| 00:35 | 35 | Numba, `prange`, comparación de hilos, Amdahl | mismo notebook |
| 01:10 | 10 | Descanso (guardar el trabajo) | |
| 01:20 | 5 | Cambio a un entorno de ejecución de GPU, configuración, verificación del dispositivo | [`02_gpu`](https://colab.research.google.com/github/caschb/IEEELatamTechRepo/blob/main/notebooks/02_gpu.ipynb) |
| 01:25 | 20 | Cálculo por vecindad con CuPy, sincronización, barrido de tamaños, transferencias, precisión | mismo notebook |
| 01:45 | 20 | Más allá de una máquina: particiones, halos, comunicación, roles de MPI y Dask | [`03_parallel_models`](https://colab.research.google.com/github/caschb/IEEELatamTechRepo/blob/main/notebooks/03_parallel_models.ipynb) |
| 02:05 | 20 | Proyecto final: elegir y justificar una implementación | [`04_capstone`](https://colab.research.google.com/github/caschb/IEEELatamTechRepo/blob/main/notebooks/04_capstone.ipynb) |
| 02:25 | 10 | Revisión, preguntas, pasos siguientes | |
| 02:25 | 15 | marimo Notebook EDA + ML | [`05_marimo`](https://molab.marimo.io/notebooks/nb_uXzyncAb9w2ZT27r2FuLdN) |

Cada notebook es autónomo: su primera celda de código instala lo que falta y debe
ejecutarse de nuevo después de reiniciar el entorno de ejecución. Los tiempos se comparan
**dentro de un mismo entorno de ejecución**; un valor de otra máquina corresponde a otro experimento.

**¿No hay GPU disponible?** Se puede continuar en el entorno de ejecución de CPU. El notebook
02 lo detecta, ejecuta las celdas de CPU y muestra una tabla registrada en GPU,
identificada con el hardware correspondiente. También hay una grabación de la
demostración en GPU.

## 3. Después del evento

- Diapositivas: [`instructor/slides.md`](instructor/slides.md)
- Secciones 2.7 a 2.9 del notebook `02_gpu` (no se cubren en vivo): latencia de PCIe, fusión de kernels con `cp.RawKernel` y el modelo roofline
- Extensiones opcionales (no se cubren en vivo):
  [hilos, procesos y el GIL](https://colab.research.google.com/github/caschb/IEEELatamTechRepo/blob/main/extensions/ext_gil_and_task_pools.ipynb) y, en un entorno de ejecución de GPU, [escribir un kernel CUDA](https://colab.research.google.com/github/caschb/IEEELatamTechRepo/blob/main/extensions/ext_cuda_kernel.ipynb)
- Guía de lectura para MPI y Dask más allá de una máquina: [`instructor/reading_guide_mpi_dask.md`](instructor/reading_guide_mpi_dask.md)
- Soluciones: [`solutions/`](solutions/)

### 4. marimo y molab

El cuaderno 05 **no es un notebook de Jupyter**: es una aplicación
[marimo](https://marimo.io). Se ejecuta en su propio entorno y no se abre en Colab como
los demás.

| Propiedad | Qué implica |
|---|---|
| Grafo reactivo | Al mover un control, todas las celdas que dependen de él se recalculan solas |
| Sin estado oculto | Al borrar una celda desaparecen sus variables; no sobrevive un modelo obsoleto |
| Orden determinista | Las celdas corren en orden de dependencias, no en el orden en que se hizo clic |
| Se guarda como `.py` | El archivo es un módulo de Python: diffs limpios en git |

### Ejecutarlo en molab

[molab](https://molab.marimo.io) es el servicio de cuadernos alojados de marimo, el
equivalente a Colab para este formato. Es la vía más rápida si no se desea instalar nada:

El cuaderno ya está publicado, así que basta con abrirlo:

**[Abrir el cuaderno 05 en molab](https://molab.marimo.io/notebooks/nb_uXzyncAb9w2ZT27r2FuLdN)**

1. Abrir el enlace e iniciar sesión.
2. Crear una copia propia para poder editarla y guardar los cambios.
3. Ejecutar. La primera celda instala lo que falte.

Para partir del archivo del repositorio en lugar del cuaderno publicado (por ejemplo,
tras modificarlo), se crea un cuaderno nuevo en [molab.marimo.io](https://molab.marimo.io)
y se sube `notebooks/05_polars_dataframes.py`.

## Requisitos previos

Python básico (funciones, bucles, listas), una cuenta de Google y un navegador. Todo lo demás se instala por los notebooks.

## Para instructores y autores

| Ruta | Qué |
|---|---|
| `instructor/run_of_show.md` | Marcas de tiempo, preguntas, errores frecuentes, respuestas, triaje de configuración y dos recortes previstos |
| `instructor/validation.md` | Criterios de publicación y verificaciones manuales de Colab que el script no realiza |
| `notebooks/*.py` | Fuentes de los notebooks en formato porcentual de Jupytext; a partir de ellas se generan los archivos `.ipynb` |
| `notebooks/05_polars_dataframes.py` | Cuaderno 05: aplicación **marimo**, no Jupytext. No se convierte a `.ipynb`; se valida con `marimo export html` |
| `tools/validate.sh` | Regenera cada notebook y ejecuta los principales en modo alternativo de CPU y en modo GPU |
| `env/` | Entorno de autoría (`uv`) y lista mínima de paquetes para Colab |
| `data/reference_timings/` | Tablas de tiempos registradas con metadatos del entorno de ejecución, utilizadas por el modo alternativo de CPU |

Ciclo de autoría:

```bash
uv sync --project env            # agregar --extra gpu en una máquina con GPU NVIDIA
```

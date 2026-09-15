# Guía de lectura: MPI y Dask fuera de un solo sistema

Notebook 03 enseña los conceptos con un modelo de NumPy dentro de un solo entorno de ejecución. Esta guía está dirigida a estudiantes (y instructores) que quieren ejecutar lo real. Ninguno de ello es necesario para el taller, y ninguno de ello corre en un entorno Colab gratuito, que no está diseñado para hospedar trabajadores distribuidos.

## Comunicación con mpi4py

- Conceptos que deben estar claros desde el Notebook 03: rangos, `comm.rank` y `comm.size`, comunicación punto a punto (`Sendrecv`) versus colectiva (`reduce`, `bcast`, `gather`), halos, y la relación de comunicación a procesamiento de `2P/n`.
- El tutorial de mpi4py (https://mpi4py.readthedocs.io/en/stable/tutorial.html) cubre exactamente las llamadas de `Sendrecv` y `reduce` mostradas en el apartado 3.4. Lee "Punto a Punto" y "Colectiva" y nota la diferencia entre los métodos en minúsculas (objetos pickled) y mayúsculas (buffers, arrays NumPy). El stencil utiliza los métodos en mayúsculas.
- Para probarlo en un portátil: `pip install mpi4py` necesita una biblioteca MPI (en Linux `apt install libopenmpi-dev`, en macOS `brew install open-mpi`); luego `mpirun -np 4 python stencil_mpi.py`. Cuatro rangos en un portátil ya muestran el argumento de correctitud (calor total independiente del número de rangos), pero no la velocidad.
- En un clúster, el lanzamiento se define mediante un script de trabajo. La
  documentación de cada sistema determina el lanzador, las opciones y los módulos
  disponibles.
- Paralelismo híbrido: una hilera de Numba por rangos, tamaño a las hilas que el rangos recibió (`NUMBA_NUM_THREADS`), o una GPU por rangos con CuPy. El código dentro de un rangos es el de los Notebooks 1 y 2, sin cambios.

## Planificación de tareas con Dask

- Conceptos del Notebook 03: tareas independientes intercambian cero bytes entre sí; el tráfico es solo los argumentos entrantes y los resultados salientes.
- Empieza con `dask.distributed` en tu propio sistema: `pip install "dask[distributed]"`, luego `Client()` sin argumentos inicia trabajadores locales. Las llamadas `client.map` y `client.gather` del apartado 3.5 corren como se escriben. La URL del cuadro de estado que el cliente imprime muestra tareas moviéndose entre trabajadores.
- El mismo código en un cluster: `dask-jobqueue` (https://jobqueue.dask.org) crea trabajadores como trabajos de lotes (`SLURMCluster`, `PBSCluster`, ...). Lee "Cómo funciona esto" y "Configuración" antes de la API. La página de Dask "Despliegue de Clusters Dask" lista las otras opciones (Kubernetes, nube).
- Datos más grandes que la RAM: `dask.array` y `dask.dataframe` descompone el trabajo de NumPy y pandas a través de los trabajadores con la misma API. Prueba `dask.array.zeros((20000, 20000), chunks=(2000, 2000))` y los cortes del stencil sobre él; el gráfico que construye es el intercambio de halos del Notebook 03, hecho por ti.

## ¿Qué elegir?

| Pregunta | Sí | No |
|---|---|---|
| Los tareas necesitan los datos de las otras tareas durante el cálculo? | Comunicación punto a punto | Planificador de tareas |
| Un array o tabla excede la RAM de un solo sistema? | `dask.array` / `dask.dataframe` (o MPI con descomposición explícita) | Manténlo en un solo sistema |
| El trabajo por tarea es menos de un milisegundo? | Agrupa las tareas en lotes primero; ninguno de los dos no ayuda con ese detalle | Bien |
| El código es un modelo de aprendizaje profundo? | PyTorch DDP / NCCL, no MPI directamente | Como arriba |

## ¿Qué medir antes de escalar

1. Las bases de cálculo serial y en un solo sistema del Notebook 1 y 2.
2. La fracción de tiempo en comunicación con un número pequeño de trabajadores. Si ya es grande, más trabajadores lo hacen peor.
3. Si el resultado es idéntico para diferentes números de trabajadores (el `array_equal` del Notebook 03, o una cantidad conservada). Un resultado de escalado sin ese chequeo no es un resultado.

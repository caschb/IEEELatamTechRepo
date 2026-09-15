# Archivo histórico: Cluster de Kabré edición

Todo lo que se encuentra en esta carpeta pertenecía a la primera edición del curso,
que se ejecutó en el cluster Kabré en CeNAT (Jupyter a través de OnDemand en un
`nukwa-l40s` nodo, trabajos MPI a través de SLURM). Se mantiene para referencia solo y
no forma parte del curso activo. Nada aquí ha sido validado contra los notebooks actuales
o el actual `env/pyproject.toml`.

| Ruta | ¿Qué era |
|---|---|
| `env/setup_kabre.sh` | Construyó el entorno de venv uv contra los módulos de MPI y CUDA del cluster y registró un kernel de Jupyter |
| `env/smoke.py`, `env/run_smoke.sh`, `env/run_smoke.sbatch` | Prueba de gasolina del entorno venv en un nodo con GPU |
| `env/uv.lock` | Archivo de bloqueo del entorno del cluster (mpi4py, dask-jobqueue, vinculaciones de CUDA 12) |
| `mpi/` | Trabajo MPI4PY de dos nodos standalone y un diagnóstico anidado de `sbatch` |
| `notebooks/check_env.py`, `notebooks/00_check_environment.ipynb` | Revisión de la llegada de kernel, GPU, MPI y SLURM |
| `notebooks/03_multinode.py`, `.ipynb` | Estilo de halo de MPI4PY y `SLURMCluster` de dask-jobqueue, enviados desde el cuaderno |
| `notebooks/execute.sh` | Ejecución sin cabeza de cuadernos en una asignación |

El curso activo se ejecuta completamente en Google Colab; consulte el README del repositorio.
Las procedimientos específicos del cluster (nombres de módulos, `mpirun` flags, la
solución de `orte_keep_fqdn_hostnames` y la submisión desde un entorno limpiado) se mantienen en los archivos
arriba y en la historia de git.

Note: Los enlaces dentro de los cuadernos no se traducen.

# %% [markdown]
# # 3. Más allá de un nodo
#
# ## 7. Un nodo tiene decenas de núcleos, una GPU y unas cuantas cientos de GB. Cuando eso no es suficiente, el cluster tiene más nodos, y nada cambia en Python excepto una cosa: **los nodos no comparten memoria**. Cada byte que otro nodo necesita tiene que ser enviado por el rednece.
#
# Dos formas de programar eso:
#
# - **MPI** (`mpi4py`): mensajes explícitos. Más rápido, más control, más código. La lengua franca de la HPC; cada código de simulación habla de ella.
# - **Dask**: un agente de programación que distribuye tareas a los hilos en otros nodos. Poca código, excelente para tareas paralelas y para datos más grandes que un nodo.
#
# Ambos necesitan SLURM para darte los nodos. Este cuaderno corre en un solo nodo; los trabajos que envía corren en otros.

# %%
import os, subprocess, time, textwrap
from pathlib import Path

WORK = Path.home() / "hpc-course-jobs"
WORK.mkdir(exist_ok=True)
VENV_PYTHON = Path(os.environ["VIRTUAL_ENV"] if "VIRTUAL_ENV" in os.environ else
                   Path(__import__("sys").executable).parent.parent) / "bin/python"
print("jobs go in", WORK, "\nranks will run", VENV_PYTHON)

# This kernel is itself a SLURM job. sbatch passes SLURM_* variables on to the
# jobs we submit from here (even with --export=NONE), and the nukwa QOS sets
# SLURM_MEM_PER_CPU, which the srun that mpirun uses to start its daemons then
# requests on kura nodes and fails. So submit from a scrubbed environment.
CLEAN_ENV = {k: v for k, v in os.environ.items() if not k.startswith("SLURM_")}

def submit(script, *args, **sbatch_opts):
    """sbatch a script from this notebook, wait for it, and return its output."""
    opts = [f"--{k.replace('_', '-')}={v}" for k, v in sbatch_opts.items()]
    cmd = ["sbatch", "--parsable", "--wait", *opts, str(script), *map(str, args)]
    out = subprocess.run(cmd, capture_output=True, text=True, cwd=WORK, env=CLEAN_ENV)
    if out.returncode:
        raise RuntimeError(out.stderr)
    return (WORK / f"slurm-{out.stdout.strip()}.out").read_text()

# %% [markdown]
# ## 3.1 MPI en diez líneas
#
# Cada rango ejecuta el mismo script. `comm.rank` le dice quién es. Operaciones colectivas (`reduce`, `bcast`, `scatter`, `gather`) hacen la comunicación.

# %%
_ = (WORK / "hello.py").write_text(textwrap.dedent('''
    from mpi4py import MPI
    import numpy as np

    comm = MPI.COMM_WORLD
    rank, size = comm.rank, comm.size
    local = np.random.default_rng(rank).random(1_000_000)
    total = comm.reduce(local.sum(), op=MPI.SUM, root=0)
    print(f"rank {rank}/{size} on {MPI.Get_processor_name()} local mean {local.mean():.4f}", flush=True)
    if rank == 0:
        print(f"global mean {total / (size * local.size):.4f}", flush=True)
'''))

_ = (WORK / "mpi.sbatch").write_text(textwrap.dedent(f'''
    #!/bin/bash
    #SBATCH --job-name=mpi-course
    #SBATCH --partition=kura-wide
    #SBATCH --time=0-00:05:00
    #SBATCH --cpus-per-task=1
    # nodes and tasks come from the command line: sbatch --nodes=N --ntasks-per-node=M
    module load gcc/12.4.0 openmpi/4.0.5
    export NUMBA_NUM_THREADS=${{SLURM_CPUS_PER_TASK:-1}}   # one Numba pool per rank, sized to the rank
    mpirun --mca orte_keep_fqdn_hostnames 1 {VENV_PYTHON} "$@" 2>&1 | grep -v openib
''').lstrip())

# %%
print(submit("mpi.sbatch", "hello.py", nodes=2, ntasks_per_node=4))

# %% [markdown]
# Eight ranks, dos nombres de host. La línea `sbatch` es toda la historia de "¿Cómo utilizo más nodos": `--nodes` y `--ntasks-per-node` deciden, `mpirun` lee la asignación, el script no cambia.
#
# Reglas del cluster que valen la pena repetir: inicia con `mpirun` y la bandera `orte_keep_fqdn_hostnames` en Kabré, no `srun --mpi=pmi2`; los ramos vienen de `--ntasks`, los hilos de `--cpus-per-task`.
#
# ## 3.2 El stencil a lo largo de los nodos: descomposición del dominio
#
# Divide el grid en filas horizontales, una por rama. Cada paso, una fila necesita la fila justo arriba y justo debajo, que pertenece a sus vecinos. Estas son **filas de halo** (huesos) que se intercambian en cada iteración. Todo lo demás es el stencil de Numba de la sesión 1, sin cambios.

# %%
_ = (WORK / "stencil_mpi.py").write_text(textwrap.dedent('''
    import sys, time
    import numpy as np
    from mpi4py import MPI
    from numba import njit, prange

    @njit(parallel=True)
    def step(u, unew):
        n, m = u.shape
        for i in prange(1, n - 1):
            for j in range(1, m - 1):
                unew[i, j] = 0.25 * (u[i-1, j] + u[i+1, j] + u[i, j-1] + u[i, j+1])

    comm = MPI.COMM_WORLD
    rank, size = comm.rank, comm.size
    n, iters = int(sys.argv[1]), int(sys.argv[2])
    rows = n // size                                   # interior rows owned by this rank
    up = rank - 1 if rank > 0 else MPI.PROC_NULL      # PROC_NULL: sends and receives become no-ops
    down = rank + 1 if rank < size - 1 else MPI.PROC_NULL

    u = np.zeros((rows + 2, n))                        # +2 halo rows
    if rank == 0:
        u[1, :] = 100.0                                # hot top edge lives in rank 0's first real row
    unew = u.copy()
    step(u, unew)                                      # compile before timing

    comm.Barrier(); t0 = MPI.Wtime()
    for _ in range(iters):
        # send my first real row up, receive neighbour's last real row into my top halo (and vice versa)
        comm.Sendrecv(u[1], dest=up, recvbuf=u[0], source=up)
        comm.Sendrecv(u[rows], dest=down, recvbuf=u[rows + 1], source=down)
        if rank == 0: u[0, :] = u[1, :]                # top boundary: keep the hot edge
        step(u, unew)
        u, unew = unew, u
    elapsed = MPI.Wtime() - t0

    total_heat = comm.reduce(u[1:-1].sum(), op=MPI.SUM, root=0)
    if rank == 0:
        cells = n * n * iters
        print(f"{size:3d} ranks on {n}x{n}, {iters} steps: {elapsed:6.2f} s "
              f"({cells/elapsed/1e9:5.2f} Gcell-updates/s)  total heat {total_heat:.1f}", flush=True)
'''))

# %%
N, ITERS = 12000, 100
for nodes, per_node in ((1, 1), (1, 4), (2, 4), (4, 4)):
    out = submit("mpi.sbatch", "stencil_mpi.py", N, ITERS,
                 nodes=nodes, ntasks_per_node=per_node, cpus_per_task=5, hint="nomultithread")
    print(f"{nodes} node(s) x {per_node} ranks x 5 threads: {out.strip()}")

# %% [markdown]
# Hybrid paralelismo: MPI entre nodos, hilos de Numba dentro de un rango. `total heat`
# es el chequeo de correctitud: debe no cambiar con el número de rango.
#
# ¿Por qué no hay escala perfecta? Cada rango intercambia dos filas por paso. Eso es muy pequeño
# compared with la banda, por lo que este problema escala bien. Los problemas que requieren
# comunicación all-to-all (FFTs, álgebra lineal densa) no lo hacen, y en esos casos
# la red decide. Medir la fracción de tiempo en comunicación antes de comprar más nodos.
#
# ## 3.3 Dask en muchos nodos, desde la notebook
#
# Para tareas independientes, el MPI es excesivo. `dask-jobqueue` pide a SLURM trabajadores de
# tareas y los conecta a un agente que corre aquí, en el kernel. El código que escribiste contra `LocalCluster`
# en la sesión 1 no cambia.

# %%
from dask.distributed import Client
from dask_jobqueue import SLURMCluster

for k in list(os.environ):                 # dask-jobqueue calls sbatch too: same scrub as above
    if k.startswith("SLURM_"):
        del os.environ[k]

cluster = SLURMCluster(
    queue="kura", cores=20, processes=4, memory="40GB", walltime="00:10:00",
    job_extra_directives=["--exclusive"],                       # one node per job, so two jobs = two hosts
    python=str(VENV_PYTHON), log_directory=str(WORK),
    scheduler_options={"dashboard_address": ":0"},
)

print(cluster.job_script())

# %%
cluster.scale(jobs=2)                      # two SLURM jobs = two nodes, 4 workers each
client = Client(cluster)
client.wait_for_workers(8, timeout=300)
print(client)

# %%
import numpy as np

def simulate(alpha, n=1000, iters=200):
    """One parameter of a sweep: run the stencil with diffusion coefficient alpha, return mean temperature."""
    u = np.zeros((n, n)); u[0, :] = 100.0
    for _ in range(iters):
        u[1:-1, 1:-1] += alpha * (u[:-2, 1:-1] + u[2:, 1:-1] + u[1:-1, :-2] + u[1:-1, 2:] - 4 * u[1:-1, 1:-1])
    return alpha, float(u.mean())

alphas = np.linspace(0.05, 0.25, 32)
t0 = time.perf_counter()
results = client.gather(client.map(simulate, alphas))
print(f"{len(alphas)} simulations on {len(client.scheduler_info()['workers'])} workers "
      f"across nodes: {time.perf_counter() - t0:.1f} s")
print({w["host"] for w in client.scheduler_info()["workers"].values()})
results[:3]

# %%
client.close(); cluster.close()

# %% [markdown]
# Adaptive mode (`cluster.adapt(minimum_jobs=0, maximum_jobs=8)`) permite que Dask
# reclame y libere nodos según lo que demanda el gráfico de tareas. Ese es el equivalente
# del cluster al "escalar a cero".
#
# ## 3.4 ¿Cuál es el caso?
#
# | Tienes | Usa |
# |---|---|
# | Tareas independientes (sweeps, archivos, folds) | Dask (`dask-jobqueue`) o un array de tareas SLURM |
# | Un gran array o dataframe que no cabe en la RAM | `dask.array` / `dask.dataframe` en `SLURMCluster` |
# | Simulaciones fuertemente acopladas, intercambio de halo, comunicación personalizada | `mpi4py` + Numba |
# | Deep Learning en GPUs | PyTorch DDP, que utiliza NCCL, el hermano de MPI del GPU |
#
# ## Conclusión para toda la tarde
#
# 1. Medir primero. Profilera, luego optimiza la línea que importa.
# 2. Vectoriza, luego compila (Numba). Diez a cien veces, sin paralelismo aún.
# 3. Hilo para el código compilado o de I/O, proceso para el código Python, `prange` para los bucles.
# 4. GPU: el mismo código NumPy con CuPy; agrupa el trabajo; mueve el dato una vez; sincroniza antes de medir; conoce tu tasa de float64.
# 5. Multi-nodo: MPI para el trabajo fuertemente acoplado, Dask para el trabajo independiente. SLURM distribuye los nodos de la misma manera.
# 6. Mantén el entorno reproducible (`uv`, un fichero de bloqueo), así el número que obtuviste hoy es el mismo que obtendrás el próximo mes.

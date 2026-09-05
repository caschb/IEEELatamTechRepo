# Paste into one cell of the OnDemand Jupyter session with the hpc-course kernel.
import os, shutil, socket, subprocess, sys, time

def ok(cond, msg): print(("PASS " if cond else "FAIL ") + msg)

import numpy as np
import numba
from numba import njit, prange

@njit(parallel=True)
def sum_sq(a):
    s = 0.0
    for i in prange(a.shape[0]):
        s += a[i] * a[i]
    return s


def main():
    print("python    ", sys.executable)
    print("host      ", socket.gethostname())
    print("job       ", os.environ.get("SLURM_JOB_ID"), "partition", os.environ.get("SLURM_JOB_PARTITION"))
    print("cpus      ", os.environ.get("SLURM_CPUS_ON_NODE"), "cpus-per-task", os.environ.get("SLURM_CPUS_PER_TASK"),
          "affinity", len(os.sched_getaffinity(0)))
    print("modules   ", os.environ.get("LOADEDMODULES") or "(none)")
    ok("/.venv/" in sys.executable, "kernel is the uv venv")
    ok(socket.gethostname().startswith("nukwa"), "running on a nukwa (GPU) node, not a login node")

    x = np.random.rand(20_000_000)
    sum_sq(x)
    t = time.perf_counter(); r = sum_sq(x); dt_numba = time.perf_counter() - t
    t = time.perf_counter(); r_np = (x * x).sum(); dt_np = time.perf_counter() - t
    ok(np.isclose(r, r_np), f"numba {numba.__version__} parallel: {dt_numba*1e3:.1f} ms vs numpy {dt_np*1e3:.1f} ms "
                            f"on {numba.get_num_threads()} threads")
    ok(numba.get_num_threads() == len(os.sched_getaffinity(0)),
       "numba thread count matches the CPUs SLURM gave us (else set NUMBA_NUM_THREADS)")

    try:
        import cupy as cp
        props = cp.cuda.runtime.getDeviceProperties(0)
        xg = cp.asarray(x); (xg * xg).sum(); cp.cuda.Device().synchronize()
        t = time.perf_counter(); rg = float((xg * xg).sum()); cp.cuda.Device().synchronize(); dt_gpu = time.perf_counter() - t
        ok(np.isclose(rg, r_np), f"cupy {cp.__version__} on {props['name'].decode()} "
                                 f"({props['totalGlobalMem']/2**30:.0f} GB): {dt_gpu*1e3:.2f} ms")
    except Exception as e:
        ok(False, f"cupy: {e!r}")

    try:
        from numba import cuda
        ok(cuda.is_available(), f"numba.cuda sees {cuda.get_current_device().name}")
    except Exception as e:
        ok(False, f"numba.cuda: {e!r}")

    try:
        from mpi4py import MPI
        ok(MPI.COMM_WORLD.size == 1, f"mpi4py {MPI.Get_version()} singleton init, "
                                     f"{MPI.Get_library_version().split(',')[0]}")
    except Exception as e:
        ok(False, f"mpi4py: {e!r}")

    try:
        from dask.distributed import Client, LocalCluster
        with LocalCluster(n_workers=2, threads_per_worker=1, dashboard_address=None) as cl, Client(cl) as c:
            ok(c.submit(lambda a: a + 1, 41).result() == 42, f"dask.distributed LocalCluster round trip")
    except Exception as e:
        ok(False, f"dask: {e!r}")

    for tool in ("sbatch", "squeue", "srun", "mpirun"):
        ok(shutil.which(tool) is not None, f"{tool} on PATH (needed for the multi-node block)"
           + ("" if shutil.which(tool) or tool != "mpirun" else " -> load openmpi/4.0.5 in the sbatch script"))

    r = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.used,memory.total,utilization.gpu", "--format=csv,noheader"],
                       capture_output=True, text=True)
    print("nvidia-smi", r.stdout.strip() or r.stderr.strip())


if __name__ == "__main__":
    main()

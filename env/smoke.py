import time

import numpy as np

import numba
from numba import njit, prange

print("numba", numba.__version__, "threads", numba.get_num_threads())


@njit(parallel=True)
def sum_sq(a):
    s = 0.0
    for i in prange(a.shape[0]):
        s += a[i] * a[i]
    return s


x = np.random.rand(10_000_000)
sum_sq(x)
t = time.perf_counter()
r = sum_sq(x)
print(f"numba parallel sum_sq: {time.perf_counter() - t:.4f}s ok={np.isclose(r, (x * x).sum())}")

import cupy as cp

dev = cp.cuda.Device()
print("cupy", cp.__version__, "device", cp.cuda.runtime.getDeviceProperties(dev.id)["name"].decode())
xg = cp.asarray(x)
(xg * xg).sum()  # warm up: context creation and kernel compile
cp.cuda.Stream.null.synchronize()
t = time.perf_counter()
rg = float((xg * xg).sum())
cp.cuda.Stream.null.synchronize()
print(f"cupy sum_sq: {time.perf_counter() - t:.4f}s ok={np.isclose(rg, r)}")

from numba import cuda

print("numba.cuda available:", cuda.is_available(), cuda.get_current_device().name)

from mpi4py import MPI

comm = MPI.COMM_WORLD
print(f"mpi4py rank {comm.rank}/{comm.size} on {MPI.Get_processor_name()} lib={MPI.Get_library_version().splitlines()[0]}")

import dask, distributed, joblib, jupyterlab, line_profiler

print("dask", dask.__version__, "jupyterlab", jupyterlab.__version__)

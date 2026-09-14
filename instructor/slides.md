---
marp: true
paginate: true
---

# Modern Parallel Programming in Python for HPC and AI

Measure. Vectorise. Compile. Then, and only then, parallelise.

Three hours, one running example, everything in Colab.

---

# Outcomes

1. Establish a correct reference result and measure repeated execution fairly.
2. Compare Python, NumPy and compiled Numba; explain why more threads may not help.
3. Run an array operation with CuPy, check it, and separate compute time from transfer-inclusive time.
4. Explain when separate-memory workers must communicate, and choose a task scheduler or message passing accordingly.

No target speedup. Correct results, explained measurements.

---

# The running example: a heat-diffusion stencil

- `n x n` grid, top edge held at 100, other edges at 0.
- Each step: every interior cell becomes the average of its four neighbours.
- Edges never change: they are the boundary condition.
- The plain-Python loop is the **reference**. Every faster version must agree with it.

```python
unew[1:-1, 1:-1] = 0.25 * (u[:-2, 1:-1] + u[2:, 1:-1] + u[1:-1, :-2] + u[1:-1, 2:])
```

Slices are views. Each `+` allocates an intermediate array.

---

# What a timing contains

- **Warm up** first: compilation, imports, caches, allocation.
- **Repeat**; report min (least interference) or median (typical). Say which.
- `%timeit` prints **mean and std. dev.** across runs; `.best` on request.
- **Same experiment**: same runtime, workload, dtype, steps.
- On a GPU: **synchronise** before starting and before stopping the clock.

A number from another machine is a different experiment.

---

# Amdahl's law

    speedup(p) = 1 / (s + (1 - s) / p)

10% serial, 4 cores: 3.1x. 16 cores: 6.4x. Unlimited: 10x.

The serial fraction sets the ceiling. In practice the ceiling comes sooner:
memory bandwidth, coordination, shared cores.

For our stencil, a poor fit to Amdahl is usually **memory bandwidth**, not
serial code.

---

# Host memory and device memory

```
   CPU  <-- RAM -->              PCIe (tens of GB/s)            <-- VRAM --> GPU
                    cp.asarray(x)  ------------------------->
                    x.get()        <-------------------------
```

- Move data once, compute a lot, move back once.
- `cp.asarray` inside a loop is usually slower than NumPy.
- Report **compute-only** and **transfer-inclusive** scopes separately.

---

# Timing scope, on one slide

| scope | includes |
|---|---|
| compute-only | kernels on data already on the device, sync at both ends |
| transfer-inclusive | upload, kernels, download, sync at both ends |
| wrong | launch only, no sync |

"40x faster": synchronised? transfers inside? same workload and dtype?

---

# Partitions and halos

```
   worker 0            worker 1            worker 2
+-----------+       +-----------+       +-----------+
| halo (top)|       | halo      | <---- | last real |
| real rows |       | real rows |       | real rows |
| real rows | ----> | halo      |       | ...       |
| halo (bot)| <---- | first real|       |           |
+-----------+       +-----------+       +-----------+
```

- Each step: two rows per interior boundary, one each way.
- Communication / computation per worker: `2P / n`. Grows with P, shrinks with n.
- Plus a fixed latency per message. Small problems do not scale across machines.

---

# Two programming models

- **Message passing (MPI, `mpi4py`)**: every rank runs the same script; you write
  the messages. Coupled simulations, halos, custom communication.
- **Task scheduling (Dask, job arrays)**: independent tasks, results collected.
  Parameter sweeps, files, folds; data larger than one machine.

The Python code is the same for 2 workers and 2000.

---

# The decision table

| You have | Reach for |
|---|---|
| Loop over array elements | NumPy first, then Numba `@njit` |
| Same loop, several cores | Numba `parallel=True` + `prange`, threads <= cores |
| Millions of identical, independent element operations | CuPy, data kept on the device |
| Many independent Python tasks | processes, `joblib`, Dask |
| Independent tasks across machines | task scheduler (Dask, job array) |
| Coupled simulation across machines | `mpi4py` (+ Numba or CuPy per rank) |
| Any performance claim | measure before and after, same runtime |

---

# When more hardware does not help

- The workload is too small: launch or thread start-up dominates.
- Data crosses the PCIe link every iteration.
- The kernel is memory-bound and the bus is already saturated.
- A serial fraction caps the speedup (Amdahl).
- The decomposition is communication-bound (`2P/n` large, latency dominates).

Find out which one **before** asking for more machines.

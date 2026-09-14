# Reading guide: MPI and Dask beyond one machine

Notebook 03 teaches the concepts with a NumPy model inside one runtime. This
guide is for students (and instructors) who want to run the real thing. None
of it is required for the workshop, and none of it runs in a free Colab
runtime, which is not meant to host distributed workers.

## Message passing with mpi4py

- Concepts to have straight from notebook 03: ranks, `comm.rank` and
  `comm.size`, point-to-point (`Sendrecv`) versus collective (`reduce`,
  `bcast`, `gather`), halos, and the `2P/n` communication-to-compute ratio.
- The mpi4py tutorial (https://mpi4py.readthedocs.io/en/stable/tutorial.html)
  covers exactly the `Sendrecv` and `reduce` calls shown in section 3.4. Read
  "Point-to-point" and "Collective" and note the difference between the
  lower-case (pickled objects) and upper-case (buffers, NumPy arrays) methods.
  The stencil uses the upper-case ones.
- To try it on a laptop: `pip install mpi4py` needs an MPI library (on Linux
  `apt install libopenmpi-dev`, on macOS `brew install open-mpi`); then
  `mpirun -np 4 python stencil_mpi.py`. Four ranks on one laptop already show
  the correctness argument (total heat independent of rank count), not the
  speed.
- On a cluster the launch is a job script; the cluster's own documentation
  decides the launcher and flags. The earlier edition of this course, in
  `archive/kabre/`, shows one such setup and its pitfalls.
- Hybrid parallelism: one Numba thread pool per rank, sized to the cores the
  rank was given (`NUMBA_NUM_THREADS`), or one GPU per rank with CuPy. The
  Python inside the rank is the code from notebooks 1 and 2, unchanged.

## Task scheduling with Dask

- Concepts from notebook 03: independent tasks exchange zero bytes with each
  other; the only traffic is arguments in and results out.
- Start with `dask.distributed` on your own machine: `pip install "dask[distributed]"`,
  then `Client()` with no arguments starts local workers. The `client.map` and
  `client.gather` calls from section 3.5 run as written. The dashboard link
  the client prints shows tasks moving between workers.
- Same code on a cluster: `dask-jobqueue` (https://jobqueue.dask.org) creates
  the workers as batch jobs (`SLURMCluster`, `PBSCluster`, ...). Read "How
  this works" and "Configuration" before the API. Dask's own "Deploy Dask
  Clusters" page lists the other options (Kubernetes, cloud).
- Larger-than-memory data: `dask.array` and `dask.dataframe` chunk NumPy and
  pandas work across workers with the same API. Try `dask.array.zeros((20000,
  20000), chunks=(2000, 2000))` and the stencil slices on it; the graph it
  builds is the halo exchange from notebook 03, done for you.

## Choosing

| Question | If yes | If no |
|---|---|---|
| Do tasks need each other's data during the computation? | message passing | task scheduler |
| Does one array or table exceed one machine's RAM? | `dask.array` / `dask.dataframe` (or MPI with explicit decomposition) | keep it on one machine |
| Is the per-task work under a millisecond? | batch tasks together first; neither tool helps with that granularity | fine |
| Is the code a deep-learning model? | PyTorch DDP / NCCL, not MPI directly | as above |

## What to measure before scaling out

1. The serial and single-machine baselines from notebooks 1 and 2.
2. The fraction of time in communication at a small worker count. If it is
   already large, more workers make it worse.
3. Whether the result is identical across worker counts (the `array_equal`
   check from notebook 03, or a conserved quantity). A scaling result without
   that check is not a result.

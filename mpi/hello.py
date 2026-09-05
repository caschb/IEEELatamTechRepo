from mpi4py import MPI
import numpy as np

comm = MPI.COMM_WORLD
rank, size = comm.rank, comm.size
local = np.random.default_rng(rank).random(1_000_000)
total = comm.reduce(local.sum(), op=MPI.SUM, root=0)
print(f"rank {rank}/{size} on {MPI.Get_processor_name()} local mean {local.mean():.4f}", flush=True)
if rank == 0:
    print(f"global mean {total / (size * local.size):.4f}", flush=True)

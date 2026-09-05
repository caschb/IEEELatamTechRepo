#!/bin/bash
# Exercises the env the way OnDemand will launch the kernel (no modules loaded),
# then with the modules, so any dependence on module paths shows up here.
cd /data/casch/hpc-course/env
echo "===== bare environment (module list follows)"; module list 2>&1
env -u LD_LIBRARY_PATH uv run --no-sync python -u smoke.py 2>&1 | grep -v "^  dev = Device"
echo "===== with modules"
module load gcc/12.4.0 openmpi/4.0.5 cuda/12.4.0
uv run --no-sync python -u smoke.py 2>&1 | grep -v "^  dev = Device"
echo "===== mpirun 4 ranks"
mpirun --mca orte_keep_fqdn_hostnames 1 -np 4 --oversubscribe .venv/bin/python -c "from mpi4py import MPI; c=MPI.COMM_WORLD; print(c.rank, c.size, MPI.Get_processor_name())" 2>&1 | grep -v openib

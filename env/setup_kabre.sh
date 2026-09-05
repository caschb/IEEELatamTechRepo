#!/bin/bash
# Run on a nukwa node (sbatch, not srun: mpi4py import breaks inside srun steps).
# Builds the uv env against the cluster MPI and CUDA modules and registers a
# Jupyter kernel that the OnDemand Jupyter app lists as "Python (hpc-course)".
set -eu
module load gcc/12.4.0 openmpi/4.0.5 cuda/12.4.0
cd /data/casch/hpc-course/env   # sbatch runs a spooled copy, so $0 is useless here

# ~/.bashrc points uv at node-local /tmp. The interpreter the venv symlinks to
# must live on the shared filesystem or the kernel only works on one node.
export UV_PYTHON_INSTALL_DIR=/data/casch/.local/share/uv/python
export UV_CACHE_DIR=/data/casch/.cache/uv

rm -rf .venv
uv python install 3.12
uv sync
uv run python -m ipykernel install --user --name hpc-course --display-name "Python (hpc-course)"
echo "--- interpreter"
readlink -f .venv/bin/python3
echo "--- kernelspec"
cat ~/.local/share/jupyter/kernels/hpc-course/kernel.json

#!/bin/bash
#SBATCH --job-name=mpi-hello
#SBATCH --partition=kura-wide
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=1
#SBATCH --time=0-00:05:00
#SBATCH --output=/data/casch/hpc-course/mpi/hello-%j.out
module load gcc/12.4.0 openmpi/4.0.5
cd /data/casch/hpc-course/mpi
mpirun --mca orte_keep_fqdn_hostnames 1 /data/casch/hpc-course/env/.venv/bin/python hello.py 2>&1 | grep -v openib

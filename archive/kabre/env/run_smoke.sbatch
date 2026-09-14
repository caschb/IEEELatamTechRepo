#!/bin/bash
#SBATCH --partition=nukwa-l40s
#SBATCH --time=0-00:10:00
#SBATCH --cpus-per-task=8
#SBATCH --ntasks=1
#SBATCH --job-name=hpc-smoke
#SBATCH --output=/data/casch/hpc-course/env/smoke-%j.out
/data/casch/hpc-course/env/run_smoke.sh

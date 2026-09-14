#!/bin/bash
# Executes the given notebooks in place on the current allocation, for validation.
set -u
cd /data/casch/hpc-course/notebooks
for nb in "$@"; do
  echo "===== $nb  $(date +%T)  host $(hostname)  cpus $(nproc)"
  /data/casch/hpc-course/env/.venv/bin/jupyter nbconvert --to notebook --execute --inplace \
      --ExecutePreprocessor.timeout=1800 --ExecutePreprocessor.kernel_name=python3 "$nb" 2>&1 | grep -v "dev = Device"
  echo "===== exit ${PIPESTATUS[0]}  $(date +%T)"
done

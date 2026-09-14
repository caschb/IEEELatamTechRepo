#!/bin/bash
# Authoring validation, runnable on any machine with uv:
#   1. regenerate every .ipynb from its jupytext .py source (notebooks/, extensions/, solutions/)
#   2. execute the core notebooks headless in CPU-fallback mode, and again in GPU mode if
#      an NVIDIA GPU is present, writing executed copies under validation/ (gitignored)
# It does not replace the manual Colab checks in instructor/validation.md.
set -u
cd "$(dirname "$0")/.."
UV="uv run --project env"
[ -n "${GPU_EXTRA:-}" ] && UV="$UV --extra gpu"

echo "== regenerate notebooks"
$UV jupytext --quiet --to ipynb notebooks/*.py extensions/*.py solutions/*.py || exit 1
git status --porcelain -- '*.ipynb' | sed 's/^/   changed: /'

CORE="notebooks/00_colab_ready.ipynb solutions/00_stencil_practice_solution.ipynb notebooks/00_stencil_practice.ipynb \
      notebooks/01_measure_and_multicore.ipynb notebooks/02_gpu.ipynb notebooks/03_parallel_models.ipynb notebooks/04_capstone.ipynb"
status=0
run_mode() {   # $1 = mode label, $2 = env assignment
  local mode=$1 envset=$2
  mkdir -p "validation/$mode"
  for nb in $CORE ${EXTRA_NOTEBOOKS:-}; do
    local out="validation/$mode/$(basename "$nb")"
    local t0=$(date +%s)
    if env $envset $UV jupyter nbconvert --to notebook --execute --ExecutePreprocessor.timeout=900 \
         --output "$(realpath -m "$out")" "$nb" >"$out.log" 2>&1; then
      echo "   PASS $mode $nb ($(( $(date +%s) - t0 )) s)"
    else
      echo "   FAIL $mode $nb  (see $out.log)"; status=1
    fi
  done
}
echo "== execute, CPU fallback mode"
run_mode cpu HPC_COURSE_FORCE_CPU=1
if command -v nvidia-smi >/dev/null && nvidia-smi -L >/dev/null 2>&1; then
  echo "== execute, GPU mode"
  run_mode gpu HPC_COURSE_FORCE_CPU=
else
  echo "== no GPU here: GPU mode not executed"
fi
exit $status

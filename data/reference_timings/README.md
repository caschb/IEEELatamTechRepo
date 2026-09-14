# Recorded timing tables

Notebooks 01 and 02 load these files when the live runtime cannot produce the
comparison itself (a single-core runtime for the thread sweep, no GPU for the
device timings). Each file starts with `# key: value` lines describing the
runtime that produced it, followed by a CSV table. The notebooks print that
metadata next to the numbers so that recorded results are never mistaken for
live ones.

| File | Produced by | Used by |
|---|---|---|
| `01_threads.csv` | notebook 01, "Checkpoint 3" cell | notebook 01, thread-scaling section, when fewer than two thread counts can be tested |
| `02_gpu.csv` | notebook 02, "Checkpoint 2" cell | notebook 02, size sweep and transfer sections, in CPU-fallback mode |
| `04_capstone.csv` | notebook 04 | instructor reference only |

## Status: interim data

The current files come from the author's workstation (32-core CPU, NVIDIA
GeForce RTX 4090, recorded in the metadata lines), **not from Google Colab**.
They exist so that the fallback path is exercised end to end. Release gate 4 in
`instructor/validation.md` replaces them with a documented Colab GPU run:

1. Open notebooks 01 and 02 in Colab on a fresh GPU runtime, run all cells.
2. Download `timings_01_cpu.csv` and `timings_02_gpu.csv` from the Files panel.
3. Copy them here as `01_threads.csv` and `02_gpu.csv`, commit, push.
4. Re-run notebook 02 on a CPU runtime and confirm the "RECORDED GPU RUN" line
   shows the Colab runtime and GPU.

The notebooks fetch the files from the `main` branch of this repository over
HTTPS; the authoring validation script reads the local copies instead.

## The GPU demonstration recording

Record the instructor's GPU run of notebook 02 (screen capture, 5 to 8 minutes,
from "Change runtime type" to the final timing table) and put the link in
`instructor/run_of_show.md`. The recording is the second fallback, after this
table, for students without a GPU.

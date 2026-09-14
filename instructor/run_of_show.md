# Run of show: 180 minutes, Colab edition

All students in Colab. Instructor shares a Colab screen on a GPU runtime from
block 5 onward, and a CPU runtime before that (so the students see the same
thread counts they will get). Keep `solutions/` and this file open in another
window. Two planned cuts if the session slips: the profiling demo (block 2,
5 min) first, then the float32 experiment (block 6, 5 min). Never cut
correctness checks, the GPU timing/transfer lesson, the capstone or the
closing discussion. GPU setup stops at its ten-minute boundary; students who
do not have a GPU by 01:30 use the fallback, no exceptions.

Recording of the GPU demonstration: *(add link after gate 5 in validation.md)*.

## 00:00 Welcome and readiness (10 min)

- 00:00 Outcomes on one slide (the four in the README). "You succeed by
  producing correct results and explaining your measurements. There is no
  target speedup."
- 00:02 Readiness check: everyone opens `01_measure_and_multicore` and runs the
  setup cell. Hands up when it prints the runtime dictionary. Meanwhile:
  - **Triage.** Setup fails on pip: ask them to rerun the cell once (transient
    network); still failing, they pair with a neighbour for this block and
    retry at the break. Cell hangs at "installing": Runtime > Restart session,
    rerun. Nothing runs at all: Runtime > Disconnect and delete runtime, reopen
    the link. Do not debug individuals past 00:08.
  - Students who did not do the preparation: point them to the one-page recap
    (`prep/primer.md` section 1 and the "What to remember" cell at the end of
    `00_stencil_practice`) and the completed stencil in `solutions/`. Do not
    reteach slicing to the room.
- 00:06 Recap of the stencil on the slide: hot edge, average of four
  neighbours, boundaries never change, the pure-Python loop is the reference.
- 00:08 Run section 1.1 together. Checkpoint 1: the assertion passed.

## 00:10 Measurement (25 min), notebook 01 sections 1.2 to 1.4

- 00:10 Prompt: "Why is one `time.time()` pair not a measurement?" Collect
  three answers (first call, other processes, resolution). Run `best_of`.
- 00:14 `%timeit`: point at the words "mean +- std. dev." in the output. Common
  misconception: that `%timeit` reports the best time. It reports the mean of
  the runs; `.best` is available if you ask for it.
- 00:17 Predict-run-explain: n=128 vs n=256 ratio. Expected answer 4x, and the
  explanation is "time proportional to cells, interpreter overhead per cell".
  Colab CPU timings for pure Python are 50 to 150 ms at n=128.
- 00:21 Profiling demo (CUT 1). Show `%lprun` once; say what you would look for
  in real code. Skip silently if late.
- 00:26 NumPy: views versus intermediates. Ask: "How many arrays does this line
  allocate?" Answer: four (three sums and the product); the slices allocate
  none. Run the baseline table. Checkpoint 2: everyone has three NumPy rows.
- 00:33 Prompt for section 1.5: "Is NumPy limited by arithmetic or by memory
  traffic?" Do not answer yet.

## 00:35 Numba, threads, Amdahl (35 min), sections 1.5 to 1.7

- 00:35 Numba serial. Emphasise the compile-on-first-call line in the output.
  Misconception: "Numba is faster because it is parallel." It is not parallel
  here; it is faster because it stops allocating intermediates.
- 00:42 `prange` and the thread list. Show `MAX_THREADS` on the shared screen:
  a Colab CPU runtime usually reports 2. Explain why the notebook asks the
  runtime instead of assuming.
- 00:46 Predict-run-explain: "Will two threads halve the time?" Typical Colab
  result: between 1.0x and 1.6x; sometimes slower. Explanations to draw out:
  shared physical core, memory bandwidth, thread start-up on a small grid.
- 00:52 Recorded thread sweep if the runtime has one core. Say out loud what
  machine the recording came from (the notebook prints it).
- 00:55 Amdahl plot. Exercise: pick the curve closest to the measurement and
  say what "serial fraction" it implies; then explain why that is a *model*
  and memory bandwidth is the more likely cause for this stencil.
- 01:02 Runtime-aware limit: `set_num_threads(min(wanted, cpu_count))`.
  Over-subscription is silent.
- 01:05 Checkpoint 3: save the CSV. Ask two students to read one row of their
  table and their Numba-vs-NumPy ratio. Different numbers, same shape: that is
  the point.

## 01:10 Break (10 min)

Tell everyone to save (Ctrl+S). Students who want to try the GPU switch early
may, but the block starts at 01:20 regardless.

## 01:20 GPU setup (10 min, hard stop), notebook 02 setup cell

- 01:20 Runtime > Change runtime type > GPU. Run the setup cell. It prints
  `MODE: GPU` or `MODE: CPU fallback`.
- Triage: "no GPU available" from Colab: fallback, immediately. CuPy import
  error on a GPU runtime: rerun once; still failing, fallback. Restart loops:
  fallback. Do not spend the room's time on one machine.
- 01:28 Show the CPU-fallback banner on the screen so those students know what
  they will see: live CPU rows, recorded GPU rows with a hardware label, and
  the same questions.
- 01:30 Move on whatever the state of the room.

## 01:30 CuPy, synchronisation, sweep, transfers, precision (40 min), sections 2.1 to 2.6

- 01:30 Section 2.1. The `xp` pattern. Checkpoint 1: same numbers on a
  different memory.
- 01:35 Section 2.2. Run the no-sync versus sync cell. Misconception to
  surface: "the GPU is so fast the time is zero." Ask what the clock contained.
- 01:41 Section 2.3 predict: "At which n will the GPU overtake the CPU on
  this runtime?" Run. Explain-what-you-see questions 1 and 2. Do not state a
  crossover size; ask the room for theirs, then for the fallback table's, and
  point out they differ.
- 01:52 Section 2.4 transfers. Predict: upload cheaper or dearer than one
  step? Typical answer on Colab T4: an upload of 32 MB costs several stencil
  steps. Show the two scopes side by side. Rule: say which scope you quote.
- 02:01 Section 2.5 float32 (CUT 2). Point out the check with a looser
  tolerance, and that the f64/f32 ratio is a property of the resource you are
  bound by, not of "the GPU".
- 02:06 Section 2.6 briefly, then Checkpoint 2: save the CSV. Exit question:
  three things to ask about "40x faster" (synchronised, transfers included,
  same workload and dtype).

## 02:10 Beyond one machine (20 min), notebook 03

- 02:10 One slide: separate machines, separate memory. The two programming
  models.
- 02:13 Section 3.2 exercise before running: rows per boundary per step, and
  bytes. Answer: two rows per interior boundary, `2*(P-1)*n*8` bytes in total,
  `2n*8` per worker in the middle. Run; the reconstructed result is
  `array_equal`, not just close.
- 02:19 Section 3.3 model plot and the latency question. Expected answer: at
  n=256, P=128 a worker computes 512 cells (about 0.5 us) and waits for two
  messages (about 20 us of latency): mostly waiting. At n=16384 it computes
  2 million cells (2 ms) per two messages: mostly computing.
- 02:24 Sections 3.4 and 3.5 as reading with the code on screen: match each
  `Sendrecv` line with a line of `exchange_halos`; then the sweep with no
  halos at all. State plainly that this notebook makes no multi-node
  performance claim.
- 02:28 Checkpoint: one sentence per capstone workload, which table row.

## 02:30 Capstone (20 min), notebook 04

- 02:30 Two workloads, three deliverables each. Students edit `CHOICE_A` and
  `CHOICE_B`, run, then write the recommendation cell. Circulate.
- Expected results and reasoning: `solutions/capstone_solution.md`. The
  teachable surprise is workload A, where `numba_par` is often slower than
  `numba`.
- 02:45 Two volunteers read their recommendation. Ask each: "Which number in
  your table is that sentence based on?"
- 02:48 Save the CSV and the notebook.

## 02:50 Debrief (10 min)

- Exit question: "When would more hardware fail to help?" Collect: small
  workload (launch/thread overhead), transfers inside a loop, memory-bound
  kernel, serial fraction, communication-bound decomposition.
- The decision table slide. Where to go next: the extensions, the reading
  guide, the slides link.

## Misconceptions to listen for, all session

| Heard | Correct |
|---|---|
| "`%timeit` gives the best time" | Mean and std over runs; `.best` on request |
| "Numba is faster because it is parallel" | Serial Numba wins by removing intermediates |
| "Processes always beat threads" | Depends on whether the code holds the GIL and on task size |
| "The GPU is Nx faster" | Only with sync, stated scope, same workload and dtype, this runtime |
| "float32 is twice as fast on GPUs" | For a memory-bound kernel it is about bytes; compute-bound depends on the card |
| "Bigger cluster, faster program" | Communication-to-compute ratio `2P/n` grows with P |

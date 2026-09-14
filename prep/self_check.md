# Self-check (5 to 10 minutes)

Answer each question before opening its explanation. Getting one wrong is
fine; read the explanation and move on. Together these cover what the first
ten live minutes assume.

## 1. Slicing

`u` is a 6x6 array. Which expression has the same shape as `u[1:-1, 1:-1]`
and selects, for each interior cell, the cell to its **left**?

(a) `u[1:-1, 2:]`  (b) `u[1:-1, :-2]`  (c) `u[:-2, 1:-1]`  (d) `u[:, :-2]`

<details><summary>Answer</summary>

**(b).** Keep the rows the same (`1:-1`) and shift the columns one to the left:
`:-2` selects columns 0 to 3, which are the left neighbours of columns 1 to 4.
(a) is the right neighbour, (c) the one above, (d) has 6 rows and so cannot be
added to the 4x4 interior.
</details>

## 2. Boundary preservation

In the stencil, `unew[1:-1, 1:-1] = 0.25 * (...)` updates only the interior.
After 100 steps, what is the value of the top row `u[0, :]`?

(a) It has cooled toward the average  (b) Still 100 everywhere  (c) 25  (d) Undefined

<details><summary>Answer</summary>

**(b).** The top row is never written, so it stays at 100. It is the boundary
condition, the "hot edge" that drives the whole simulation. The same is true
of the other three edges, which stay at 0. Every faster implementation must
preserve this, and the notebooks check it explicitly.
</details>

## 3. Speedup

Version A takes 8.0 s. Version B takes 0.5 s on the same machine. A colleague
runs version B on a faster laptop in 0.2 s. What is the speedup of B over A?

(a) 40x  (b) 16x  (c) 2.5x  (d) Cannot say without more information

<details><summary>Answer</summary>

**(b).** 8.0 / 0.5 = 16, measured on the same machine. The laptop number is a
different experiment: different hardware, so it cannot be compared with A. In
the workshop every comparison is made within one runtime.
</details>

## 4. Separate device memory

`x` is a NumPy array in RAM. Which statement about a GPU computation on it
is true?

(a) The GPU reads `x` from RAM directly, so no copy is needed
(b) `x` must be copied to GPU memory first; the result must be copied back to be used in NumPy
(c) Copies are so fast they never matter
(d) CuPy arrays and NumPy arrays share memory

<details><summary>Answer</summary>

**(b).** The GPU has its own memory. `cp.asarray(x)` copies up; `.get()`
copies down. The copies go over a link far slower than either memory, so a
loop that copies every iteration can be slower than not using the GPU at all.
</details>

## 5. Timing scope

You want to time 20 stencil steps on the GPU. Which procedure is right?

(a) Start the clock, run 20 steps, stop the clock
(b) Run 20 steps once to warm up; then start the clock, run 20 steps, stop the clock
(c) Run 20 steps once to warm up; synchronise; start the clock, run 20 steps, synchronise, stop the clock; repeat a few times and report the minimum or median
(d) Time one step and multiply by 20

<details><summary>Answer</summary>

**(c).** Warm-up removes compilation and allocation from the measurement.
Synchronising before stopping the clock is essential on a GPU, because the
calls return before the work is done; without it you time the *launch*.
Repeats show the spread. (d) misses the fact that the first step is not
representative and that per-step overhead can dominate at small sizes.
</details>

## One more, for the discussion at the end of the workshop

When would buying more hardware **not** make a program faster? Think of two
reasons. (You will meet at least three during the live session.)

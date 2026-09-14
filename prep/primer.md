# Primer: the five ideas the workshop builds on (15 minutes)

Read this once before the event. Each section ends with the one sentence to
remember.

## 1. Arrays: shape, slicing, dtype, views and copies

A NumPy array is a block of numbers of one type (`dtype`) with a `shape`.
`u = np.zeros((4, 6))` is 4 rows by 6 columns of `float64`, 8 bytes each.

Slicing selects a rectangle without copying it. `u[1:-1, 1:-1]` is every row
except the first and last, and every column except the first and last: the
**interior**. Shifting a slice by one selects the neighbours of the interior:

| slice | selects |
|---|---|
| `u[:-2, 1:-1]` | the cell **above** each interior cell |
| `u[2:, 1:-1]` | the cell **below** |
| `u[1:-1, :-2]` | the cell to the **left** |
| `u[1:-1, 2:]` | the cell to the **right** |

All four have the same shape as the interior, so they can be added together.

A slice is a **view**: it shares memory with the original, so writing into it
changes the original. Arithmetic is different: `a + b` allocates a **new
array** for the result, even if `a` and `b` are views. Four additions produce
four temporary arrays. That matters for speed later.

`dtype` decides the bytes per number. `float64` is the default; `float32`
halves the memory and the traffic, at the cost of precision.

*Remember: slices are free views; arithmetic allocates.*

## 2. CPU memory and GPU memory are different places

A GPU has its own memory, separate from the computer's RAM. Before a GPU can
work on an array, the array must be **copied** to the device; to look at the
result from Python, it must be copied back. The link between the two (PCIe)
is much slower than the memory on either side. CuPy, the library used in the
workshop, looks exactly like NumPy but keeps its arrays on the GPU; the copies
are `cp.asarray(x)` (up) and `x.get()` (down).

*Remember: move data once, compute a lot, move it back once.*

## 3. Concurrency is not parallelism

**Concurrency** is dealing with several things at once (taking turns);
**parallelism** is doing several things at the same instant (several cores).
Python threads are concurrent but, for ordinary Python code, not parallel: an
interpreter lock (the GIL) lets one thread run Python at a time. Compiled code
(NumPy internals, Numba) can release that lock, so it can run in parallel on
threads. Separate **processes** are always parallel but do not share memory,
so data must be copied to them.

*Remember: threads for compiled code, processes for Python code, and measure.*

## 4. Speedup and Amdahl's law

Speedup = time before / time after, on the same machine, same problem. Ten
seconds to two seconds is a speedup of 5.

If a fraction `s` of the work cannot be parallelised, then with `p` cores:

    speedup(p) = 1 / (s + (1 - s) / p)

Worked example: a program spends 10% of its time reading a file (serial) and
90% in a loop that parallelises perfectly. With 4 cores:

    1 / (0.10 + 0.90 / 4) = 1 / 0.325 = 3.08

With 16 cores: `1 / (0.10 + 0.9/16) = 6.4`. With infinitely many: `1 / 0.10 = 10`.
The serial 10% caps the speedup at 10 no matter how much hardware you add.
In practice the cap comes sooner, because the parallel part also shares
memory bandwidth and pays for coordination.

*Remember: the serial fraction sets the ceiling; find it before buying cores.*

## 5. What a timing measures

A timing is only meaningful if you can say what was inside the clock:

- **Warm-up.** The first call pays for compilation, imports and caches. Time
  the second call onward.
- **Repeats.** Take several repeats and report the minimum (least interference)
  or the median (typical); say which.
- **Scope.** For a GPU: did the clock include copying data to and from the
  device? Did you wait for the GPU to finish (`synchronize`) before stopping
  the clock? A GPU call returns *before* the work is done.
- **Same experiment.** Same machine, same problem size, same `dtype`, same
  number of steps. Otherwise you are comparing two different things.

*Remember: warm up, repeat, synchronise, and say what the clock contained.*

## Four questions to check yourself

1. Which slice selects the cell below each interior cell of `u`?
2. If `v = u[1:-1, 1:-1]` and you do `v[:] = 0`, does `u` change?
3. A program is 25% serial. What is its speedup with 4 cores? With unlimited cores?
4. You time `y = f(x_gpu)` and get 0.01 ms. What is probably wrong?

Answers: 1. `u[2:, 1:-1]`. 2. Yes; `v` is a view. 3. `1/(0.25+0.75/4) = 2.29`;
at most 4. 4. The clock stopped before the GPU finished; synchronise first.

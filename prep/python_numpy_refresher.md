# Optional refresher: the Python and NumPy the workshop uses

Skip this if you write Python regularly. Otherwise, read it before the primer.
It covers only what the course notebooks use. Try each snippet in a Colab cell.

## Python

```python
def average_of_four(a, b, c, d):        # a function with four arguments
    return 0.25 * (a + b + c + d)

for i in range(1, 5):                   # 1, 2, 3, 4 (the end is excluded)
    print(i, average_of_four(i, i, i, i))

grid = [[0.0] * 4 for _ in range(3)]    # a list of 3 lists of 4 zeros
grid[0][1] = 100.0                      # row 0, column 1
print(len(grid), len(grid[0]))          # 3 4

a, b = b, a                             # swap two names; the notebooks do this each step
```

`import time; t0 = time.perf_counter(); ...; elapsed = time.perf_counter() - t0`
is how the notebooks measure wall-clock seconds.

## NumPy

```python
import numpy as np
u = np.zeros((4, 6))                    # 4 rows, 6 columns, float64
u[0, :] = 100.0                         # whole first row
print(u.shape, u.dtype, u.nbytes)       # (4, 6) float64 192

v = u[1:-1, 1:-1]                       # a view of the interior: rows 1..2, cols 1..4
v[:] = 5.0                              # writes into u as well
print(u)

w = u.copy()                            # an independent copy
w[:] = 0                                # u is unchanged

a = np.arange(6).reshape(2, 3)          # [[0 1 2], [3 4 5]]
print(a[:, 1:], a[:, :-1])              # drop first column / drop last column
print(a + a, a * 0.5, a.sum(), a.mean())
print(np.allclose(a * 0.5 * 2, a))      # True: compare floats with a tolerance
```

Things to notice:

- Indexing is `[row, column]`, both starting at 0; negative indices count from the end.
- A slice `start:stop` excludes `stop`; `:-1` means "all but the last".
- Arithmetic between arrays of the same shape is element by element and produces a new array.
- `astype(np.float32)` converts the dtype; `float32` uses half the bytes of `float64`.

## Jupyter and Colab

- **Shift+Enter** runs a cell and moves to the next one.
- A cell that starts with `%timeit` times the line that follows it several times.
- Variables defined in one cell are available in later cells, until the
  runtime restarts. After a restart, run from the top.

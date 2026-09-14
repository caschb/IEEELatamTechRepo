# %% [markdown]
# # 0b. The running example: a heat-diffusion stencil (SOLUTION)
#
# Every session of the workshop uses one small program, so that we can spend the
# live time on measuring and speeding it up rather than on explaining it. This
# notebook shows you what it does. No physics or numerical-methods background
# is assumed. Standard CPU runtime.
#
# Save a copy in Drive first (**File > Save a copy in Drive**).

# %%
import numpy as np
import matplotlib.pyplot as plt

def init_grid(n):
    u = np.zeros((n, n))
    u[0, :] = 100.0          # the top edge is held at 100 degrees; everything else starts at 0
    return u

u = init_grid(8)
print(u)

# %% [markdown]
# ## What one step does
#
# Picture a square metal plate. The top edge is kept hot. At every step, each
# interior cell takes the **average of its four neighbours** (up, down, left,
# right). Heat spreads down from the hot edge, one row per step at first. The
# edge cells never change: they are the boundary condition.
#
# Here is that rule written with plain loops. It is slow but obviously right,
# which makes it our **reference**.

# %%
def step_python(u, unew):
    n, m = len(u), len(u[0])
    for i in range(1, n - 1):
        for j in range(1, m - 1):
            unew[i][j] = 0.25 * (u[i-1][j] + u[i+1][j] + u[i][j-1] + u[i][j+1])
    return unew

u = init_grid(8)
after_one = step_python(u.tolist(), u.tolist())
print(np.array(after_one))

# %% [markdown]
# Only row 1 changed: each of its interior cells now holds `0.25 * 100 = 25`.
# The corners of row 1 (columns 0 and 7) are edges and stay at 0.
#
# ## Visualising it

# %%
def run_python(n, iters):
    u, unew = init_grid(n).tolist(), init_grid(n).tolist()
    for _ in range(iters):
        step_python(u, unew)
        u, unew = unew, u
    return np.array(u)

fig, axes = plt.subplots(1, 3, figsize=(10, 3.2))
for ax, iters in zip(axes, (0, 10, 100)):
    im = ax.imshow(run_python(40, iters), vmin=0, vmax=100, cmap="inferno")
    ax.set(title=f"after {iters} steps", xticks=[], yticks=[])
fig.colorbar(im, ax=axes, label="temperature"); plt.show()

# %% [markdown]
# ## Your task: the same step with NumPy slices
#
# Loops in Python are slow. NumPy lets us update all interior cells at once with
# **slices**. `u[1:-1, 1:-1]` is every interior cell. Its neighbour *above* is
# `u[:-2, 1:-1]` (rows shifted up by one), its neighbour *below* is `u[2:, 1:-1]`,
# and its neighbour to the *left* is `u[1:-1, :-2]`.
#
# **Complete the fourth term**: the neighbour to the *right*. Replace the
# placeholder and run the cell. (Hint: shift the columns, not the rows.)

# %%
def step_numpy(u, unew):
    right = u[1:-1, 2:]            # solution: columns shifted right by one
    unew[1:-1, 1:-1] = 0.25 * (u[:-2, 1:-1] + u[2:, 1:-1] + u[1:-1, :-2] + right)
    return unew

# %% [markdown]
# ## Check against the reference
#
# Two checks. First, the interior sum after one step on an 8x8 grid is exactly
# `150.0` (six interior cells of row 1 at 25 each). That check is weak: after one
# step almost every cell is still zero, so a wrong slice can pass it. The second
# check, the full grid after 20 steps against the loop version, is the real one.

# %%
u = init_grid(8)
one = step_numpy(u, u.copy())
print("interior sum after one step:", one[1:-1, 1:-1].sum(), "(reference: 150.0)")

def run_numpy(n, iters):
    u = init_grid(n); unew = u.copy()
    for _ in range(iters):
        step_numpy(u, unew)
        u, unew = unew, u
    return u

mine, reference = run_numpy(32, 20), run_python(32, 20)
if np.allclose(mine, reference):
    print("PASS: your NumPy step matches the reference after 20 steps")
else:
    bad = np.argwhere(~np.isclose(mine, reference))
    print(f"FAIL: {len(bad)} cells differ, first at (row, col) = {tuple(int(v) for v in bad[0])}. "
          "Check which neighbour your fourth slice really selects.")

# %% [markdown]
# ## What to remember for the workshop
#
# - `init_grid(n)`: `n x n` grid, hot top edge, zero elsewhere.
# - `step_*(u, unew)`: reads `u`, writes `unew`; the caller swaps them each step.
# - The pure-Python version is the reference. Every faster version must agree
#   with it, and we will check that before timing anything.
#
# Save the notebook. Then read `prep/self_check.md`.

# %% [markdown]
# # 0a. Is my Colab ready? (preparation, 10 to 15 minutes)
#
# Goal: before the workshop, make sure you can open, run, edit and save a
# notebook in Google Colab, and that the packages the course needs install on a
# **standard CPU runtime**. You do **not** need a GPU for this. A completed CPU
# check is all the preparation requires.
#
# ## Step 1: make your own copy
#
# Colab opened this notebook read-only from GitHub. Choose **File > Save a copy
# in Drive** so your edits and outputs are kept. Work in the copy from now on.
#
# ## Step 2: run a cell
#
# Click the cell below and press **Shift+Enter** (or the play button).

# %%
print("Hello from Colab. This cell ran.")

# %% [markdown]
# ## Step 3: edit a cell
#
# Change the number in the next cell to your favourite one, then run it. The
# check on the second line should say `PASS`.

# %%
favourite = 7            # <- change this
print("PASS: you edited and ran a cell" if favourite != 7 else "not yet: change the number and run again")

# %% [markdown]
# ## Step 4: install what the course needs
#
# Every course notebook starts with a setup cell like this one. It imports each
# package and installs it with pip **only if the import fails**, so on a standard
# Colab runtime it is usually quick. You must run it again whenever the runtime
# restarts, because a fresh runtime has none of your installs.

# %%
import importlib, importlib.util, os, platform, subprocess, sys

def ensure(module, package=None):
    """Import `module`, installing `package` with pip only if the import fails."""
    if importlib.util.find_spec(module) is None:
        print("installing", package or module)
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", package or module], check=True)
    return importlib.import_module(module)

np = ensure("numpy")
numba = ensure("numba")
psutil = ensure("psutil")
ensure("matplotlib"); import matplotlib.pyplot as plt
ensure("line_profiler")
print("all packages import")

# %% [markdown]
# ## Step 5: look at the machine you were given
#
# Colab hands out different virtual machines at different times. Knowing what you
# have is part of measuring honestly.

# %%
IN_COLAB = "COLAB_RELEASE_TAG" in os.environ or "google.colab" in sys.modules
print("running in Colab:", IN_COLAB)
print("python           ", platform.python_version())
print("numpy / numba    ", np.__version__, "/", numba.__version__)
print("logical CPUs     ", os.cpu_count())
print("Numba thread cap ", numba.config.NUMBA_NUM_THREADS)
print("RAM              ", round(psutil.virtual_memory().total / 2**30, 1), "GB")

# %% [markdown]
# ## Step 6: a tiny compiled function
#
# This checks that Numba can compile on this runtime. The first call takes a
# second or two (compilation); the second is fast.

# %%
import time
from numba import njit

@njit
def sum_of_squares(a):
    s = 0.0
    for i in range(a.shape[0]):
        s += a[i] * a[i]
    return s

x = np.random.default_rng(0).random(2_000_000)
t0 = time.perf_counter(); r1 = sum_of_squares(x); t_first = time.perf_counter() - t0
t0 = time.perf_counter(); r2 = sum_of_squares(x); t_second = time.perf_counter() - t0
assert np.isclose(r1, (x * x).sum())
print(f"first call (compiles): {t_first*1e3:7.1f} ms   second call: {t_second*1e3:6.1f} ms   PASS")

# %% [markdown]
# ## Step 7: restart and rerun
#
# Choose **Runtime > Restart session**, then **Runtime > Run all**. Everything
# above should pass again without you doing anything else. This is exactly what
# you will do during the workshop if a runtime disconnects.
#
# If instead you choose **Runtime > Disconnect and delete runtime**, the next
# run will need to install packages again. That is expected.
#
# ## Step 8: your status report
#
# Run the cell below and keep its output. If something failed, paste it where
# the preparation README tells you to report problems.

# %%
report = {
    "colab": IN_COLAB, "python": platform.python_version(),
    "numpy": np.__version__, "numba": numba.__version__,
    "cpus": os.cpu_count(), "ram_gb": round(psutil.virtual_memory().total / 2**30, 1),
    "numba_compiles": bool(np.isclose(r1, r2)),
}
print("STATUS " + " ".join(f"{k}={v}" for k, v in report.items()))
print("READY for the workshop" if report["numba_compiles"] else "NOT READY: see the error above")

# %% [markdown]
# **Done.** Save the notebook (Ctrl+S). Next: `00_stencil_practice`.

# Preparation package (45 to 60 minutes)

Do this before the workshop, ideally a few days ahead so there is time to fix
problems. You need a browser, a Google account, and a normal computer. A
standard **CPU** Colab runtime is enough; you do not need a GPU for anything
here, and a completed CPU check is all the workshop requires.

## What you will be able to do afterwards

- Open a course notebook in Colab, save your own copy, run and edit cells,
  install the packages, and recover after a runtime restart.
- Recognise the small program (a heat-diffusion stencil) that every live
  session builds on, and write one line of NumPy slicing for it.
- Explain, in a sentence each: what a speedup is, why a GPU has its own memory,
  and what a timing should and should not include.

## Prerequisites

Basic Python: defining a function, a `for` loop, a list. If that is rusty, do
the optional [Python and NumPy refresher](python_numpy_refresher.md) first. It
is not counted in the hour.

## Do these in order

| # | Item | Time | Done when |
|---|---|---:|---|
| 1 | Read [`primer.md`](primer.md) | 15 min | You can answer the four questions at its end |
| 2 | [Is my Colab ready?](https://colab.research.google.com/github/caschb/IEEELatamTechRepo/blob/main/notebooks/00_colab_ready.ipynb) | 10-15 min | The last cell prints `READY for the workshop` and you saved the notebook |
| 3 | [The running example](https://colab.research.google.com/github/caschb/IEEELatamTechRepo/blob/main/notebooks/00_stencil_practice.ipynb) | 10-15 min | The check cell prints `PASS` and you saved the notebook |
| 4 | [`self_check.md`](self_check.md) | 5-10 min | You read the explanation for any question you got wrong |

## Checklist

- [ ] I saved my own copy of each notebook (File > Save a copy in Drive).
- [ ] The readiness notebook printed `READY for the workshop`.
- [ ] I restarted the runtime and ran everything again; it still passed.
- [ ] The stencil practice check printed `PASS`.
- [ ] I copied the `STATUS ...` line from the readiness notebook somewhere I can find it.

## Saving and downloading your work

Colab saves your copy to Google Drive automatically (also Ctrl+S). To keep a
file the notebook wrote (for example a timing table), open the **Files** panel
on the left, right-click the file, **Download**. Files in the runtime disappear
when the runtime is deleted; your notebook copy in Drive does not.

## Two things that surprise people

- **A new runtime has nothing installed.** The setup cell at the top of every
  notebook installs what is missing. Run it again after any restart. It is fast
  the second time.
- **Runtimes disconnect.** Colab ends idle sessions and enforces usage limits.
  Nothing is lost that you saved; rerun from the top.

## If something does not work

Paste the `STATUS ...` line (or the error) and the step number into the
workshop's help channel that the organisers announced, or email the instructor.
Say which browser you use. During the event, the first ten minutes are for
fixing exactly these problems, so do not worry if something is still open.

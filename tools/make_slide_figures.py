"""Regenerate the figures used by instructor/slides.md.

    uv run --project env python tools/make_slide_figures.py

Charts read data/reference_timings/*.csv, so they show the recorded reference
machine, not the machine running this script. The only live measurement is the
warm-up figure (fig_timing_runs), which is labelled with the host it ran on.
"""

import csv
import platform
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle

ROOT = Path(__file__).resolve().parent.parent
TIMINGS = ROOT / "data" / "reference_timings"
OUT = ROOT / "instructor" / "figures"

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#b9b8b3"
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
BLUE_LIGHT, ORANGE_LIGHT, AQUA_LIGHT = "#cde2fb", "#fbd9cb", "#c9eedf"
GRAY_FILL = "#f0efec"

plt.rcParams.update({
    "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE,
    "font.size": 15,
    "axes.edgecolor": MUTED,
    "axes.labelcolor": INK2,
    "axes.titlecolor": INK,
    "axes.titlesize": 17,
    "axes.titleweight": "bold",
    "axes.titlelocation": "left",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.color": "#e6e5e1",
    "grid.linewidth": 0.8,
    "xtick.color": INK2,
    "ytick.color": INK2,
    "legend.frameon": False,
    "lines.linewidth": 2,
    "svg.hashsalt": "slides",
})


def save(fig, name):
    fig.savefig(OUT / f"{name}.svg", bbox_inches="tight", metadata={"Date": None})
    plt.close(fig)


def read_timings(name):
    lines = (TIMINGS / name).read_text().splitlines()
    meta = dict(l[2:].split(": ", 1) for l in lines if l.startswith("# "))
    rows = list(csv.DictReader(l for l in lines if not l.startswith("#")))
    return meta, rows


def source_note(ax, meta, y=-0.2):
    ax.annotate(f"Referencia: {meta['gpu']}, {meta['cpus']} núcleos lógicos",
                xy=(0, y), xycoords="axes fraction", fontsize=11, color=INK2)


def box(ax, x, y, w, h, text, fc, ec=None, fs=15, color=INK, weight="normal"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=0.08",
                                fc=fc, ec=ec or fc, lw=1.5))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs,
            color=color, weight=weight)


def arrow(ax, p, q, color=INK2, style="-|>", lw=2, rad=0.0):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle=style, mutation_scale=18, color=color,
                                 lw=lw, connectionstyle=f"arc3,rad={rad}"))


def canvas(w, h, xlim, ylim):
    fig, ax = plt.subplots(figsize=(w, h))
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.axis("off")
    return fig, ax


def fig_roadmap():
    fig, ax = canvas(13, 2.2, (0, 13), (0, 2.2))
    steps = [("Medir", "referencia\ncorrecta"), ("Vectorizar", "NumPy"),
             ("Compilar", "Numba"), ("Multinúcleo", "prange"),
             ("GPU", "CuPy"), ("Varias\nmáquinas", "MPI, Dask")]
    for i, (title, sub) in enumerate(steps):
        x = 0.1 + i * 2.15
        box(ax, x, 0.8, 1.8, 1.2, title, BLUE if i == 0 else BLUE_LIGHT,
            color=SURFACE if i == 0 else INK, weight="bold")
        ax.text(x + 0.9, 0.45, sub, ha="center", va="center", fontsize=13, color=INK2)
        if i < len(steps) - 1:
            arrow(ax, (x + 1.82, 1.4), (x + 2.13, 1.4))
    save(fig, "roadmap")


def diffuse(n, steps):
    u = np.zeros((n, n))
    u[0, :] = 100.0
    unew = u.copy()
    for _ in range(steps):
        unew[1:-1, 1:-1] = 0.25 * (u[:-2, 1:-1] + u[2:, 1:-1] + u[1:-1, :-2] + u[1:-1, 2:])
        u, unew = unew, u
    return u


def fig_stencil():
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.6), gridspec_kw={"width_ratios": [1, 1, 1]})
    ax = axes[0]
    ax.set_xlim(-0.5, 4.5)
    ax.set_ylim(-0.5, 4.5)
    ax.set_aspect("equal")
    ax.axis("off")
    for i in range(5):
        for j in range(5):
            fc = GRAY_FILL
            if (i, j) == (2, 2):
                fc = ORANGE
            elif abs(i - 2) + abs(j - 2) == 1:
                fc = BLUE_LIGHT
            ax.add_patch(Rectangle((j - 0.45, i - 0.45), 0.9, 0.9, fc=fc, ec=MUTED))
    for (i, j) in [(1, 2), (3, 2), (2, 1), (2, 3)]:
        ax.text(j, i, "1/4", ha="center", va="center", fontsize=13, color=INK)
    ax.text(2, 2, "nuevo", ha="center", va="center", fontsize=11, color=SURFACE, weight="bold")
    ax.set_title("Promedio de 4 vecinos", loc="center")

    for ax, steps, title in [(axes[1], 0, "Paso 0"), (axes[2], 3000, "Después de 3000 pasos")]:
        im = ax.imshow(diffuse(64, steps), cmap="inferno", vmin=0, vmax=100)
        ax.set_title(title, loc="center")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.grid(False)
    axes[1].set_xlabel("arriba fijo en 100, los demás bordes en 0", fontsize=12)
    cb = fig.colorbar(im, ax=axes[1:], fraction=0.025, pad=0.02)
    cb.set_label("temperatura")
    cb.outline.set_visible(False)
    save(fig, "stencil")


def fig_timing_runs():
    from numba import njit

    @njit
    def step(u, unew):
        n, m = u.shape
        for i in range(1, n - 1):
            for j in range(1, m - 1):
                unew[i, j] = 0.25 * (u[i-1, j] + u[i+1, j] + u[i, j-1] + u[i, j+1])
        return unew

    u = diffuse(512, 0)
    unew = u.copy()
    times = []
    for _ in range(30):
        t0 = time.perf_counter()
        step(u, unew)
        times.append(time.perf_counter() - t0)
    times = np.array(times) * 1e3
    warm = times[1:]

    fig, ax = plt.subplots(figsize=(12, 4.6))
    x = np.arange(1, len(times) + 1)
    ax.scatter(x[1:], warm, s=60, color=BLUE, zorder=3, label="ejecuciones repetidas")
    ax.scatter(x[:1], times[:1], s=90, color=ORANGE, zorder=3, label="primera llamada")
    ax.axhline(np.median(warm), color=INK2, lw=1.5, ls="--")
    ax.axhline(warm.min(), color=AQUA, lw=1.5)
    ax.text(len(times) + 0.8, np.median(warm) * 1.15, "mediana", va="bottom", color=INK2,
            fontsize=13)
    ax.text(len(times) + 0.8, warm.min() / 1.15, "mínimo", va="top", color=INK2, fontsize=13)
    ax.annotate("incluye la compilación de Numba", xy=(1, times[0]), xytext=(3.5, times[0]),
                va="center", fontsize=13, color=INK2,
                arrowprops=dict(arrowstyle="-", color=INK2))
    ax.set_yscale("log")
    ax.set_xlabel("ejecución")
    ax.set_ylabel("tiempo por paso (ms, log)")
    ax.set_title("Numba, n = 512: la primera ejecución no es una medición")
    ax.set_xlim(0, len(times) + 4)
    ax.legend(loc="upper right")
    ax.annotate(f"Medido en {platform.node()} al generar la figura", xy=(0, -0.22),
                xycoords="axes fraction", fontsize=11, color=INK2)
    save(fig, "timing_runs")


def fig_numpy_numba():
    meta, rows = read_timings("01_threads.csv")
    sizes = [128, 512, 1024]
    t = {(r["impl"], int(r["n"])): float(r["min_s"]) * 1e3 for r in rows if r["threads"] == "1"}
    fig, ax = plt.subplots(figsize=(10, 4.8))
    xs = np.arange(len(sizes))
    w = 0.36
    for k, (impl, color, label) in enumerate([("numpy", BLUE, "NumPy"), ("numba", ORANGE, "Numba @njit")]):
        vals = [t[(impl, n)] for n in sizes]
        ax.bar(xs + (k - 0.5) * w * 1.06, vals, w, color=color, label=label, zorder=3)
    for i, n in enumerate(sizes):
        ratio = t[("numpy", n)] / t[("numba", n)]
        ax.text(xs[i], t[("numpy", n)] * 1.5, f"{ratio:.1f}x", ha="center", fontsize=14,
                color=INK, weight="bold")
    ax.set_yscale("log")
    ax.set_xticks(xs, [f"n = {n}" for n in sizes])
    ax.set_ylabel("tiempo por paso (ms, log)")
    ax.set_ylim(top=max(t.values()) * 4)
    ax.grid(axis="x", visible=False)
    ax.set_title("Un solo hilo: Numba evita los arrays intermedios")
    ax.legend(loc="upper left")
    source_note(ax, meta)
    save(fig, "numpy_numba")


def fig_amdahl():
    meta, rows = read_timings("01_threads.csv")
    pr = {int(r["threads"]): float(r["min_s"]) for r in rows if r["impl"] == "numba_prange"}
    p = np.array(sorted(pr))
    speed = pr[1] / np.array([pr[k] for k in p])

    fig, ax = plt.subplots(figsize=(11, 5.2))
    pp = np.logspace(0, np.log2(32), 200, base=2)
    for s, color in [(0.01, BLUE), (0.05, AQUA), (0.10, ORANGE)]:
        ax.plot(pp, 1 / (s + (1 - s) / pp), color=color)
        ax.text(32 * 1.08, 1 / (s + (1 - s) / 32), f"{int(s * 100)}% serial", color=INK2,
                fontsize=13, va="center")
    ax.plot(p, speed, "o-", color=INK, ms=9, lw=1.5, zorder=4)
    ax.text(p[-1] * 1.08, speed[-1], "medido\n(prange,\nn = 1024)", fontsize=13, color=INK,
            va="center")
    ax.set_xscale("log", base=2)
    ax.set_xticks([1, 2, 4, 8, 16, 32], ["1", "2", "4", "8", "16", "32"])
    ax.set_xlabel("hilos")
    ax.set_ylabel("aceleración")
    ax.set_xlim(0.9, 32)
    ax.set_ylim(0, 26)
    ax.set_title("La fracción serial pone el techo; la memoria llega antes")
    source_note(ax, meta)
    save(fig, "amdahl")


def fig_host_device():
    fig, ax = canvas(13, 3.6, (0, 13), (0, 3.6))
    box(ax, 0.2, 1.3, 1.8, 1.2, "CPU", BLUE, color=SURFACE, weight="bold")
    box(ax, 2.6, 1.3, 2.2, 1.2, "RAM\n(host)", BLUE_LIGHT)
    box(ax, 8.2, 1.3, 2.2, 1.2, "VRAM\n(dispositivo)", ORANGE_LIGHT)
    box(ax, 11.0, 1.3, 1.8, 1.2, "GPU", ORANGE, color=SURFACE, weight="bold")
    arrow(ax, (2.0, 1.9), (2.6, 1.9), style="<|-|>")
    arrow(ax, (10.4, 1.9), (11.0, 1.9), style="<|-|>")
    arrow(ax, (4.9, 2.2), (8.1, 2.2), color=INK, lw=3)
    arrow(ax, (8.1, 1.6), (4.9, 1.6), color=INK, lw=3)
    ax.text(6.5, 2.45, "cp.asarray(x)", ha="center", family="monospace", fontsize=14)
    ax.text(6.5, 1.15, "x.get()", ha="center", va="top", family="monospace", fontsize=14)
    ax.text(6.5, 0.35, "PCIe: decenas de GB/s", ha="center", fontsize=14, color=INK2)
    ax.text(11.6, 0.35, "VRAM: cientos de GB/s", ha="center", fontsize=14, color=INK2)
    ax.text(1.4, 0.35, "RAM: decenas de GB/s", ha="center", fontsize=14, color=INK2)
    save(fig, "host_device")


def fig_gpu_crossover():
    meta, rows = read_timings("02_gpu.csv")
    fig, ax = plt.subplots(figsize=(10, 5))
    for impl, color, label in [("numpy", BLUE, "NumPy (CPU)"), ("cupy", ORANGE, "CuPy (GPU)")]:
        pts = {}
        for r in rows:
            if r["impl"] == impl and r["dtype"] == "float64" and r["scope"] == "compute":
                pts.setdefault(int(r["n"]), float(r["min_s"]) * 1e3)
        ns = sorted(pts)
        ax.plot(ns, [pts[n] for n in ns], "o-", color=color, ms=9)
        ax.text(ns[-1] * 1.12, pts[ns[-1]], label, color=INK, fontsize=14, va="center")
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xticks([128, 512, 1024, 2048], ["128", "512", "1024", "2048"])
    ax.set_xlim(100, 4500)
    ax.set_xlabel("n (cuadrícula n x n)")
    ax.set_ylabel("tiempo por paso (ms, log)")
    ax.set_title("Con poca carga, lanzar el kernel domina")
    source_note(ax, meta)
    save(fig, "gpu_crossover")


def fig_timing_scopes():
    fig, ax = canvas(13, 4.4, (0, 13), (0, 4.4))
    ax.text(0.1, 3.3, "CPU", fontsize=14, weight="bold", va="center")
    ax.text(0.1, 2.3, "GPU", fontsize=14, weight="bold", va="center")
    box(ax, 1.2, 3.0, 0.5, 0.6, "", INK2)
    ax.text(1.85, 3.3, "lanzar el kernel (retorna de inmediato)", va="center", fontsize=12,
            color=INK2)
    box(ax, 1.2, 2.0, 2.4, 0.6, "subida", BLUE_LIGHT, fs=13)
    box(ax, 3.7, 2.0, 5.6, 0.6, "cálculo", ORANGE, color=SURFACE, fs=13, weight="bold")
    box(ax, 9.4, 2.0, 2.4, 0.6, "bajada", BLUE_LIGHT, fs=13)

    def bracket(x0, x1, y, label, color):
        ax.plot([x0, x0, x1, x1], [y + 0.15, y, y, y + 0.15], color=color, lw=2.5)
        ax.text(x1 + 0.15, y, label, va="center", fontsize=13, color=INK)

    bracket(1.2, 1.7, 1.45, "erróneo: solo el lanzamiento", INK2)
    bracket(3.7, 9.3, 0.95, "solo cálculo", ORANGE)
    bracket(1.2, 11.8, 0.4, "incluye transferencias", BLUE)
    for x in (1.2, 11.8):
        ax.plot([x, x], [0.3, 2.7], color=INK2, ls=":", lw=1.5)
    for x in (1.2, 11.8):
        ax.text(x, 2.85, "sincronizar", ha="center", fontsize=11, color=INK2)
    save(fig, "timing_scopes")


def fig_transfer_bars():
    meta, rows = read_timings("02_gpu.csv")
    t = {}
    for r in rows:
        if r["n"] == "2048" and r["dtype"] == "float64":
            t.setdefault((r["impl"], r["scope"]), float(r["min_s"]) * 1e3)
    bars = [("NumPy", t[("numpy", "compute")], BLUE),
            ("CuPy, incluye transferencias", t[("cupy", "transfer_inclusive")], ORANGE_LIGHT),
            ("CuPy, solo cálculo", t[("cupy", "compute")], ORANGE)]
    fig, ax = plt.subplots(figsize=(11, 3.6))
    for i, (label, val, color) in enumerate(bars):
        ax.barh(i, val, 0.6, color=color, zorder=3)
        ax.text(val * 1.15, i, f"{val:.1f} ms  ({bars[0][1] / val:.0f}x)" if i else f"{val:.0f} ms",
                va="center", fontsize=14, color=INK)
    ax.set_yticks(range(3), [b[0] for b in bars])
    ax.set_xscale("log")
    ax.set_xlim(1, 3000)
    ax.invert_yaxis()
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("tiempo por paso, n = 2048 (ms, log)")
    ax.set_title("Mismo cálculo, tres números distintos")
    source_note(ax, meta, y=-0.42)
    save(fig, "transfer_bars")


def fig_halos():
    fig, ax = canvas(13, 5.2, (0, 13), (0, 5.2))
    rows_real = 4
    for w in range(3):
        x = 0.6 + w * 4.3
        y0 = 0.6
        ax.text(x + 1.5, 4.85, f"trabajador {w}", ha="center", fontsize=15, weight="bold")
        y = 4.3
        real = ["real"] * rows_real
        if w > 0:
            real[0] = "envía"
        if w < 2:
            real[-1] = "envía"
        cells = ["halo" if w > 0 else "borde"] + real + ["halo" if w < 2 else "borde"]
        for k, kind in enumerate(cells):
            fc = {"halo": ORANGE_LIGHT, "real": BLUE_LIGHT, "envía": "#86b6ef",
                  "borde": GRAY_FILL}[kind]
            ec = ORANGE if kind == "halo" else MUTED
            ax.add_patch(Rectangle((x, y - 0.6 * (k + 1)), 3.0, 0.55, fc=fc, ec=ec,
                                   lw=2 if kind == "halo" else 1, ls="--" if kind == "halo" else "-"))
            if kind != "real":
                ax.text(x + 1.5, y - 0.6 * (k + 1) + 0.275, kind, ha="center", va="center",
                        fontsize=12, color=INK2)
    arrow(ax, (3.6, 1.575), (4.9, 3.975), color=ORANGE, rad=-0.2)
    arrow(ax, (4.9, 3.375), (3.6, 0.975), color=BLUE, rad=-0.2)
    arrow(ax, (7.9, 1.575), (9.2, 3.975), color=ORANGE, rad=-0.2)
    arrow(ax, (9.2, 3.375), (7.9, 0.975), color=BLUE, rad=-0.2)
    ax.text(6.5, 0.1, "Cada paso: por cada borde interno, una fila en cada dirección",
            ha="center", fontsize=14, color=INK2)
    save(fig, "halos")


def fig_comm_model():
    # Same model as notebook 03, section 3.3: 1 ns per cell update, and per step
    # two messages of one row each at 10 us latency + 10 GB/s.
    fig, ax = plt.subplots(figsize=(11, 5))
    P = np.array([2, 4, 8, 16, 32, 64, 128])
    for n, color in [(256, ORANGE), (2048, AQUA), (16384, BLUE)]:
        compute = n * n / P * 1e-9
        comm = 2 * (10e-6 + n * 8 / 10e9)
        frac = compute / (compute + comm)
        ax.plot(P, frac * 100, "o-", color=color, ms=8)
        ax.text(P[-1] * 1.12, frac[-1] * 100, f"n = {n}", va="center", fontsize=14, color=INK)
    ax.set_xscale("log", base=2)
    ax.set_xticks(P, [str(p) for p in P])
    ax.set_xlim(1.6, 300)
    ax.set_ylim(0, 105)
    ax.set_xlabel("trabajadores P")
    ax.set_ylabel("% del paso calculando")
    ax.set_title("Problemas pequeños pasan el tiempo esperando mensajes")
    ax.annotate("Modelo del notebook 03: 1 ns por celda, 10 us de latencia por mensaje, 10 GB/s",
                xy=(0, -0.2), xycoords="axes fraction", fontsize=11, color=INK2)
    save(fig, "comm_model")


def fig_models():
    fig, (a, b) = plt.subplots(1, 2, figsize=(14, 5))
    for ax in (a, b):
        ax.set_xlim(0, 6.5)
        ax.set_ylim(0, 5)
        ax.axis("off")
    a.set_title("Paso de mensajes (MPI)", loc="center")
    for k in range(4):
        box(a, 0.3 + k * 1.6, 2.0, 1.2, 1.0, f"rango {k}", BLUE_LIGHT, fs=13)
        if k < 3:
            arrow(a, (1.5 + k * 1.6, 2.7), (1.9 + k * 1.6, 2.7), color=ORANGE, style="-|>")
            arrow(a, (1.9 + k * 1.6, 2.3), (1.5 + k * 1.6, 2.3), color=ORANGE, style="-|>")
    a.text(3.25, 3.8, "mismo script en cada rango", ha="center", fontsize=13, color=INK2)
    a.text(3.25, 1.2, "los rangos hablan entre sí\ncada paso (halos)", ha="center",
           fontsize=13, color=INK2)

    b.set_title("Administrador de tareas (Dask)", loc="center")
    box(b, 2.25, 3.8, 2.0, 0.9, "planificador", AQUA, color=SURFACE, fs=13, weight="bold")
    for k in range(4):
        x = 0.3 + k * 1.6
        box(b, x, 2.1, 1.2, 0.9, f"tarea {k}", BLUE_LIGHT, fs=13)
        arrow(b, (3.25, 3.8), (x + 0.6, 3.0))
        arrow(b, (x + 0.6, 2.1), (3.25, 1.3))
    box(b, 2.25, 0.4, 2.0, 0.9, "resultados", GRAY_FILL, fs=13)
    b.text(6.4, 0.85, "sin comunicación\nentre tareas", ha="right", va="center", fontsize=12,
           color=INK2)
    save(fig, "models")


def fig_eager_lazy():
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    rng = np.random.default_rng(3)
    ncol, nrow = 8, 14
    keep_cols = {1, 3}
    keep_rows = set(rng.choice(nrow, 3, replace=False))
    names = ["time", "mag", "lat", "depth", "lon", "place", "type", "net"]
    for ax, lazy in zip(axes, (False, True)):
        ax.set_xlim(-0.2, ncol + 0.2)
        ax.set_ylim(-1.2, nrow + 1.4)
        ax.axis("off")
        for c in range(ncol):
            ax.text(c + 0.5, nrow + 0.4, names[c], ha="center", fontsize=11, color=INK2,
                    rotation=0)
            for r in range(nrow):
                used = c in keep_cols and r in keep_rows
                if lazy:
                    fc = ORANGE if used else SURFACE
                    ec = "#e6e5e1"
                else:
                    fc = ORANGE if used else BLUE_LIGHT
                    ec = SURFACE
                ax.add_patch(Rectangle((c + 0.05, nrow - 1 - r + 0.05), 0.9, 0.9, fc=fc, ec=ec))
    axes[0].set_title("Ansiosa: read_csv lee todo", loc="center")
    axes[1].set_title("Diferida: scan_csv + collect", loc="center")
    axes[0].text(ncol / 2, -0.8, "se lee a memoria y luego se descarta casi todo", ha="center",
                 fontsize=13, color=INK2)
    axes[1].text(ncol / 2, -0.8, "filtro y columnas empujados al escaneo", ha="center",
                 fontsize=13, color=INK2)
    save(fig, "eager_lazy")


def fig_oversubscription():
    fig, ax = canvas(13, 4.4, (0, 13), (0, 4.4))
    ax.text(0.2, 4.05, "Máquina: 4 núcleos físicos, 8 lógicos", fontsize=14, weight="bold")
    for c in range(4):
        x = 0.2 + c * 1.5
        ax.add_patch(FancyBboxPatch((x, 2.2), 1.3, 1.4, boxstyle="round,pad=0,rounding_size=0.08",
                                    fc=GRAY_FILL, ec=MUTED))
        for h in range(2):
            ax.add_patch(Rectangle((x + 0.12 + h * 0.56, 2.4), 0.5, 0.9, fc=BLUE_LIGHT, ec=BLUE))
        ax.text(x + 0.65, 2.0, f"núcleo {c}", ha="center", va="top", fontsize=11, color=INK2)
    ax.text(3.1, 0.9, "2 hilos de hardware comparten\nlas unidades de ejecución",
            ha="center", fontsize=12, color=INK2)

    ax.text(6.9, 4.05, "n_jobs=-1 con BLAS de 8 hilos", fontsize=14, weight="bold")
    for j in range(8):
        for t in range(8):
            ax.add_patch(Rectangle((6.9 + t * 0.34, 3.35 - j * 0.34), 0.28, 0.28, fc=ORANGE,
                                   ec=SURFACE))
    ax.text(9.8, 2.2, "8 procesos\nx 8 hilos\n= 64 hilos", fontsize=15, va="center", color=INK)
    ax.text(6.9, 0.45, "compitiendo por 4 núcleos: más lento, no más rápido", fontsize=13,
            color=INK2)
    save(fig, "oversubscription")


def fig_decision():
    fig, ax = canvas(14, 6.2, (0, 14), (0, 6.2))
    q = dict(fc=GRAY_FILL, fs=13)
    a = dict(fc=BLUE_LIGHT, fs=13)
    box(ax, 5.0, 5.2, 4.0, 0.8, "¿Una máquina alcanza?", **q)
    box(ax, 1.0, 3.6, 4.2, 0.8, "¿Bucle sobre elementos\nde un array?", **q)
    box(ax, 9.3, 3.6, 4.2, 0.8, "¿Las partes se\ncomunican cada paso?", **q)
    arrow(ax, (6.0, 5.2), (3.1, 4.4))
    ax.text(4.1, 4.9, "sí", fontsize=13, color=INK2)
    arrow(ax, (8.0, 5.2), (11.4, 4.4))
    ax.text(9.8, 4.9, "no", fontsize=13, color=INK2)

    box(ax, 0.0, 1.8, 3.0, 1.0, "NumPy, luego\nNumba @njit", **a)
    box(ax, 3.3, 1.8, 3.2, 1.0, "tareas Python\nindependientes:\njoblib, procesos", **a)
    arrow(ax, (2.2, 3.6), (1.5, 2.8))
    ax.text(1.2, 3.2, "sí", fontsize=13, color=INK2)
    arrow(ax, (4.0, 3.6), (4.9, 2.8))
    ax.text(4.7, 3.2, "no", fontsize=13, color=INK2)

    box(ax, 0.0, 0.2, 3.0, 1.0, "¿varios núcleos?\nprange, hilos <= núcleos", fc=AQUA_LIGHT, fs=12)
    box(ax, 3.3, 0.2, 3.2, 1.0, "¿millones de operaciones\nidénticas? CuPy, datos\nen el dispositivo",
        fc=ORANGE_LIGHT, fs=12)
    arrow(ax, (1.5, 1.8), (1.5, 1.2))
    arrow(ax, (2.8, 1.8), (4.2, 1.2))

    box(ax, 8.0, 1.8, 2.8, 1.0, "mpi4py\n(+ Numba o CuPy\npor rango)", **a)
    box(ax, 11.2, 1.8, 2.8, 1.0, "administrador de\ntareas: Dask,\narrays de tareas", **a)
    arrow(ax, (10.6, 3.6), (9.4, 2.8))
    ax.text(9.6, 3.2, "sí", fontsize=13, color=INK2)
    arrow(ax, (12.2, 3.6), (12.6, 2.8))
    ax.text(12.6, 3.2, "no", fontsize=13, color=INK2)

    box(ax, 8.0, 0.2, 6.0, 1.0, "En todos los casos: medir antes y\ndespués, en el mismo entorno",
        fc=BLUE, color=SURFACE, fs=13, weight="bold")
    save(fig, "decision")


def fig_capstone():
    meta, rows = read_timings("04_capstone.csv")
    t = {(r["workload"], r["impl"]): float(r["min_s"]) * 1e3 for r in rows}
    fig, (a, b) = plt.subplots(1, 2, figsize=(14, 4.6))
    panels = [
        (a, "A: 48 cuadrículas pequeñas (n = 128)", "A_sweep",
         [("numpy", "NumPy", BLUE), ("numba", "Numba", ORANGE), ("numba_par", "Numba prange", AQUA)]),
        (b, "B: una cuadrícula grande (n = 2048)", "B_single",
         [("numpy", "NumPy", BLUE), ("numba_par", "Numba prange", AQUA), ("cupy", "CuPy", ORANGE),
          ("cupy_transfer_inclusive", "CuPy + transf.", ORANGE_LIGHT)]),
    ]
    for ax, title, wl, impls in panels:
        vals = [t[(wl, k)] for k, _, _ in impls]
        ax.barh(range(len(impls)), vals, 0.6, color=[c for _, _, c in impls], zorder=3)
        for i, v in enumerate(vals):
            ax.text(v * 1.15, i, f"{v:.1f} ms", va="center", fontsize=13)
        ax.set_yticks(range(len(impls)), [l for _, l, _ in impls])
        ax.set_xscale("log")
        ax.set_xlim(min(vals) / 2, max(vals) * 8)
        ax.invert_yaxis()
        ax.grid(axis="y", visible=False)
        ax.set_title(title, fontsize=15)
        ax.set_xlabel("tiempo total (ms, log)")
    source_note(a, meta, y=-0.3)
    save(fig, "capstone")


if __name__ == "__main__":
    OUT.mkdir(exist_ok=True)
    for f in [fig_roadmap, fig_stencil, fig_timing_runs, fig_numpy_numba, fig_amdahl,
              fig_host_device, fig_gpu_crossover, fig_timing_scopes, fig_transfer_bars,
              fig_halos, fig_comm_model, fig_models, fig_eager_lazy, fig_oversubscription,
              fig_decision, fig_capstone]:
        f()
        print("wrote", f.__name__.removeprefix("fig_"))

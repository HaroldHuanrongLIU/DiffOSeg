"""Regenerate slide charts from frozen evidence with the project's matplotlib."""
from io import StringIO
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".course-build/matplotlib"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ASSETS = ROOT / "presentation/assets"
BLUE, LIGHT, GREY = "#002C55", "#005F96", "#D7D7D7"
plt.rcParams.update({"font.family": "DejaVu Serif", "font.size": 13,
                     "svg.hashsalt": "diffoseg-course-slides", "axes.unicode_minus": False})


def save(fig, name):
    svg = StringIO()
    fig.savefig(svg, format="svg", facecolor="white", metadata={"Date": None})
    (ASSETS / name).write_text(
        "\n".join(line.rstrip() for line in svg.getvalue().splitlines()) + "\n"
    )
    plt.close(fig)


def main():
    ASSETS.mkdir(exist_ok=True)
    pa = json.loads((ROOT / "results/course_project/pa_results.json").read_text())["primary"]
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 1.8))
    fig.subplots_adjust(left=.075, right=.98, top=.74, bottom=.26, wspace=.30)
    for idx, (ax, scale, title, limits) in enumerate(zip(
        axes, (1, 100), ("GED30: P minus A", "DiceSoft30: P minus A (pp)"),
        ((-.0065, .0065), (-.6, .6)),
    )):
        mean = pa["difference"][idx] * scale
        low, high = [v * scale for v in pa["difference_ci95"][idx]]
        ax.axvline(0, color="#9a9a9a", linestyle="--", linewidth=1.2)
        ax.errorbar(mean, 0, xerr=[[mean-low], [high-mean]], fmt="o", color=BLUE,
                    ecolor=LIGHT, elinewidth=3, capsize=7, markersize=7)
        ax.set(xlim=limits, ylim=(-.5, .5), yticks=[])
        ax.set_title(title, fontsize=16, color=BLUE, pad=22)
        ax.text(.5, 1.04, f"{mean:+.6f}" if idx == 0 else f"{mean:+.4f} pp",
                transform=ax.transAxes, ha="center", color=BLUE, fontsize=15)
        ax.set_xticks((-.006, 0, .006) if idx == 0 else (-.5, 0, .5))
        ax.tick_params(axis="x", labelsize=13)
        for s in ("left", "right", "top"):
            ax.spines[s].set_visible(False)
        ax.spines["bottom"].set_color(GREY)
    save(fig, "pa-paired-intervals.svg")

    pref = json.loads((ROOT / "results/course_project/formal_e.json").read_text())
    values = [100 * pref["means"][key] for key in ("e11", "reference", "candidate")]
    fig, ax = plt.subplots(figsize=(5.7, 3.0))
    fig.subplots_adjust(left=.24, right=.96, bottom=.23, top=.91)
    for y, val in zip((2, 1, 0), values):
        ax.scatter(val, y, s=80, color=BLUE if y == 0 else LIGHT, zorder=3)
        ax.text(val, y+.20, f"{val:.4f}%", ha="center", fontsize=13, color=BLUE)
    ax.set(yticks=(2, 1, 0), yticklabels=("Reproduced\nDiffOSeg", "Pref-A", "Final E"),
           xlim=(87.5, 91.05), ylim=(-.4, 2.55), xticks=(88, 89, 90, 91),
           xlabel="Four-fold preference Dice (%)")
    ax.tick_params(axis="y", length=0, pad=12)
    ax.grid(axis="x", color=GREY, linewidth=.7)
    for s in ("left", "right", "top"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(GREY)
    save(fig, "preference-results.svg")
    print("FIGURES_GENERATED: paired intervals and preference results")


if __name__ == "__main__":
    main()

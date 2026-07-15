"""Desktop-only: draw the two-core Pico architecture block diagram (report Fig., C1)."""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import params as PA

OUT = os.path.join(os.path.dirname(__file__), "..", "Py_Code", "images", "part4")
os.makedirs(OUT, exist_ok=True)

fig, ax = plt.subplots(figsize=(10, 4.4)); ax.axis("off")
ax.set_xlim(0, 10); ax.set_ylim(0, 5)


def box(x, y, w, h, title, lines, fc):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.06",
                                fc=fc, ec="k", lw=1.6))
    ax.text(x + w / 2, y + h - 0.35, title, ha="center", va="top", fontsize=12, fontweight="bold")
    for i, ln in enumerate(lines):
        ax.text(x + w / 2, y + h - 0.85 - 0.42 * i, ln, ha="center", va="top", fontsize=9)


box(0.4, 1.2, 4.2, 2.9, "Core 0  (main)  --  Controller",
    [r"hardware Timer @ $T_s = N\,dt$ = %d ms" % int(PA.PICO_TS * 1000),
     "read state x, compute u",
     "RBF: adapt weights (C3)  /  ADP: frozen (C4)",
     "measure exec time (avg, max)  (C5)",
     r"uses g known;  f NOT evaluated here"], "#cfe3ff")

box(5.4, 1.2, 4.2, 2.9, "Core 1  (second)  --  Process",
    [r"paced real-time loop @ $dt$ = %d ms" % int(PA.PICO_DT * 1000),
     "one scalar RK4 step of x' = f(x)+g u",
     "ZOH: hold u between updates",
     r"owns the unknown drift f(x)",
     "simulates the full nonlinear plant"], "#ffe0cc")

# arrows
ax.add_patch(FancyArrowPatch((5.4, 3.2), (4.6, 3.2), arrowstyle="-|>", mutation_scale=18, lw=2, color="navy"))
ax.text(5.0, 3.45, r"state $x=[y,\dot y,\ddot y]$", ha="center", fontsize=10, color="navy")
ax.add_patch(FancyArrowPatch((4.6, 1.9), (5.4, 1.9), arrowstyle="-|>", mutation_scale=18, lw=2, color="darkred"))
ax.text(5.0, 1.55, r"control $u$ (ZOH)", ha="center", fontsize=10, color="darkred")

ax.text(5.0, 4.6, "Raspberry Pi Pico  (RP2040, dual core, ulab)",
        ha="center", fontsize=12, fontweight="bold")
fig.tight_layout(); fig.savefig(f"{OUT}/architecture.png", dpi=140); plt.close()
print(f"saved {OUT}/architecture.png")

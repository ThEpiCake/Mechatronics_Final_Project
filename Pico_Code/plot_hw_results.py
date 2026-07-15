"""
Desktop post-processing of the ON-CHIP experiment logs (Part 4 hardware results).

Reads the CSV logs captured from pico_experiment.py running on the physical Pico
(Py_Code/data/pico_hw_rbf.csv, pico_hw_adp.csv), re-runs the matching desktop
references, and produces the report's hardware figures in Py_Code/images/part4/:

    pico_hw_rbf.png     Pico tracking vs y_d and vs the desktop two-core emulation
    pico_hw_adp.png     Pico ADP regulation vs the fine desktop rollout
    pico_hw_timing.png  on-chip controller execution-time histogram + per-tick trace

and prints every number quoted in Section 5 of the report.
"""
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "Py_Code"))

import params as PA
import pico_control as CT
from desktop_sim import run_two_core
import adp_offpolicy as ADP

DATA = os.path.join(HERE, "..", "Py_Code", "data")
OUT = os.path.join(HERE, "..", "Py_Code", "images", "part4")
os.makedirs(OUT, exist_ok=True)


def load(path):
    """Parse a pico_experiment capture: '#' metadata lines + k,y,u,exec,tot CSV."""
    meta = {}
    rows = []
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if line.startswith("#"):
                for tok in line[1:].split():
                    if "=" in tok:
                        k, v = tok.split("=", 1)
                        meta[k] = v
            elif line and line[0].isdigit():
                rows.append([float(v) for v in line.split(",")])
    a = np.array(rows)
    d = dict(meta=meta, k=a[:, 0], y=a[:, 1], u=a[:, 2],
             exec_us=a[:, 3], tot_us=a[:, 4])
    d["t"] = (d["k"]) * float(meta["Ts"])
    return d


def stats(v):
    v = v[1:]                       # skip the cold first tick
    return v.mean(), v.max()


if __name__ == "__main__":
    Ts, N = PA.PICO_TS, PA.PICO_N
    hw_r = load(os.path.join(DATA, "pico_hw_rbf.csv"))
    hw_a = load(os.path.join(DATA, "pico_hw_adp.csv"))
    T_r = len(hw_r["y"]) * Ts
    T_a = len(hw_a["y"]) * Ts
    print(f"loaded: RBF {len(hw_r['y'])} ticks ({T_r:.0f} s), "
          f"ADP {len(hw_a['y'])} ticks ({T_a:.0f} s); meta {hw_r['meta']}")

    # ---- desktop references under the SAME two-core scheme ----
    dr = run_two_core("rbf", T=T_r, x0=[0.0, 0.0, 0.0])
    da = run_two_core("adp", T=T_a, x0=[0.7, 0.5, -1.0])
    tRoll, XRoll, _, _ = ADP.rollout([0.7, 0.5, -1.0], np.array(PA.ADP_W),
                                     T=T_a, dt=1e-3)

    # desktop y at the controller ticks: y_log[i] holds t=(i+1)*dt -> tick j = j*N-1
    def at_ticks(d, K):
        idx = np.arange(K) * N - 1
        y = d["y"][np.clip(idx, 0, len(d["y"]) - 1)]
        y[0] = 0.0 if d is dr else 0.7
        return y

    yd = np.array([CT.y_ref(t)[0] for t in hw_r["t"]])

    # ---- RBF numbers ----
    m = int(0.6 * len(hw_r["y"]))
    rms_hw = np.sqrt(np.mean((hw_r["y"][m:] - yd[m:]) ** 2))
    dev_r = np.abs(hw_r["y"] - at_ticks(dr, len(hw_r["y"])))
    ex_avg_r, ex_max_r = stats(hw_r["exec_us"])
    tot_avg_r, tot_max_r = stats(hw_r["tot_us"])
    print(f"[RBF hw] steady RMS |e| = {rms_hw:.3e} m   "
          f"max |y_pico - y_desktop| = {dev_r.max():.3e} m")
    print(f"[RBF hw] exec avg/max = {ex_avg_r:.0f}/{ex_max_r:.0f} us ; "
          f"tick incl gc avg/max = {tot_avg_r:.0f}/{tot_max_r:.0f} us  (Ts={Ts*1e6:.0f} us)")

    # ---- ADP numbers ----
    idx_roll = np.clip((hw_a["k"] * Ts / 1e-3).astype(int), 0, len(XRoll) - 1)
    dev_a = np.abs(hw_a["y"] - XRoll[idx_roll, 0])
    ex_avg_a, ex_max_a = stats(hw_a["exec_us"])
    tot_avg_a, tot_max_a = stats(hw_a["tot_us"])
    print(f"[ADP hw] max |y_pico - y_rollout| = {dev_a.max():.3e} m")
    print(f"[ADP hw] exec avg/max = {ex_avg_a:.0f}/{ex_max_a:.0f} us ; "
          f"tick incl gc avg/max = {tot_avg_a:.0f}/{tot_max_a:.0f} us  (Ts={Ts*1e6:.0f} us)")

    # ---- figure: RBF tracking ----
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.2))
    ax[0].plot(hw_r["t"], yd, "k--", lw=1.4, label=r"$y_d$")
    ax[0].plot(hw_r["t"], hw_r["y"], "b-", lw=1.0, label="Pico (on chip)")
    ax[0].plot(dr["t"], dr["y"], "r:", lw=1.4, label="desktop emulation")
    ax[0].set_title(f"On-chip RBF-NN tracking (steady RMS {rms_hw:.2e} m)")
    ax[0].set_xlabel("t [s]"); ax[0].set_ylabel("y [m]")
    ax[0].legend(); ax[0].grid(alpha=.3)
    ax[1].plot(hw_r["t"], dev_r, "m-", lw=0.9)
    ax[1].set_title(f"|Pico $-$ desktop|  (max {dev_r.max():.1e} m)")
    ax[1].set_xlabel("t [s]"); ax[1].set_ylabel("deviation [m]"); ax[1].grid(alpha=.3)
    fig.tight_layout(); fig.savefig(f"{OUT}/pico_hw_rbf.png", dpi=140); plt.close()

    # ---- figure: ADP regulation ----
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.2))
    ax[0].plot(tRoll, XRoll[:, 0], "r--", lw=1.8, label="desktop rollout")
    ax[0].plot(hw_a["t"], hw_a["y"], "b-", lw=1.0, label="Pico (on chip)")
    ax[0].set_title(f"On-chip ADP regulation (max dev {dev_a.max():.1e} m)")
    ax[0].set_xlabel("t [s]"); ax[0].set_ylabel("y [m]")
    ax[0].legend(); ax[0].grid(alpha=.3)
    ax[1].plot(hw_a["t"], hw_a["u"], "g-", lw=1.0)
    ax[1].set_title("Pico ADP control input (on chip)")
    ax[1].set_xlabel("t [s]"); ax[1].set_ylabel("u [V]"); ax[1].grid(alpha=.3)
    fig.tight_layout(); fig.savefig(f"{OUT}/pico_hw_adp.png", dpi=140); plt.close()

    # ---- figure: on-chip timing ----
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.2))
    ax[0].hist(hw_r["exec_us"][1:], bins=40, alpha=.7,
               label=f"RBF update (avg {ex_avg_r:.0f}, max {ex_max_r:.0f} us)")
    ax[0].hist(hw_a["exec_us"][1:], bins=40, alpha=.7,
               label=f"ADP update (avg {ex_avg_a:.0f}, max {ex_max_a:.0f} us)")
    ax[0].axvline(Ts * 1e6, color="r", ls="--", lw=2, label=f"$T_s$ = {Ts*1e6:.0f} us")
    ax[0].set_xlabel("controller-update execution time [us]"); ax[0].set_ylabel("count")
    ax[0].set_title("On-chip execution time (RP2040 @ 200 MHz, ulab)")
    ax[0].legend(); ax[0].grid(alpha=.3)
    k = hw_r["k"][1:200]
    ax[1].plot(k, hw_r["exec_us"][1:200], "b.-", ms=2.5, lw=0.6, label="RBF update")
    ax[1].plot(k, hw_r["tot_us"][1:200], "c.", ms=2.5, label="full tick (incl. GC)")
    ax[1].axhline(Ts * 1e6, color="r", ls="--", lw=2, label=f"$T_s$")
    ax[1].set_xlabel("controller tick $k$"); ax[1].set_ylabel("time [us]")
    ax[1].set_title("Per-tick trace (GC every %d ticks)" % 10)
    ax[1].legend(); ax[1].grid(alpha=.3)
    fig.tight_layout(); fig.savefig(f"{OUT}/pico_hw_timing.png", dpi=140); plt.close()
    print(f"saved hardware figures -> {OUT}/")

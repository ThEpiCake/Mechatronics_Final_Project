"""
Desktop validation of the Pico code (Part 4) WITHOUT the board.

It runs the exact two-core scheme numerically -- plant_core1.rk4_step advancing the
plant at dt (the core-1 task) and a pico_control controller updating every N steps
at Ts (the core-0 task) -- for both controllers, checks the responses against the
Section 3/4 computer results, and measures the per-update execution time (the C5
timing method; the representative on-chip numbers are recorded on the Pico).

The MicroPython files (plant_core1.py, pico_control.py, params.py) are imported
unchanged; only the two-core *orchestration* differs from main.py (a plain loop
here, hardware timers + _thread on the board).
"""
import os
import sys
import time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import plant_core1 as PL
import pico_control as CT
import params as PA

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "Py_Code"))
import plant as DP           # desktop reference
import config as DC
import adp_offpolicy as ADP

OUT = os.path.join(os.path.dirname(__file__), "..", "Py_Code", "images", "part4")
os.makedirs(OUT, exist_ok=True)


def run_two_core(kind, T, x0):
    """Numerically emulate the two-core real-time scheme."""
    dt, N, Ts = PA.PICO_DT, PA.PICO_N, PA.PICO_TS
    n = int(round(T / dt))
    x = np.array(x0, dtype=float)
    ctrl = CT.RBFController() if kind == "rbf" else CT.ADPController()
    u = 0.0
    t_log, y_log, yd_log, u_log, texec = [], [], [], [], []
    for k in range(n):
        t = k * dt
        if k % N == 0:                       # core 0: controller update at Ts
            t0 = time.perf_counter()
            u = ctrl.update(x, t, Ts) if kind == "rbf" else ctrl.control(x)
            texec.append(time.perf_counter() - t0)
        x = PL.rk4_step(x, u, dt)            # core 1: one plant step at dt (ZOH u)
        t_log.append((k + 1) * dt); y_log.append(float(x[0])); u_log.append(u)
        yd_log.append(CT.y_ref((k + 1) * dt)[0])
    return dict(t=np.array(t_log), y=np.array(y_log), yd=np.array(yd_log),
                u=np.array(u_log), texec=np.array(texec) * 1e6)   # us


if __name__ == "__main__":
    print("== Pico Part 4: desktop validation of the two-core scheme ==")
    print(f"dt={PA.PICO_DT*1e3:.1f} ms (core1), Ts={PA.PICO_TS*1e3:.1f} ms (core0), N={PA.PICO_N}")

    # ---- C3: adaptive RBF-NN tracking on the two-core scheme ----
    dr = run_two_core("rbf", T=30.0, x0=[0.0, 0.0, 0.0])
    m = int(0.6 * len(dr["t"]))
    rms_rbf = np.sqrt(np.mean((dr["y"][m:] - dr["yd"][m:]) ** 2))
    print(f"[C3 RBF]  steady RMS tracking error = {rms_rbf:.3e} m "
          f"(desktop discrete @200Hz was ~9e-3)")

    # ---- C4: frozen ADP policy on the two-core scheme, vs desktop rollout ----
    x0 = [0.7, 0.5, -1.0]
    da = run_two_core("adp", T=6.0, x0=x0)
    w = np.array(PA.ADP_W)
    tR, XR, UR, _ = ADP.rollout(x0, w, T=6.0, dt=1e-3)   # desktop reference
    # align on common TIMES: da["y"][k] is at t=(k+1)*PICO_DT, XR[i] at t=i*1e-3
    r = int(round(PA.PICO_DT / 1e-3))
    idx = (np.arange(len(da["y"])) + 1) * r
    L = np.sum(idx < len(XR))
    max_dev = np.max(np.abs(da["y"][:L] - XR[idx[:L], 0]))
    print(f"[C4 ADP]  Pico vs desktop max |y-y_ref| = {max_dev:.3e} m (should be ~0)")

    # ---- C5: execution-time measurement ----
    for kind, d in (("RBF", dr), ("ADP", da)):
        avg, mx = np.mean(d["texec"]), np.max(d["texec"])
        print(f"[C5 {kind}]  update time: avg={avg:.1f} us  max={mx:.1f} us  "
              f"(< Ts={PA.PICO_TS*1e6:.0f} us ? {'YES' if mx < PA.PICO_TS*1e6 else 'NO'})  "
              f"[desktop NumPy proxy]")

    # ---------------- figures ----------------
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(dr["t"], dr["yd"], "k--", lw=1.4, label=r"$y_d$")
    ax.plot(dr["t"], dr["y"], "b-", lw=1.1, label="Pico two-core (RBF)")
    ax.set_title(f"Pico two-core RBF-NN tracking (steady RMS {rms_rbf:.2e} m)")
    ax.set_xlabel("t [s]"); ax.set_ylabel("y [m]"); ax.legend(); ax.grid(alpha=.3)
    fig.tight_layout(); fig.savefig(f"{OUT}/pico_rbf.png", dpi=140); plt.close()

    fig, ax = plt.subplots(1, 2, figsize=(12, 4.2))
    ax[0].plot(tR, XR[:, 0], "r--", lw=1.8, label="desktop rollout")
    ax[0].plot(da["t"], da["y"], "b-", lw=1.1, label="Pico two-core (ADP)")
    ax[0].set_title(f"ADP regulation: Pico vs desktop (max dev {max_dev:.1e} m)")
    ax[0].set_xlabel("t [s]"); ax[0].set_ylabel("y [m]"); ax[0].legend(); ax[0].grid(alpha=.3)
    ax[1].plot(da["t"], da["u"], "g-", lw=1.1)
    ax[1].set_title("Pico ADP control input"); ax[1].set_xlabel("t [s]"); ax[1].set_ylabel("u [V]")
    ax[1].grid(alpha=.3)
    fig.tight_layout(); fig.savefig(f"{OUT}/pico_adp.png", dpi=140); plt.close()

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(dr["texec"], bins=40, alpha=.7, label=f"RBF (max {np.max(dr['texec']):.0f} us)")
    ax.hist(da["texec"], bins=40, alpha=.7, label=f"ADP (max {np.max(da['texec']):.0f} us)")
    ax.axvline(PA.PICO_TS * 1e6, color="r", ls="--", lw=2, label=f"$T_s$={PA.PICO_TS*1e6:.0f} us")
    ax.set_title("Per-update execution time (desktop NumPy proxy)")
    ax.set_xlabel("execution time [us]"); ax.set_ylabel("count"); ax.legend(); ax.grid(alpha=.3)
    fig.tight_layout(); fig.savefig(f"{OUT}/pico_timing.png", dpi=140); plt.close()
    print(f"saved Part-4 figures -> {OUT}/")

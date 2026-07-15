"""
Branch A -- discrete-time (emulation / indirect digital-design) realization of the
adaptive RBF-NN tracking controller (report Section 3.3-3.4, grading A4-A5).

Emulation design (Franklin, Powell & Emami-Naeini 1998, ch. 8): the controller is
designed in continuous time (rbf_adaptive.py) and then implemented at a sampling
period Ts, holding u constant between samples (ZOH). The continuous plant is
integrated exactly (RK4 sub-stepping) between samples; only the controller and the
weight update run at the sample instants.

Per-sample calculation order:
  (1) read the full state x_k (all three states measured);
  (2) evaluate the reference, tracking errors xi_k and RBF features phi_k;
  (3) control law  u_k = sat( (What_k . phi_k + y_d''' - l2 e'' - l1 e' - l0 e)/c );
  (4) hold u_k and advance the true plant over one Ts by ZOH (RK4 sub-steps);
  (5) forward-Euler update of the weights  What_{k+1} = What_k + Ts * weight_dot(...).

As Ts grows the ZOH delay and the forward-Euler weight update eventually destabilise
the loop -- shown by sweeping Ts and by the three representative tracking plots.
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import plant as P
import config as C
import rbf_adaptive as R    # shares LAMBDA, bP, CENTERS, control law, weight_dot


def simulate_discrete(Ts, T=30.0, adaptive=True, nsub=None):
    """Discrete emulation of the RBF tracking controller at sample period Ts."""
    L2, L1, L0 = C.LAM
    nsub = max(2, int(np.ceil(Ts / 5e-4))) if nsub is None else nsub  # keep RK4 error small
    n = int(round(T / Ts))
    x = np.zeros(3)
    W = np.zeros(C.N_RBF)
    t_log, y_log, yd_log, u_log = [], [], [], []
    diverged = False
    for k in range(n):
        tk = k * Ts
        yd, yd1, yd2, yd3 = C.y_ref(tk)
        e, ed, edd = x[0] - yd, x[1] - yd1, x[2] - yd2
        xi = np.array([e, ed, edd])
        if adaptive:
            phi = C.rbf_phi(x, R.CENTERS)
            Dhat = W @ phi
        else:
            phi, Dhat = None, P.drift_D(x)
        u = float(np.clip((Dhat + yd3 - L2 * edd - L1 * ed - L0 * e) / P.c, -C.U_MAX, C.U_MAX))
        if not np.isfinite(u) or abs(x[0]) > 1e3:
            diverged = True
            break
        # advance the true plant over Ts with ZOH input u (RK4 sub-steps)
        h = Ts / nsub
        for _ in range(nsub):
            x = P.rk4_step(x, u, h)
        # forward-Euler weight update
        if adaptive:
            W = W + Ts * R.weight_dot(W, R.bP @ xi, phi)
        t_log.append(tk + Ts); y_log.append(x[0]); yd_log.append(C.y_ref(tk + Ts)[0]); u_log.append(u)
    return dict(t=np.array(t_log), y=np.array(y_log), yd=np.array(yd_log),
                u=np.array(u_log), diverged=diverged,
                e=(np.array(y_log) - np.array(yd_log)) if y_log else np.array([]))


def ss_rms(d, frac=0.3):
    if d["diverged"] or len(d["t"]) < 5:
        return np.inf
    m = int((1 - frac) * len(d["t"]))
    return float(np.sqrt(np.mean(d["e"][m:] ** 2)))


if __name__ == "__main__":
    print("== Branch A: discrete emulation, sampling-rate sweep ==")
    rates = [1000, 500, 200, 100, 50, 40, 30, 25, 20, 15, 10]
    print(f"{'fs[Hz]':>7} {'Ts[ms]':>7} | {'ss-RMS|e|':>12}")
    results = {}
    for fs in rates:
        Ts = 1.0 / fs
        d = simulate_discrete(Ts, T=30.0)
        results[fs] = d
        r = ss_rms(d)
        print(f"{fs:>7} {1000/fs:>7.1f} | {'DIVERGES' if not np.isfinite(r) else f'{r:.5f}':>12}")

    # continuous reference (fine-grained) for overlay
    dc = R.simulate(adaptive=True, T=30.0)

    os.makedirs("images/part2", exist_ok=True)
    # pick three representative rates: fine, intermediate, degrading -> set in config.TS_LIST
    reps = [200, 40, 15]
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.2), sharey=True)
    for a, fs in zip(ax, reps):
        d = results[fs]
        a.plot(dc["t"], dc["y"], "g-", lw=1.0, alpha=.6, label="continuous")
        a.plot(d["t"], d["yd"], "k--", lw=1.3, label=r"$y_d$")
        a.plot(d["t"], d["y"], "b-", lw=1.1, label=f"discrete {fs} Hz")
        a.set_title(f"$f_s={fs}$ Hz ($T_s={1000/fs:.0f}$ ms)" +
                    ("  -- degrades" if not np.isfinite(ss_rms(d)) or ss_rms(d) > 0.05 else ""))
        a.set_xlabel("t [s]"); a.legend(fontsize=8); a.grid(alpha=.3); a.set_ylim(-0.9, 0.9)
    ax[0].set_ylabel("y [m]")
    fig.suptitle("Continuous vs discrete-time realization at three sampling periods")
    fig.tight_layout(); fig.savefig("images/part2/discrete_compare.png", dpi=140); plt.close()

    # RMS vs Ts summary
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for fs in rates:
        r = ss_rms(results[fs])
        bad = (not np.isfinite(r)) or r > 0.05          # degraded / diverged
        ax.semilogy(1000 / fs, r if np.isfinite(r) else 1.0,
                    "rx" if bad else "bo", ms=9, mew=2)
    ax.axhline(ss_rms(dict(t=dc["t"], e=dc["e"], diverged=False)), color="g", ls="--",
               label="continuous ss-RMS")
    ax.set_title("Discrete emulation: steady RMS tracking error vs sample period\n"
                 "(blue = matches the continuous design, red x = degrades/diverges)")
    ax.set_xlabel(r"$T_s$ [ms]"); ax.set_ylabel("ss-RMS |e| [m]")
    ax.legend(); ax.grid(alpha=.3, which="both")
    fig.tight_layout(); fig.savefig("images/part2/discrete_rms.png", dpi=140); plt.close()
    print("saved discrete figures -> images/part2/")

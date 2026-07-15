"""
Branch A -- adaptive RBF neural-network TRACKING controller (report Section 3,
grading A1-A3, A5-continuous).

Plant (control-affine, relative degree 3):   x' = f(x) + g u ,  y = x1 .
Unknown drift D(x) (Eq. drift in plant.py) approximated by an RBF network
    Dhat(x) = What^T phi(x) ,   phi = Gaussian RBFs on a grid (config.rbf_*).

Tracking errors  xi = [e, e', e''] ,  e = y - y_d .  With the feedback-linearizing
control (c and phi known; What adapted online)
    u = (1/c) [ What^T phi(x) + y_d''' - l2 e'' - l1 e' - l0 e ] ,
the error dynamics are
    xi' = Lambda xi + b ( What^T phi - D ) ,   b = [0,0,1]^T ,
with Lambda the Hurwitz companion matrix of the desired error poles (config.ERR_POLES).

Lyapunov design.  With P = P^T > 0 solving  Lambda^T P + P Lambda = -Q  and
    V = xi^T P xi + (1/gamma) Wtil^T Wtil ,   Wtil = What - W* ,
the weight-adaptation law is the Lyapunov gradient law with PROJECTION,
    What' = Proj( -gamma ( b^T P xi ) phi(x) ),   |w_i| <= W_MAX,
which gives  V' <= -xi^T Q xi - 2 (b^T P xi) eps,  i.e. the tracking error and the
weights are uniformly ultimately bounded, with the residual set fixed by the RBF
approximation error eps.  (weight_dot also supports sigma-modification as an
alternative robustness term; the design in the report uses projection only,
config.SIGMA_MOD = 0.)  The nominal controller (A1) replaces What^T phi by the true
D(x) and yields the exact linear error dynamics xi' = Lambda xi.
"""

import os
import numpy as np
from scipy.integrate import solve_ivp
from scipy.linalg import solve_continuous_lyapunov
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import plant as P
import config as C

# ----- error-dynamics companion matrix, Lyapunov P, RBF centers -------------
L2, L1, L0 = C.LAM                        # gains (from poles)
LAMBDA = np.array([[0.0, 1.0, 0.0],
                   [0.0, 0.0, 1.0],
                   [-L0, -L1, -L2]])
P_LYAP = solve_continuous_lyapunov(LAMBDA.T, -C.LYAP_Q)   # Lambda^T P + P Lambda = -Q
b_vec = np.array([0.0, 0.0, 1.0])
bP = b_vec @ P_LYAP                        # row vector b^T P  (= 3rd row of P)
CENTERS = C.rbf_centers()


def _sat(u):
    return float(np.clip(u, -C.U_MAX, C.U_MAX))


def weight_dot(W, bPxi, phi):
    """Lyapunov gradient weight-adaptation law with projection to |w_i| <= W_MAX
    (and optional sigma-modification).  The projection cancels any drive that would
    push a saturated weight further out, which keeps ||W|| bounded while preserving
    the Lyapunov inequality.  Reused by the discrete controller and the Pico port."""
    tau = -C.GAMMA_ADAPT * bPxi * phi - C.GAMMA_ADAPT * C.SIGMA_MOD * W
    out = tau.copy()
    out[(W >= C.W_MAX) & (tau > 0)] = 0.0
    out[(W <= -C.W_MAX) & (tau < 0)] = 0.0
    return out


def controller(x, W, t, adaptive):
    """Return (u, xi, phi, Dhat) for the current state/weights."""
    yd, yd1, yd2, yd3 = C.y_ref(t)
    e = x[0] - yd
    ed = x[1] - yd1
    edd = x[2] - yd2
    xi = np.array([e, ed, edd])
    if adaptive:
        phi = C.rbf_phi(x, CENTERS)
        Dhat = W @ phi
    else:
        phi = None
        Dhat = P.drift_D(x)                # nominal: true drift (f known)
    u = _sat((Dhat + yd3 - L2 * edd - L1 * ed - L0 * e) / P.c)
    return u, xi, phi, Dhat


def simulate(adaptive, T=20.0, x0=(0.0, 0.0, 0.0), dt_out=2e-3):
    """Closed-loop simulation. State = [x(3), What(N_RBF)] (weights only if adaptive)."""
    n = C.N_RBF

    def ode(t, S):
        x = S[:3]
        W = S[3:] if adaptive else None
        u, xi, phi, _ = controller(x, W, t, adaptive)
        xdot = P.dynamics(x, u)
        if not adaptive:
            return list(xdot)
        Wdot = weight_dot(W, bP @ xi, phi)
        return list(xdot) + list(Wdot)

    S0 = list(x0) + ([0.0] * n if adaptive else [])
    te = np.arange(0, T + dt_out / 2, dt_out)
    sol = solve_ivp(ode, [0, T], S0, t_eval=te, method="RK45", rtol=1e-7, atol=1e-9)
    x = sol.y[:3]
    t = sol.t
    yd = np.array([C.y_ref(tt)[0] for tt in t])
    # reconstruct control, RBF output and true drift along the trajectory
    u = np.empty(len(t)); Dhat = np.empty(len(t)); Dtrue = np.empty(len(t))
    Wn = np.zeros(len(t))
    for i, tt in enumerate(t):
        W = sol.y[3:, i] if adaptive else None
        ui, _, _, Dh = controller(x[:, i], W, tt, adaptive)
        u[i] = ui; Dhat[i] = Dh; Dtrue[i] = P.drift_D(x[:, i])
        if adaptive:
            Wn[i] = np.linalg.norm(W)
    return dict(t=t, x=x, y=x[0], yd=yd, e=x[0] - yd, u=u,
                Dhat=Dhat, Dtrue=Dtrue, Wnorm=Wn,
                W=(sol.y[3:] if adaptive else None))


# ============================================================================
# Figures (A5, continuous part)
# ============================================================================
def make_figures(da, dn, outdir="images/part2"):
    os.makedirs(outdir, exist_ok=True)

    # tracking: adaptive vs nominal vs reference
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(dn["t"], dn["yd"], "k--", lw=1.6, label=r"$y_d$ (reference)")
    ax.plot(dn["t"], dn["y"], "r:", lw=1.5, label=r"$y$ nominal ($f$ known)")
    ax.plot(da["t"], da["y"], "b-", lw=1.3, label=r"$y$ adaptive (RBF, cold start)")
    ax.set_title("Trajectory tracking: nominal vs adaptive RBF-NN")
    ax.set_xlabel("t [s]"); ax.set_ylabel("y [m]"); ax.legend(loc="upper right"); ax.grid(alpha=.3)
    fig.tight_layout(); fig.savefig(f"{outdir}/tracking.png", dpi=140); plt.close()

    # tracking error
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.semilogy(dn["t"], np.abs(dn["e"]) + 1e-18, "r:", lw=1.4, label="nominal")
    ax.semilogy(da["t"], np.abs(da["e"]) + 1e-18, "b-", lw=1.3, label="adaptive")
    ax.set_title(r"Tracking error $|e(t)|$")
    ax.set_xlabel("t [s]"); ax.set_ylabel("|e| [m]"); ax.legend(); ax.grid(alpha=.3, which="both")
    fig.tight_layout(); fig.savefig(f"{outdir}/tracking_error.png", dpi=140); plt.close()

    # control input
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(da["t"], da["u"], "b-", lw=1.1, label="adaptive")
    ax.plot(dn["t"], dn["u"], "r:", lw=1.1, label="nominal")
    ax.axhline(C.U_MAX, color="gray", ls=":", lw=.7); ax.axhline(-C.U_MAX, color="gray", ls=":", lw=.7)
    ax.set_title("Control input $u(t)$"); ax.set_xlabel("t [s]"); ax.set_ylabel("u [V]")
    ax.legend(); ax.grid(alpha=.3)
    fig.tight_layout(); fig.savefig(f"{outdir}/control.png", dpi=140); plt.close()

    # RBF approximation of the drift
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(da["t"], da["Dtrue"], "k-", lw=1.6, label=r"true drift $D(x)$")
    ax.plot(da["t"], da["Dhat"], "b--", lw=1.3, label=r"RBF estimate $\widehat{D}(x)=\widehat{W}^\top\phi$")
    ax.set_title("Online RBF approximation of the unknown drift")
    ax.set_xlabel("t [s]"); ax.set_ylabel("drift"); ax.legend(); ax.grid(alpha=.3)
    fig.tight_layout(); fig.savefig(f"{outdir}/rbf_drift.png", dpi=140); plt.close()

    # weight norm
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(da["t"], da["Wnorm"], "purple", lw=1.4)
    ax.set_title(r"RBF weight norm $\|\widehat{W}(t)\|$ (bounded)")
    ax.set_xlabel("t [s]"); ax.set_ylabel(r"$\|\widehat{W}\|$"); ax.grid(alpha=.3)
    fig.tight_layout(); fig.savefig(f"{outdir}/weights.png", dpi=140); plt.close()

    # full state histories x2, x3 (x1 = y is the tracking figure) -- A5
    fig, ax = plt.subplots(2, 1, figsize=(9, 5.6), sharex=True)
    ydd = np.gradient(da["yd"], da["t"])           # y_d' for visual reference
    ax[0].plot(da["t"], da["x"][1], "b-", lw=1.1, label="adaptive")
    ax[0].plot(dn["t"], dn["x"][1], "r:", lw=1.1, label="nominal")
    ax[0].plot(da["t"], ydd, "k--", lw=1.0, label=r"$\dot y_d$")
    ax[0].set_ylabel(r"$x_2=\dot y$ [m/s]"); ax[0].legend(loc="upper right"); ax[0].grid(alpha=.3)
    ax[1].plot(da["t"], da["x"][2], "b-", lw=1.1, label="adaptive")
    ax[1].plot(dn["t"], dn["x"][2], "r:", lw=1.1, label="nominal")
    ax[1].set_ylabel(r"$x_3=\ddot y$ [m/s$^2$]"); ax[1].set_xlabel("t [s]")
    ax[1].legend(loc="upper right"); ax[1].grid(alpha=.3)
    fig.suptitle("State histories under the nominal and adaptive controllers")
    fig.tight_layout(); fig.savefig(f"{outdir}/states.png", dpi=140); plt.close()

    # representative individual weight estimates -- A5
    fig, ax = plt.subplots(figsize=(9, 4))
    Wh = da["W"]
    idx = np.argsort(np.abs(Wh[:, -1]))[-6:][::-1]     # 6 largest final |w_i|
    for i in idx:
        ax.plot(da["t"], Wh[i], lw=1.1, label=rf"$\widehat w_{{{i+1}}}$")
    ax.set_title("Representative RBF weight estimates (6 largest at $t=T$)")
    ax.set_xlabel("t [s]"); ax.set_ylabel(r"$\widehat w_i$"); ax.legend(ncol=3, fontsize=9)
    ax.grid(alpha=.3)
    fig.tight_layout(); fig.savefig(f"{outdir}/weights_traces.png", dpi=140); plt.close()


if __name__ == "__main__":
    print("== Branch A: adaptive RBF-NN tracking (continuous) ==")
    print(f"error poles {C.ERR_POLES}, gains (l2,l1,l0)=({L2},{L1},{L0})")
    print(f"P eig = {np.round(np.linalg.eigvalsh(P_LYAP),4)} (>0), RBF centers = {C.N_RBF}")

    dn = simulate(adaptive=False, T=30.0)
    da = simulate(adaptive=True, T=30.0)

    def ss_rms(d, frac=0.5):
        m = int((1 - frac) * len(d["t"]))
        return np.sqrt(np.mean(d["e"][m:] ** 2))

    print(f"[nominal ] max|e|={np.max(np.abs(dn['e'])):.3e}  ss-RMS|e|={ss_rms(dn):.3e}")
    print(f"[adaptive] max|e|={np.max(np.abs(da['e'])):.3e}  ss-RMS|e|={ss_rms(da):.3e}")
    m = int(0.5 * len(da["t"]))
    drift_rms = np.sqrt(np.mean((da["Dhat"][m:] - da["Dtrue"][m:]) ** 2))
    print(f"[adaptive] steady RBF drift error RMS = {drift_rms:.3e}, "
          f"final ||W|| = {da['Wnorm'][-1]:.2f}, max|u| = {np.max(np.abs(da['u'])):.1f}")
    make_figures(da, dn)
    print("saved figures -> images/part2/")

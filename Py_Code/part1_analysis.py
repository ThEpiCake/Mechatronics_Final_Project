"""
Part 1 -- process modelling and analysis (report Section 2, grading M1-M4).

Produces:
  * M4  significance of the nonlinearity: zero-input free response and phase
        portraits of the NONLINEAR model vs its local LINEARIZATION, at a small
        and a large amplitude within the operating range; the hardening restoring
        force k1 y + k3 y^3 vs k1 y.
  * M3  Lyapunov analysis of the zero-input system x' = f(x): phase-portrait
        convergence to the origin, the energy Lyapunov function V(t) decreasing
        with V_dot(t) <= 0 along a trajectory, and the sign of V_dot over the
        (y', F) plane (negative definite quadratic form).

All figures are written to images/part1/.
"""

import os
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import plant as P
import config as C

OUT = "images/part1"
os.makedirs(OUT, exist_ok=True)


def free_response(x0, T, linear=False, n=2000):
    """Zero-input (u=0) response from x0; nonlinear or linearized model."""
    fun = (lambda t, x: P.dynamics_linear(x, 0.0)) if linear else \
          (lambda t, x: P.dynamics(x, 0.0))
    te = np.linspace(0, T, n)
    sol = solve_ivp(fun, [0, T], np.asarray(x0, float), t_eval=te,
                    method="RK45", rtol=1e-10, atol=1e-12)
    return sol.t, sol.y


# ============================================================================
# M4 -- significance of the nonlinear dynamics
# ============================================================================
def fig_nonlinear_vs_linear():
    T = 6.0
    cases = [("small", 0.15), ("large", 0.7)]    # release from rest y(0)=y0
    devs = {}
    fig, axes = plt.subplots(2, 1, figsize=(7.2, 6.4), sharex=True)
    for ax, (name, y0) in zip(axes, cases):
        t, xn = free_response([y0, 0, 0], T, linear=False)
        _, xl = free_response([y0, 0, 0], T, linear=True)
        ax.plot(t, xn[0], "b-", lw=1.8, label="nonlinear")
        ax.plot(t, xl[0], "r--", lw=1.6, label="linearization")
        ax.set_title(f"{name} amplitude: $y(0)={y0}$ m")
        ax.set_ylabel("y [m]"); ax.legend(); ax.grid(alpha=.3)
        devs[name] = np.max(np.abs(xn[0] - xl[0]))
    axes[1].set_xlabel("t [s]")
    fig.tight_layout(); fig.savefig(f"{OUT}/free_response.png", dpi=140); plt.close()

    # phase portrait for the large case
    t, xn = free_response([0.7, 0, 0], T, linear=False)
    _, xl = free_response([0.7, 0, 0], T, linear=True)
    fig, ax = plt.subplots(figsize=(6.2, 5.2))
    ax.plot(xn[0], xn[1], "b-", lw=1.8, label="nonlinear")
    ax.plot(xl[0], xl[1], "r--", lw=1.6, label="linearization")
    ax.plot(0, 0, "k*", ms=12)
    ax.set_title(r"Phase portrait $(y,\dot y)$, $y(0)=0.7$ m")
    ax.set_xlabel("y [m]"); ax.set_ylabel(r"$\dot y$ [m/s]"); ax.legend(); ax.grid(alpha=.3)
    fig.tight_layout(); fig.savefig(f"{OUT}/phase_nl_vs_lin.png", dpi=140); plt.close()

    # hardening restoring force
    y = np.linspace(-C.X_BOX[0, 1], C.X_BOX[0, 1], 400)
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    ax.plot(y, P.k1 * y, "r--", lw=1.6, label=r"linear $k_1 y$")
    ax.plot(y, P.k1 * y + P.k3 * y ** 3, "b-", lw=1.8,
            label=r"Duffing $k_1 y + k_3 y^3$")
    ax.set_title("Restoring force: hardening spring"); ax.set_xlabel("y [m]")
    ax.set_ylabel("force [N]"); ax.legend(); ax.grid(alpha=.3)
    fig.tight_layout(); fig.savefig(f"{OUT}/restoring_force.png", dpi=140); plt.close()
    return devs


# ============================================================================
# M3 -- Lyapunov analysis of the zero-input system
# ============================================================================
def fig_phase_convergence():
    """Fan of initial conditions -> all trajectories spiral to the origin."""
    T = 6.0
    fig, ax = plt.subplots(figsize=(6.4, 5.4))
    thetas = np.linspace(0, 2 * np.pi, 12, endpoint=False)
    for th in thetas:
        x0 = [0.7 * np.cos(th), 1.3 * np.sin(th), 0.0]
        _, x = free_response(x0, T, linear=False)
        ax.plot(x[0], x[1], lw=1.0, alpha=.8)
        ax.plot(x0[0], x0[1], "b.", ms=6)
    ax.plot(0, 0, "r*", ms=14, label="origin")
    ax.set_title("Zero-input trajectories converge to the origin")
    ax.set_xlabel("y [m]"); ax.set_ylabel(r"$\dot y$ [m/s]"); ax.legend(); ax.grid(alpha=.3)
    fig.tight_layout(); fig.savefig(f"{OUT}/lyap_phase.png", dpi=140); plt.close()


def fig_V_decreasing():
    """Energy Lyapunov function decreasing, with V_dot <= 0, along a trajectory."""
    T = 6.0
    gamma = P.GAMMA_LYAP
    t, x = free_response([0.7, 1.0, 0.0], T, linear=False)
    V = np.array([P.lyapunov_V(x[:, i], gamma) for i in range(x.shape[1])])
    Vd = np.array([P.lyapunov_Vdot(x[:, i], gamma) for i in range(x.shape[1])])
    fig, ax = plt.subplots(2, 1, figsize=(6.6, 6.4), sharex=True)
    ax[0].semilogy(t, V + 1e-18, "b-", lw=1.8)
    ax[0].set_title(r"Lyapunov function $V(t)$ (log scale)")
    ax[0].set_ylabel("V"); ax[0].grid(alpha=.3, which="both")
    ax[1].plot(t, Vd, "r-", lw=1.8); ax[1].axhline(0, color="k", lw=.8)
    ax[1].set_title(r"$\dot V(t) \leq 0$ along the trajectory")
    ax[1].set_xlabel("t [s]"); ax[1].set_ylabel(r"$\dot V$"); ax[1].grid(alpha=.3)
    fig.tight_layout(); fig.savefig(f"{OUT}/lyap_V.png", dpi=140); plt.close()
    return float(Vd.max())


def fig_Vdot_field():
    """Sign of V_dot = -c1 y'^2 - (gamma/tau) F^2 + y' F over the (y',F) plane."""
    gamma = P.GAMMA_LYAP
    yd = np.linspace(-3, 3, 220)
    F = np.linspace(-60, 60, 220)
    YD, FF = np.meshgrid(yd, F)
    Vd = -P.c1 * YD ** 2 - (gamma / P.tau) * FF ** 2 + YD * FF
    fig, ax = plt.subplots(figsize=(6.6, 5.0))
    pc = ax.contourf(YD, FF, Vd, levels=30, cmap="viridis")
    ax.contour(YD, FF, Vd, levels=[0], colors="w", linewidths=1.5)
    fig.colorbar(pc, ax=ax, label=r"$\dot V$")
    ax.set_title(r"$\dot V(\dot y, F) \leq 0$ (negative definite quadratic form)")
    ax.set_xlabel(r"$\dot y$ [m/s]"); ax.set_ylabel("F [N]")
    fig.tight_layout(); fig.savefig(f"{OUT}/lyap_Vdot_field.png", dpi=140); plt.close()
    return float(Vd.max())


if __name__ == "__main__":
    print("== Part 1 analysis ==")
    print(f"linearization eigenvalues: {np.round(np.linalg.eigvals(P.A_lin), 4)}")

    devs = fig_nonlinear_vs_linear()
    print(f"[M4] max |y_nl - y_lin|:  small(y0=0.15) = {devs['small']:.4f} m, "
          f"large(y0=0.7) = {devs['large']:.4f} m")
    ratio = (P.k3 * 0.7 ** 3) / (P.k1 * 0.7)
    print(f"[M4] cubic/linear force ratio at y=0.7 m: {ratio:.2%} (significant nonlinearity)")

    fig_phase_convergence()
    vmax_traj = fig_V_decreasing()
    vmax_field = fig_Vdot_field()
    gamma = P.GAMMA_LYAP
    print(f"[M3] Lyapunov gamma = {gamma:.4f}  (> tau/(4 c1) = {P.tau/(4*P.c1):.4f})")
    print(f"[M3] max V_dot along trajectory = {vmax_traj:.3e} (<=0 expected)")
    print(f"[M3] max V_dot over (y',F) grid  = {vmax_field:.3e} (<=0 expected)")
    print(f"saved figures -> {OUT}/")

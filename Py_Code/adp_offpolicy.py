"""
Branch B -- nonlinear OFF-POLICY adaptive dynamic programming (report Section 4,
grading B1-B8).

Optimal regulation of x' = f(x) + g u (f unknown, g known) for the cost
    J = \int_0^inf (x^T Q x + R u^2) dt ,   Q>0, R>0.

Policy iteration (Lewis/Vamvoudakis; Jiang & Jiang off-policy IRL):
  * value  V_i(x)   = x^T P_i x + c_i^T phi_nl(x)                       (Eq. B4-6)
  * policy u_{i+1}  = -(1/2) R^{-1} g(x)^T grad V_i(x)                  (Eq. B4-7)

Off-policy Bellman equation along the REAL data (behaviour input u, real trajectory),
which never uses f:
  V_i(x(t+T)) - V_i(x(t)) = -\int_t^{t+T}(x^T Q x + R u_i^2) dtau
                            - 2\int_t^{t+T} R u_{i+1} (u - u_i) dtau .

Writing V_i = w_i . Phi_V(x) with Phi_V = [quadratic monomials ; phi_nl], and using
u_{i+1} = -(1/(2R)) zeta(x)^T w_i with zeta(x) = c * dPhi_V/dx3 (g=[0,0,c]), each data
window j gives ONE linear equation in the single unknown vector w_i:
    [ dPhi_V_j - \int_j zeta (u - u_i) dtau ] . w_i = -\int_j (x^T Q x + R u_i^2) dtau .
With many more windows than unknowns, w_i is found by least squares (pseudoinverse);
the same data set is reused for every iteration.  u_i is the current policy evaluated
on the stored states (u_0 = 0).
"""

import os
import numpy as np
from scipy.linalg import solve_continuous_are
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import plant as P
import config as C

R = C.ADP_R
Q = C.ADP_Q
MON = C.ADP_MONOMIALS                      # nonlinear monomial exponents (m,3)
N_NL = MON.shape[0]
N_W = 6 + N_NL                             # 6 quadratic + nonlinear unknowns


# ----- value-function basis Phi_V(x) and its x3-derivative ------------------
def phi_V(x):
    """Full value-function basis: 6 quadratic monomials + nonlinear monomials."""
    x1, x2, x3 = x
    quad = [x1 * x1, x1 * x2, x1 * x3, x2 * x2, x2 * x3, x3 * x3]
    nl = [x1 ** e[0] * x2 ** e[1] * x3 ** e[2] for e in MON]
    return np.array(quad + nl)


def dphi_V_dx3(x):
    """d Phi_V / d x3  (only the x3-dependent terms survive)."""
    x1, x2, x3 = x
    dquad = [0.0, 0.0, x1, 0.0, x2, 2.0 * x3]
    dnl = []
    for e in MON:
        if e[2] == 0:
            dnl.append(0.0)
        else:
            dnl.append(e[2] * x1 ** e[0] * x2 ** e[1] * x3 ** (e[2] - 1))
    return np.array(dquad + dnl)


def zeta(x):
    """zeta(x) = c * dPhi_V/dx3 ; policy u = -(1/(2R)) zeta(x)^T w."""
    return P.c * dphi_V_dx3(x)


def policy(x, w):
    return -(1.0 / (2.0 * R)) * (zeta(x) @ w)


def P_matrix(w):
    """Reconstruct the symmetric quadratic matrix P_i from the first 6 weights."""
    p = w[:6]
    return np.array([[p[0], 0.5 * p[1], 0.5 * p[2]],
                     [0.5 * p[1], p[3], 0.5 * p[4]],
                     [0.5 * p[2], 0.5 * p[4], p[5]]])


# ----- B3: off-policy data generation ---------------------------------------
def generate_data(T=None, dt=None):
    """Apply a bounded, sufficiently exciting behaviour input (u_0=0 + exploration)
    to the TRUE plant and record (t, X, U).  f is used only by the simulator."""
    dt = C.ADP_DT if dt is None else dt
    T = C.ADP_N_WINDOWS * C.ADP_WINDOW if T is None else T
    n = int(round(T / dt))
    X = np.empty((n + 1, 3)); U = np.empty(n + 1)
    x = C.ADP_X0.copy()
    for k in range(n + 1):
        t = k * dt
        u = C.ADP_PROBE_AMP * np.sum(np.sin(C.ADP_PROBE_FREQS * t)) / len(C.ADP_PROBE_FREQS)
        u = float(np.clip(u, -C.U_MAX, C.U_MAX))
        X[k] = x; U[k] = u
        if k < n:
            x = P.rk4_step(x, u, dt)
    return np.arange(n + 1) * dt, X, U


# ----- B6: off-policy policy iteration ---------------------------------------
def policy_iteration(X, U, dt, verbose=True):
    """Off-policy IRL policy iteration reusing the same dataset each step."""
    Wlen = int(round(C.ADP_WINDOW / dt))          # samples per window
    n_win = (len(X) - 1) // Wlen
    # precompute per-sample basis, zeta, stage-cost-without-control
    Phi = np.array([phi_V(x) for x in X])          # (N, N_W)
    Z = np.array([zeta(x) for x in X])             # (N, N_W)
    xQx = np.einsum("ni,ij,nj->n", X, Q, X)        # x^T Q x per sample

    dPhi = np.empty((n_win, N_W))
    for j in range(n_win):
        dPhi[j] = Phi[(j + 1) * Wlen] - Phi[j * Wlen]

    def win_trapz(integrand):
        """Trapezoidal integral of a per-sample (N,...) array over each window."""
        out = []
        for j in range(n_win):
            s = slice(j * Wlen, (j + 1) * Wlen + 1)
            out.append(np.trapz(integrand[s], dx=dt, axis=0))
        return np.array(out)

    w = np.zeros(N_W)                              # value weights (define policy u_1 after solve)
    history = {"w": [], "P_eig": [], "res": []}
    for k in range(C.ADP_MAX_ITERS):
        # current policy u_k on the stored states (u_0 = 0)
        u_cur = np.zeros(len(X)) if k == 0 else -(1.0 / (2.0 * R)) * (Z @ w)
        # regressor A_j and target b_j for this iteration (reuse same data)
        A = dPhi - win_trapz(Z * (U - u_cur)[:, None])          # (n_win, N_W)
        b = -win_trapz(xQx + R * u_cur ** 2)                    # (n_win,)
        w_new, *_ = np.linalg.lstsq(A, b, rcond=None)
        res = np.linalg.norm(A @ w_new - b) / np.sqrt(n_win)
        delta = np.linalg.norm(w_new - w)
        w = w_new
        Pe = np.linalg.eigvalsh(P_matrix(w))
        history["w"].append(w.copy()); history["P_eig"].append(Pe); history["res"].append(res)
        if verbose:
            print(f"  iter {k:2d}: ||dw||={delta:.3e}  residual={res:.3e}  "
                  f"eig(P)={np.round(Pe,3)}  {'PD' if np.all(Pe>0) else 'NOT PD'}")
        if delta < C.ADP_TOL and k > 0:
            break
    return w, history


# ----- B7: validation -- learned policy vs initial policy u0=0 --------------
def rollout(x0, w, T=6.0, dt=1e-3, use_policy=True):
    """Closed-loop rollout; returns (t, X, U, J_T) with J_T the accumulated cost."""
    n = int(round(T / dt))
    X = np.empty((n + 1, 3)); U = np.empty(n + 1)
    x = np.array(x0, float); J = 0.0
    for k in range(n + 1):
        u = float(np.clip(policy(x, w), -C.U_MAX, C.U_MAX)) if use_policy else 0.0
        X[k] = x; U[k] = u
        stage = x @ Q @ x + R * u * u
        J += stage * dt * (0.5 if k in (0, n) else 1.0)
        if k < n:
            x = P.rk4_step(x, u, dt)
    return np.arange(n + 1) * dt, X, U, J


if __name__ == "__main__":
    print("== Branch B: off-policy ADP ==")
    print(f"Q=diag{np.diag(Q).tolist()} R={R}; unknowns={N_W} (6 quad + {N_NL} nonlinear)")
    dt = C.ADP_DT
    t, X, U = generate_data()
    print(f"data: T={t[-1]:.1f}s, {len(X)} samples, |x|max per axis={np.round(np.max(np.abs(X),0),3)} "
          f"(box {C.X_BOX[:,1]}), |u|max={np.max(np.abs(U)):.1f}")

    w, hist = policy_iteration(X, U, dt)
    print(f"stopping criterion ||w_i+1 - w_i|| < {C.ADP_TOL} met after {len(hist['w'])} iterations")
    Pfin = P_matrix(w)
    print(f"learned quadratic P=\n{np.round(Pfin,3)}")
    print(f"eig(P) = {np.round(np.linalg.eigvalsh(Pfin),4)}  (all>0 => locally PD)")

    # sanity: compare quadratic part with the LQR/ARE solution of the linearization
    Pare = solve_continuous_are(P.A_lin, P.B_lin.reshape(3, 1), Q, np.array([[R]]))
    print(f"ARE (linearization) P=\n{np.round(Pare,3)}")
    print(f"||P_adp - P_are||/||P_are|| = {np.linalg.norm(Pfin-Pare)/np.linalg.norm(Pare):.3%}")

    # nonlinear value-function coefficients
    print(f"nonlinear coeffs c = {np.round(w[6:],4)}")

    # B7 validation over several initial conditions
    ics = [[0.6, 0, 0], [-0.5, 1.0, 0], [0.4, -0.8, 2.0], [0.7, 0.5, -1.0]]
    print("\n[B7] accumulated cost J_T (learned vs initial u0=0):")
    Jl, J0 = [], []
    for x0 in ics:
        _, _, _, jl = rollout(x0, w, use_policy=True)
        _, _, _, j0 = rollout(x0, w, use_policy=False)
        Jl.append(jl); J0.append(j0)
        print(f"  x0={x0}:  J_learned={jl:8.3f}   J_(u0=0)={j0:8.3f}   improvement={100*(j0-jl)/j0:5.1f}%")

    # ---------------- figures ----------------
    OUT = "images/part3"; os.makedirs(OUT, exist_ok=True)

    # data-collection state stays in the operating box
    fig, ax = plt.subplots(figsize=(9, 3.6))
    for i, lab in enumerate([r"$y$ [m]", r"$\dot y$ [m/s]", r"$\ddot y$ [m/s$^2$]"]):
        ax.plot(t, X[:, i], lw=0.7, label=lab)
    ax.set_title(f"Off-policy data collection (first 6 s of the {t[-1]:.0f} s record)")
    ax.set_xlabel("t [s]"); ax.set_ylabel("state (per-signal units)")
    ax.legend(ncol=3); ax.grid(alpha=.3)
    ax.set_xlim(0, min(6, t[-1]))
    fig.tight_layout(); fig.savefig(f"{OUT}/data.png", dpi=140); plt.close()

    # policy-iteration convergence: P eigenvalues, residual, coefficient evolution
    Peig = np.array(hist["P_eig"]); res = np.array(hist["res"])
    Wev = np.array(hist["w"])                      # (iterations, 12)
    it = np.arange(1, len(Wev) + 1)
    fig, ax = plt.subplots(1, 3, figsize=(15, 4))
    for i in range(3):
        ax[0].plot(it, Peig[:, i], "o-", lw=1.4, label=f"$\\lambda_{i+1}(P)$")
    ax[0].axhline(0, color="k", lw=.7)
    ax[0].set_title("Eigenvalues of $P_i$ (stay $>0$)")
    ax[0].set_xlabel("iteration"); ax[0].set_ylabel("eig($P_i$)"); ax[0].legend(); ax[0].grid(alpha=.3)
    ax[1].semilogy(it, res + 1e-18, "s-", color="purple", lw=1.4)
    ax[1].set_title("Off-policy least-squares residual")
    ax[1].set_xlabel("iteration"); ax[1].set_ylabel("residual RMS"); ax[1].grid(alpha=.3, which="both")
    for i in range(Wev.shape[1]):
        ax[2].plot(it, Wev[:, i], "o-", lw=1.0, ms=3)
    ax[2].set_title(f"Value-function coefficients $w_i$ ({Wev.shape[1]} traces)")
    ax[2].set_xlabel("iteration"); ax[2].set_ylabel("$w_i$"); ax[2].grid(alpha=.3)
    fig.tight_layout(); fig.savefig(f"{OUT}/convergence.png", dpi=140); plt.close()

    # B7: learned vs initial policy from several ICs -- ALL states + input
    sel = [0, 1, 3]                                # IC1, IC2, IC4 (IC4 = the Part-4 demo IC)
    cols = ["tab:blue", "tab:green", "tab:red"]
    fig, ax = plt.subplots(4, 1, figsize=(10, 10.5), sharex=True)
    labs = [r"$x_1=y$ [m]", r"$x_2=\dot y$ [m/s]", r"$x_3=\ddot y$ [m/s$^2$]"]
    for c, j in zip(cols, sel):
        tl, Xl, Ul, _ = rollout(ics[j], w, use_policy=True)
        t0, X0r, _, _ = rollout(ics[j], w, use_policy=False)
        for i in range(3):
            ax[i].plot(tl, Xl[:, i], color=c, lw=1.3,
                       label=f"IC{j+1} learned" if i == 0 else None)
            ax[i].plot(t0, X0r[:, i], color=c, lw=1.0, ls=":", alpha=.75,
                       label=f"IC{j+1} initial $u_0=0$" if i == 0 else None)
        ax[3].plot(tl, Ul, color=c, lw=1.2, label=f"IC{j+1}")
    for i in range(3):
        ax[i].set_ylabel(labs[i]); ax[i].grid(alpha=.3)
    ax[0].legend(ncol=3, fontsize=9)
    ax[3].set_ylabel("u [V]"); ax[3].set_xlabel("t [s]"); ax[3].grid(alpha=.3)
    ax[3].legend(ncol=3, fontsize=9, title="learned policy")
    fig.suptitle("Learned vs initial policy: all state variables and the control input")
    fig.tight_layout(); fig.savefig(f"{OUT}/regulation.png", dpi=140); plt.close()

    # cost comparison bar
    fig, ax = plt.subplots(figsize=(8, 4))
    xpos = np.arange(len(ics))
    ax.bar(xpos - 0.2, J0, 0.4, label=r"initial $u_0=0$", color="salmon")
    ax.bar(xpos + 0.2, Jl, 0.4, label="learned policy", color="steelblue")
    ax.set_xticks(xpos); ax.set_xticklabels([f"IC{i+1}" for i in range(len(ics))])
    ax.set_title(r"Accumulated cost $J_T$: learned policy vs initial policy")
    ax.set_ylabel(r"$J_T$"); ax.legend(); ax.grid(alpha=.3, axis="y")
    fig.tight_layout(); fig.savefig(f"{OUT}/cost.png", dpi=140); plt.close()
    print(f"\nsaved figures -> {OUT}/")

    # export identified coefficients for the Pico (Part 4)
    np.savez("data/adp_policy.npz", w=w, MON=MON, Q=np.diag(Q), R=R, c=P.c)
    print("saved data/adp_policy.npz")

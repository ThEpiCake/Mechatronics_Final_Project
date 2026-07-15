"""
Single source of truth for all project design choices (operating range, input
limits, reference trajectory, RBF network, adaptation gains, ADP weights, sampling
periods, Pico timing).  Every simulation module imports from here so that the
report, the desktop code and the Pico code stay consistent and reproducible.
"""

import numpy as np

# ============================================================================
# Operating range and input limit (spec 1.7)
# ============================================================================
# Declared operating range of the PROCESS, in the companion state x=[y,y',y''].  It
# encloses both the Branch-A tracking tube and the Branch-B regulation transients
# (releasing the stiff spring from y=0.6 m produces accelerations up to ~7 m/s^2).
# The process is strongly nonlinear over this range: the Duffing cubic force
# k3 y^2/k1 reaches ~1 (equal to the linear force) at y = 0.7 m (spec 1.3).
X_BOX = np.array([[-0.8, 0.8],     # y   [m]
                  [-2.0, 2.0],     # y'  [m/s]
                  [-11.0, 11.0]])  # y'' [m/s^2]
U_MAX = 40.0                       # |u| <= U_MAX [V]  (actuator input saturation)

# ============================================================================
# Reference trajectory for Branch A (smooth, feasible): y_d(t) = A sin(w t)
# ============================================================================
REF_A = 0.5        # amplitude [m]  (peaks reach the strongly-nonlinear region)
REF_W = 2.0        # frequency [rad/s]


def y_ref(t):
    """Return (y_d, y_d', y_d'', y_d''') for the tracking reference."""
    A, w = REF_A, REF_W
    s, csn = np.sin(w * t), np.cos(w * t)
    return (A * s, A * w * csn, -A * w ** 2 * s, -A * w ** 3 * csn)


# ============================================================================
# Branch A: nominal tracking error dynamics  e''' + l2 e'' + l1 e' + l0 e = 0
# ============================================================================
# Desired (Hurwitz) error poles -> companion gains via characteristic polynomial.
ERR_POLES = np.array([-6.0, -7.0, -8.0])
_poly = np.poly(ERR_POLES)                 # [1, l2, l1, l0]
LAM = _poly[1:]                            # (l2, l1, l0)

# ============================================================================
# Branch A: RBF network approximating the unknown drift D(x) (3-D input)
# ============================================================================
# Centers on a regular grid inside X_BOX; distances measured in NORMALIZED
# coordinates xn = (x - center)/half_range so a single width handles the very
# different per-axis scales.  A NORMALIZED (partition-of-unity) Gaussian network is
# used: phi_i = g_i / sum_j g_j.  This keeps the basis well conditioned and the
# ideal weights close to the local values of the approximated drift, so a bounded
# adaptation law can realise them (a plain Gaussian basis is nearly linearly
# dependent here and needs astronomically large weights).
# The RBF centers cover only the Branch-A tracking region (the tube around the
# reference), which is smaller than the full process operating range -- this keeps
# the network resolution high where the tracking controller actually operates.
RBF_BOX = np.array([[-0.7, 0.7],   # y   [m]
                    [-1.3, 1.3],   # y'  [m/s]
                    [-3.0, 3.0]])  # y'' [m/s^2]
RBF_GRID = (6, 6, 6)               # centers per axis -> 216 Gaussians
RBF_WIDTH = 0.24                   # Gaussian width in normalized units (~0.6x grid spacing:
#                                    LOCAL, so the ideal weights ~ local drift values ~ O(300))


def _rbf_halfrange():
    return 0.5 * (RBF_BOX[:, 1] - RBF_BOX[:, 0])


def rbf_centers():
    """(N,3) array of RBF centers on the grid inside RBF_BOX (physical units)."""
    axes = [np.linspace(RBF_BOX[i, 0], RBF_BOX[i, 1], RBF_GRID[i]) for i in range(3)]
    g0, g1, g2 = np.meshgrid(*axes, indexing="ij")
    return np.stack([g0.ravel(), g1.ravel(), g2.ravel()], axis=1)


def rbf_phi(x, centers):
    """Normalized Gaussian RBF vector phi(x) (partition of unity). x:(3,), centers:(N,3)."""
    half = _rbf_halfrange()
    xn = (np.asarray(x, float) - centers) / half          # (N,3)
    g = np.exp(-np.sum(xn ** 2, axis=1) / (2.0 * RBF_WIDTH ** 2))
    s = g.sum()
    return g / s if s > 1e-12 else g


N_RBF = int(np.prod(RBF_GRID))

# ----- adaptation law gains (Lyapunov weight update + projection) ------------
# The weights are updated by the Lyapunov gradient law and kept bounded by a
# PROJECTION onto |w_i| <= W_MAX (rather than sigma-modification, which biases the
# weights toward zero and here prevents them from reaching the ideal values).
GAMMA_ADAPT = 40000.0  # learning rate (scalar; applied to every weight)
SIGMA_MOD = 0.0        # sigma-modification leakage (0 -> boundedness from projection)
W_MAX = 600.0          # projection bound on |weight|
LYAP_Q = np.diag([50.0, 10.0, 1.0])   # Q in  Lambda^T P + P Lambda = -Q  (error Lyapunov)

# ============================================================================
# Branch A discrete realization (A4/A5): sampling periods to compare
# ============================================================================
# One fine (matches continuous), one intermediate, one that clearly degrades.
TS_LIST = [0.005, 0.025, 1.0 / 15.0]   # [s]  (200 Hz, 40 Hz, 15 Hz -> degrades)

# ============================================================================
# Branch B: off-policy ADP
# ============================================================================
ADP_Q = np.diag([300.0, 20.0, 1.0])    # state penalty  (Q = Q^T > 0)
ADP_R = 0.1                            # control penalty (R = R^T > 0, scalar)

# Nonlinear value-function basis phi_{V,nl}: monomials of total degree 3-4 in
# (x1,x2,x3), vanishing at the origin together with their first derivatives, no
# constant/linear terms (spec B4).  Chosen to reflect the Duffing structure
# (quartic potential x1^4 and its couplings).  Each row = exponents (e1,e2,e3).
ADP_MONOMIALS = np.array([
    [4, 0, 0],   # x1^4   (Duffing quartic potential -> dominant correction)
    [3, 1, 0],   # x1^3 x2
    [2, 2, 0],   # x1^2 x2^2
    [2, 0, 1],   # x1^2 x3
    [2, 1, 1],   # x1^2 x2 x3
    [3, 0, 1],   # x1^3 x3
])

# Off-policy data generation: bounded, sufficiently exciting probing input.
ADP_DT = 0.001            # integration step for data collection [s]
ADP_WINDOW = 0.05         # integral-reinforcement window length T [s]
ADP_N_WINDOWS = 400       # number of data windows (>> unknowns)
ADP_X0 = np.array([0.6, 0.0, 0.0])    # initial condition for the data run
ADP_PROBE_FREQS = np.array([1.3, 3.7, 7.1, 11.9, 17.0])   # incommensurate excitation
ADP_PROBE_AMP = 8.0       # probing amplitude [V] (bounded, within U_MAX)
ADP_MAX_ITERS = 12        # policy-iteration cap
ADP_TOL = 1e-4            # stopping tolerance on ||theta_{i+1}-theta_i||

# ============================================================================
# Part 4: Raspberry Pi Pico real-time timing
# ============================================================================
PICO_DT = 0.002           # secondary-core plant integration step [s]  (500 Hz)
PICO_N = 5                # controller runs every N plant steps -> Ts = N*dt
PICO_TS = PICO_N * PICO_DT # controller sampling period [s]  (10 ms, 100 Hz)
# Ts = 10 ms was chosen from ON-CHIP measurements: the 216-center RBF update takes
# a few ms on the RP2040 (software floats), so 5 ms left no real-time margin; the
# Section-3 sampling sweep shows the design is unchanged down to 20 Hz, so 100 Hz
# keeps a wide stability margin while max(exec) < Ts holds with headroom.


if __name__ == "__main__":
    print(f"error-dynamics gains (l2,l1,l0) = {np.round(LAM,3)}  from poles {ERR_POLES}")
    print(f"RBF: {RBF_GRID} grid -> {N_RBF} centers, width {RBF_WIDTH} (normalized)")
    print(f"adaptation: Gamma={GAMMA_ADAPT}, sigma={SIGMA_MOD}")
    print(f"ADP: Q=diag{np.diag(ADP_Q).tolist()}, R={ADP_R}, "
          f"{ADP_MONOMIALS.shape[0]} nonlinear basis fns, {ADP_N_WINDOWS} windows")
    print(f"Pico: dt={PICO_DT*1e3:.1f} ms, N={PICO_N}, Ts={PICO_TS*1e3:.1f} ms")
    c = rbf_centers()
    print(f"rbf_centers shape {c.shape}; phi(0) sum = {rbf_phi(np.zeros(3), c).sum():.3f}")

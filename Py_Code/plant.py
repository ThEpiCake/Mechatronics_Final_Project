"""
Shared nonlinear process for the Mechatronics Systems final project.

Physical process (reused from HW1, per the project spec's permission to continue
with the homework process).  A mass on a Duffing (cubic) spring with linear
viscous damping, driven by a first-order force actuator:

    m y'' + c1 y' + k1 y + k3 y^3 = F          (Duffing mass-spring-damper)
    tau F' + F = ka u                          (first-order force actuator)

State (companion form used by BOTH controllers), x = [y, y', y''] = [x1, x2, x3]:

    x1' = x2
    x2' = x3
    x3' = -a1 x3 - a2 x2 - a3 x1 - b1 x1^3 - b2 x1^2 x2 + c u

i.e. the control-affine form required by the project (Eq. 1):

    x' = f(x) + g(x) u ,     g(x) = [0, 0, c]^T  (CONSTANT -> known to controller)

with f(x) the (smooth, nonlinear) drift.  The scalar drift in the input channel

    D(x) = a1 x3 + a2 x2 + a3 x1 + b1 x1^3 + b2 x1^2 x2        (so f3 = -D(x))

is the "unknown nonlinear function" that Branch A approximates by an RBF network.

f(0) = 0 (spec 1.5 satisfied: the regulation equilibrium is already the origin,
because a3 = k1/(tau m) != 0 ties the mass to ground -> unique equilibrium).

Lyapunov analysis of the zero-input system x' = f(x) is cleanest in the physically
equivalent coordinates z = [y, y', F], related to x by the GLOBAL diffeomorphism

    F = m x3 + c1 x2 + k1 x1 + k3 x1^3   (<->   x3 = (F - c1 x2 - k1 x1 - k3 x1^3)/m)

in which the zero-input dynamics are  z1'=z2, z2'=(z3 - c1 z2 - k1 z1 - k3 z1^3)/m,
z3'=-z3/tau, and the mechanical energy + actuator term is a Lyapunov function
(see part1_analysis.py).

Coefficients (derived from the physical parameters, identical to HW1):
    a1 = c1/m + 1/tau ,  a2 = k1/m + c1/(tau m) ,  a3 = k1/(tau m) ,  c = ka/(tau m)
    b1 = k3/(tau m)   (regressor y^3) ,   b2 = 3 k3/m   (regressor y^2 y')
"""

import numpy as np

# ----- physical parameters --------------------------------------------------
# Structure (Duffing spring + first-order actuator) and the mass/stiffness/actuator
# gain (m, k1, ka) are those of HW1.  Three parameters are re-tuned for the project,
# as the spec permits ("the model may be reformulated or adapted"): a slower, more
# realistic actuator (tau: 0.05 -> 0.2 s), a stronger cubic stiffness (k3: 5 -> 40),
# and lighter damping (c1: 4 -> 1).  The result is a process that is (i) STRONGLY
# nonlinear at a moderate amplitude -- the cubic force k3 y^2/k1 is 50% of the linear
# force already at y = 0.5 m; (ii) lightly damped, so the zero-input response is
# oscillatory and the optimal regulator (Branch B) has a clear benefit over doing
# nothing; while (iii) keeping the unknown drift of moderate magnitude, which makes
# the RBF and ADP designs well conditioned and portable to the Pico numeric range.
m, c1, k1 = 1.0, 1.0, 20.0      # mass [kg], linear damping [N s/m], linear stiffness [N/m]
tau, ka = 0.2, 1.0              # actuator time constant [s], actuator gain [N/V]
k3 = 40.0                       # cubic (Duffing) stiffness [N/m^3]

# ----- derived linear jerk coefficients -------------------------------------
a1 = c1 / m + 1.0 / tau                 # 24
a2 = k1 / m + c1 / (tau * m)            # 100
a3 = k1 / (tau * m)                     # 400   (!= 0  -> unique equilibrium at origin)
c = ka / (tau * m)                      # 20    (input gain: g = [0,0,c])

# ----- derived nonlinear jerk coefficients ----------------------------------
b1 = k3 / (tau * m)             # 100   coefficient of y^3
b2 = 3.0 * k3 / m               # 15    coefficient of y^2 y'  (cross term)

# constant input vector g(x) = g  (control-affine, g known to the controller)
G = np.array([0.0, 0.0, c])


def drift_D(x):
    """Scalar nonlinear drift in the input channel: x3' = c u - D(x).
    This is the 'unknown' function the RBF network (Branch A) approximates."""
    x1, x2, x3 = x[0], x[1], x[2]
    return a1 * x3 + a2 * x2 + a3 * x1 + b1 * x1 ** 3 + b2 * x1 ** 2 * x2


def f(x):
    """Drift field f(x) of the control-affine model x' = f(x) + g u."""
    x1, x2, x3 = x[0], x[1], x[2]
    return np.array([x2, x3, -drift_D(x)])


def g(x=None):
    """Known input field g(x) = [0,0,c] (constant)."""
    return G


def dynamics(x, u):
    """Full control-affine vector field x' = f(x) + g u."""
    return f(x) + G * u


def rk4_step(x, u, dt):
    """One classical RK4 integration step with ZOH input u (held over dt).
    This is the exact integrator ported to the Pico secondary core (Part 4)."""
    k1_ = dynamics(x, u)
    k2_ = dynamics(x + 0.5 * dt * k1_, u)
    k3_ = dynamics(x + 0.5 * dt * k2_, u)
    k4_ = dynamics(x + dt * k3_, u)
    return x + (dt / 6.0) * (k1_ + 2 * k2_ + 2 * k3_ + k4_)


# ----- linearization about the origin (for the M4 comparison) ---------------
# Dropping the nonlinear terms b1 x1^3 and b2 x1^2 x2 gives the Jacobian A:
A_lin = np.array([[0.0, 1.0, 0.0],
                  [0.0, 0.0, 1.0],
                  [-a3, -a2, -a1]])
B_lin = np.array([0.0, 0.0, c])


def f_linear(x):
    """Linearized drift A_lin x (zero-input) for the nonlinear-vs-linear study."""
    return A_lin @ np.asarray(x, dtype=float)


def dynamics_linear(x, u):
    return A_lin @ np.asarray(x, dtype=float) + B_lin * u


# ----- physical actuator force + energy-based Lyapunov function (Part 1) -----
def force_F(x):
    """Physical actuator force F as a function of the companion state x.
    z = [y, y', F] is a global diffeomorphism of x = [y, y', y'']."""
    x1, x2, x3 = x[0], x[1], x[2]
    return m * x3 + c1 * x2 + k1 * x1 + k3 * x1 ** 3


def lyapunov_V(x, gamma):
    """Lyapunov candidate for the zero-input system, in physical coordinates:
        V = 1/2 m y'^2 + 1/2 k1 y^2 + 1/4 k3 y^4 + 1/2 gamma F^2 .
    Positive definite and radially unbounded (k3>0)."""
    x1, x2 = x[0], x[1]
    F = force_F(x)
    return 0.5 * m * x2 ** 2 + 0.5 * k1 * x1 ** 2 + 0.25 * k3 * x1 ** 4 + 0.5 * gamma * F ** 2


def lyapunov_Vdot(x, gamma):
    """Time derivative of V along the ZERO-INPUT flow x' = f(x).
    In physical coordinates this collapses to the quadratic form
        Vdot = -c1 y'^2 - (gamma/tau) F^2 + y' F ,
    negative definite in (y', F) whenever gamma > tau/(4 c1)."""
    x2 = x[1]
    F = force_F(x)
    return -c1 * x2 ** 2 - (gamma / tau) * F ** 2 + x2 * F


# gamma making Vdot negative (semi)definite: need c1*(gamma/tau) - 1/4 > 0
GAMMA_LYAP = tau / (4.0 * c1) * 4.0     # = tau/c1  (comfortably > tau/(4 c1))


if __name__ == "__main__":
    print("Control-affine Duffing jerk plant  x' = f(x) + g u ,  g = [0,0,c]")
    print(f"  a1={a1:.1f} a2={a2:.1f} a3={a3:.1f} c={c:.1f} | b1={b1:.1f}(y^3) b2={b2:.1f}(y^2 y')")
    print(f"  f(0) = {f(np.zeros(3))}  (origin is an equilibrium)")
    # unique equilibrium check: -a3 x1 - b1 x1^3 = 0 -> x1(a3+b1 x1^2)=0 -> x1=0
    print(f"  a3,b1 > 0  -> a3 + b1 x1^2 > 0  -> origin is the UNIQUE zero-input equilibrium")
    # open-loop linear stability (Routh on s^3+a1 s^2+a2 s+a3)
    print(f"  linear open-loop: a1*a2={a1*a2:.0f} > a3={a3:.0f} -> {a1*a2 > a3} (Hurwitz)")
    print(f"  linear eigenvalues: {np.round(np.linalg.eigvals(A_lin), 4)}")
    # Lyapunov spot check on a grid: Vdot <= 0 everywhere, =0 only at y'=F=0
    gamma = GAMMA_LYAP
    print(f"  gamma_lyap = {gamma:.4f}  (need > tau/(4 c1) = {tau/(4*c1):.4f})")
    rng = np.random.default_rng(0)
    Xs = rng.uniform(-3, 3, size=(20000, 3))
    vdots = np.array([lyapunov_Vdot(x, gamma) for x in Xs])
    print(f"  Vdot over 20000 random states in [-3,3]^3: max = {vdots.max():.3e} (<=0 expected)")

"""
Secondary-core (core 1) task: real-time simulation of the continuous nonlinear
process.  One RK4 integration step is performed per hardware-timer tick (period dt),
with the controller's input u held constant between controller updates (ZOH).

f(x) is evaluated HERE only -- it never appears in the controller code, honouring
the "f unknown to the controller, simulated inside the process" requirement (C1).

Runs unchanged on the Pico (ulab) and on a desktop (NumPy fallback) for validation.
The RK4 step is written in scalar arithmetic (no array temporaries): on the RP2040
(Cortex-M0+, software floats) small-vector ulab calls are dominated by per-call
overhead, and the scalar form brings the step from ~2.9 ms to well under the plant
period; @micropython.native compiles it with the native emitter on the board.
"""
try:
    from ulab import numpy as np           # on the Pico
except ImportError:
    import numpy as np                     # on a desktop (validation)

try:
    import micropython                     # on the Pico: enables @micropython.native
except ImportError:
    class _shim:                           # desktop: identity decorator
        @staticmethod
        def native(fun):
            return fun
    micropython = _shim()

import params as PA


def f(x):
    """Drift f(x) of x' = f(x) + g u  (control-affine, g=[0,0,C_IN])."""
    x1, x2, x3 = x[0], x[1], x[2]
    D = PA.A1 * x3 + PA.A2 * x2 + PA.A3 * x1 + PA.B1 * x1 * x1 * x1 + PA.B2 * x1 * x1 * x2
    return np.array([x2, x3, -D])


def dyn(x, u):
    fx = f(x)
    return np.array([fx[0], fx[1], fx[2] + PA.C_IN * u])


@micropython.native
def rk4_scalars(x1, x2, x3, u, dt,
                A1=PA.A1, A2=PA.A2, A3=PA.A3, B1=PA.B1, B2=PA.B2, C_IN=PA.C_IN):
    """One RK4 step with ZOH input u held over dt, in scalar form (the core-1
    real-time loop).  The model constants are bound as default arguments so
    lookups stay local; no array temporaries are created on the plant core."""
    bu = C_IN * u
    h = 0.5 * dt

    k1a = x2; k1b = x3
    k1c = bu - (A1 * x3 + A2 * x2 + A3 * x1 + B1 * x1 * x1 * x1 + B2 * x1 * x1 * x2)

    e1 = x1 + h * k1a; e2 = x2 + h * k1b; e3 = x3 + h * k1c
    k2a = e2; k2b = e3
    k2c = bu - (A1 * e3 + A2 * e2 + A3 * e1 + B1 * e1 * e1 * e1 + B2 * e1 * e1 * e2)

    e1 = x1 + h * k2a; e2 = x2 + h * k2b; e3 = x3 + h * k2c
    k3a = e2; k3b = e3
    k3c = bu - (A1 * e3 + A2 * e2 + A3 * e1 + B1 * e1 * e1 * e1 + B2 * e1 * e1 * e2)

    e1 = x1 + dt * k3a; e2 = x2 + dt * k3b; e3 = x3 + dt * k3c
    k4a = e2; k4b = e3
    k4c = bu - (A1 * e3 + A2 * e2 + A3 * e1 + B1 * e1 * e1 * e1 + B2 * e1 * e1 * e2)

    w = dt / 6.0
    return (x1 + w * (k1a + 2.0 * (k2a + k3a) + k4a),
            x2 + w * (k1b + 2.0 * (k2b + k3b) + k4b),
            x3 + w * (k1c + 2.0 * (k2c + k3c) + k4c))


def rk4_step(x, u, dt):
    """Array-in / array-out wrapper around rk4_scalars (desktop validation and
    any caller holding the state as a vector)."""
    y1, y2, y3 = rk4_scalars(float(x[0]), float(x[1]), float(x[2]), u, dt)
    return np.array([y1, y2, y3])

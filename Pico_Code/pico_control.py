"""
Main-core (core 0) controllers for the Pico: the adaptive RBF-NN tracking
controller (C3) and the frozen ADP optimal-regulation policy (C4).

All vector/matrix work uses ulab.numpy (the RBF feature vector and its inner
products are vectorised, not Python loops -- C5).  The same code runs on a desktop
via the NumPy fallback so the Pico results can be checked against the computer
results under matching conditions.
"""
try:
    from ulab import numpy as np           # on the Pico
    _ON_PICO = True
except ImportError:
    import numpy as np                     # on a desktop (validation)
    _ON_PICO = False

import math
import params as PA


def _sat(u, umax):
    return umax if u > umax else (-umax if u < -umax else u)


def y_ref(t):
    """(y_d, y_d', y_d'', y_d''') for y_d = A sin(w t)."""
    A, w = PA.REF_A, PA.REF_W
    s, c = math.sin(w * t), math.cos(w * t)
    return A * s, A * w * c, -A * w * w * s, -A * w * w * w * c


# ---------------------------------------------------------------------------
# Branch A -- adaptive RBF-NN tracking controller
# ---------------------------------------------------------------------------
class RBFController:
    def __init__(self):
        b = PA.RBF_BOX
        g = PA.RBF_GRID
        ax = [self._lin(b[i][0], b[i][1], g[i]) for i in range(3)]
        c0, c1, c2 = [], [], []
        for v0 in ax[0]:
            for v1 in ax[1]:
                for v2 in ax[2]:                 # 'ij' ravel order (matches desktop)
                    c0.append(v0); c1.append(v1); c2.append(v2)
        self.C0 = np.array(c0); self.C1 = np.array(c1); self.C2 = np.array(c2)
        self.n = len(c0)
        self.ih0 = 2.0 / (b[0][1] - b[0][0])
        self.ih1 = 2.0 / (b[1][1] - b[1][0])
        self.ih2 = 2.0 / (b[2][1] - b[2][0])
        self.i2w2 = 1.0 / (2.0 * PA.RBF_WIDTH * PA.RBF_WIDTH)
        # pre-scaled PER-AXIS centers: d_i = x_i*m_i - A_i gives
        # (x_i-c_i)*ih_i*sqrt(i2w2) in one small (grid-length) vector op, so
        # exp(-(d0^2+d1^2+d2^2)) needs no further scaling
        s = math.sqrt(self.i2w2)
        self.m0 = self.ih0 * s; self.A0 = np.array(ax[0]) * self.m0
        self.m1 = self.ih1 * s; self.A1 = np.array(ax[1]) * self.m1
        self.m2 = self.ih2 * s; self.A2 = np.array(ax[2]) * self.m2
        self.g0, self.g1, self.g2 = g[0], g[1], g[2]
        self.W = np.zeros(self.n)
        self.l2, self.l1, self.l0 = PA.LAM
        self.bp0, self.bp1, self.bp2 = PA.BP

    @staticmethod
    def _lin(a, b, n):
        return [a] if n == 1 else [a + (b - a) * i / (n - 1) for i in range(n)]

    def _gauss(self, x):
        """Unnormalized Gaussian vector g(x), exploiting the SEPARABILITY of the
        grid: exp(-(d0^2+d1^2+d2^2)) over the 6x6x6 Cartesian product equals the
        outer product of three per-axis 6-element Gaussians.  All exp() calls act
        on grid-length vectors; only two broadcast multiplications touch the full
        216-element vector -- on the RP2040 @ 200 MHz this brings the full RBF
        update to the measured ~2.9 ms.  The (g0*g1)*g2 broadcast order reproduces the 'ij' ravel order
        of the flat center list, so W indexing is unchanged."""
        d0 = x[0] * self.m0 - self.A0
        d1 = x[1] * self.m1 - self.A1
        d2 = x[2] * self.m2 - self.A2
        d0 *= d0; d1 *= d1; d2 *= d2
        d0 *= -1.0; d1 *= -1.0; d2 *= -1.0
        e0 = np.exp(d0); e1 = np.exp(d1); e2 = np.exp(d2)
        a = e0.reshape((self.g0, 1)) * e1.reshape((1, self.g1))
        b = a.reshape((self.g0 * self.g1, 1)) * e2.reshape((1, self.g2))
        return b.reshape((self.n,))

    def features(self, x):
        """Normalized Gaussian RBF vector phi(x) (vectorised ulab op)."""
        g = self._gauss(x)
        s = np.sum(g)
        return g / s if s > 1e-12 else g   # underflow guard, same as the desktop

    def update(self, x, t, dt):
        """One controller update: read state x at time t, return u; adapt weights.
        dt is the controller sampling period Ts (forward-Euler weight update).
        The normalization phi = g/sum(g) is folded into the two scalar results
        (W.phi = W.g/sum(g); W-update scaled by 1/sum(g)), saving a 216-element
        vector division per update on the Pico."""
        yd, yd1, yd2, yd3 = y_ref(t)
        e = x[0] - yd; ed = x[1] - yd1; edd = x[2] - yd2
        g = self._gauss(x)
        s = np.sum(g)
        # underflow guard (matches the desktop rbf_phi): far outside the center
        # box the float32 sum can reach 0, and dividing would give NaN weights
        isg = 1.0 / s if s > 1e-12 else 0.0
        Dhat = np.dot(self.W, g) * isg
        u = _sat((Dhat + yd3 - self.l2 * edd - self.l1 * ed - self.l0 * e) / PA.C_IN, PA.U_MAX)
        # Lyapunov gradient weight update + projection (forward Euler, in place)
        bPxi = self.bp0 * e + self.bp1 * ed + self.bp2 * edd
        g *= dt * PA.GAMMA_ADAPT * bPxi * isg
        self.W -= g
        self.W = np.clip(self.W, -PA.W_MAX, PA.W_MAX)      # projection
        return u


# ---------------------------------------------------------------------------
# Branch B -- frozen ADP optimal-regulation policy  u = -(1/2R) zeta(x)^T w
# ---------------------------------------------------------------------------
class ADPController:
    def __init__(self):
        self.w = np.array(PA.ADP_W)
        self.mon = PA.ADP_MON
        self.k = -1.0 / (2.0 * PA.ADP_R)

    def zeta(self, x):
        """c * d/dx3 of the value-function basis [quadratic ; nonlinear]."""
        x1, x2, x3 = x[0], x[1], x[2]
        z = [0.0, 0.0, x1, 0.0, x2, 2.0 * x3]              # d/dx3 of quadratic monomials
        for e in self.mon:                                 # d/dx3 of nonlinear monomials
            if e[2] == 0:
                z.append(0.0)
            else:
                z.append(e[2] * x1 ** e[0] * x2 ** e[1] * x3 ** (e[2] - 1))
        return np.array(z) * PA.C_IN

    def control(self, x):
        return _sat(self.k * np.dot(self.zeta(x), self.w), PA.U_MAX)

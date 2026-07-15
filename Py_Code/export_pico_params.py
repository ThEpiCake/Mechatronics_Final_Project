"""
Desktop tool: freeze the design constants (plant coefficients, RBF network geometry
and Lyapunov gain b^T P, ADP value-function weights, timing) into a plain-Python
module ../Pico_Code/params.py that MicroPython can import directly (no NumPy/SciPy).
"""
import numpy as np
import plant as P
import config as C
import rbf_adaptive as R
import adp_offpolicy as A

# recompute the frozen ADP weights (same as adp_offpolicy __main__)
t, X, U = A.generate_data()
w_adp, _ = A.policy_iteration(X, U, C.ADP_DT, verbose=False)

def fmt(a):
    a = np.asarray(a)
    if a.ndim == 1:
        return "[" + ", ".join(f"{v:.10g}" for v in a) + "]"
    return "[" + ", ".join(fmt(r) for r in a) + "]"

lines = []
lines.append('"""Auto-generated frozen parameters for the Pico (see Py_Code/export_pico_params.py).')
lines.append('Do not edit by hand -- regenerate from the desktop design."""')
lines.append("")
lines.append("# --- plant coefficients (x' = f(x) + g u,  g=[0,0,C_IN]) ---")
lines.append(f"A1, A2, A3 = {P.a1}, {P.a2}, {P.a3}")
lines.append(f"B1, B2 = {P.b1}, {P.b2}")
lines.append(f"C_IN = {P.c}")
lines.append(f"U_MAX = {C.U_MAX}")
lines.append("")
lines.append("# --- reference trajectory  y_d = REF_A sin(REF_W t) ---")
lines.append(f"REF_A, REF_W = {C.REF_A}, {C.REF_W}")
lines.append("")
lines.append("# --- Branch A: RBF network + tracking-error / adaptation gains ---")
lines.append(f"RBF_BOX = {fmt(C.RBF_BOX)}")
lines.append(f"RBF_GRID = {tuple(C.RBF_GRID)}")
lines.append(f"RBF_WIDTH = {C.RBF_WIDTH}")
lines.append(f"LAM = {fmt(C.LAM)}          # (l2, l1, l0) error-dynamics gains")
lines.append(f"BP = {fmt(R.bP)}   # b^T P  (Lyapunov row for the weight update)")
lines.append(f"GAMMA_ADAPT = {C.GAMMA_ADAPT}")
lines.append(f"W_MAX = {C.W_MAX}")
lines.append("")
lines.append("# --- Branch B: frozen ADP value-function weights & basis ---")
lines.append(f"ADP_W = {fmt(w_adp)}")
lines.append(f"ADP_MON = {fmt(C.ADP_MONOMIALS)}   # nonlinear monomial exponents")
lines.append(f"ADP_R = {C.ADP_R}")
lines.append("")
lines.append("# --- real-time timing ---")
lines.append(f"PICO_DT = {C.PICO_DT}     # plant integration step [s] (core 1)")
lines.append(f"PICO_N = {C.PICO_N}          # controller runs every N steps")
lines.append(f"PICO_TS = {C.PICO_TS}     # controller period Ts = N*dt [s] (core 0)")
lines.append("")

with open("../Pico_Code/params.py", "w") as f:
    f.write("\n".join(lines) + "\n")
print("wrote ../Pico_Code/params.py")
print(f"  ADP_W = {np.round(w_adp,4)}")
print(f"  BP    = {np.round(R.bP,5)}  LAM = {C.LAM}")

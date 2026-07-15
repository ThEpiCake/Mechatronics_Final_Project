"""Auto-generated frozen parameters for the Pico (see Py_Code/export_pico_params.py).
Do not edit by hand -- regenerate from the desktop design."""

# --- plant coefficients (x' = f(x) + g u,  g=[0,0,C_IN]) ---
A1, A2, A3 = 6.0, 25.0, 100.0
B1, B2 = 200.0, 120.0
C_IN = 5.0
U_MAX = 40.0

# --- reference trajectory  y_d = REF_A sin(REF_W t) ---
REF_A, REF_W = 0.5, 2.0

# --- Branch A: RBF network + tracking-error / adaptation gains ---
RBF_BOX = [[-0.7, 0.7], [-1.3, 1.3], [-3, 3]]
RBF_GRID = (6, 6, 6)
RBF_WIDTH = 0.24
LAM = [21, 146, 336]          # (l2, l1, l0) error-dynamics gains
BP = [0.0744047619, 0.1120192308, 0.02914377289]   # b^T P  (Lyapunov row for the weight update)
GAMMA_ADAPT = 40000.0
W_MAX = 600.0

# --- Branch B: frozen ADP value-function weights & basis ---
ADP_W = [142.4532391, 47.05655143, 1.507198037, 10.40130824, 0.8544764147, 0.06514417075, -4.209970205, -0.162800595, -0.4582616195, 0.008331340857, -0.5569282389, -0.7593206006]
ADP_MON = [[4, 0, 0], [3, 1, 0], [2, 2, 0], [2, 0, 1], [2, 1, 1], [3, 0, 1]]   # nonlinear monomial exponents
ADP_R = 0.1

# --- real-time timing ---
PICO_DT = 0.002     # plant integration step [s] (core 1)
PICO_N = 5          # controller runs every N steps
PICO_TS = 0.01     # controller period Ts = N*dt [s] (core 0)


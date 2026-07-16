"""
Bounded, logged run of the two-core real-time scheme (Part 4 hardware experiment).

Core allocation (same intent as main.py):

    core 1 : the nonlinear process.  A paced real-time loop performs one scalar
             RK4 step (plant_core1.rk4_scalars) every dt, with catch-up if a step
             is transiently delayed (e.g. by a GC pause on the other core).  A
             plain machine.Timer is NOT used for the plant: on the RP2040 port
             MicroPython soft-timer callbacks are always dispatched on core 0
             (verified on hardware via the SIO CPUID register), so a paced loop
             inside the _thread task is the only way to actually run the process
             on the second core.
    core 0 : the controller, triggered by a hardware Timer every Ts = N*dt.  It
             copies the three state scalars under the lock, computes u, publishes
             it, and logs (y, u, exec time).  gc.collect() runs every GC_EVERY
             ticks so collection happens at a controlled point, never mid-update.

The run lasts a fixed simulated time T; the log is printed as CSV over USB so the
desktop side (plot_hw_results.py) can draw the Pico-vs-computer overlays and the
timing histogram from real on-chip data.

Usage from the host:

    mpremote exec "import pico_experiment; pico_experiment.run('rbf', 30.0)"
    mpremote exec "import pico_experiment; pico_experiment.run('adp',  8.0)"

Each call shuts the timer down and releases the second core, so both runs can be
captured in one session.
"""
import _thread
import array
import gc
import machine
import utime
from machine import Timer

import params as PA
import plant_core1 as PL
import pico_control as CT

machine.freq(200_000_000)      # officially supported RP2040 fast clock

_CPUID = 0xD0000000            # SIO CPUID register: 0 on core 0, 1 on core 1
GC_EVERY = 10                  # controller ticks between deterministic collections

# shared between the cores (single-writer each; lock guards the exchange)
_s1 = _s2 = _s3 = 0.0          # state scalars, written by core 1
_u = 0.0                       # input, written by core 0
_lock = _thread.allocate_lock()
_plant_ticks = 0
_plant_cpu = -1
_plant_late_max = 0            # worst plant-step start delay [us]
_core1_stop = False
_core1_done = False
_go = False                    # plant holds x0 until the first u is published


def _core1_task():
    """Plant real-time loop on the second core: one RK4 step every dt."""
    global _s1, _s2, _s3, _plant_ticks, _plant_cpu, _core1_done, _plant_late_max
    _plant_cpu = machine.mem32[_CPUID] & 1
    dt = PA.PICO_DT
    dt_us = int(dt * 1e6)
    x1, x2, x3 = _s1, _s2, _s3
    while not _go and not _core1_stop:      # aligned start: wait for u(x0)
        pass
    nxt = utime.ticks_add(utime.ticks_us(), dt_us)
    while not _core1_stop:
        late = utime.ticks_diff(utime.ticks_us(), nxt)
        if late < 0:
            continue                        # busy-wait: the core is dedicated
        if late > _plant_late_max:
            _plant_late_max = late
        nxt = utime.ticks_add(nxt, dt_us)
        with _lock:
            u = _u
        x1, x2, x3 = PL.rk4_scalars(x1, x2, x3, u, dt)
        with _lock:
            _s1 = x1; _s2 = x2; _s3 = x3
        _plant_ticks += 1
    _core1_done = True


def run(mode="rbf", T=20.0):
    global _s1, _s2, _s3, _u, _core1_stop, _core1_done, _plant_ticks, _plant_late_max, _go
    K = int(round(T / PA.PICO_TS))
    Ts = PA.PICO_TS

    y_log = array.array("f", bytearray(4 * K))
    u_log = array.array("f", bytearray(4 * K))
    exec_log = array.array("H", bytearray(2 * K))   # controller update [us]
    tot_log = array.array("H", bytearray(2 * K))    # full tick incl. log/gc [us]

    ctrl = CT.RBFController() if mode == "rbf" else CT.ADPController()
    _s1, _s2, _s3 = (0.0, 0.0, 0.0) if mode == "rbf" else (0.7, 0.5, -1.0)
    _u = 0.0
    _plant_ticks = 0
    _plant_late_max = 0
    _core1_stop = False
    _core1_done = False
    _go = False

    box = {"k": 0, "cpu": -1}

    def _ctrl_tick(timer):
        global _u, _go
        k = box["k"]
        if k >= K:
            return
        box["cpu"] = machine.mem32[_CPUID] & 1
        ta = utime.ticks_us()
        with _lock:
            x = (_s1, _s2, _s3)
        if mode == "rbf":
            u = ctrl.update(x, k * Ts, Ts)
        else:
            u = ctrl.control(x)
        with _lock:
            _u = u
        te = utime.ticks_diff(utime.ticks_us(), ta)   # read + compute + publish (C5)
        _go = True                 # release the plant on the first published u
        y_log[k] = x[0]
        u_log[k] = u
        if k % GC_EVERY == 0:
            gc.collect()      # collection at a controlled point, never mid-update
        tt = utime.ticks_diff(utime.ticks_us(), ta)
        exec_log[k] = te if te < 65535 else 65535
        tot_log[k] = tt if tt < 65535 else 65535
        box["k"] = k + 1

    gc.collect()
    _thread.start_new_thread(_core1_task, ())
    utime.sleep_ms(100)                       # let the plant loop start
    tim0 = Timer(period=int(Ts * 1000), mode=Timer.PERIODIC, callback=_ctrl_tick)

    t_wall = utime.ticks_ms()
    while box["k"] < K:
        utime.sleep_ms(50)
    wall = utime.ticks_diff(utime.ticks_ms(), t_wall)

    tim0.deinit()
    _core1_stop = True
    while not _core1_done:
        utime.sleep_ms(5)

    ex = exec_log[1:]                          # skip the cold first tick
    print("# mode=%s T=%.1f Ts=%g dt=%g K=%d plant_ticks=%d wall_ms=%d plant_late_max_us=%d" %
          (mode, T, Ts, PA.PICO_DT, K, _plant_ticks, wall, _plant_late_max))
    print("# freq_hz=%d ctrl_cpu=%d plant_cpu=%d micropython=%s" %
          (machine.freq(), box["cpu"], _plant_cpu,
           ".".join(str(v) for v in __import__("sys").implementation.version[:3])))
    print("# exec_us avg=%.1f max=%d | tick_us(incl log/gc) avg=%.1f max=%d | Ts_us=%d" %
          (sum(ex) / len(ex), max(ex), sum(tot_log[1:]) / (K - 1), max(tot_log[1:]),
           int(Ts * 1e6)))
    print("k,y,u,exec_us,tot_us")
    for k in range(K):
        print("%d,%.6f,%.5f,%d,%d" % (k, y_log[k], u_log[k], exec_log[k], tot_log[k]))
    print("# done")

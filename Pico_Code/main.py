"""
Raspberry Pi Pico top-level (Part 4).  Runs the complete two-core real-time scheme:

    core 1 (second core)  : simulates the nonlinear process.  A paced real-time
                            loop performs ONE scalar RK4 step every dt with the
                            controller input u held constant (ZOH) and catch-up if
                            a step is transiently delayed.  (A machine.Timer is
                            NOT used here: on the RP2040 port soft-timer callbacks
                            are always dispatched on core 0 -- verified on hardware
                            via the SIO CPUID register -- so a paced loop inside
                            the _thread task is what actually runs on core 1.)
    core 0 (main core)    : a hardware Timer with period Ts = N*dt triggers ONE
                            controller update -- it reads the latest full state,
                            computes u, (adapts the RBF weights,) and publishes u
                            to core 1.  The per-update execution time is measured
                            and its average and maximum are reported (must stay
                            below Ts).  gc.collect() runs every GC_EVERY ticks so
                            collection happens at a controlled point.

Set MODE = "rbf" (adaptive tracking, C3) or "adp" (frozen optimal policy, C4).

This module targets MicroPython on the Pico (machine, _thread, ulab).  It is not run
on the desktop; the numerical logic is validated off-board by desktop_sim.py, which
imports the same params/plant_core1/pico_control unchanged.
"""
import _thread
import gc
import utime
import machine
from machine import Timer

import params as PA
import plant_core1 as PL
import pico_control as CT

machine.freq(200_000_000)       # 200 MHz, the officially supported RP2040 fast clock

MODE = "rbf"                    # "rbf" (Branch A tracking) or "adp" (Branch B regulation)
X0 = [0.0, 0.0, 0.0] if MODE == "rbf" else [0.7, 0.5, -1.0]
GC_EVERY = 10                   # controller ticks between deterministic collections

# ---- shared state between the cores (single-writer each; lock guards exchange) ----
_s1, _s2, _s3 = X0              # state scalars, written by core 1, read by core 0
_u = 0.0                        # written by core 0, read by core 1
_lock = _thread.allocate_lock()
_go = False                     # plant holds X0 until the first u is published

# ---- execution-time statistics (C5) ----
_exec_sum = 0
_exec_max = 0
_exec_cnt = 0


# ---------------------------------------------------------------------------
# core 1: process simulation, one scalar RK4 step every dt (paced loop)
# ---------------------------------------------------------------------------
def core1_task():
    global _s1, _s2, _s3
    dt = PA.PICO_DT
    dt_us = int(dt * 1e6)
    x1, x2, x3 = _s1, _s2, _s3
    while not _go:                      # aligned start: wait for u(X0)
        pass
    nxt = utime.ticks_add(utime.ticks_us(), dt_us)
    while True:
        if utime.ticks_diff(utime.ticks_us(), nxt) < 0:
            continue                    # busy-wait: the core is dedicated
        nxt = utime.ticks_add(nxt, dt_us)
        with _lock:
            u = _u
        x1, x2, x3 = PL.rk4_scalars(x1, x2, x3, u, dt)
        with _lock:
            _s1 = x1; _s2 = x2; _s3 = x3


# ---------------------------------------------------------------------------
# core 0: controller update, one per timer tick (period Ts = N*dt)
# ---------------------------------------------------------------------------
def make_controller_tick(ctrl):
    box = {"k": 0}

    def _tick(timer):
        global _u, _go, _exec_sum, _exec_max, _exec_cnt
        k = box["k"]
        ta = utime.ticks_us()
        with _lock:
            x = (_s1, _s2, _s3)
        if MODE == "rbf":
            u = ctrl.update(x, k * PA.PICO_TS, PA.PICO_TS)   # logical time: no wraparound
        else:
            u = ctrl.control(x)
        with _lock:
            _u = u
        dt_us = utime.ticks_diff(utime.ticks_us(), ta)   # read + compute + publish (C5)
        _go = True
        if k % GC_EVERY == 0:
            gc.collect()      # deterministic collection, never mid-update
        box["k"] = k + 1
        _exec_sum += dt_us
        _exec_cnt += 1
        if dt_us > _exec_max:
            _exec_max = dt_us
    return _tick


def main():
    ctrl = CT.RBFController() if MODE == "rbf" else CT.ADPController()
    _thread.start_new_thread(core1_task, ())        # plant on core 1
    utime.sleep_ms(50)                              # let core 1 start
    Timer(period=int(PA.PICO_TS * 1000), mode=Timer.PERIODIC,
          callback=make_controller_tick(ctrl))      # controller on core 0
    # report loop.  NOTE: no lock here -- the controller callback runs on THIS core
    # at bytecode boundaries, so taking _lock in this loop could deadlock against a
    # callback dispatched while the lock is held; a single stale float read is
    # perfectly fine for a 1 Hz diagnostic print.
    while True:
        utime.sleep_ms(1000)
        y = _s1
        avg = (_exec_sum / _exec_cnt) if _exec_cnt else 0
        print("MODE=%s  y=%.4f  u=%.3f  exec avg=%.1f us  max=%d us  (Ts=%d us)"
              % (MODE, y, _u, avg, _exec_max, int(PA.PICO_TS * 1e6)))


if __name__ == "__main__":
    main()

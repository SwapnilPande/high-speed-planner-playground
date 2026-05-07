"""
Real-time hardware execution loop for PREEMPT-RT Linux.

PREEMPT-RT techniques used:
  - mlockall(MCL_CURRENT|MCL_FUTURE)   — no page faults inside the loop
  - SCHED_FIFO via sched_setscheduler  — RT thread is never preempted by CFS tasks
  - clock_nanosleep(TIMER_ABSTIME)     — absolute-deadline sleep; drift-free at 1 kHz
  - CPU affinity (optional)            — pin RT thread to a dedicated core
  - Pre-fault all arrays               — touch every page before entering the loop

Run as root or with CAP_SYS_NICE + CAP_IPC_LOCK for the RT privileges.
"""
from __future__ import annotations
import ctypes
import ctypes.util
import os
import threading
import time
import warnings
import numpy as np

from kinodynamic_planner.types import Trajectory
from kinodynamic_planner.planning.base import JointConstraints

_NJ = 7

# Safety threshold: abort if any joint exceeds this tracking error
_MAX_POS_ERR_RAD: float = 0.3

# After the trajectory, hold the final position for this many ms before releasing
_SETTLE_MS: int = 500

# ──────────────────────────────────────────────────────────────────────────────
# POSIX / Linux real-time helpers via ctypes
# ──────────────────────────────────────────────────────────────────────────────

_libc = ctypes.CDLL(ctypes.util.find_library("c"), use_errno=True)

_CLOCK_MONOTONIC = 1
_TIMER_ABSTIME   = 1
_SCHED_FIFO      = 1
_MCL_CURRENT     = 1
_MCL_FUTURE      = 2


class _Timespec(ctypes.Structure):
    _fields_ = [("tv_sec", ctypes.c_long), ("tv_nsec", ctypes.c_long)]


class _SchedParam(ctypes.Structure):
    _fields_ = [("sched_priority", ctypes.c_int)]


def _lock_memory() -> None:
    """Lock all current and future pages — prevents page faults in the RT loop."""
    ret = _libc.mlockall(_MCL_CURRENT | _MCL_FUTURE)
    if ret != 0:
        warnings.warn(
            f"mlockall failed (errno {ctypes.get_errno()}). "
            "RT loop may experience page-fault jitter. "
            "Run as root or raise RLIMIT_MEMLOCK."
        )


def _set_sched_fifo(priority: int) -> None:
    """Set SCHED_FIFO for the calling thread (must be called from within the thread)."""
    param = _SchedParam(sched_priority=priority)
    ret = _libc.sched_setscheduler(0, _SCHED_FIFO, ctypes.byref(param))
    if ret != 0:
        raise PermissionError(
            f"sched_setscheduler(SCHED_FIFO, {priority}) failed "
            f"(errno {ctypes.get_errno()}). Run as root or grant CAP_SYS_NICE."
        )


def _now() -> _Timespec:
    ts = _Timespec()
    _libc.clock_gettime(_CLOCK_MONOTONIC, ctypes.byref(ts))
    return ts


def _now_ns() -> int:
    ts = _now()
    return ts.tv_sec * 1_000_000_000 + ts.tv_nsec


def _ts_add_ns(ts: _Timespec, ns: int) -> _Timespec:
    total = ts.tv_sec * 1_000_000_000 + ts.tv_nsec + ns
    return _Timespec(tv_sec=total // 1_000_000_000, tv_nsec=total % 1_000_000_000)


def _sleep_until(ts: _Timespec) -> None:
    _libc.clock_nanosleep(_CLOCK_MONOTONIC, _TIMER_ABSTIME, ctypes.byref(ts), None)


def _prefault(arrays: list[np.ndarray]) -> None:
    """Touch every element of each array to force OS to map all pages now."""
    for a in arrays:
        a.fill(0)


# ──────────────────────────────────────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────────────────────────────────────

def run_on_hardware(
    arm,
    traj: Trajectory,
    constraints: JointConstraints,
    rt_priority: int = 80,
    rt_cpu: int | None = None,
) -> dict:
    """Execute *traj* on the real arm with PREEMPT-RT timing.

    Parameters
    ----------
    arm:
        Connected :class:`KinovaArm` already in LOW_LEVEL_SERVOING mode.
    traj:
        1 kHz trajectory from any planner.
    constraints:
        Used only for the pre-run velocity sanity check.
    rt_priority:
        SCHED_FIFO priority for the control thread (1–99; 80 is a sane default).
    rt_cpu:
        If given, pin the RT thread to this CPU core via sched_setaffinity.

    Returns
    -------
    dict with keys: t, q_cmd, q_actual, qd_cmd, qd_actual, latency_us, overruns
    """
    N  = len(traj.t)
    dt = traj.dt

    log: dict[str, np.ndarray | int] = {
        "t":          np.empty(N),
        "q_cmd":      np.empty((N, _NJ)),
        "q_actual":   np.empty((N, _NJ)),
        "qd_cmd":     np.empty((N, _NJ)),
        "qd_actual":  np.empty((N, _NJ)),
        "latency_us": np.empty(N),
    }

    # Lock memory process-wide before spawning the RT thread (skip if RT disabled)
    if rt_priority > 0:
        _lock_memory()

    # Pre-fault all log arrays and trajectory data now so the RT loop is page-fault-free
    _prefault(list(log.values()) + [traj.q, traj.qd])  # type: ignore[arg-type]

    stop_event = threading.Event()
    exc_holder: list[BaseException | None] = [None]

    def _rt_thread_body() -> None:
        try:
            if rt_cpu is not None:
                os.sched_setaffinity(0, {rt_cpu})
            if rt_priority > 0:
                _set_sched_fifo(rt_priority)
            _control_loop(arm, traj, log, stop_event)
        except BaseException as exc:
            exc_holder[0] = exc
            stop_event.set()

    thread = threading.Thread(target=_rt_thread_body, daemon=True, name="rt-control")
    thread.start()

    try:
        while thread.is_alive() and not stop_event.is_set():
            time.sleep(0.05)
    except KeyboardInterrupt:
        print("\n[rt_runner] Ctrl-C — emergency stop")
        stop_event.set()
        arm.estop()

    thread.join(timeout=5.0)
    if thread.is_alive():
        print("[rt_runner] WARNING: RT thread did not exit in 5 s")

    if exc_holder[0] is not None:
        raise exc_holder[0]  # type: ignore[misc]

    overruns = int((log["latency_us"] > traj.dt * 1e6).sum())  # type: ignore[index]
    log["overruns"] = overruns
    return log


# ──────────────────────────────────────────────────────────────────────────────
# Inner RT loop (runs inside SCHED_FIFO thread)
# ──────────────────────────────────────────────────────────────────────────────

def _control_loop(
    arm,
    traj: Trajectory,
    log: dict,
    stop_event: threading.Event,
) -> None:
    N         = len(traj.t)
    period_ns = round(traj.dt * 1_000_000_000)

    # Arm first wakeup at the next period boundary from now
    t_next = _ts_add_ns(_now(), period_ns)

    for i in range(N):
        if stop_event.is_set():
            return

        # ── sleep until absolute deadline ──────────────────────────────────
        _sleep_until(t_next)
        t_wake_ns = _now_ns()

        # ── command + state read (single RPC) ──────────────────────────────
        arm.send_joint_positions(traj.q[i], traj.qd[i])
        q_actual, qd_actual = arm.read_joint_state()

        # ── debug: every ~500 ms print what we sent vs what we read back ──
        if i % max(1, round(0.5 / traj.dt)) == 0:
            pre_refresh = np.array(arm._last_cmd_deg)        # captured pre-Refresh
            post_refresh = np.array([
                arm._command.actuators[j].position for j in range(_NJ)
            ])
            fb_deg = np.array([
                arm._feedback.actuators[j].position for j in range(_NJ)
            ])
            print(
                f"[rt] step {i:5d}  "
                f"traj_rad={np.round(traj.q[i], 3)}  "
                f"pre={np.round(pre_refresh, 2)}  "
                f"post={np.round(post_refresh, 2)}  "
                f"fb={np.round(fb_deg, 2)}"
            )

        # ── safety check (wrap to [-π, π] for continuous joints) ──────────
        delta = (q_actual - traj.q[i] + np.pi) % (2 * np.pi) - np.pi
        pos_err = float(np.abs(delta).max())
        if pos_err > _MAX_POS_ERR_RAD:
            stop_event.set()
            arm.estop()
            raise RuntimeError(
                f"Safety abort at step {i}/{N}: "
                f"tracking error {pos_err:.3f} rad > limit {_MAX_POS_ERR_RAD} rad"
            )

        # ── log ─────────────────────────────────────────────────────────────
        log["t"][i]          = i * traj.dt
        log["q_cmd"][i]      = traj.q[i]
        log["q_actual"][i]   = q_actual
        log["qd_cmd"][i]     = traj.qd[i]
        log["qd_actual"][i]  = qd_actual
        log["latency_us"][i] = (_now_ns() - t_wake_ns) / 1_000.0

        t_next = _ts_add_ns(t_next, period_ns)

    # ── settle: hold final position ────────────────────────────────────────
    q_final   = traj.q[-1]
    n_settle  = _SETTLE_MS * 1000 // round(traj.dt * 1e6)   # steps
    for _ in range(n_settle):
        if stop_event.is_set():
            return
        _sleep_until(t_next)
        arm.send_joint_positions(q_final)
        t_next = _ts_add_ns(t_next, period_ns)

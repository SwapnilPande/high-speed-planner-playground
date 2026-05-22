"""Plan a trajectory and execute it on the real Kinova Gen3 arm.

Examples
--------
# Dry-run (simulate in MuJoCo, no hardware):
  python scripts/run_on_arm.py --dry-run --planner toppra

# Real arm:
  python scripts/run_on_arm.py --arm-ip 192.168.1.10 --planner toppra

# Real arm, pin RT thread to core 3, save execution log:
  python scripts/run_on_arm.py --arm-ip 192.168.1.10 --planner toppra \
      --rt-cpu 3 --log run_log.npy
"""
from __future__ import annotations
import argparse
import time
import numpy as np

from kinodynamic_planner.types import Trajectory


# ── Joint-space test move ─────────────────────────────────────────────────────
# Joint angles are written in pendant-display form (degrees in [0, 360°)).
# `_canonical` wraps each into (-π, π] so bounded joints (1, 3, 5) land in
# their physical range and continuous joints stay well-defined for planning.
def _canonical(q_deg: np.ndarray) -> np.ndarray:
    q = q_deg * np.pi / 180.0
    return (q + np.pi) % (2.0 * np.pi) - np.pi

Q_START = _canonical(np.array([322.0, 303, 180.0, 213.0, 0.0, 350.0, 85.0]))
Q_WAYPOINT = _canonical(np.array([322, 80, 181, 230.0, 0, 120, 90.0]))
Q_GOAL  = _canonical(np.array([322.0,  80, 180.0, 254.0, 0.0,  94.0, 88.0]))

# Continuous joints can rotate past ±π — bounded joints (indices 1, 3, 5) cannot.
_CONTINUOUS = np.array([True, False, True, False, True, False, True])


def _unwrap_to_shortest(q_target: np.ndarray, q_ref: np.ndarray) -> np.ndarray:
    """Shift continuous-joint values of q_target by ±2π so the path from q_ref
    is the shortest angular route. Bounded joints are left untouched.

    Without this, a target on the opposite side of the ±π seam from the start
    causes the planner to sweep ~2π in joint space instead of the short arc.
    """
    q = q_target.copy()
    delta = q[_CONTINUOUS] - q_ref[_CONTINUOUS]
    delta = (delta + np.pi) % (2.0 * np.pi) - np.pi
    q[_CONTINUOUS] = q_ref[_CONTINUOUS] + delta
    return q

# ── CLI ───────────────────────────────────────────────────────────────────────

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Plan + execute a trajectory on the Kinova Gen3."
    )
    p.add_argument("--planner", choices=["toppra", "ruckig", "min-jerk", "kinova"],
                   default="toppra",
                   help="kinova = use the arm's onboard high-level planner "
                        "(baseline; ignores --control-hz and all RT options)")
    p.add_argument("--arm-ip",   default="192.168.1.10")
    p.add_argument("--username", default="admin")
    p.add_argument("--password", default="admin")
    p.add_argument("--rt-priority", type=int, default=80,
                   help="SCHED_FIFO priority for the RT thread (1-99)")
    p.add_argument("--rt-cpu", type=int, default=None,
                   help="Pin RT thread to this CPU core (optional)")
    p.add_argument("--dry-run", action="store_true",
                   help="Simulate in MuJoCo instead of connecting to hardware")
    p.add_argument("--sim-model", default="models/kinova_gen3/scene.xml",
                   help="MuJoCo scene model (for --dry-run)")
    p.add_argument("--plan-model", default="models/kinova_gen3/gen3_plan.xml",
                   help="MuJoCo planning model (for TOPP-RA inverse dynamics)")
    p.add_argument("--log", default=None, metavar="FILE",
                   help="Save execution log to .npy file")
    p.add_argument("--yes", "-y", action="store_true",
                   help="Skip confirmation prompt")
    p.add_argument("--speed-scale", type=float, default=1.0,
                   help="Scale v_max/a_max/j_max by this factor (e.g. 0.25 for a slow first run)")
    p.add_argument("--control-hz", type=float, default=1000.0,
                   help="Trajectory sample rate / RT loop frequency. Drop to 500 if Python overhead caps the loop below 1 kHz.")
    return p


# ── Planning ─────────────────────────────────────────────────────────────────

def _make_planner(args, dt):
    if args.planner == "toppra":
        from kinodynamic_planner.planning.toppra_planner import TOPPRAPlanner
        return TOPPRAPlanner(args.plan_model, dt=dt)
    elif args.planner == "ruckig":
        from kinodynamic_planner.planning.ruckig_planner import RuckigPlanner
        return RuckigPlanner(dt=dt)
    else:
        from kinodynamic_planner.planning.min_jerk import MinJerkPlanner
        return MinJerkPlanner(dt=dt)


def _concat(traj1: Trajectory, traj2: Trajectory) -> Trajectory:
    # Drop traj2's first sample — it duplicates traj1's last (both at waypoint).
    dt = traj1.dt
    N = len(traj1.t) + len(traj2.t) - 1
    t = np.arange(N) * dt
    q  = np.concatenate([traj1.q,  traj2.q[1:]],  axis=0)
    qd = np.concatenate([traj1.qd, traj2.qd[1:]], axis=0)
    qdd = None
    if traj1.qdd is not None and traj2.qdd is not None:
        qdd = np.concatenate([traj1.qdd, traj2.qdd[1:]], axis=0)
    return Trajectory(t=t, q=q, qd=qd, qdd=qdd)


def _plan(args, constraints):
    dt = 1.0 / args.control_hz
    planner = _make_planner(args, dt)
    # Unwrap each subsequent target relative to the previous so continuous
    # joints take the short arc instead of crossing the ±π seam the long way.
    wp   = _unwrap_to_shortest(Q_WAYPOINT, Q_START)
    goal = _unwrap_to_shortest(Q_GOAL,     wp)

    # Planners that support waypoints (min-jerk) flow through the waypoint in
    # one trajectory; others fall back to two stop-at-waypoint legs.
    if hasattr(planner, "plan_waypoints"):
        traj = planner.plan_waypoints([Q_START, wp, goal], constraints)
        print(f"  Stitched  : {traj.duration:.3f} s  "
              f"({len(traj.t)} steps, flows through waypoint)")
        return traj

    leg1 = planner.plan(Q_START, wp,   constraints)
    leg2 = planner.plan(wp,      goal, constraints)
    print(f"  Leg 1     : {leg1.duration:.3f} s  ({len(leg1.t)} steps)")
    print(f"  Leg 2     : {leg2.duration:.3f} s  ({len(leg2.t)} steps)")
    return _concat(leg1, leg2)


# ── Dry-run (MuJoCo) ─────────────────────────────────────────────────────────

def _dry_run(args, traj) -> dict:
    from kinodynamic_planner.sim.simulator import Simulator
    from kinodynamic_planner.runner.playback import run_playback

    sim = Simulator(model_path=args.sim_model, control_hz=1.0 / traj.dt)
    print("[dry-run] Running in MuJoCo simulator (no hardware)")
    log = run_playback(sim, traj, render=False)
    sim.close()
    return log


# ── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    args = _build_parser().parse_args()

    from kinodynamic_planner.planning.base import JointConstraints
    constraints = JointConstraints.kinova_gen3()
    if args.speed_scale != 1.0:
        if args.speed_scale <= 0.0 or args.speed_scale > 10.0:
            raise ValueError(f"--speed-scale must be in (0, 1], got {args.speed_scale}")
        s = args.speed_scale
        constraints = JointConstraints(
            v_max=constraints.v_max * s,
            a_max=constraints.a_max * s,
            j_max=constraints.j_max * s,
        )
        print(f"[constraints] Scaled v/a/j limits by {s}")

    if args.planner == "kinova":
        if args.dry_run:
            raise SystemExit(
                "--planner kinova runs Kinova's onboard planner — no dry-run "
                "(MuJoCo doesn't model the firmware's planner). Use a different "
                "planner for dry-run, or omit --dry-run."
            )
        _run_kinova_baseline(args)
        return

    # ── Plan ──────────────────────────────────────────────────────────────
    print(f"Planning with {args.planner}...")
    traj = _plan(args, constraints)
    print(f"  Duration : {traj.duration:.3f} s  ({len(traj.t)} steps @ {1/traj.dt:.0f} Hz)")

    _print_trajectory(traj)

    # ── Dry-run ───────────────────────────────────────────────────────────
    if args.dry_run:
        log = _dry_run(args, traj)
        _print_log_summary(log)
        _save_log(log, args.log)
        return

    # ── Hardware execution ────────────────────────────────────────────────
    from kinodynamic_planner.hardware.kinova_arm import KinovaArm
    from kinodynamic_planner.hardware.rt_runner import run_on_hardware

    print(f"\nConnecting to arm at {args.arm_ip}...")
    with KinovaArm(ip=args.arm_ip, username=args.username, password=args.password) as arm:

        # Clear any latched faults from a prior run before commanding motion
        print("[reset] Clearing faults...")
        arm.clear_faults()

        # Show where the arm actually is so we can compare with the target.
        q_now, _ = arm.read_joint_state()
        print(f"[reset] Current pose (rad): {np.round(q_now, 3)}")
        print(f"[reset] Current pose (deg): {np.round(np.degrees(q_now), 1)}")
        target_deg_api = (np.degrees(traj.q[0]) % 360.0)
        print(f"[reset] Target  pose (rad): {np.round(traj.q[0], 3)}")
        print(f"[reset] Target  pose (deg, API form): {np.round(target_deg_api, 1)}")

        # Auto-reset: move arm to trajectory start using high-level API
        print(f"[reset] Moving to start position ...")
        arm.move_to_joints(traj.q[0])
        print("[reset] Done.")

        # Hold here until operator signals go
        if not args.yes:
            print(f"\n  Planner   : {args.planner}")
            print(f"  Duration  : {traj.duration:.3f} s")
            print(f"  RT prio   : SCHED_FIFO / priority {args.rt_priority}")
            if args.rt_cpu is not None:
                print(f"  RT CPU    : core {args.rt_cpu}")
            input("\nArm is at start. Press Enter to execute...")

        print("\n[hardware] Switching to low-level servoing...")
        arm.set_low_level_servoing()

        print(f"[hardware] Executing trajectory ({traj.duration:.3f} s)...")
        t0 = time.monotonic()
        log = run_on_hardware(
            arm, traj, constraints,
            rt_priority=args.rt_priority,
            rt_cpu=args.rt_cpu,
        )
        elapsed = time.monotonic() - t0

        print(f"[hardware] Done in {elapsed:.3f} s wall time")

    _print_log_summary(log)
    _save_log(log, args.log)


# ── Kinova onboard-planner baseline ──────────────────────────────────────────

def _run_kinova_baseline(args) -> None:
    """Execute Q_START → Q_WAYPOINT → Q_GOAL using the arm's onboard high-level
    planner. The arm computes and runs its own trajectory; we record actual
    joint state via cyclic feedback so it can be compared against other planners.
    """
    from kinodynamic_planner.hardware.kinova_arm import KinovaArm

    print(f"\nConnecting to arm at {args.arm_ip}...")
    with KinovaArm(ip=args.arm_ip, username=args.username, password=args.password) as arm:
        print("[reset] Clearing faults...")
        arm.clear_faults()

        q_now, _ = arm.read_joint_state()
        print(f"[reset] Current pose (deg): {np.round(np.degrees(q_now), 1)}")

        print("[reset] Moving to start position ...")
        arm.move_to_joints(Q_START)
        print("[reset] Done.")

        if not args.yes:
            print(f"\n  Planner   : kinova (onboard high-level)")
            input("\nArm is at start. Press Enter to execute...")

        print("\n[hardware] Executing via Kinova onboard planner...")
        t0 = time.monotonic()
        leg1 = arm.move_to_joints_logged(Q_WAYPOINT)
        leg2 = arm.move_to_joints_logged(Q_GOAL)
        elapsed = time.monotonic() - t0
        print(f"[hardware] Done in {elapsed:.3f} s wall time")
        print(f"  Leg 1 : {leg1['t'][-1]:.3f} s  ({len(leg1['t'])} samples)")
        print(f"  Leg 2 : {leg2['t'][-1]:.3f} s  ({len(leg2['t'])} samples)")

    # Stitch legs: offset leg2 timestamps so the log is continuous.
    t_offset = leg1["t"][-1]
    t  = np.concatenate([leg1["t"], leg2["t"] + t_offset])
    q  = np.concatenate([leg1["q"], leg2["q"]])
    qd = np.concatenate([leg1["qd"], leg2["qd"]])

    # Build a log dict compatible with _print_log_summary / downstream plotting.
    # No commanded trajectory exists for this baseline — q_cmd is set to the
    # active leg's goal so the "tracking error" line reports distance-to-goal.
    n1, n2 = len(leg1["t"]), len(leg2["t"])
    q_cmd = np.vstack([np.tile(Q_WAYPOINT, (n1, 1)), np.tile(Q_GOAL, (n2, 1))])
    log = {
        "t":         t,
        "q_cmd":     q_cmd,
        "q_actual":  q,
        "qd_cmd":    np.zeros_like(q),
        "qd_actual": qd,
    }
    _print_log_summary(log)
    _save_log(log, args.log)


# ── Reporting ─────────────────────────────────────────────────────────────────

def _print_trajectory(traj, max_rows: int = 50) -> None:
    """Print a downsampled view of the trajectory: time, q (rad), qd (rad/s)."""
    N = len(traj.t)
    idx = np.unique(np.linspace(0, N - 1, min(max_rows, N)).astype(int))
    header = "  step      t(s)   " + "      ".join(f"q{i}" for i in range(7)) + "   |   " + "    ".join(f"qd{i}" for i in range(7))
    print(f"\nTrajectory ({len(idx)} of {N} steps shown):")
    print(header)
    for i in idx:
        q_str  = " ".join(f"{v:+7.3f}" for v in traj.q[i])
        qd_str = " ".join(f"{v:+7.3f}" for v in traj.qd[i])
        print(f"  {i:5d}  {traj.t[i]:7.3f}   {q_str}   |   {qd_str}")
    print()


def _print_log_summary(log: dict) -> None:
    delta = (log["q_actual"] - log["q_cmd"] + np.pi) % (2 * np.pi) - np.pi
    q_err = np.abs(delta)
    print(f"\nTracking:")
    print(f"  Mean position error : {q_err.mean():.4f} rad")
    print(f"  Max  position error : {q_err.max():.4f} rad")
    if "latency_us" in log:
        lat = log["latency_us"]
        print(f"\nRT loop latency (step work time, not sleep):")
        print(f"  Mean : {lat.mean():.1f} µs")
        print(f"  Max  : {lat.max():.1f} µs")
        overruns = int(log.get("overruns", (lat > 1000).sum()))
        print(f"  Overruns (>1 ms) : {overruns} / {len(lat)} "
              f"({100*overruns/len(lat):.2f}%)")


def _save_log(log: dict, path: str | None) -> None:
    if path:
        np.save(path, log)
        print(f"\nLog saved to {path}")


if __name__ == "__main__":
    main()

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


# ── Joint-space test move ─────────────────────────────────────────────────────
# Joint angles are written in pendant-display form (degrees in [0, 360°)).
# `_canonical` wraps each into (-π, π] so bounded joints (1, 3, 5) land in
# their physical range and continuous joints stay well-defined for planning.
def _canonical(q_deg: np.ndarray) -> np.ndarray:
    q = q_deg * np.pi / 180.0
    return (q + np.pi) % (2.0 * np.pi) - np.pi

Q_START = _canonical(np.array([0.0, 295, 180.0, 213.0, 0.0, 345.0, 95.0]))
Q_GOAL  = _canonical(np.array([0.0,  82, 180.0, 292.0, 0.0,  60.0, 95.0]))

# ── CLI ───────────────────────────────────────────────────────────────────────

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Plan + execute a trajectory on the Kinova Gen3."
    )
    p.add_argument("--planner", choices=["toppra", "ruckig", "min-jerk"],
                   default="toppra")
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

def _plan(args, constraints):
    dt = 1.0 / args.control_hz
    if args.planner == "toppra":
        from kinodynamic_planner.planning.toppra_planner import TOPPRAPlanner
        return TOPPRAPlanner(args.plan_model, dt=dt).plan(Q_START, Q_GOAL, constraints)
    elif args.planner == "ruckig":
        from kinodynamic_planner.planning.ruckig_planner import RuckigPlanner
        return RuckigPlanner(dt=dt).plan(Q_START, Q_GOAL, constraints)
    else:
        from kinodynamic_planner.planning.min_jerk import MinJerkPlanner
        return MinJerkPlanner(dt=dt).plan(Q_START, Q_GOAL, constraints)


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

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
import sys
import time
import numpy as np


# ── Joint-space test move ─────────────────────────────────────────────────────
Q_START = np.zeros(7)
Q_GOAL  = np.array([0.0, -0.8, 0.0, -1.5, 0.0, 1.2, 0.0])

# Arm must be within this tolerance of Q_START before the RT loop begins
_PREFLIGHT_TOL_RAD = 0.05


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
    return p


# ── Planning ─────────────────────────────────────────────────────────────────

def _plan(args, constraints):
    if args.planner == "toppra":
        from kinodynamic_planner.planning.toppra_planner import TOPPRAPlanner
        return TOPPRAPlanner(args.plan_model).plan(Q_START, Q_GOAL, constraints)
    elif args.planner == "ruckig":
        from kinodynamic_planner.planning.ruckig_planner import RuckigPlanner
        return RuckigPlanner().plan(Q_START, Q_GOAL, constraints)
    else:
        from kinodynamic_planner.planning.min_jerk import MinJerkPlanner
        return MinJerkPlanner().plan(Q_START, Q_GOAL, constraints)


# ── Pre-flight ────────────────────────────────────────────────────────────────

def _preflight(arm, traj) -> None:
    """Verify the arm is close enough to the trajectory start to begin safely."""
    q_current, _ = arm.read_joint_state()
    err = float(np.abs(q_current - traj.q[0]).max())
    if err > _PREFLIGHT_TOL_RAD:
        print(f"\n[preflight] FAIL — arm is {err:.3f} rad from trajectory start "
              f"(limit {_PREFLIGHT_TOL_RAD} rad).")
        print(f"  Current  : {np.round(q_current, 3)}")
        print(f"  Expected : {np.round(traj.q[0], 3)}")
        print("\nMove the arm to the home position (all zeros) and retry.")
        sys.exit(1)
    print(f"[preflight] OK  — arm at start (max error {err:.4f} rad)")


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

    # ── Plan ──────────────────────────────────────────────────────────────
    print(f"Planning with {args.planner}...")
    traj = _plan(args, constraints)
    print(f"  Duration : {traj.duration:.3f} s  ({len(traj.t)} steps @ {1/traj.dt:.0f} Hz)")

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

        # Pre-flight: verify arm position
        _preflight(arm, traj)

        # Confirm before executing
        if not args.yes:
            print(f"\n  Planner   : {args.planner}")
            print(f"  Duration  : {traj.duration:.3f} s")
            print(f"  RT prio   : SCHED_FIFO / priority {args.rt_priority}")
            if args.rt_cpu is not None:
                print(f"  RT CPU    : core {args.rt_cpu}")
            answer = input("\nExecute on real arm? [y/N] ").strip().lower()
            if answer not in ("y", "yes"):
                print("Aborted.")
                return

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

def _print_log_summary(log: dict) -> None:
    q_err = np.abs(log["q_actual"] - log["q_cmd"])
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

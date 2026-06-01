"""Connect to the Kinova arm and print the current joint configuration.

Useful for setting up waypoints: put the arm in admittance / teach mode, move
it to the desired pose by hand, and copy the printed line into
`scripts/run_on_arm.py` (Q_START / Q_WAYPOINT / Q_GOAL / Q_GRAB).

Examples
--------
# Stream joint angles at 10 Hz until Ctrl-C:
  python scripts/print_joints.py --arm-ip 192.168.1.10

# Single snapshot:
  python scripts/print_joints.py --arm-ip 192.168.1.10 --once
"""
from __future__ import annotations
import argparse
import time
import numpy as np


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Print the Kinova Gen3's current joint configuration."
    )
    p.add_argument("--arm-ip",   default="192.168.1.10")
    p.add_argument("--username", default="admin")
    p.add_argument("--password", default="admin")
    p.add_argument("--once", action="store_true",
                   help="Print a single snapshot and exit (default: stream until Ctrl-C)")
    p.add_argument("--rate-hz", type=float, default=10.0,
                   help="Polling rate when streaming (default: 10 Hz)")
    p.add_argument("--label", default="Q_NEW",
                   help="Variable name used in the copy-paste line (e.g. Q_GRAB)")
    return p


def _print_pose(q_rad: np.ndarray, label: str) -> None:
    """Print one joint reading in three forms: canonical rad, pendant deg,
    and a copy-pasteable `_canonical(np.array([...]))` line."""
    deg_api = (np.degrees(q_rad) % 360.0)
    rad_str = "  ".join(f"{v:+7.4f}" for v in q_rad)
    deg_str = "  ".join(f"{v:7.2f}" for v in deg_api)
    deg_csv = ", ".join(f"{v:.1f}" for v in deg_api)
    print(f"  rad (canonical) : [{rad_str}]")
    print(f"  deg (pendant)   : [{deg_str}]")
    print(f"  paste-ready     : {label} = _canonical(np.array([{deg_csv}]))")


def main() -> None:
    args = _build_parser().parse_args()

    from kinodynamic_planner.hardware.kinova_arm import KinovaArm

    print(f"Connecting to arm at {args.arm_ip}...")
    with KinovaArm(ip=args.arm_ip, username=args.username, password=args.password) as arm:
        if args.once:
            arm.refresh_feedback()
            q, _ = arm.read_joint_state()
            print()
            _print_pose(q, args.label)
            return

        period = 1.0 / args.rate_hz
        print(f"Streaming at {args.rate_hz:.1f} Hz. Move the arm in teach mode "
              f"and press Ctrl-C when the pose looks right.\n")
        try:
            while True:
                t0 = time.monotonic()
                arm.refresh_feedback()
                q, _ = arm.read_joint_state()
                # Overwrite-in-place by leading each block with \r and an
                # ANSI clear: keeps the terminal scroll quiet during teach.
                deg_api = (np.degrees(q) % 360.0)
                deg_str = "  ".join(f"{v:7.2f}" for v in deg_api)
                print(f"\rq (deg, pendant): [{deg_str}]", end="", flush=True)
                dt = time.monotonic() - t0
                if dt < period:
                    time.sleep(period - dt)
        except KeyboardInterrupt:
            print()  # newline after the in-place line
            arm.refresh_feedback()
            q, _ = arm.read_joint_state()
            print(f"\nFinal pose:")
            _print_pose(q, args.label)


if __name__ == "__main__":
    main()

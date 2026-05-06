"""Plan a min-jerk trajectory from Q_START to Q_GOAL and play it back in MuJoCo."""
import argparse
import numpy as np


def main():
    parser = argparse.ArgumentParser(
        description="Min-jerk trajectory demo on the Kinova Gen3"
    )
    parser.add_argument("--model",  default="models/kinova_gen3/scene.xml")
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--log",    default=None, help="Save playback log to .npy")
    args = parser.parse_args()

    from kinodynamic_planner.planning.min_jerk import MinJerkPlanner
    from kinodynamic_planner.planning.base import JointConstraints
    from kinodynamic_planner.sim.simulator import Simulator
    from kinodynamic_planner.runner.playback import run_playback

    q_start = np.zeros(7)
    q_goal  = np.array([0.0, -0.8, 0.0, -1.5, 0.0, 1.2, 0.0])
    constraints = JointConstraints.kinova_gen3()

    planner = MinJerkPlanner()
    traj = planner.plan(q_start, q_goal, constraints)

    print(f"Min-jerk plan:")
    print(f"  Steps   : {len(traj.t)}")
    print(f"  Duration: {traj.duration:.3f} s")
    print(f"  dt      : {traj.dt*1000:.1f} ms")
    print(f"  Peak |qd| : {np.abs(traj.qd).max():.4f} rad/s  "
          f"(limit {constraints.v_max.min():.2f})")
    print(f"  Peak |qdd|: {np.abs(traj.qdd).max():.4f} rad/s² "
          f"(limit {constraints.a_max.min():.2f})")

    sim = Simulator(model_path=args.model, control_hz=1.0 / traj.dt)
    log = run_playback(sim, traj, render=args.render)
    sim.close()

    q_err = np.abs(log["q_actual"] - log["q_cmd"])
    print(f"  Mean position error: {q_err.mean():.4f} rad")
    print(f"  Max  position error: {q_err.max():.4f} rad")

    if args.log:
        np.save(args.log, log)
        print(f"Log saved to {args.log}")


if __name__ == "__main__":
    main()

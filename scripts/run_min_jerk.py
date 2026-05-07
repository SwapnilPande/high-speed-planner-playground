"""Plan a min-jerk trajectory from Q_START to Q_GOAL and play it back in MuJoCo."""
import argparse
import time
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

    peak_qd_joint = int(np.abs(traj.qd).max(axis=0).argmax())
    peak_qd_val = np.abs(traj.qd).max()
    peak_qdd_joint = int(np.abs(traj.qdd).max(axis=0).argmax())
    peak_qdd_val = np.abs(traj.qdd).max()
    print(f"  Peak |qd| : {peak_qd_val:.4f} rad/s  "
          f"(joint {peak_qd_joint} limit {constraints.v_max[peak_qd_joint]:.2f})")
    print(f"  Peak |qdd|: {peak_qdd_val:.4f} rad/s² "
          f"(joint {peak_qdd_joint} limit {constraints.a_max[peak_qdd_joint]:.2f})")

    sim = Simulator(model_path=args.model, control_hz=1.0 / traj.dt)

    if args.render:
        sim.render()  # opens viewer zoomed out (distance=3.0)
        start_ee = sim.get_ee_position(q_start, body_name="bracelet_link")
        goal_ee  = sim.get_ee_position(q_goal,  body_name="bracelet_link")
        sim.add_sphere_marker(start_ee, rgba=(0.0, 1.0, 0.0, 0.8))  # green = start
        sim.add_sphere_marker(goal_ee,  rgba=(1.0, 0.0, 0.0, 0.8))  # red   = goal
        sim.sync_viewer()
        input("Press Enter to play...")

    log = run_playback(sim, traj, render=args.render)

    if args.render:
        # Settle: hold current position for 2 s so the arm reaches Q_GOAL
        settle_steps = int(2.0 / traj.dt)
        for _ in range(settle_steps):
            sim.step(np.zeros(7))
            sim.sync_viewer()
        print("Trajectory complete. Close the viewer window to exit.")
        while sim.is_viewer_open():
            sim.sync_viewer()
            time.sleep(0.05)

    sim.close()

    q_err = np.abs(log["q_actual"] - log["q_cmd"])
    print(f"  Mean position error: {q_err.mean():.4f} rad")
    print(f"  Max  position error: {q_err.max():.4f} rad")

    if args.log:
        np.save(args.log, log)
        print(f"Log saved to {args.log}")


if __name__ == "__main__":
    main()

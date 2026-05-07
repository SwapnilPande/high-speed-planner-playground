"""Plan a minimum-time trajectory via iLQR and play it back in MuJoCo."""
import argparse
import json
import pathlib
import time
import numpy as np


def main():
    parser = argparse.ArgumentParser(
        description="iLQR minimum-time trajectory demo on the Kinova Gen3"
    )
    parser.add_argument("--model",   default="models/kinova_gen3/scene.xml")
    parser.add_argument("--render",  action="store_true")
    parser.add_argument("--record",  default=None, metavar="FILE", help="Save rendered video to .mp4")
    parser.add_argument("--camera",  default="camera.json", help="Camera state file (save/load)")
    parser.add_argument("--log",     default=None, help="Save playback log to .npy")
    args = parser.parse_args()

    from kinodynamic_planner.planning.ilqr import ILQRPlanner
    from kinodynamic_planner.planning.min_jerk import MinJerkPlanner
    from kinodynamic_planner.planning.base import JointConstraints
    from kinodynamic_planner.sim.simulator import Simulator
    from kinodynamic_planner.sim.recorder import VideoRecorder
    from kinodynamic_planner.runner.playback import run_playback

    q_start = np.zeros(7)
    q_goal  = np.array([0.0, -0.8, 0.0, -1.5, 0.0, 1.2, 0.0])
    constraints = JointConstraints.kinova_gen3()

    print("Planning min-jerk reference...")
    mj_traj = MinJerkPlanner().plan(q_start, q_goal, constraints)
    print(f"  Min-jerk duration: {mj_traj.duration:.3f} s")

    print("Running iLQR bisection...")
    planner = ILQRPlanner()
    traj = planner.plan(q_start, q_goal, constraints)

    speedup = mj_traj.duration / traj.duration
    print(f"\niLQR plan:")
    print(f"  Steps   : {len(traj.t)}")
    print(f"  Duration: {traj.duration:.3f} s  ({speedup:.2f}x faster than min-jerk)")
    print(f"  dt      : {traj.dt*1000:.1f} ms")

    peak_qd_joint = int(np.abs(traj.qd).max(axis=0).argmax())
    peak_qd_val   = np.abs(traj.qd).max()
    peak_qdd_joint = int(np.abs(traj.qdd).max(axis=0).argmax())
    peak_qdd_val   = np.abs(traj.qdd).max()
    print(f"  Peak |qd| : {peak_qd_val:.4f} rad/s  "
          f"(joint {peak_qd_joint} limit {constraints.v_max[peak_qd_joint]:.2f})")
    print(f"  Peak |qdd|: {peak_qdd_val:.4f} rad/s² "
          f"(joint {peak_qdd_joint} limit {constraints.a_max[peak_qdd_joint]:.2f})")

    sim = Simulator(model_path=args.model, control_hz=1.0 / traj.dt)
    camera_path = pathlib.Path(args.camera)

    recorder = None
    if args.record:
        recorder = VideoRecorder(sim, fps=30)
        recorder.set_dt(traj.dt)

    if args.render:
        sim.render()
        if camera_path.exists():
            sim.set_camera_state(json.loads(camera_path.read_text()))
        start_ee = sim.get_ee_position(q_start, body_name="bracelet_link")
        goal_ee  = sim.get_ee_position(q_goal,  body_name="bracelet_link")
        sim.add_sphere_marker(start_ee, rgba=(0.0, 1.0, 0.0, 0.8))
        sim.add_sphere_marker(goal_ee,  rgba=(1.0, 0.0, 0.0, 0.8))
        sim.sync_viewer()
        input("Press Enter to play...")

    log = run_playback(sim, traj, render=args.render, recorder=recorder)

    if args.render or recorder:
        settle_steps = int(2.0 / traj.dt)
        for i in range(settle_steps):
            sim.step_pos(q_goal)
            if args.render and i % max(1, round(0.01 / traj.dt)) == 0:
                sim.sync_viewer()
            if recorder and i % recorder.every == 0:
                recorder.capture(sim.get_camera_state())

    if args.render:
        print("Trajectory complete. Close the viewer window to exit.")
        while sim.is_viewer_open():
            sim.sync_viewer()
            time.sleep(0.05)
        camera_path.write_text(json.dumps(sim.get_camera_state() or {}, indent=2))
        print(f"Camera state saved to {camera_path}")

    sim.close()

    q_err = np.abs(log["q_actual"] - log["q_cmd"])
    print(f"  Mean position error: {q_err.mean():.4f} rad")
    print(f"  Max  position error: {q_err.max():.4f} rad")

    if recorder and args.record:
        recorder.save(args.record)

    if args.log:
        np.save(args.log, log)
        print(f"Log saved to {args.log}")


if __name__ == "__main__":
    main()

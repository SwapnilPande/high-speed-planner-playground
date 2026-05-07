"""Plan a jerk-limited time-optimal trajectory via Ruckig and play it back."""
import argparse
import json
import pathlib
import time
import numpy as np


def main():
    parser = argparse.ArgumentParser(
        description="Ruckig trajectory demo on the Kinova Gen3"
    )
    parser.add_argument("--model",   default="models/kinova_gen3/scene.xml")
    parser.add_argument("--render",  action="store_true")
    parser.add_argument("--record",  default=None, metavar="FILE")
    parser.add_argument("--camera",  default="camera.json")
    parser.add_argument("--log",     default=None)
    args = parser.parse_args()

    from kinodynamic_planner.planning.ruckig_planner import RuckigPlanner
    from kinodynamic_planner.planning.min_jerk import MinJerkPlanner
    from kinodynamic_planner.planning.base import JointConstraints
    from kinodynamic_planner.sim.simulator import Simulator
    from kinodynamic_planner.sim.recorder import VideoRecorder
    from kinodynamic_planner.runner.playback import run_playback

    q_start = np.zeros(7)
    q_goal  = np.array([0.0, -0.8, 0.0, -1.5, 0.0, 1.2, 0.0])
    constraints = JointConstraints.kinova_gen3()

    mj_traj = MinJerkPlanner().plan(q_start, q_goal, constraints)
    rk_traj = RuckigPlanner().plan(q_start, q_goal, constraints)

    speedup = mj_traj.duration / rk_traj.duration
    print(f"Ruckig plan:")
    print(f"  Min-jerk duration : {mj_traj.duration:.3f} s")
    print(f"  Ruckig duration   : {rk_traj.duration:.3f} s  ({speedup:.2f}x speedup)")
    print(f"  Steps             : {len(rk_traj.t)}")

    peak_qd_j  = int(np.abs(rk_traj.qd).max(axis=0).argmax())
    peak_qdd_j = int(np.abs(rk_traj.qdd).max(axis=0).argmax())
    print(f"  Peak |qd|  : {np.abs(rk_traj.qd).max():.4f} rad/s  "
          f"(joint {peak_qd_j} limit {constraints.v_max[peak_qd_j]:.2f})")
    print(f"  Peak |qdd| : {np.abs(rk_traj.qdd).max():.4f} rad/s²  "
          f"(joint {peak_qdd_j} limit {constraints.a_max[peak_qdd_j]:.2f})")

    sim = Simulator(model_path=args.model, control_hz=1.0 / rk_traj.dt)
    camera_path = pathlib.Path(args.camera)

    recorder = None
    if args.record:
        recorder = VideoRecorder(sim, fps=30)
        recorder.set_dt(rk_traj.dt)

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

    log = run_playback(sim, rk_traj, render=args.render, recorder=recorder)

    if args.render or recorder:
        settle_steps = int(2.0 / rk_traj.dt)
        for i in range(settle_steps):
            sim.step_pos(q_goal)
            if args.render and i % max(1, round(0.01 / rk_traj.dt)) == 0:
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

"""Plan a time-optimal trajectory via TOPP-RA (dynamics-aware) and compare to Ruckig."""
import argparse
import json
import pathlib
import time
import numpy as np


def main():
    parser = argparse.ArgumentParser(
        description="TOPP-RA trajectory demo on the Kinova Gen3"
    )
    parser.add_argument("--model",   default="models/kinova_gen3/scene.xml")
    parser.add_argument("--plan-model", default="models/kinova_gen3/gen3_plan.xml")
    parser.add_argument("--render",  action="store_true")
    parser.add_argument("--record",  default=None, metavar="FILE")
    parser.add_argument("--camera",  default="camera.json")
    parser.add_argument("--log",     default=None)
    args = parser.parse_args()

    from kinodynamic_planner.planning.toppra_planner import TOPPRAPlanner
    from kinodynamic_planner.planning.ruckig_planner import RuckigPlanner
    from kinodynamic_planner.planning.min_jerk import MinJerkPlanner
    from kinodynamic_planner.planning.base import JointConstraints
    from kinodynamic_planner.sim.simulator import Simulator
    from kinodynamic_planner.sim.recorder import VideoRecorder
    from kinodynamic_planner.runner.playback import run_playback

    q_start = np.zeros(7)
    q_goal  = np.array([0.0, 0.8, 0.0, 1.5, 0.0, -1.2, 0.0])
    constraints = JointConstraints.kinova_gen3()

    mj_traj = MinJerkPlanner().plan(q_start, q_goal, constraints)
    rk_traj = RuckigPlanner().plan(q_start, q_goal, constraints)
    tp_traj = TOPPRAPlanner(args.plan_model).plan(q_start, q_goal, constraints)

    print("Trajectory comparison:")
    print(f"  Min-jerk  : {mj_traj.duration:.3f} s  (baseline)")
    print(f"  Ruckig    : {rk_traj.duration:.3f} s  ({mj_traj.duration/rk_traj.duration:.2f}x speedup over min-jerk)")
    print(f"  TOPP-RA   : {tp_traj.duration:.3f} s  ({mj_traj.duration/tp_traj.duration:.2f}x speedup over min-jerk, "
          f"{rk_traj.duration/tp_traj.duration:.2f}x speedup over Ruckig)")

    peak_qd  = np.abs(tp_traj.qd).max(axis=0)
    peak_qdd = np.abs(tp_traj.qdd).max(axis=0)
    print(f"\n  TOPP-RA peak |qd|  per joint: {np.round(peak_qd, 3)}  (limits: {constraints.v_max})")
    print(f"  TOPP-RA peak |qdd| per joint: {np.round(peak_qdd, 1)}")

    sim = Simulator(model_path=args.model, control_hz=1.0 / tp_traj.dt)
    camera_path = pathlib.Path(args.camera)

    recorder = None
    if args.record:
        recorder = VideoRecorder(sim, fps=30)
        recorder.set_dt(tp_traj.dt)

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

    log = run_playback(sim, tp_traj, render=args.render, recorder=recorder)

    if args.render or recorder:
        settle_steps = int(2.0 / tp_traj.dt)
        for i in range(settle_steps):
            sim.step_pos(q_goal)
            if args.render and i % max(1, round(0.01 / tp_traj.dt)) == 0:
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
    print(f"\n  Mean position error: {q_err.mean():.4f} rad")
    print(f"  Max  position error: {q_err.max():.4f} rad")

    if recorder and args.record:
        recorder.save(args.record)
    if args.log:
        np.save(args.log, log)
        print(f"Log saved to {args.log}")


if __name__ == "__main__":
    main()

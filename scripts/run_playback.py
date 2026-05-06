"""Play back a joint trajectory on the Kinova Gen3 MuJoCo simulation."""
import argparse
import numpy as np


def main():
    parser = argparse.ArgumentParser(description="Play back a trajectory in MuJoCo")
    parser.add_argument("--traj", required=True, help="Path to trajectory file (.npy or .csv)")
    parser.add_argument("--model", default="models/kinova_gen3/scene.xml", help="Path to MJCF model")
    parser.add_argument("--render", action="store_true", help="Launch MuJoCo viewer")
    parser.add_argument("--hz", type=float, default=None, help="Control frequency override (Hz)")
    parser.add_argument("--log", default=None, help="Save log to .npy file")
    args = parser.parse_args()

    from kinodynamic_planner.sim.simulator import Simulator
    from kinodynamic_planner.runner.playback import load_trajectory, run_playback

    traj = load_trajectory(args.traj)
    control_hz = args.hz if args.hz is not None else 1.0 / traj.dt

    print(f"Trajectory: {len(traj.t)} steps, duration={traj.duration:.2f}s, hz={control_hz:.1f}")
    print(f"Model: {args.model}")

    sim = Simulator(model_path=args.model, control_hz=control_hz)
    log = run_playback(sim, traj, render=args.render)
    sim.close()

    q_err = np.abs(log["q_actual"] - log["q_cmd"])
    print(f"Mean joint position error: {q_err.mean():.4f} rad")
    print(f"Max  joint position error: {q_err.max():.4f} rad")

    if args.log:
        np.save(args.log, log)
        print(f"Log saved to {args.log}")


if __name__ == "__main__":
    main()

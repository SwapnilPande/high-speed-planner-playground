"""Generate test trajectories and save to disk."""
import numpy as np
import argparse


def wave(t: np.ndarray) -> np.ndarray:
    """
    Wave: each joint has a different frequency and phase, creating a
    ripple that propagates up the kinematic chain. Produces complex,
    visually interesting full-arm motion.
    """
    # Frequencies (Hz) and phase offsets (rad) per joint
    freqs  = np.array([0.20, 0.35, 0.55, 0.35, 0.70, 0.25, 0.50])
    phases = np.linspace(0, np.pi, 7)           # π/6 offset between consecutive joints
    amps   = np.array([0.60, 0.50, 0.60, 0.50, 0.60, 0.40, 0.40])  # rad/s
    return amps[None, :] * np.sin(2 * np.pi * freqs[None, :] * t[:, None] + phases[None, :])


def sweep(t: np.ndarray) -> np.ndarray:
    """
    Slow base rotation with faster distal joints — arm sweeps around
    while the wrist flicks.
    """
    qd = np.zeros((len(t), 7))
    qd[:, 0] = 0.40 * np.sin(2 * np.pi * 0.15 * t)          # base yaw, slow
    qd[:, 1] = 0.50 * np.sin(2 * np.pi * 0.30 * t)          # shoulder pitch
    qd[:, 2] = 0.50 * np.sin(2 * np.pi * 0.30 * t + np.pi)  # elbow yaw, antiphase
    qd[:, 3] = 0.40 * np.sin(2 * np.pi * 0.60 * t)          # elbow pitch
    qd[:, 4] = 0.40 * np.sin(2 * np.pi * 0.80 * t + np.pi / 3)
    qd[:, 5] = 0.35 * np.sin(2 * np.pi * 0.50 * t + np.pi / 2)
    qd[:, 6] = 0.35 * np.sin(2 * np.pi * 1.00 * t)          # wrist, fast
    return qd


TRAJECTORIES = {"wave": wave, "sweep": sweep}


def main():
    parser = argparse.ArgumentParser(description="Generate a test joint-space trajectory")
    parser.add_argument("--out", default="test_traj.npy")
    parser.add_argument("--duration", type=float, default=8.0)
    parser.add_argument("--hz", type=float, default=100.0)
    parser.add_argument("--type", choices=list(TRAJECTORIES), default="wave",
                        help="Trajectory shape (default: wave)")
    args = parser.parse_args()

    N = int(args.duration * args.hz)
    t = np.linspace(0, args.duration, N)

    qd = TRAJECTORIES[args.type](t)
    q = np.cumsum(qd, axis=0) / args.hz  # Euler integration for position reference

    np.save(args.out, {"t": t, "q": q, "qd": qd})
    print(f"Saved '{args.type}' trajectory: {N} steps @ {args.hz}Hz, {args.duration}s -> {args.out}")


if __name__ == "__main__":
    main()

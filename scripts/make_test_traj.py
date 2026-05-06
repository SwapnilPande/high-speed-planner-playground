"""Generate a sinusoidal test trajectory and save to disk."""
import numpy as np
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="test_traj.npy")
    parser.add_argument("--duration", type=float, default=3.0)
    parser.add_argument("--hz", type=float, default=100.0)
    args = parser.parse_args()

    N = int(args.duration * args.hz)
    t = np.linspace(0, args.duration, N)

    # Gentle sinusoidal velocities within safe limits (rad/s)
    freq = 0.5  # Hz
    amp = np.array([0.3, 0.2, 0.3, 0.2, 0.3, 0.2, 0.3])
    qd = amp[None, :] * np.sin(2 * np.pi * freq * t[:, None])
    q = np.cumsum(qd, axis=0) / args.hz  # integrate for position reference

    np.save(args.out, {"t": t, "q": q, "qd": qd})
    print(f"Saved trajectory: {N} steps @ {args.hz}Hz -> {args.out}")

if __name__ == "__main__":
    main()

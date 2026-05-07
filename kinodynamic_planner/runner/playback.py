from __future__ import annotations
import pathlib
import time
import numpy as np
from kinodynamic_planner.types import Trajectory


def run_playback(
    sim,
    traj: Trajectory,
    render: bool = False,
) -> dict:
    sim.reset()
    if render:
        sim.render()

    N = len(traj.t)
    n_joints = traj.q.shape[1]
    log = {
        "t": np.empty(N),
        "q_cmd": np.empty((N, n_joints)),
        "q_actual": np.empty((N, n_joints)),
        "qd_cmd": np.empty((N, n_joints)),
        "qd_actual": np.empty((N, n_joints)),
    }

    t_wall_start = time.monotonic()
    for i in range(N):
        sim.step_pos(traj.q[i])
        state = sim.get_state()

        log["t"][i] = state.t
        log["q_cmd"][i] = traj.q[i]
        log["q_actual"][i] = state.q
        log["qd_cmd"][i] = traj.qd[i]
        log["qd_actual"][i] = state.qd

        if render:
            sim.sync_viewer()
            # Pace to real time: sleep until the next step's wall-clock deadline
            t_deadline = t_wall_start + traj.t[i]
            sleep_s = t_deadline - time.monotonic()
            if sleep_s > 0:
                time.sleep(sleep_s)

    return log


def load_trajectory(path: str) -> Trajectory:
    p = pathlib.Path(path)
    if p.suffix == ".npy":
        data = np.load(path, allow_pickle=True).item()
        return Trajectory(t=data["t"], q=data["q"], qd=data["qd"])
    elif p.suffix == ".csv":
        arr = np.loadtxt(path, delimiter=",", skiprows=1)
        t = arr[:, 0]
        q = arr[:, 1:8]
        qd = arr[:, 8:15]
        return Trajectory(t=t, q=q, qd=qd)
    else:
        raise ValueError(f"Unsupported trajectory file extension: {p.suffix!r}. Use .npy or .csv")

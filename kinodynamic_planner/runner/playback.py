from __future__ import annotations
import pathlib
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
    log = {
        "t": np.empty(N),
        "q_cmd": np.empty((N, 7)),
        "q_actual": np.empty((N, 7)),
        "qd_cmd": np.empty((N, 7)),
        "qd_actual": np.empty((N, 7)),
    }

    for i in range(N):
        qd_cmd = traj.qd[i]
        sim.step(qd_cmd)
        state = sim.get_state()

        log["t"][i] = state.t
        log["q_cmd"][i] = traj.q[i]
        log["q_actual"][i] = state.q
        log["qd_cmd"][i] = qd_cmd
        log["qd_actual"][i] = state.qd

        if render:
            sim.sync_viewer()

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

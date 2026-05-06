from __future__ import annotations
import pathlib
import numpy as np
from kinodynamic_planner.types import Trajectory, RobotState


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

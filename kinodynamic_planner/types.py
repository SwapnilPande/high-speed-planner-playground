from __future__ import annotations
from dataclasses import dataclass
import numpy as np


@dataclass
class RobotState:
    t: float
    q: np.ndarray    # joint positions (7,)
    qd: np.ndarray   # joint velocities (7,)
    tau: np.ndarray  # joint torques (7,)


@dataclass
class Trajectory:
    t: np.ndarray           # time stamps (N,)
    q: np.ndarray           # joint positions (N, 7)
    qd: np.ndarray          # joint velocities (N, 7)
    qdd: np.ndarray | None = None  # joint accelerations (N, 7), None if not computed

    @property
    def duration(self) -> float:
        return float(self.t[-1] - self.t[0])

    @property
    def dt(self) -> float:
        if len(self.t) < 2:
            raise ValueError("Trajectory must have at least 2 timesteps to compute dt")
        return float(self.t[1] - self.t[0])

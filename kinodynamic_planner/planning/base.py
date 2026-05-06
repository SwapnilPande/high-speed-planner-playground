from __future__ import annotations
from dataclasses import dataclass
from typing import Protocol
import numpy as np
from kinodynamic_planner.types import Trajectory


@dataclass
class JointConstraints:
    v_max: np.ndarray   # (7,) rad/s   — per-joint velocity limits
    a_max: np.ndarray   # (7,) rad/s²  — per-joint acceleration limits
    j_max: np.ndarray   # (7,) rad/s³  — per-joint jerk limits

    @staticmethod
    def kinova_gen3() -> "JointConstraints":
        """Conservative published limits for the Kinova Gen3 7DOF."""
        return JointConstraints(
            v_max=np.array([1.39, 1.39, 1.39, 1.39, 1.22, 1.22, 1.22]),
            a_max=np.full(7, 8.0),
            j_max=np.full(7, 50.0),
        )


class Planner(Protocol):
    def plan(
        self,
        q_start: np.ndarray,
        q_goal: np.ndarray,
        constraints: JointConstraints,
    ) -> Trajectory: ...

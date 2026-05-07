from __future__ import annotations
import numpy as np
import ruckig
from kinodynamic_planner.types import Trajectory
from kinodynamic_planner.planning.base import JointConstraints

_NJ = 7


class RuckigPlanner:
    """Time-optimal joint-space planner using Ruckig (jerk-limited OTG).

    Produces q, qd, qdd at the configured rate respecting v_max, a_max, and
    j_max from JointConstraints. Each joint is planned independently at the
    per-joint limits; Ruckig synchronises them so all joints finish together.
    """

    def __init__(self, dt: float = 1e-3) -> None:
        self._dt = dt

    def plan(
        self,
        q_start: np.ndarray,
        q_goal: np.ndarray,
        constraints: JointConstraints,
    ) -> Trajectory:
        otg  = ruckig.Ruckig(_NJ, self._dt)
        inp  = ruckig.InputParameter(_NJ)
        traj = ruckig.Trajectory(_NJ)

        inp.current_position     = q_start.tolist()
        inp.current_velocity     = [0.0] * _NJ
        inp.current_acceleration = [0.0] * _NJ

        inp.target_position     = q_goal.tolist()
        inp.target_velocity     = [0.0] * _NJ
        inp.target_acceleration = [0.0] * _NJ

        inp.max_velocity     = constraints.v_max.tolist()
        inp.max_acceleration = constraints.a_max.tolist()
        inp.max_jerk         = constraints.j_max.tolist()

        result = otg.calculate(inp, traj)
        if result == ruckig.Result.ErrorInvalidInput:
            raise ValueError("Ruckig: invalid input (check constraint values)")

        N  = max(1, round(traj.duration / self._dt))
        t  = np.linspace(0.0, traj.duration, N + 1)
        qs, qds, qdds = [], [], []
        for ti in t:
            pos, vel, acc = traj.at_time(ti)
            qs.append(pos)
            qds.append(vel)
            qdds.append(acc)

        return Trajectory(
            t   = t,
            q   = np.array(qs),
            qd  = np.array(qds),
            qdd = np.array(qdds),
        )

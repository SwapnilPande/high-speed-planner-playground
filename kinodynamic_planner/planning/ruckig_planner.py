from __future__ import annotations
import numpy as np
import ruckig
from kinodynamic_planner.types import Trajectory
from kinodynamic_planner.planning.base import JointConstraints

_NJ = 7
_DT = 1e-3   # 1 kHz output, matching the rest of the stack


class RuckigPlanner:
    """Time-optimal joint-space planner using Ruckig (jerk-limited OTG).

    Produces q, qd, qdd at 1 kHz respecting v_max, a_max, and j_max from
    JointConstraints.  Each joint is planned independently at the per-joint
    limits; Ruckig synchronises them so all joints finish simultaneously.
    """

    def plan(
        self,
        q_start: np.ndarray,
        q_goal: np.ndarray,
        constraints: JointConstraints,
    ) -> Trajectory:
        otg  = ruckig.Ruckig(_NJ, _DT)
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

        # Sample at 1 kHz
        N  = max(1, round(traj.duration / _DT))
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

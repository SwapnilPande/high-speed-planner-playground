from __future__ import annotations
import numpy as np
from kinodynamic_planner.types import Trajectory
from kinodynamic_planner.planning.base import JointConstraints

_SAMPLE_HZ = 100.0

# Peak values of the normalised 5th-order basis and its derivatives
_V_PEAK = 15.0 / 8.0                        # |s′|_max  at τ=0.5
_A_PEAK = 10.0 * np.sqrt(3.0) / 3.0         # |s″|_max  at τ=(3-√3)/6
_J_PEAK = 60.0                               # |s‴|_max  at τ=0 and τ=1


class MinJerkPlanner:
    """Plans a minimum-jerk joint trajectory between two configurations.

    Time T is chosen as the smallest value that satisfies every per-joint
    velocity, acceleration, and jerk limit simultaneously.
    """

    def plan(
        self,
        q_start: np.ndarray,
        q_goal: np.ndarray,
        constraints: JointConstraints,
    ) -> Trajectory:
        dq = q_goal - q_start
        if np.allclose(dq, 0.0):
            raise ValueError(
                "q_start and q_goal are identical; cannot plan a trajectory."
            )

        abs_dq = np.abs(dq)

        # Minimum T from each constraint class
        T_v = _V_PEAK * np.max(abs_dq / constraints.v_max)
        T_a = np.sqrt(_A_PEAK * np.max(abs_dq / constraints.a_max))
        T_j = np.cbrt(_J_PEAK * np.max(abs_dq / constraints.j_max))
        T = max(T_v, T_a, T_j)

        # Sample at fixed rate
        N = max(2, int(np.ceil(T * _SAMPLE_HZ)) + 1)
        t = np.linspace(0.0, T, N)
        tau = t / T

        # Normalised basis and derivatives
        s   = 10*tau**3 - 15*tau**4 +  6*tau**5
        sd  = 30*tau**2 - 60*tau**3 + 30*tau**4   # ds/dτ
        sdd = 60*tau    -180*tau**2 +120*tau**3    # d²s/dτ²

        q   = q_start[None, :] + dq[None, :] * s[:, None]
        qd  = dq[None, :] * sd[:, None]  / T
        qdd = dq[None, :] * sdd[:, None] / T**2

        return Trajectory(t=t, q=q, qd=qd, qdd=qdd)

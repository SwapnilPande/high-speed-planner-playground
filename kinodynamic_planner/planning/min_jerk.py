from __future__ import annotations
import numpy as np
from scipy.interpolate import make_interp_spline
from kinodynamic_planner.types import Trajectory
from kinodynamic_planner.planning.base import JointConstraints

# Peak values of the normalised 5th-order basis and its derivatives
_V_PEAK = 15.0 / 8.0                        # |s′|_max  at τ=0.5
_A_PEAK = 10.0 * np.sqrt(3.0) / 3.0         # |s″|_max  at τ=(3-√3)/6
_J_PEAK = 60.0                               # |s‴|_max  at τ=0 and τ=1


class MinJerkPlanner:
    """Plans a minimum-jerk joint trajectory between two configurations.

    Time T is chosen as the smallest value that satisfies every per-joint
    velocity, acceleration, and jerk limit simultaneously.
    """

    def __init__(self, dt: float = 1e-3) -> None:
        self._dt = dt

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

        N = max(2, int(np.ceil(T / self._dt)) + 1)
        t = np.linspace(0.0, T, N)
        tau = t / T

        # Normalised basis and derivatives
        s = 10*tau**3 - 15*tau**4 + 6*tau**5
        sd = 30*tau**2 - 60*tau**3 + 30*tau**4    # ds/dτ
        sdd = 60*tau - 180*tau**2 + 120*tau**3    # d²s/dτ²

        q = q_start[None, :] + dq[None, :] * s[:, None]
        qd = dq[None, :] * sd[:, None] / T
        qdd = dq[None, :] * sdd[:, None] / T**2

        return Trajectory(t=t, q=q, qd=qd, qdd=qdd)

    def plan_waypoints(
        self,
        waypoints: np.ndarray,
        constraints: JointConstraints,
    ) -> Trajectory:
        """Plan a minimum-jerk trajectory passing through every waypoint.

        Chaining single-segment `plan()` calls forces a full stop at each
        intermediate waypoint. Instead this fits one quintic spline through
        all waypoints, so the arm flows through interior waypoints without
        stopping; both endpoints are still brought to rest.

        Total duration is scaled so the trajectory just meets the binding
        velocity, acceleration, or jerk limit. `plan()` gets this from a
        closed form, but that form is specific to the rest-to-rest basis,
        so here the peaks are measured off the spline instead.
        """
        W = np.asarray(waypoints, dtype=float)

        # Knot times: space each segment by its largest per-joint move, so
        # longer legs get proportionally more time. Absolute scale is set
        # below by the limit rescale.
        seg = np.max(np.abs(np.diff(W, axis=0)), axis=1)
        knots = np.concatenate([[0.0], np.cumsum(seg)])
        spline = self._fit_spline(knots, W)

        # Measure peak |qd|, |qdd|, |qddd| on a grid dense enough to bound
        # the sampled-trajectory peaks, then scale time so the binding limit
        # is hit exactly: scaling time by k divides qd/qdd/qddd by k/k²/k³,
        # so a single scale lands every limit in range.
        dense = np.linspace(0.0, knots[-1], 20001)
        peak_v = np.abs(spline(dense, nu=1)).max(axis=0)
        peak_a = np.abs(spline(dense, nu=2)).max(axis=0)
        peak_j = np.abs(spline(dense, nu=3)).max(axis=0)
        k = max(
            np.max(peak_v / constraints.v_max),
            np.sqrt(np.max(peak_a / constraints.a_max)),
            np.cbrt(np.max(peak_j / constraints.j_max)),
        )
        knots = knots * k
        spline = self._fit_spline(knots, W)

        T = knots[-1]
        N = max(2, int(np.ceil(T / self._dt)) + 1)
        t = np.linspace(0.0, T, N)
        return Trajectory(
            t=t, q=spline(t),
            qd=spline(t, nu=1), qdd=spline(t, nu=2), qddd=spline(t, nu=3),
        )

    @staticmethod
    def _fit_spline(knots: np.ndarray, waypoints: np.ndarray):
        """Quintic interpolating spline through `waypoints`, clamped to rest
        (zero velocity and acceleration) at both endpoints."""
        zero = np.zeros(waypoints.shape[1])
        return make_interp_spline(
            knots, waypoints, k=5,
            bc_type=([(1, zero), (2, zero)], [(1, zero), (2, zero)]),
        )

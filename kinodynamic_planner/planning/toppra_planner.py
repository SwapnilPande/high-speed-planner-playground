from __future__ import annotations
import numpy as np
import mujoco
import toppra
import toppra.algorithm as algo
import toppra.constraint as constraint
from kinodynamic_planner.types import Trajectory
from kinodynamic_planner.planning.base import JointConstraints

_NJ = 7
_N_WAYPOINTS = 50   # geometric path resolution
_N_GRIDPTS   = 200  # TOPP-RA constraint gridpoints (denser → less interpolation overshoot)
_V_MARGIN    = 0.95 # velocity safety factor: constrain to 95% of limit


class TOPPRAPlanner:
    """Time-optimal joint-space planner using TOPP-RA (dynamics-aware).

    Uses MuJoCo inverse dynamics to account for gravity and inertia at
    each configuration, so torque limits — not conservative kinematic
    a_max — determine the speed profile.
    """

    def __init__(self, model_path: str, dt: float = 1e-3) -> None:
        self._model = mujoco.MjModel.from_xml_path(model_path)
        self._data  = mujoco.MjData(self._model)
        self._dt    = dt

    def _inv_dyn(self, q: np.ndarray, qd: np.ndarray, qdd: np.ndarray) -> np.ndarray:
        self._data.qpos[:_NJ] = q
        self._data.qvel[:_NJ] = qd
        self._data.qacc[:_NJ] = qdd
        mujoco.mj_inverse(self._model, self._data)
        return self._data.qfrc_inverse[:_NJ].copy()

    def plan(
        self,
        q_start: np.ndarray,
        q_goal: np.ndarray,
        constraints: JointConstraints,
        tau_max: np.ndarray | None = None,
    ) -> Trajectory:
        if tau_max is None:
            # Kinova Gen3 forcerange from gen3_plan.xml
            tau_max = np.array([105., 105., 105., 105., 52., 52., 52.])

        # Geometric path: straight line in joint space
        ss = np.linspace(0.0, 1.0, _N_WAYPOINTS)
        qs = q_start[None, :] + (q_goal - q_start)[None, :] * ss[:, None]
        path = toppra.SplineInterpolator(ss, qs)

        tau_lim = np.column_stack([-tau_max, tau_max])   # (7, 2)
        fs_coef = np.zeros(_NJ)

        vlim = np.column_stack([-_V_MARGIN * constraints.v_max,
                                  _V_MARGIN * constraints.v_max])  # (7, 2)

        pc_torque = constraint.JointTorqueConstraint(self._inv_dyn, tau_lim, fs_coef)
        pc_vel    = constraint.JointVelocityConstraint(vlim)

        gridpoints = np.linspace(0.0, 1.0, _N_GRIDPTS)
        instance = algo.TOPPRA([pc_torque, pc_vel], path, gridpoints=gridpoints)
        traj_param = instance.compute_trajectory(sd_start=0, sd_end=0)

        if traj_param is None:
            raise RuntimeError("TOPP-RA failed to find a feasible trajectory")

        T = traj_param.duration
        N = max(1, round(T / self._dt))
        t = np.linspace(0.0, T, N + 1)

        qs_out  = traj_param(t)
        qds_out = traj_param(t, 1)
        qdds_out = traj_param(t, 2)

        return Trajectory(
            t   = t,
            q   = np.array(qs_out),
            qd  = np.array(qds_out),
            qdd = np.array(qdds_out),
        )

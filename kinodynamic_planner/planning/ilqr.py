from __future__ import annotations
import numpy as np
import mujoco
from kinodynamic_planner.types import Trajectory
from kinodynamic_planner.planning.base import JointConstraints
from kinodynamic_planner.planning.min_jerk import MinJerkPlanner

_NJ = 7    # joints
_NX = 14   # state dim: q + qd
_NU = 7    # control dim: tau

# Cost weights
_Q_q   = 1e6    # terminal position weight (per joint)
_Q_qd  = 1e4    # terminal velocity weight (per joint)
_R     = 1e-4   # running torque weight (per joint)
_W_V   = 1e3    # soft velocity limit weight (per joint per step)
_W_A   = 1e2    # soft acceleration limit weight (per joint per step)

_MU0       = 1e-4   # initial Quu regularisation
_ALPHA     = [1.0, 0.5, 0.25, 0.1, 0.05]
_CONV_TOL  = 1e-3   # feedforward norm convergence threshold
_TERM_TOL  = 0.05   # max |q_final_j - q_goal_j| (rad) for bisection feasibility
_V_TOL     = 0.01   # allowable velocity overshoot past v_max (rad/s)
_A_TOL     = 0.1    # allowable acceleration overshoot past a_max (rad/s²)
_N_BISECT  = 6      # bisection iterations
_MAX_ITER  = 50     # max iLQR iterations per solve


class ILQRPlanner:
    """Minimum-time kinodynamic planner using iLQR with MuJoCo torque dynamics.

    Plans u*(t) torques in simulation respecting v_max and a_max from
    JointConstraints, then returns q*(t) for position control execution.
    """

    def __init__(self, model_path: str = "models/kinova_gen3/gen3_plan.xml"):
        self._model   = mujoco.MjModel.from_xml_path(model_path)
        self._data    = mujoco.MjData(self._model)
        self._tau_min = self._model.actuator_forcerange[:_NU, 0].copy()
        self._tau_max = self._model.actuator_forcerange[:_NU, 1].copy()
        self._dt      = float(self._model.opt.timestep)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def plan(
        self,
        q_start: np.ndarray,
        q_goal: np.ndarray,
        constraints: JointConstraints,
        n_bisect: int = _N_BISECT,
        max_iter: int = _MAX_ITER,
    ) -> Trajectory:
        """Plan a minimum-time trajectory respecting v_max and a_max from constraints.

        Returns q*(t), qd*(t) at 1 kHz for position control execution.
        """
        v_max   = constraints.v_max
        a_max   = constraints.a_max
        mj_traj = MinJerkPlanner().plan(q_start, q_goal, constraints)
        T_hi    = mj_traj.duration
        T_lo    = T_hi * 0.3

        best_xs: list[np.ndarray] | None = None
        best_us: list[np.ndarray] | None = None
        T_best  = T_hi

        for _ in range(n_bisect):
            T_try = (T_lo + T_hi) / 2
            xs, us, feasible = self._solve_ilqr(
                q_start, q_goal, T_try, mj_traj, v_max, a_max, max_iter
            )
            if feasible:
                T_hi, T_best, best_xs, best_us = T_try, T_try, xs, us
            else:
                T_lo = T_try

        if best_xs is None:
            xs, us, _ = self._solve_ilqr(
                q_start, q_goal, T_hi, mj_traj, v_max, a_max, max_iter
            )
            best_xs, best_us, T_best = xs, us, T_hi

        N   = len(best_xs) - 1
        t   = np.linspace(0.0, T_best, N + 1)
        q   = np.array([x[:_NJ]  for x in best_xs])
        qd  = np.array([x[_NJ:]  for x in best_xs])
        qdd = np.gradient(qd, t, axis=0)
        return Trajectory(t=t, q=q, qd=qd, qdd=qdd)

    def _solve_ilqr(
        self,
        q_start: np.ndarray,
        q_goal: np.ndarray,
        T: float,
        mj_traj: Trajectory,
        v_max: np.ndarray,
        a_max: np.ndarray,
        max_iter: int,
    ) -> tuple[list[np.ndarray], list[np.ndarray], bool]:
        N  = max(2, round(T * 1000))
        x0 = np.concatenate([q_start, np.zeros(_NJ)])

        us = self._warm_start(mj_traj, T, N)
        xs = self._rollout(x0, us)
        mu = _MU0

        for _ in range(max_iter):
            As, Bs = self._compute_jacobians(xs, us)
            ks, Ks = self._backward_pass(xs, us, As, Bs, q_goal, v_max, a_max, mu)

            xs_new, us_new, _ = self._forward_pass(
                x0, xs, us, ks, Ks, q_goal, v_max, a_max
            )
            if xs_new is None:
                mu = min(mu * 10, 1e6)
                continue

            xs, us = xs_new, us_new
            mu = max(_MU0, mu / 10)

            k_norm = float(np.sqrt(sum(np.dot(k, k) for k in ks)))
            if k_norm < _CONV_TOL:
                break

        q_err_max = float(np.max(np.abs(xs[-1][:_NJ] - q_goal)))
        qd_arr    = np.array([x[_NJ:] for x in xs])
        v_ok      = bool(np.all(np.abs(qd_arr) <= v_max[None, :] + _V_TOL))
        qdd_arr   = np.diff(qd_arr, axis=0) / self._dt
        a_ok      = bool(np.all(np.abs(qdd_arr) <= a_max[None, :] + _A_TOL))
        feasible  = (q_err_max < _TERM_TOL) and v_ok and a_ok

        return xs, us, feasible

    # ------------------------------------------------------------------
    # Dynamics
    # ------------------------------------------------------------------

    def _dynamics(self, x: np.ndarray, u: np.ndarray) -> np.ndarray:
        """Single MuJoCo step with motor torque control."""
        mujoco.mj_resetData(self._model, self._data)
        self._data.qpos[:_NJ] = x[:_NJ]
        self._data.qvel[:_NJ] = x[_NJ:]
        self._data.ctrl[:_NU] = np.clip(u, self._tau_min, self._tau_max)
        mujoco.mj_step(self._model, self._data)
        return np.concatenate([
            self._data.qpos[:_NJ].copy(),
            self._data.qvel[:_NJ].copy(),
        ])

    # ------------------------------------------------------------------
    # Jacobians
    # ------------------------------------------------------------------

    def _compute_jacobians(
        self, xs: list[np.ndarray], us: list[np.ndarray]
    ) -> tuple[list[np.ndarray], list[np.ndarray]]:
        """Finite-difference Jacobians A_k, B_k at each step via mjd_transitionFD."""
        N  = len(us)
        A  = np.zeros((_NX, _NX))
        B  = np.zeros((_NX, _NU))
        As: list[np.ndarray] = []
        Bs: list[np.ndarray] = []
        for k in range(N):
            mujoco.mj_resetData(self._model, self._data)
            self._data.qpos[:_NJ] = xs[k][:_NJ]
            self._data.qvel[:_NJ] = xs[k][_NJ:]
            self._data.ctrl[:_NU] = np.clip(us[k], self._tau_min, self._tau_max)
            mujoco.mjd_transitionFD(
                self._model, self._data, 1e-6, True, A, B, None, None
            )
            As.append(A.copy())
            Bs.append(B.copy())
        return As, Bs

    # ------------------------------------------------------------------
    # Rollout and warm start
    # ------------------------------------------------------------------

    def _rollout(
        self, x0: np.ndarray, us: list[np.ndarray]
    ) -> list[np.ndarray]:
        """Forward-simulate from x0 under control sequence us."""
        xs = [x0.copy()]
        for u in us:
            xs.append(self._dynamics(xs[-1], u))
        return xs

    def _warm_start(
        self, mj_traj: Trajectory, T: float, N: int
    ) -> list[np.ndarray]:
        """Resample min-jerk traj to N steps; compute torques via mj_inverse."""
        t_new  = np.linspace(0.0, T, N)
        t_mj   = mj_traj.t
        q_rs   = np.stack([np.interp(t_new, t_mj, mj_traj.q[:, j])   for j in range(_NJ)], axis=1)
        qd_rs  = np.stack([np.interp(t_new, t_mj, mj_traj.qd[:, j])  for j in range(_NJ)], axis=1)
        qdd_rs = np.stack([np.interp(t_new, t_mj, mj_traj.qdd[:, j]) for j in range(_NJ)], axis=1)

        us: list[np.ndarray] = []
        for k in range(N):
            self._data.qpos[:_NJ] = q_rs[k]
            self._data.qvel[:_NJ] = qd_rs[k]
            self._data.qacc[:_NJ] = qdd_rs[k]
            mujoco.mj_inverse(self._model, self._data)
            tau = np.clip(self._data.qfrc_inverse[:_NJ], self._tau_min, self._tau_max)
            us.append(tau.copy())
        return us

    # ------------------------------------------------------------------
    # iLQR passes
    # ------------------------------------------------------------------

    def _cost(
        self,
        xs: list[np.ndarray],
        us: list[np.ndarray],
        q_goal: np.ndarray,
        v_max: np.ndarray,
        a_max: np.ndarray,
    ) -> float:
        total = 0.0
        N = len(us)
        for k in range(N):
            total += 0.5 * _R * float(np.dot(us[k], us[k]))
            excess_v = np.maximum(0.0, np.abs(xs[k][_NJ:]) - v_max)
            total += 0.5 * _W_V * float(np.dot(excess_v, excess_v))
            qdd = (xs[k + 1][_NJ:] - xs[k][_NJ:]) / self._dt
            excess_a = np.maximum(0.0, np.abs(qdd) - a_max)
            total += 0.5 * _W_A * float(np.dot(excess_a, excess_a))
        dq = xs[-1][:_NJ] - q_goal
        total += 0.5 * _Q_q  * float(np.dot(dq, dq))
        total += 0.5 * _Q_qd * float(np.dot(xs[-1][_NJ:], xs[-1][_NJ:]))
        excess_v_N = np.maximum(0.0, np.abs(xs[-1][_NJ:]) - v_max)
        total += 0.5 * _W_V * float(np.dot(excess_v_N, excess_v_N))
        return total

    def _backward_pass(
        self,
        xs: list[np.ndarray],
        us: list[np.ndarray],
        As: list[np.ndarray],
        Bs: list[np.ndarray],
        q_goal: np.ndarray,
        v_max: np.ndarray,
        a_max: np.ndarray,
        mu: float,
    ) -> tuple[list[np.ndarray], list[np.ndarray]]:
        N = len(us)

        # Terminal value function
        Vx = np.zeros(_NX)
        Vx[:_NJ] = _Q_q  * (xs[-1][:_NJ] - q_goal)
        Vx[_NJ:] = _Q_qd * xs[-1][_NJ:]
        excess_v_N = np.maximum(0.0, np.abs(xs[-1][_NJ:]) - v_max)
        Vx[_NJ:] += _W_V * np.sign(xs[-1][_NJ:]) * excess_v_N

        Vxx = np.zeros((_NX, _NX))
        Vxx[:_NJ, :_NJ] = np.eye(_NJ) * _Q_q
        Vxx[_NJ:, _NJ:] = np.eye(_NJ) * (_Q_qd + _W_V * (excess_v_N > 0))

        ks: list[np.ndarray] = [np.empty(0)] * N
        Ks: list[np.ndarray] = [np.empty(0)] * N

        for k in range(N - 1, -1, -1):
            Ak, Bk = As[k], Bs[k]

            # Acceleration penalty: spans x_k and x_{k+1}
            qdd_k    = (xs[k + 1][_NJ:] - xs[k][_NJ:]) / self._dt
            sign_a   = np.sign(qdd_k)
            excess_a = np.maximum(0.0, np.abs(qdd_k) - a_max)

            # ∂p_k/∂x_{k+1}[7:] folded into Vx before computing Q
            Vx_adj = Vx.copy()
            Vx_adj[_NJ:] += _W_A * sign_a * excess_a / self._dt

            # Velocity penalty and acceleration backward term at x_k
            qd_k     = xs[k][_NJ:]
            sign_v   = np.sign(qd_k)
            excess_v = np.maximum(0.0, np.abs(qd_k) - v_max)

            ell_x = np.zeros(_NX)
            ell_x[_NJ:] += _W_V * sign_v * excess_v
            ell_x[_NJ:] -= _W_A * sign_a * excess_a / self._dt

            ell_xx_diag = np.zeros(_NX)
            ell_xx_diag[_NJ:] += _W_V * (excess_v > 0).astype(float)
            ell_xx_diag[_NJ:] += _W_A / self._dt**2 * (excess_a > 0).astype(float)

            Qu  = Bk.T @ Vx_adj + _R * us[k]
            Quu = Bk.T @ Vxx @ Bk + _R * np.eye(_NU)
            Qux = Bk.T @ Vxx @ Ak
            Qx  = Ak.T @ Vx_adj + ell_x
            Qxx = Ak.T @ Vxx @ Ak + np.diag(ell_xx_diag)

            Quu_reg = Quu + mu * np.eye(_NU)
            Kk = -np.linalg.solve(Quu_reg, Qux)
            kk = -np.linalg.solve(Quu_reg, Qu)

            Vx  = Qx  + Kk.T @ Quu @ kk
            Vxx = Qxx + Kk.T @ Quu @ Kk
            Vxx = (Vxx + Vxx.T) / 2

            ks[k] = kk
            Ks[k] = Kk

        return ks, Ks

    def _forward_pass(
        self,
        x0: np.ndarray,
        xs: list[np.ndarray],
        us: list[np.ndarray],
        ks: list[np.ndarray],
        Ks: list[np.ndarray],
        q_goal: np.ndarray,
        v_max: np.ndarray,
        a_max: np.ndarray,
    ) -> tuple[list[np.ndarray] | None, list[np.ndarray] | None, float | None]:
        cost_old = self._cost(xs, us, q_goal, v_max, a_max)
        N = len(us)

        for alpha in _ALPHA:
            xs_new: list[np.ndarray] = [x0.copy()]
            us_new: list[np.ndarray] = []
            dx = np.zeros(_NX)
            for k in range(N):
                du    = Ks[k] @ dx + alpha * ks[k]
                u_new = np.clip(us[k] + du, self._tau_min, self._tau_max)
                x_next = self._dynamics(xs_new[-1], u_new)
                dx = x_next - xs[k + 1]
                xs_new.append(x_next)
                us_new.append(u_new)
            cost_new = self._cost(xs_new, us_new, q_goal, v_max, a_max)
            if cost_new < cost_old:
                return xs_new, us_new, cost_new

        return None, None, None

import numpy as np
import pytest
from kinodynamic_planner.planning.ilqr import ILQRPlanner

_MODEL = "models/kinova_gen3/gen3_plan.xml"
_NJ = 7
_NX = 14
_NU = 7


@pytest.fixture
def planner():
    return ILQRPlanner(_MODEL)


@pytest.fixture
def x0():
    return np.concatenate([np.zeros(_NJ), np.zeros(_NJ)])


@pytest.fixture
def u_zero():
    return np.zeros(_NU)


def test_dynamics_output_shape(planner, x0, u_zero):
    x_next = planner._dynamics(x0, u_zero)
    assert x_next.shape == (_NX,)


def test_dynamics_zero_torque_gravity_drop(planner, x0, u_zero):
    x_next = planner._dynamics(x0, u_zero)
    assert not np.allclose(x_next[_NJ:], x0[_NJ:])


def test_dynamics_torque_clipped(planner, x0):
    u_huge = np.full(_NU, 1e6)
    u_max  = np.array([105.]*4 + [52.]*3)
    x_huge = planner._dynamics(x0, u_huge)
    x_max  = planner._dynamics(x0, u_max)
    np.testing.assert_allclose(x_huge, x_max, atol=1e-10)


def test_jacobian_shapes(planner, x0, u_zero):
    xs = [x0, planner._dynamics(x0, u_zero)]
    us = [u_zero]
    As, Bs = planner._compute_jacobians(xs, us)
    assert len(As) == 1
    assert As[0].shape == (_NX, _NX)
    assert Bs[0].shape == (_NX, _NU)


def test_jacobian_B_nonzero(planner, x0, u_zero):
    xs = [x0, planner._dynamics(x0, u_zero)]
    us = [u_zero]
    _, Bs = planner._compute_jacobians(xs, us)
    assert np.abs(Bs[0]).max() > 1e-8


# ---------------------------------------------------------------------------
# Task 3: rollout and warm start
# ---------------------------------------------------------------------------

from tests.planning.conftest import Q_START, Q_GOAL
from kinodynamic_planner.planning.base import JointConstraints
from kinodynamic_planner.planning.min_jerk import MinJerkPlanner


def test_rollout_length(planner, x0):
    us = [np.zeros(_NU)] * 5
    xs = planner._rollout(x0, us)
    assert len(xs) == 6


def test_rollout_first_state(planner, x0):
    us = [np.zeros(_NU)] * 5
    xs = planner._rollout(x0, us)
    np.testing.assert_array_equal(xs[0], x0)


def test_warm_start_shape(planner):
    constraints = JointConstraints.kinova_gen3()
    mj_traj = MinJerkPlanner().plan(Q_START, Q_GOAL, constraints)
    T = mj_traj.duration
    N = max(2, round(T * 1000))
    xs_ref, us = planner._warm_start(mj_traj, T, N)
    assert len(xs_ref) == N + 1
    assert xs_ref[0].shape == (_NX,)
    assert len(us) == N
    assert us[0].shape == (_NU,)


def test_warm_start_torques_within_limits(planner):
    constraints = JointConstraints.kinova_gen3()
    mj_traj = MinJerkPlanner().plan(Q_START, Q_GOAL, constraints)
    T = mj_traj.duration
    N = max(2, round(T * 1000))
    _, us = planner._warm_start(mj_traj, T, N)
    u_arr = np.array(us)
    assert np.all(u_arr >= planner._tau_min - 1e-9)
    assert np.all(u_arr <= planner._tau_max + 1e-9)


# ---------------------------------------------------------------------------
# Task 4: backward and forward passes
# ---------------------------------------------------------------------------

from kinodynamic_planner.planning.base import JointConstraints as _JC


def test_backward_pass_shapes(planner, x0):
    q_goal = np.array([0.0, -0.8, 0.0, -1.5, 0.0, 1.2, 0.0])
    v_max  = _JC.kinova_gen3().v_max
    a_max  = _JC.kinova_gen3().a_max
    N      = 3
    us     = [np.zeros(_NU)] * N
    xs     = planner._rollout(x0, us)
    As, Bs = planner._compute_jacobians(xs, us)
    ks, Ks = planner._backward_pass(xs, us, As, Bs, q_goal, v_max, a_max, mu=1e-4)
    assert len(ks) == N and len(Ks) == N
    assert ks[0].shape == (_NU,)
    assert Ks[0].shape == (_NU, _NX)


def test_forward_pass_returns_new_trajectory(planner, x0):
    # Use q_goal=zeros (hold position): gravity makes zero-torque suboptimal,
    # gravity-comp torques are within forcerange, so the line search succeeds.
    q_goal = np.zeros(_NJ)
    v_max  = _JC.kinova_gen3().v_max
    a_max  = _JC.kinova_gen3().a_max
    N      = 100
    us     = [np.zeros(_NU)] * N
    xs     = planner._rollout(x0, us)
    As, Bs = planner._compute_jacobians(xs, us)
    ks, Ks = planner._backward_pass(xs, us, As, Bs, q_goal, v_max, a_max, mu=1e-4)
    xs_new, us_new, cost_new = planner._forward_pass(x0, xs, us, ks, Ks, q_goal, v_max, a_max)
    assert xs_new is not None
    assert len(xs_new) == N + 1
    assert len(us_new) == N
    assert isinstance(cost_new, float)


def test_cost_decreases_after_one_iter(planner, x0):
    # Same hold-position scenario: gravity-comp torques clearly beat zero torques.
    q_goal   = np.zeros(_NJ)
    v_max    = _JC.kinova_gen3().v_max
    a_max    = _JC.kinova_gen3().a_max
    N        = 100
    us       = [np.zeros(_NU)] * N
    xs       = planner._rollout(x0, us)
    cost_old = planner._cost(xs, us, q_goal, v_max, a_max)
    As, Bs   = planner._compute_jacobians(xs, us)
    ks, Ks   = planner._backward_pass(xs, us, As, Bs, q_goal, v_max, a_max, mu=1e-4)
    _, _, cost_new = planner._forward_pass(x0, xs, us, ks, Ks, q_goal, v_max, a_max)
    assert cost_new < cost_old


# ---------------------------------------------------------------------------
# Task 5: plan() integration tests (slow)
# ---------------------------------------------------------------------------

import pytest
from kinodynamic_planner.planning.ilqr import _TERM_TOL, _V_TOL, _A_TOL


@pytest.mark.slow
def test_plan_returns_trajectory_type(planner):
    from kinodynamic_planner.types import Trajectory
    constraints = JointConstraints.kinova_gen3()
    traj = planner.plan(Q_START, Q_GOAL, constraints)
    assert isinstance(traj, Trajectory)


@pytest.mark.slow
def test_plan_boundary_positions(planner):
    constraints = JointConstraints.kinova_gen3()
    traj = planner.plan(Q_START, Q_GOAL, constraints)
    np.testing.assert_allclose(traj.q[0], Q_START, atol=1e-6)
    np.testing.assert_allclose(traj.q[-1], Q_GOAL, atol=_TERM_TOL)


@pytest.mark.slow
def test_plan_velocity_limits_respected(planner):
    constraints = JointConstraints.kinova_gen3()
    traj = planner.plan(Q_START, Q_GOAL, constraints)
    peak_qd = np.abs(traj.qd).max(axis=0)
    assert np.all(peak_qd <= constraints.v_max + _V_TOL)


@pytest.mark.slow
def test_plan_accel_limits_respected(planner):
    constraints = JointConstraints.kinova_gen3()
    traj = planner.plan(Q_START, Q_GOAL, constraints)
    qdd = np.diff(traj.qd, axis=0) / traj.dt
    peak_qdd = np.abs(qdd).max(axis=0)
    assert np.all(peak_qdd <= constraints.a_max + _A_TOL)


@pytest.mark.slow
def test_plan_faster_than_min_jerk(planner):
    constraints = JointConstraints.kinova_gen3()
    mj_traj   = MinJerkPlanner().plan(Q_START, Q_GOAL, constraints)
    ilqr_traj = planner.plan(Q_START, Q_GOAL, constraints)
    assert ilqr_traj.duration < mj_traj.duration


@pytest.mark.slow
def test_plan_monotone_time(planner):
    constraints = JointConstraints.kinova_gen3()
    traj = planner.plan(Q_START, Q_GOAL, constraints)
    assert np.all(np.diff(traj.t) > 0)

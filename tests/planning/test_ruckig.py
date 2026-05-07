import numpy as np
import pytest
from kinodynamic_planner.planning.ruckig_planner import RuckigPlanner
from kinodynamic_planner.planning.min_jerk import MinJerkPlanner
from kinodynamic_planner.planning.base import JointConstraints
from kinodynamic_planner.types import Trajectory
from tests.planning.conftest import Q_START, Q_GOAL

_NJ = 7


@pytest.fixture
def planner():
    return RuckigPlanner()


@pytest.fixture
def constraints():
    return JointConstraints.kinova_gen3()


@pytest.fixture
def traj(planner, constraints):
    return planner.plan(Q_START, Q_GOAL, constraints)


def test_returns_trajectory_type(traj):
    assert isinstance(traj, Trajectory)


def test_monotone_time(traj):
    assert np.all(np.diff(traj.t) > 0)


def test_boundary_positions(traj):
    np.testing.assert_allclose(traj.q[0],  Q_START, atol=1e-9)
    np.testing.assert_allclose(traj.q[-1], Q_GOAL,  atol=1e-6)


def test_boundary_velocities_zero(traj):
    np.testing.assert_allclose(traj.qd[0],  np.zeros(_NJ), atol=1e-9)
    np.testing.assert_allclose(traj.qd[-1], np.zeros(_NJ), atol=1e-6)


def test_velocity_limits_respected(traj, constraints):
    assert np.all(np.abs(traj.qd) <= constraints.v_max + 1e-6)


def test_accel_limits_respected(traj, constraints):
    assert np.all(np.abs(traj.qdd) <= constraints.a_max + 1e-6)


def test_sampled_at_1kHz(traj):
    # dt should be within 0.5 µs of 1 ms
    dt = np.diff(traj.t)
    assert np.allclose(dt, 1e-3, atol=5e-7)


def test_faster_than_min_jerk(planner, constraints):
    mj_traj = MinJerkPlanner().plan(Q_START, Q_GOAL, constraints)
    rk_traj = planner.plan(Q_START, Q_GOAL, constraints)
    assert rk_traj.duration < mj_traj.duration


def test_zero_move_is_instant(planner, constraints):
    traj = planner.plan(Q_START, Q_START, constraints)
    assert traj.duration == pytest.approx(0.0, abs=1e-9)
    np.testing.assert_allclose(traj.q[0], Q_START, atol=1e-9)

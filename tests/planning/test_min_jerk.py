import numpy as np
import pytest
from kinodynamic_planner.planning.min_jerk import MinJerkPlanner
from kinodynamic_planner.types import Trajectory
from tests.planning.conftest import Q_START, Q_GOAL


def test_plan_returns_trajectory(constraints):
    traj = MinJerkPlanner().plan(Q_START, Q_GOAL, constraints)
    assert isinstance(traj, Trajectory)


def test_boundary_positions(constraints):
    traj = MinJerkPlanner().plan(Q_START, Q_GOAL, constraints)
    np.testing.assert_allclose(traj.q[0],  Q_START, atol=1e-10)
    np.testing.assert_allclose(traj.q[-1], Q_GOAL,  atol=1e-10)


def test_boundary_velocities_zero(constraints):
    traj = MinJerkPlanner().plan(Q_START, Q_GOAL, constraints)
    np.testing.assert_allclose(traj.qd[0],  np.zeros(7), atol=1e-10)
    np.testing.assert_allclose(traj.qd[-1], np.zeros(7), atol=1e-10)


def test_boundary_accelerations_zero(constraints):
    traj = MinJerkPlanner().plan(Q_START, Q_GOAL, constraints)
    np.testing.assert_allclose(traj.qdd[0],  np.zeros(7), atol=1e-10)
    np.testing.assert_allclose(traj.qdd[-1], np.zeros(7), atol=1e-10)


def test_velocity_constraint_satisfied(constraints):
    traj = MinJerkPlanner().plan(Q_START, Q_GOAL, constraints)
    assert np.all(np.abs(traj.qd) <= constraints.v_max[None, :] + 1e-6)


def test_acceleration_constraint_satisfied(constraints):
    traj = MinJerkPlanner().plan(Q_START, Q_GOAL, constraints)
    assert np.all(np.abs(traj.qdd) <= constraints.a_max[None, :] + 1e-6)


def test_time_monotone(constraints):
    traj = MinJerkPlanner().plan(Q_START, Q_GOAL, constraints)
    assert np.all(np.diff(traj.t) > 0)


def test_qdd_shape(constraints):
    traj = MinJerkPlanner().plan(Q_START, Q_GOAL, constraints)
    assert traj.qdd is not None
    assert traj.qdd.shape == traj.q.shape


def test_equal_start_goal_raises(constraints):
    with pytest.raises(ValueError, match="identical"):
        MinJerkPlanner().plan(Q_START, Q_START, constraints)

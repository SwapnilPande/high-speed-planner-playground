import numpy as np
import pytest
from kinodynamic_planner.planning.min_jerk import MinJerkPlanner
from kinodynamic_planner.planning.base import JointConstraints
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


# ── plan_waypoints: smooth trajectory through intermediate waypoints ───────

Q_WAYPOINT = np.array([0.4, -0.3, -0.2, -0.6, 0.3, 0.5, -0.4])
WAYPOINTS = [Q_START, Q_WAYPOINT, Q_GOAL]


@pytest.fixture
def wp_traj(constraints):
    return MinJerkPlanner().plan_waypoints(WAYPOINTS, constraints)


def test_plan_waypoints_returns_trajectory(wp_traj):
    assert isinstance(wp_traj, Trajectory)


def test_plan_waypoints_reaches_every_waypoint(wp_traj):
    # the trajectory passes through every waypoint (closest sample within a
    # few mrad — interior waypoints land between samples, not exactly on one)
    for wp in WAYPOINTS:
        closest = np.linalg.norm(wp_traj.q - wp, axis=1).min()
        assert closest < 1e-2


def test_plan_waypoints_starts_and_ends_at_rest(wp_traj):
    np.testing.assert_allclose(wp_traj.qd[0],   np.zeros(7), atol=1e-9)
    np.testing.assert_allclose(wp_traj.qd[-1],  np.zeros(7), atol=1e-9)
    np.testing.assert_allclose(wp_traj.qdd[0],  np.zeros(7), atol=1e-9)
    np.testing.assert_allclose(wp_traj.qdd[-1], np.zeros(7), atol=1e-9)


def test_plan_waypoints_does_not_stop_at_interior_waypoint(wp_traj):
    # the whole point: nonzero velocity as the arm passes the interior waypoint
    i = np.linalg.norm(wp_traj.q - Q_WAYPOINT, axis=1).argmin()
    assert np.linalg.norm(wp_traj.qd[i]) > 0.1


def test_plan_waypoints_time_is_monotone(wp_traj):
    assert np.all(np.diff(wp_traj.t) > 0)


def test_plan_waypoints_respects_velocity_limits(wp_traj, constraints):
    assert np.all(np.abs(wp_traj.qd) <= constraints.v_max[None, :] + 1e-6)


def test_plan_waypoints_respects_acceleration_limits(wp_traj, constraints):
    assert np.all(np.abs(wp_traj.qdd) <= constraints.a_max[None, :] + 1e-6)


def test_plan_waypoints_respects_jerk_limits(wp_traj, constraints):
    assert wp_traj.qddd is not None
    assert np.all(np.abs(wp_traj.qddd) <= constraints.j_max[None, :] + 1e-6)


def test_plan_waypoints_respects_a_tight_acceleration_limit(constraints):
    # acceleration is slack under the Kinova's own limits; force it to bind
    # so the rescale's acceleration term is exercised.
    tight = JointConstraints(
        v_max=constraints.v_max,
        a_max=np.full(7, 0.2),
        j_max=constraints.j_max,
    )
    traj = MinJerkPlanner().plan_waypoints(WAYPOINTS, tight)
    assert np.all(np.abs(traj.qdd) <= tight.a_max[None, :] + 1e-6)

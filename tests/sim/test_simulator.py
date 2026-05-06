import numpy as np
import pytest
from kinodynamic_planner.types import RobotState


def test_simulator_loads(sim):
    assert sim is not None


def test_initial_state_shape(sim):
    state = sim.get_state()
    assert isinstance(state, RobotState)
    assert state.q.shape == (7,)
    assert state.qd.shape == (7,)
    assert state.tau.shape == (7,)


def test_initial_state_at_zero(sim):
    state = sim.get_state()
    np.testing.assert_allclose(state.q, np.zeros(7), atol=1e-6)
    np.testing.assert_allclose(state.qd, np.zeros(7), atol=1e-6)


def test_reset_restores_zero(sim):
    # Move arm, then reset
    qd = np.full(7, 0.1)
    for _ in range(50):
        sim.step(qd)
    sim.reset()
    state = sim.get_state()
    np.testing.assert_allclose(state.q, np.zeros(7), atol=1e-4)
    np.testing.assert_allclose(state.qd, np.zeros(7), atol=1e-4)


def test_reset_to_custom_q0(sim):
    q0 = np.array([0.1, 0.2, 0.0, -0.1, 0.0, 0.3, 0.0])
    sim.reset(q0=q0)
    state = sim.get_state()
    np.testing.assert_allclose(state.q, q0, atol=1e-6)


def test_step_advances_time(sim):
    state0 = sim.get_state()
    sim.step(np.zeros(7))
    state1 = sim.get_state()
    assert state1.t > state0.t


def test_velocity_command_moves_joints(sim):
    qd_cmd = np.array([0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    for _ in range(200):
        sim.step(qd_cmd)
    state = sim.get_state()
    # Joint 0 should have moved in the commanded direction
    assert state.q[0] > 0.05


def test_joint_limit_clipping(sim):
    # Command beyond joint 0 limit (+3.14)
    qd_cmd = np.array([10.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    for _ in range(1000):
        sim.step(qd_cmd)
    state = sim.get_state()
    assert state.q[0] <= 3.14 + 1e-3

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


def test_unlimited_joints_are_not_clipped(sim):
    # All joints in minimal_7dof.xml are limited, so create a sim from XML with one unlimited joint
    import mujoco
    import tempfile, pathlib
    xml = """
<mujoco model="unlimited_test">
  <option timestep="0.01"/>
  <compiler angle="radian"/>
  <worldbody>
    <body name="link0">
      <joint name="j_unlimited" type="hinge" axis="0 0 1"/>
      <geom type="box" size="0.05 0.05 0.1"/>
      <body name="link1" pos="0 0 0.2">
        <joint name="j_limited" type="hinge" axis="0 1 0" range="-1.0 1.0" limited="true"/>
        <geom type="box" size="0.05 0.05 0.1"/>
        <body name="link2" pos="0 0 0.2">
          <joint name="j2" type="hinge" axis="0 0 1" range="-1.0 1.0" limited="true"/>
          <geom type="box" size="0.05 0.05 0.1"/>
          <body name="link3" pos="0 0 0.2">
            <joint name="j3" type="hinge" axis="0 1 0" range="-1.0 1.0" limited="true"/>
            <geom type="box" size="0.05 0.05 0.1"/>
            <body name="link4" pos="0 0 0.2">
              <joint name="j4" type="hinge" axis="0 0 1" range="-1.0 1.0" limited="true"/>
              <geom type="box" size="0.05 0.05 0.1"/>
              <body name="link5" pos="0 0 0.2">
                <joint name="j5" type="hinge" axis="0 1 0" range="-1.0 1.0" limited="true"/>
                <geom type="box" size="0.05 0.05 0.1"/>
                <body name="link6" pos="0 0 0.2">
                  <joint name="j6" type="hinge" axis="0 0 1" range="-1.0 1.0" limited="true"/>
                  <geom type="box" size="0.05 0.05 0.05"/>
                </body>
              </body>
            </body>
          </body>
        </body>
      </body>
    </body>
  </worldbody>
  <actuator>
    <position name="a0" joint="j_unlimited" kp="500" kv="50"/>
    <position name="a1" joint="j_limited" kp="500" kv="50"/>
    <position name="a2" joint="j2" kp="500" kv="50"/>
    <position name="a3" joint="j3" kp="500" kv="50"/>
    <position name="a4" joint="j4" kp="500" kv="50"/>
    <position name="a5" joint="j5" kp="500" kv="50"/>
    <position name="a6" joint="j6" kp="500" kv="50"/>
  </actuator>
</mujoco>"""
    from kinodynamic_planner.sim.simulator import Simulator
    with tempfile.NamedTemporaryFile(suffix=".xml", delete=False, mode="w") as f:
        f.write(xml)
        tmp_path = f.name
    s = Simulator(model_path=tmp_path, control_hz=100.0)
    # Command large velocity on the unlimited joint — should NOT be clipped to 0
    qd_cmd = np.array([5.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    for _ in range(200):
        s.step(qd_cmd)
    state = s.get_state()
    s.close()
    # Unlimited joint should have moved significantly past 0
    assert state.q[0] > 0.1

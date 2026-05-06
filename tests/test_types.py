import numpy as np
import pytest
from kinodynamic_planner.types import RobotState, Trajectory


def test_robot_state_fields():
    state = RobotState(
        t=0.1,
        q=np.zeros(7),
        qd=np.ones(7),
        tau=np.full(7, 0.5),
    )
    assert state.t == pytest.approx(0.1)
    assert state.q.shape == (7,)
    assert state.qd.shape == (7,)
    assert state.tau.shape == (7,)


def test_trajectory_fields():
    N = 100
    traj = Trajectory(
        t=np.linspace(0, 1, N),
        q=np.zeros((N, 7)),
        qd=np.zeros((N, 7)),
    )
    assert traj.t.shape == (N,)
    assert traj.q.shape == (N, 7)
    assert traj.qd.shape == (N, 7)


def test_trajectory_duration():
    N = 50
    traj = Trajectory(
        t=np.linspace(0, 2.0, N),
        q=np.zeros((N, 7)),
        qd=np.zeros((N, 7)),
    )
    assert traj.duration == pytest.approx(2.0)


def test_trajectory_dt():
    N = 101
    traj = Trajectory(
        t=np.linspace(0, 1.0, N),
        q=np.zeros((N, 7)),
        qd=np.zeros((N, 7)),
    )
    assert traj.dt == pytest.approx(0.01)

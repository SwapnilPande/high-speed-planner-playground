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

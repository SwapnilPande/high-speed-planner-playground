import numpy as np
import pytest
from kinodynamic_planner.planning.base import JointConstraints


def test_kinova_gen3_constraints_shape():
    c = JointConstraints.kinova_gen3()
    assert c.v_max.shape == (7,)
    assert c.a_max.shape == (7,)
    assert c.j_max.shape == (7,)


def test_kinova_gen3_limits_positive():
    c = JointConstraints.kinova_gen3()
    assert np.all(c.v_max > 0)
    assert np.all(c.a_max > 0)
    assert np.all(c.j_max > 0)


def test_kinova_gen3_velocity_limits():
    c = JointConstraints.kinova_gen3()
    # Large joints (0-3): 1.39 rad/s; small joints (4-6): 1.22 rad/s
    np.testing.assert_allclose(c.v_max[:4], 1.39)
    np.testing.assert_allclose(c.v_max[4:], 1.22)

import numpy as np
import pytest
from kinodynamic_planner.planning.base import JointConstraints

# Shared test pair — reused for iLQR later
Q_START = np.zeros(7)
Q_GOAL  = np.array([0.0, -0.8, 0.0, -1.5, 0.0, 1.2, 0.0])


@pytest.fixture
def constraints():
    return JointConstraints.kinova_gen3()

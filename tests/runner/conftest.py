import pathlib
import pytest
import numpy as np
from kinodynamic_planner.types import Trajectory


FIXTURE_XML = pathlib.Path(__file__).parent.parent / "fixtures" / "minimal_7dof.xml"


@pytest.fixture
def sim():
    from kinodynamic_planner.sim.simulator import Simulator
    s = Simulator(model_path=str(FIXTURE_XML), control_hz=100.0)
    yield s
    s.close()


@pytest.fixture
def short_traj():
    N = 20
    t = np.linspace(0, 0.19, N)
    q = np.zeros((N, 7))
    qd = np.zeros((N, 7))
    qd[:, 0] = 0.1  # slow constant velocity on joint 0
    return Trajectory(t=t, q=q, qd=qd)

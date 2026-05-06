import pathlib
import pytest


FIXTURE_XML = pathlib.Path(__file__).parent.parent / "fixtures" / "minimal_7dof.xml"


@pytest.fixture
def sim():
    from kinodynamic_planner.sim.simulator import Simulator
    s = Simulator(model_path=str(FIXTURE_XML), control_hz=100.0)
    yield s
    s.close()

"""Integration tests using the real Kinova Gen3 model. Skipped if model not downloaded."""
import pathlib
import numpy as np
import pytest

KINOVA_MODEL = pathlib.Path(__file__).parent.parent.parent / "models" / "kinova_gen3" / "scene.xml"

pytestmark = pytest.mark.skipif(
    not KINOVA_MODEL.exists(),
    reason="Kinova Gen3 model not downloaded. Run: bash scripts/download_model.sh",
)


@pytest.fixture
def kinova_sim():
    from kinodynamic_planner.sim.simulator import Simulator
    s = Simulator(model_path=str(KINOVA_MODEL), control_hz=100.0)
    yield s
    s.close()


def test_kinova_loads(kinova_sim):
    assert kinova_sim is not None


def test_kinova_initial_state(kinova_sim):
    state = kinova_sim.get_state()
    assert state.q.shape == (7,)


def test_kinova_step_no_error(kinova_sim):
    for _ in range(100):
        kinova_sim.step(np.zeros(7))
    state = kinova_sim.get_state()
    assert state.t > 0


def test_kinova_playback_runs(kinova_sim):
    from kinodynamic_planner.types import Trajectory
    from kinodynamic_planner.runner.playback import run_playback
    N = 50
    t = np.linspace(0, 0.5, N)
    qd = np.zeros((N, 7))
    qd[:, 1] = 0.1  # limited joint — reliable tracking
    q = np.zeros((N, 7))
    q[:, 1] = 0.1 * t  # integrate: ramp to 0.05 rad
    traj = Trajectory(t=t, q=q, qd=qd)
    log = run_playback(kinova_sim, traj)
    assert log["q_actual"][-1, 1] > 0.001


def test_kinova_unlimited_joint_responds(kinova_sim):
    from kinodynamic_planner.types import Trajectory
    from kinodynamic_planner.runner.playback import run_playback
    N = 200
    t = np.linspace(0, 2.0, N)
    qd = np.zeros((N, 7))
    qd[:, 0] = 0.5  # base rotation — unlimited joint
    q = np.zeros((N, 7))
    q[:, 0] = 0.5 * t  # integrate: ramp to 1.0 rad
    traj = Trajectory(t=t, q=q, qd=qd)
    log = run_playback(kinova_sim, traj)
    assert log["q_actual"][-1, 0] > 0.01

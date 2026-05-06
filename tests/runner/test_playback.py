import numpy as np
import pytest
import tempfile
import pathlib
from kinodynamic_planner.types import Trajectory


def _make_traj(N: int = 50) -> dict:
    t = np.linspace(0, 1.0, N)
    q = np.random.randn(N, 7)
    qd = np.random.randn(N, 7)
    return {"t": t, "q": q, "qd": qd}


def test_load_trajectory_from_npy():
    from kinodynamic_planner.runner.playback import load_trajectory
    data = _make_traj(50)
    with tempfile.NamedTemporaryFile(suffix=".npy", delete=False) as f:
        path = f.name
    np.save(path, data)
    traj = load_trajectory(path)
    assert isinstance(traj, Trajectory)
    assert traj.t.shape == (50,)
    assert traj.q.shape == (50, 7)
    assert traj.qd.shape == (50, 7)


def test_load_trajectory_from_csv():
    from kinodynamic_planner.runner.playback import load_trajectory
    data = _make_traj(30)
    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False, mode="w") as f:
        path = f.name
        header = "t," + ",".join(f"q{i}" for i in range(7)) + "," + ",".join(f"qd{i}" for i in range(7))
        f.write(header + "\n")
        for i in range(30):
            row = [data["t"][i]] + list(data["q"][i]) + list(data["qd"][i])
            f.write(",".join(str(v) for v in row) + "\n")
    traj = load_trajectory(path)
    assert isinstance(traj, Trajectory)
    assert traj.t.shape == (30,)
    np.testing.assert_allclose(traj.q, data["q"], atol=1e-6)


def test_load_trajectory_unknown_extension():
    from kinodynamic_planner.runner.playback import load_trajectory
    with pytest.raises(ValueError, match="Unsupported"):
        load_trajectory("trajectory.txt")

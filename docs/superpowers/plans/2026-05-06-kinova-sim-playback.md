# Kinova Gen3 MuJoCo Simulator + Trajectory Playback — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Stand up a MuJoCo simulation of the Kinova Gen3 7DOF arm that can load and play back a joint-space trajectory with velocity commands.

**Architecture:** Four-layer Python package (sim, runner, types, scripts) with clean interfaces. MuJoCo Menagerie provides the Kinova Gen3 MJCF model. The Simulator wraps MuJoCo's MjModel/MjData, accepts joint velocity commands, and integrates them to position targets for position-actuated servos. Playback loads a trajectory file, steps the sim, and logs actual vs commanded state.

**Tech Stack:** Python 3.11+, uv, mujoco>=3.2, numpy, scipy, pytest

---

## File Map

| File | Responsibility |
|------|---------------|
| `pyproject.toml` | uv project config, dependencies |
| `tests/fixtures/minimal_7dof.xml` | Minimal 7-joint MuJoCo model for fast unit tests |
| `models/kinova_gen3/` | Downloaded MJCF from MuJoCo Menagerie |
| `kinodynamic_planner/types.py` | `RobotState` and `Trajectory` dataclasses |
| `kinodynamic_planner/sim/simulator.py` | `Simulator`: MuJoCo wrapper, step, reset, get_state, render |
| `kinodynamic_planner/runner/playback.py` | `load_trajectory`, `run_playback` |
| `scripts/run_playback.py` | CLI entrypoint |
| `tests/test_types.py` | Unit tests for dataclasses |
| `tests/sim/test_simulator.py` | Unit + integration tests for Simulator |
| `tests/runner/test_playback.py` | Unit tests for load_trajectory and run_playback |

---

## Task 1: Initialize uv project and package structure

**Files:**
- Create: `pyproject.toml`
- Create: `kinodynamic_planner/__init__.py`
- Create: `kinodynamic_planner/sim/__init__.py`
- Create: `kinodynamic_planner/runner/__init__.py`
- Create: `tests/__init__.py`
- Create: `tests/sim/__init__.py`
- Create: `tests/runner/__init__.py`

- [ ] **Step 1: Initialize uv project**

```bash
cd /home/swapnil/atdev/kinodynamic_planner
uv init --no-readme --python 3.11
```

Expected output: `pyproject.toml` created, `hello.py` created (delete it).

```bash
rm -f hello.py
```

- [ ] **Step 2: Replace pyproject.toml content**

Replace the generated `pyproject.toml` with:

```toml
[project]
name = "kinodynamic-planner"
version = "0.1.0"
requires-python = ">=3.11"
dependencies = [
    "mujoco>=3.2",
    "numpy>=1.26",
    "scipy>=1.12",
]

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.uv]
dev-dependencies = [
    "pytest>=8.0",
    "numpy>=1.26",
]

[tool.pytest.ini_options]
testpaths = ["tests"]
```

- [ ] **Step 3: Install dependencies**

```bash
uv sync
```

Expected: resolves and installs mujoco, numpy, scipy, pytest into `.venv/`.

- [ ] **Step 4: Create package directories**

```bash
mkdir -p kinodynamic_planner/sim kinodynamic_planner/runner
mkdir -p tests/sim tests/runner tests/fixtures
mkdir -p models scripts
touch kinodynamic_planner/__init__.py
touch kinodynamic_planner/sim/__init__.py
touch kinodynamic_planner/runner/__init__.py
touch tests/__init__.py
touch tests/sim/__init__.py
touch tests/runner/__init__.py
```

- [ ] **Step 5: Verify pytest runs**

```bash
uv run pytest --collect-only
```

Expected: `no tests ran` (0 errors, 0 collected).

- [ ] **Step 6: Init git and commit**

```bash
git init
echo ".venv/\n__pycache__/\n*.pyc\n.pytest_cache/\n*.egg-info/" > .gitignore
git add .
git commit -m "chore: initialize uv project with package structure"
```

---

## Task 2: Create minimal 7-DOF MuJoCo test fixture

**Files:**
- Create: `tests/fixtures/minimal_7dof.xml`
- Create: `tests/fixtures/__init__.py`

This XML is used only in unit tests so they run without the Kinova model.

- [ ] **Step 1: Write fixture XML**

Create `tests/fixtures/minimal_7dof.xml`:

```xml
<mujoco model="minimal_7dof">
  <option timestep="0.01"/>
  <worldbody>
    <body name="link0" pos="0 0 0">
      <joint name="j0" type="hinge" axis="0 0 1" range="-3.14 3.14"/>
      <geom type="box" size="0.05 0.05 0.1"/>
      <body name="link1" pos="0 0 0.2">
        <joint name="j1" type="hinge" axis="0 1 0" range="-2.09 2.09"/>
        <geom type="box" size="0.05 0.05 0.1"/>
        <body name="link2" pos="0 0 0.2">
          <joint name="j2" type="hinge" axis="0 0 1" range="-3.14 3.14"/>
          <geom type="box" size="0.05 0.05 0.1"/>
          <body name="link3" pos="0 0 0.2">
            <joint name="j3" type="hinge" axis="0 1 0" range="-2.09 2.09"/>
            <geom type="box" size="0.05 0.05 0.1"/>
            <body name="link4" pos="0 0 0.2">
              <joint name="j4" type="hinge" axis="0 0 1" range="-3.14 3.14"/>
              <geom type="box" size="0.05 0.05 0.1"/>
              <body name="link5" pos="0 0 0.2">
                <joint name="j5" type="hinge" axis="0 1 0" range="-2.09 2.09"/>
                <geom type="box" size="0.05 0.05 0.1"/>
                <body name="link6" pos="0 0 0.2">
                  <joint name="j6" type="hinge" axis="0 0 1" range="-3.14 3.14"/>
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
    <position name="a0" joint="j0" kp="500" kv="50"/>
    <position name="a1" joint="j1" kp="500" kv="50"/>
    <position name="a2" joint="j2" kp="500" kv="50"/>
    <position name="a3" joint="j3" kp="500" kv="50"/>
    <position name="a4" joint="j4" kp="500" kv="50"/>
    <position name="a5" joint="j5" kp="500" kv="50"/>
    <position name="a6" joint="j6" kp="500" kv="50"/>
  </actuator>
</mujoco>
```

- [ ] **Step 2: Create fixtures __init__**

```bash
touch tests/fixtures/__init__.py
```

- [ ] **Step 3: Verify MuJoCo loads the fixture**

```bash
uv run python -c "
import mujoco, pathlib
xml = pathlib.Path('tests/fixtures/minimal_7dof.xml').read_text()
m = mujoco.MjModel.from_xml_string(xml)
print(f'nq={m.nq}, nv={m.nv}, nu={m.nu}')
"
```

Expected: `nq=7, nv=7, nu=7`

- [ ] **Step 4: Commit**

```bash
git add tests/fixtures/
git commit -m "test: add minimal 7-DOF MuJoCo fixture for unit tests"
```

---

## Task 3: Download Kinova Gen3 MJCF model

**Files:**
- Create: `models/kinova_gen3/` (downloaded from MuJoCo Menagerie)
- Create: `scripts/download_model.sh`

- [ ] **Step 1: Write download script**

Create `scripts/download_model.sh`:

```bash
#!/usr/bin/env bash
set -e
DEST="$(dirname "$0")/../models"
TMP=$(mktemp -d)

echo "Cloning MuJoCo Menagerie (sparse)..."
git clone --filter=blob:none --sparse https://github.com/google-deepmind/mujoco_menagerie.git "$TMP/menagerie"
cd "$TMP/menagerie"
git sparse-checkout set kinova_gen3

echo "Copying kinova_gen3 to models/..."
mkdir -p "$DEST"
cp -r kinova_gen3 "$DEST/"

rm -rf "$TMP"
echo "Done. Model at models/kinova_gen3/"
```

```bash
chmod +x scripts/download_model.sh
```

- [ ] **Step 2: Run the download script**

```bash
bash scripts/download_model.sh
```

Expected: `models/kinova_gen3/` populated with `gen3.xml`, `scene.xml`, and asset files.

- [ ] **Step 3: Verify the model loads**

```bash
uv run python -c "
import mujoco
m = mujoco.MjModel.from_xml_path('models/kinova_gen3/scene.xml')
print(f'nq={m.nq}, nu={m.nu}, model={m.model_name}')
"
```

Expected output contains `nq=7` (or more if gripper included) and no errors.

Note the exact `nq` value — the Kinova Gen3 model in Menagerie includes only the 7 arm joints (no gripper). If `nq > 7`, set `N_ARM_JOINTS = 7` in the Simulator.

- [ ] **Step 4: Add models to gitignore and commit script**

```bash
echo "models/kinova_gen3/" >> .gitignore
git add scripts/download_model.sh .gitignore
git commit -m "chore: add model download script for Kinova Gen3 from MuJoCo Menagerie"
```

---

## Task 4: Implement data types (TDD)

**Files:**
- Create: `kinodynamic_planner/types.py`
- Create: `tests/test_types.py`

- [ ] **Step 1: Write failing tests**

Create `tests/test_types.py`:

```python
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
```

- [ ] **Step 2: Run tests — verify they fail**

```bash
uv run pytest tests/test_types.py -v
```

Expected: `ModuleNotFoundError: No module named 'kinodynamic_planner.types'`

- [ ] **Step 3: Implement types.py**

Create `kinodynamic_planner/types.py`:

```python
from __future__ import annotations
from dataclasses import dataclass
import numpy as np


@dataclass
class RobotState:
    t: float
    q: np.ndarray    # joint positions (7,)
    qd: np.ndarray   # joint velocities (7,)
    tau: np.ndarray  # joint torques (7,)


@dataclass
class Trajectory:
    t: np.ndarray    # time stamps (N,)
    q: np.ndarray    # joint positions (N, 7)
    qd: np.ndarray   # joint velocities (N, 7)

    @property
    def duration(self) -> float:
        return float(self.t[-1] - self.t[0])

    @property
    def dt(self) -> float:
        return float(self.t[1] - self.t[0])
```

- [ ] **Step 4: Run tests — verify they pass**

```bash
uv run pytest tests/test_types.py -v
```

Expected: 4 tests pass.

- [ ] **Step 5: Commit**

```bash
git add kinodynamic_planner/types.py tests/test_types.py
git commit -m "feat: add RobotState and Trajectory dataclasses"
```

---

## Task 5: Implement Simulator (TDD)

**Files:**
- Create: `kinodynamic_planner/sim/simulator.py`
- Create: `tests/sim/test_simulator.py`
- Create: `tests/sim/conftest.py`

- [ ] **Step 1: Write conftest with fixture**

Create `tests/sim/conftest.py`:

```python
import pathlib
import pytest


FIXTURE_XML = pathlib.Path(__file__).parent.parent / "fixtures" / "minimal_7dof.xml"


@pytest.fixture
def sim():
    from kinodynamic_planner.sim.simulator import Simulator
    s = Simulator(model_path=str(FIXTURE_XML), control_hz=100.0)
    yield s
    s.close()
```

- [ ] **Step 2: Write failing tests**

Create `tests/sim/test_simulator.py`:

```python
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
```

- [ ] **Step 3: Run tests — verify they fail**

```bash
uv run pytest tests/sim/test_simulator.py -v
```

Expected: `ModuleNotFoundError: No module named 'kinodynamic_planner.sim.simulator'`

- [ ] **Step 4: Implement simulator.py**

Create `kinodynamic_planner/sim/simulator.py`:

```python
from __future__ import annotations
import numpy as np
import mujoco
from kinodynamic_planner.types import RobotState

N_ARM_JOINTS = 7


class Simulator:
    def __init__(self, model_path: str, control_hz: float = 100.0) -> None:
        self._model = mujoco.MjModel.from_xml_path(model_path)
        self._data = mujoco.MjData(self._model)
        self._dt = 1.0 / control_hz
        self._model.opt.timestep = self._dt
        self._viewer = None
        self._joint_limits = self._model.jnt_range[:N_ARM_JOINTS].copy()  # (7, 2)
        mujoco.mj_resetData(self._model, self._data)

    def reset(self, q0: np.ndarray | None = None) -> None:
        mujoco.mj_resetData(self._model, self._data)
        if q0 is not None:
            self._data.qpos[:N_ARM_JOINTS] = q0
            self._data.ctrl[:N_ARM_JOINTS] = q0
            mujoco.mj_forward(self._model, self._data)

    def step(self, qd_cmd: np.ndarray) -> None:
        q_current = self._data.qpos[:N_ARM_JOINTS].copy()
        q_target = q_current + qd_cmd * self._dt
        q_target = np.clip(
            q_target,
            self._joint_limits[:, 0],
            self._joint_limits[:, 1],
        )
        self._data.ctrl[:N_ARM_JOINTS] = q_target
        mujoco.mj_step(self._model, self._data)

    def get_state(self) -> RobotState:
        return RobotState(
            t=float(self._data.time),
            q=self._data.qpos[:N_ARM_JOINTS].copy(),
            qd=self._data.qvel[:N_ARM_JOINTS].copy(),
            tau=self._data.actuator_force[:N_ARM_JOINTS].copy(),
        )

    def render(self) -> None:
        if self._viewer is None:
            self._viewer = mujoco.viewer.launch_passive(self._model, self._data)

    def sync_viewer(self) -> None:
        if self._viewer is not None and self._viewer.is_running():
            self._viewer.sync()

    def close(self) -> None:
        if self._viewer is not None:
            self._viewer.close()
            self._viewer = None
```

- [ ] **Step 5: Run tests — verify they pass**

```bash
uv run pytest tests/sim/test_simulator.py -v
```

Expected: all 8 tests pass.

- [ ] **Step 6: Commit**

```bash
git add kinodynamic_planner/sim/simulator.py tests/sim/
git commit -m "feat: implement MuJoCo Simulator with velocity command interface"
```

---

## Task 6: Implement trajectory I/O (TDD)

**Files:**
- Create: `kinodynamic_planner/runner/playback.py` (load_trajectory only)
- Create: `tests/runner/test_playback.py`

- [ ] **Step 1: Write failing tests for load_trajectory**

Create `tests/runner/test_playback.py`:

```python
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
```

- [ ] **Step 2: Run tests — verify they fail**

```bash
uv run pytest tests/runner/test_playback.py -v
```

Expected: `ModuleNotFoundError: No module named 'kinodynamic_planner.runner.playback'`

- [ ] **Step 3: Implement load_trajectory in playback.py**

Create `kinodynamic_planner/runner/playback.py`:

```python
from __future__ import annotations
import pathlib
import numpy as np
from kinodynamic_planner.types import Trajectory, RobotState


def load_trajectory(path: str) -> Trajectory:
    p = pathlib.Path(path)
    if p.suffix == ".npy":
        data = np.load(path, allow_pickle=True).item()
        return Trajectory(t=data["t"], q=data["q"], qd=data["qd"])
    elif p.suffix == ".csv":
        arr = np.loadtxt(path, delimiter=",", skiprows=1)
        t = arr[:, 0]
        q = arr[:, 1:8]
        qd = arr[:, 8:15]
        return Trajectory(t=t, q=q, qd=qd)
    else:
        raise ValueError(f"Unsupported trajectory file extension: {p.suffix!r}. Use .npy or .csv")
```

- [ ] **Step 4: Run tests — verify they pass**

```bash
uv run pytest tests/runner/test_playback.py -v
```

Expected: 3 tests pass.

- [ ] **Step 5: Commit**

```bash
git add kinodynamic_planner/runner/playback.py tests/runner/test_playback.py
git commit -m "feat: implement trajectory loading from .npy and .csv"
```

---

## Task 7: Implement run_playback (TDD)

**Files:**
- Modify: `kinodynamic_planner/runner/playback.py` (add `run_playback`)
- Modify: `tests/runner/test_playback.py` (add playback tests)
- Create: `tests/runner/conftest.py`

- [ ] **Step 1: Write conftest**

Create `tests/runner/conftest.py`:

```python
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
```

- [ ] **Step 2: Write failing tests for run_playback**

Append to `tests/runner/test_playback.py`:

```python
def test_run_playback_returns_log(sim, short_traj):
    from kinodynamic_planner.runner.playback import run_playback
    log = run_playback(sim, short_traj)
    assert "t" in log
    assert "q_cmd" in log
    assert "q_actual" in log
    assert "qd_cmd" in log
    assert "qd_actual" in log


def test_run_playback_log_length(sim, short_traj):
    from kinodynamic_planner.runner.playback import run_playback
    log = run_playback(sim, short_traj)
    N = len(short_traj.t)
    assert len(log["t"]) == N
    assert log["q_cmd"].shape == (N, 7)
    assert log["q_actual"].shape == (N, 7)


def test_run_playback_tracks_velocity_command(sim, short_traj):
    from kinodynamic_planner.runner.playback import run_playback
    log = run_playback(sim, short_traj)
    # Joint 0 should have moved in the positive direction
    assert log["q_actual"][-1, 0] > 0.01
```

- [ ] **Step 3: Run new tests — verify they fail**

```bash
uv run pytest tests/runner/test_playback.py::test_run_playback_returns_log -v
```

Expected: `ImportError` or `AttributeError` — `run_playback` not defined.

- [ ] **Step 4: Implement run_playback in playback.py**

Add to `kinodynamic_planner/runner/playback.py`:

```python
def run_playback(
    sim,
    traj: Trajectory,
    render: bool = False,
) -> dict:
    sim.reset()
    if render:
        sim.render()

    N = len(traj.t)
    log = {
        "t": np.empty(N),
        "q_cmd": np.empty((N, 7)),
        "q_actual": np.empty((N, 7)),
        "qd_cmd": np.empty((N, 7)),
        "qd_actual": np.empty((N, 7)),
    }

    for i in range(N):
        qd_cmd = traj.qd[i]
        sim.step(qd_cmd)
        state = sim.get_state()

        log["t"][i] = state.t
        log["q_cmd"][i] = traj.q[i]
        log["q_actual"][i] = state.q
        log["qd_cmd"][i] = qd_cmd
        log["qd_actual"][i] = state.qd

        if render:
            sim.sync_viewer()

    return log
```

- [ ] **Step 5: Run all tests — verify they pass**

```bash
uv run pytest tests/ -v
```

Expected: all tests pass.

- [ ] **Step 6: Commit**

```bash
git add kinodynamic_planner/runner/playback.py tests/runner/
git commit -m "feat: implement trajectory playback runner with state logging"
```

---

## Task 8: CLI script and end-to-end test

**Files:**
- Create: `scripts/run_playback.py`
- Create: `scripts/make_test_traj.py`

- [ ] **Step 1: Create test trajectory generator**

Create `scripts/make_test_traj.py`:

```python
"""Generate a sinusoidal test trajectory and save to disk."""
import numpy as np
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="test_traj.npy")
    parser.add_argument("--duration", type=float, default=3.0)
    parser.add_argument("--hz", type=float, default=100.0)
    args = parser.parse_args()

    N = int(args.duration * args.hz)
    t = np.linspace(0, args.duration, N)

    # Gentle sinusoidal velocities within safe limits (rad/s)
    freq = 0.5  # Hz
    amp = np.array([0.3, 0.2, 0.3, 0.2, 0.3, 0.2, 0.3])
    qd = amp[None, :] * np.sin(2 * np.pi * freq * t[:, None])
    q = np.cumsum(qd, axis=0) / args.hz  # integrate for position reference

    np.save(args.out, {"t": t, "q": q, "qd": qd})
    print(f"Saved trajectory: {N} steps @ {args.hz}Hz -> {args.out}")

if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Create CLI run_playback.py**

Create `scripts/run_playback.py`:

```python
"""Play back a joint trajectory on the Kinova Gen3 MuJoCo simulation."""
import argparse
import pathlib
import numpy as np


def main():
    parser = argparse.ArgumentParser(description="Play back a trajectory in MuJoCo")
    parser.add_argument("--traj", required=True, help="Path to trajectory file (.npy or .csv)")
    parser.add_argument("--model", default="models/kinova_gen3/scene.xml", help="Path to MJCF model")
    parser.add_argument("--render", action="store_true", help="Launch MuJoCo viewer")
    parser.add_argument("--hz", type=float, default=None, help="Control frequency override (Hz)")
    parser.add_argument("--log", default=None, help="Save log to .npy file")
    args = parser.parse_args()

    from kinodynamic_planner.sim.simulator import Simulator
    from kinodynamic_planner.runner.playback import load_trajectory, run_playback

    traj = load_trajectory(args.traj)
    control_hz = args.hz if args.hz is not None else 1.0 / traj.dt

    print(f"Trajectory: {len(traj.t)} steps, duration={traj.duration:.2f}s, hz={control_hz:.1f}")
    print(f"Model: {args.model}")

    sim = Simulator(model_path=args.model, control_hz=control_hz)
    log = run_playback(sim, traj, render=args.render)
    sim.close()

    q_err = np.abs(log["q_actual"] - log["q_cmd"])
    print(f"Mean joint position error: {q_err.mean():.4f} rad")
    print(f"Max  joint position error: {q_err.max():.4f} rad")

    if args.log:
        np.save(args.log, log)
        print(f"Log saved to {args.log}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 3: Generate test trajectory using the minimal model (smoke test)**

```bash
uv run python scripts/make_test_traj.py --out /tmp/test_traj.npy --duration 2.0 --hz 100
```

Expected: `Saved trajectory: 200 steps @ 100.0Hz -> /tmp/test_traj.npy`

- [ ] **Step 4: Run playback with minimal fixture model (no render)**

```bash
uv run python scripts/run_playback.py \
  --traj /tmp/test_traj.npy \
  --model tests/fixtures/minimal_7dof.xml \
  --log /tmp/playback_log.npy
```

Expected: prints trajectory info, mean/max joint error, saves log. No errors.

- [ ] **Step 5: Run playback with Kinova model and viewer**

```bash
uv run python scripts/run_playback.py \
  --traj /tmp/test_traj.npy \
  --model models/kinova_gen3/scene.xml \
  --render
```

Expected: MuJoCo viewer opens showing the Kinova Gen3 arm moving through the sinusoidal trajectory.

- [ ] **Step 6: Run full test suite one final time**

```bash
uv run pytest tests/ -v
```

Expected: all tests pass.

- [ ] **Step 7: Commit**

```bash
git add scripts/
git commit -m "feat: add CLI playback script and test trajectory generator"
```

---

## Task 9: Integration test with Kinova model

**Files:**
- Create: `tests/sim/test_simulator_integration.py`

- [ ] **Step 1: Write integration test**

Create `tests/sim/test_simulator_integration.py`:

```python
"""Integration tests using the real Kinova Gen3 model. Skipped if model not downloaded."""
import pathlib
import numpy as np
import pytest

KINOVA_MODEL = pathlib.Path("models/kinova_gen3/scene.xml")

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
    qd[:, 0] = 0.1
    traj = Trajectory(t=t, q=np.zeros((N, 7)), qd=qd)
    log = run_playback(kinova_sim, traj)
    assert log["q_actual"][-1, 0] > 0.001
```

- [ ] **Step 2: Run integration tests**

```bash
uv run pytest tests/sim/test_simulator_integration.py -v
```

Expected: 4 tests pass (or all skipped if model not downloaded).

- [ ] **Step 3: Final commit**

```bash
git add tests/sim/test_simulator_integration.py
git commit -m "test: add Kinova integration tests (skipped if model absent)"
```

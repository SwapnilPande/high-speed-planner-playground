# Kinova Gen3 MuJoCo Simulator + Trajectory Playback

**Date:** 2026-05-06  
**Status:** Approved

## Overview

A standalone Python project (managed with uv) that provides a MuJoCo simulation of the Kinova Gen3 7DOF arm and a trajectory playback runner. This is the first phase of a larger kinodynamic planning system. The architecture is designed to be extended with planners and controllers later.

## Scope

This phase covers only:
1. MuJoCo simulator wrapper for the Kinova Gen3
2. Trajectory playback (load → command → log)

Planners, IK, and controllers are out of scope for this phase.

## Project Structure

```
kinodynamic_planner/
├── pyproject.toml
├── models/
│   └── kinova_gen3/          # MJCF model files (downloaded from MuJoCo Menagerie)
├── kinodynamic_planner/
│   ├── __init__.py
│   ├── sim/
│   │   ├── __init__.py
│   │   └── simulator.py      # MuJoCo wrapper
│   └── runner/
│       ├── __init__.py
│       └── playback.py       # Trajectory playback logic
└── scripts/
    └── run_playback.py       # CLI entrypoint
```

## Data Types

```python
@dataclass
class RobotState:
    t: float               # simulation time
    q: np.ndarray          # joint positions (7,)
    qd: np.ndarray         # joint velocities (7,)
    tau: np.ndarray        # joint torques (7,)

@dataclass
class Trajectory:
    t: np.ndarray          # time stamps (N,)
    q: np.ndarray          # joint positions (N, 7)
    qd: np.ndarray         # joint velocities (N, 7)
```

## Simulator (`sim/simulator.py`)

Wraps `mujoco.MjModel` and `mujoco.MjData`. Public interface:

- `__init__(model_path: str, control_hz: float = 100.0)` — loads MJCF, sets timestep
- `reset(q0: np.ndarray | None = None)` — resets to home config or provided `q0`
- `step(qd_cmd: np.ndarray)` — applies joint velocity command, advances physics
- `get_state() -> RobotState` — returns current joint positions, velocities, torques
- `render()` — launches MuJoCo passive viewer (non-blocking)
- `close()` — tears down viewer if open

Control mode: joint velocities by default. The simulator applies velocity commands via MuJoCo's `ctrl` array with a velocity actuator model.

## Trajectory Playback (`runner/playback.py`)

Loads a trajectory from file (`.npy` dict with keys `t`, `q`, `qd`, or a `.csv` with columns `t, q0..q6, qd0..qd6`). Steps the simulator at the trajectory's native timestep, feeding `qd` commands. Logs actual state at each step.

Returns a log dict: `{"t": [...], "q_cmd": [...], "q_actual": [...], "qd_cmd": [...], "qd_actual": [...]}`.

## CLI (`scripts/run_playback.py`)

```
python scripts/run_playback.py --traj <path> [--render] [--hz 100] [--log <output_path>]
```

- `--traj` — path to trajectory file (`.npy` or `.csv`)
- `--render` — launch MuJoCo viewer
- `--hz` — control frequency override (default: inferred from trajectory `t` array)
- `--log` — save log dict to `.npy` file

## Kinova Gen3 Model

Source: MuJoCo Menagerie (`google-deepmind/mujoco_menagerie`), `kinova_gen3` directory. 7DOF, MJCF format. Joint limits, actuator models, and collision geometry included.

## Dependencies

- `mujoco >= 3.0`
- `numpy`
- `scipy` (trajectory interpolation)
- `pyyaml` (future config use)

## Success Criteria

- MuJoCo sim loads the Kinova Gen3 model without errors
- Passive viewer renders the arm in home configuration
- A synthetic sinusoidal trajectory (generated inline for testing) plays back correctly
- Logged `q_actual` tracks `q_cmd` within expected sim error bounds
- CLI runs end-to-end with `--render`

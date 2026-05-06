# MinJerk Planner Design

**Date:** 2026-05-06  
**Status:** Approved

## Overview

Add a `MinJerkPlanner` to the kinodynamic_planner package. It plans a joint-space trajectory from `q_start` to `q_goal` using a 5th-order polynomial that minimises the integral of squared jerk. Total time `T` is computed analytically from per-joint velocity, acceleration, and jerk constraints — no tuning required.

The same `q_start`/`q_goal` test pair is committed to `tests/planning/conftest.py` for reuse when iLQR is added.

## Changes to Existing Code

`kinodynamic_planner/types.py`: add `qdd: np.ndarray  # (N, 7)` to `Trajectory`. Update `duration` and `dt` properties (unchanged). All existing callers pass `qdd` as zeros where they don't compute it.

## New Files

### `kinodynamic_planner/planning/base.py`

```python
@dataclass
class JointConstraints:
    v_max: np.ndarray   # (7,) rad/s
    a_max: np.ndarray   # (7,) rad/s²
    j_max: np.ndarray   # (7,) rad/s³

    @staticmethod
    def kinova_gen3() -> "JointConstraints":
        # Kinova Gen3 published limits (conservative)
        return JointConstraints(
            v_max=np.array([1.39, 1.39, 1.39, 1.39, 1.22, 1.22, 1.22]),
            a_max=np.full(7, 8.0),
            j_max=np.full(7, 50.0),
        )

class Planner(Protocol):
    def plan(self, q_start: np.ndarray, q_goal: np.ndarray,
             constraints: JointConstraints) -> Trajectory: ...
```

### `kinodynamic_planner/planning/min_jerk.py`

`MinJerkPlanner.plan(q_start, q_goal, constraints) -> Trajectory`:

1. Compute `Δq = q_goal - q_start` per joint.
2. Compute minimum T per constraint:
   - `T_v = 1.875  * max_j(|Δqj| / v_max_j)`
   - `T_a = sqrt(1.924 * max_j(|Δqj| / a_max_j))`
   - `T_j = cbrt(60.0  * max_j(|Δqj| / j_max_j))`
   - `T   = max(T_v, T_a, T_j)`
3. Sample at `dt = 1/100` s (100 Hz, matching sim):
   - `τ = t / T`
   - `q(τ)   = q_start + Δq * (10τ³ − 15τ⁴ + 6τ⁵)`
   - `qd(τ)  = Δq/T   * 30τ²(1−τ)²`
   - `qdd(τ) = Δq/T²  * 60τ(1−τ)(1−2τ)`
4. Return `Trajectory(t, q, qd, qdd)`.

Raises `ValueError` if `q_start == q_goal` (zero displacement, T undefined).

### `tests/planning/conftest.py`

```python
Q_START = np.zeros(7)
Q_GOAL  = np.array([0.0, -0.8, 0.0, -1.5, 0.0, 1.2, 0.0])
```

### `tests/planning/test_min_jerk.py`

- Boundary conditions: `q[0] ≈ q_start`, `q[-1] ≈ q_goal`, `qd[0] ≈ 0`, `qd[-1] ≈ 0`
- Constraint satisfaction: `|qd|.max() ≤ v_max + tol` per joint
- Monotone time: `t` strictly increasing
- Equal start/goal raises `ValueError`
- Trajectory plays back in sim without joint limit violations

## Success Criteria

- All tests pass
- `uv run python scripts/run_playback.py --traj traj.npy --render` with a MinJerk-generated trajectory shows the Kinova arm moving smoothly from `Q_START` to `Q_GOAL`

# Viewer Enhancements Design

**Date:** 2026-05-07  
**Status:** Approved

## Overview

Three small UX improvements to `--render` mode in `scripts/run_min_jerk.py`:

1. **Zoom out** — viewer starts with camera pulled back so the full arm is visible
2. **EE markers** — green sphere at Q_START end-effector, red sphere at Q_GOAL end-effector, visible throughout playback
3. **Press to play** — viewer opens and shows markers; playback starts only when user presses Enter in the terminal

## Changes

### `kinodynamic_planner/sim/simulator.py`

**`render()`** — after `launch_passive`, set `viewer.cam.distance = 3.0` (current default is ~1.5).

**`get_ee_position(q: np.ndarray) -> np.ndarray`** — saves `data.qpos`, sets qpos to `q`, calls `mujoco.mj_fwdPosition`, reads `data.xpos` for body `"bracelet_link"` (last Kinova arm link, index looked up once via `model.body("bracelet_link").id`), restores qpos, returns `(3,)` array.

**`add_sphere_marker(pos: np.ndarray, rgba: tuple[float,float,float,float], radius: float = 0.03) -> None`** — appends a sphere geom to `viewer.user_scn` using `mujoco.mjv_initGeom`. No-ops if viewer is None or `user_scn` is full.

### `scripts/run_min_jerk.py`

After `sim.render()` (viewer open, camera set):
1. Call `sim.get_ee_position(q_start)` → green sphere `(0, 1, 0, 0.8)`
2. Call `sim.get_ee_position(q_goal)` → red sphere `(1, 0, 0, 0.8)`
3. Call `sim.sync_viewer()` — markers appear
4. `input("Press Enter to play...")` — blocks; viewer stays live in background thread
5. `run_playback(sim, traj, render=True)` as normal — markers persist (they live in `user_scn`, not simulation state)

## Success Criteria

- `uv run python scripts/run_min_jerk.py --render` opens the viewer zoomed out, shows two spheres at EE positions, waits for Enter, then plays smoothly
- Markers visible throughout the trajectory
- No new files, no new dependencies

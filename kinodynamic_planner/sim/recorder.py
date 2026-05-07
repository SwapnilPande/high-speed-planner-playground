from __future__ import annotations
import numpy as np
import mujoco
import imageio


class VideoRecorder:
    """Offscreen renderer that collects frames and writes an MP4."""

    def __init__(
        self,
        sim,
        fps: int = 30,
        width: int = 1280,
        height: int = 720,
    ) -> None:
        self._renderer = mujoco.Renderer(sim.model, height, width)
        self._sim = sim
        self._fps = fps
        self._frames: list[np.ndarray] = []
        # Capture every N sim steps so recorded fps matches requested fps
        self.every: int = 1  # set externally once traj.dt is known

    def set_dt(self, dt: float) -> None:
        """Compute capture stride from simulation timestep."""
        self.every = max(1, round(1.0 / (self._fps * dt)))

    def capture(self, camera_state: dict | None = None) -> None:
        cam = mujoco.MjvCamera()
        if camera_state is not None:
            cam.azimuth = camera_state["azimuth"]
            cam.elevation = camera_state["elevation"]
            cam.distance = camera_state["distance"]
            cam.lookat[:] = camera_state["lookat"]
        else:
            mujoco.mjv_defaultCamera(cam)
        self._renderer.update_scene(self._sim.data, camera=cam)
        self._frames.append(self._renderer.render().copy())

    def save(self, path: str) -> None:
        imageio.mimsave(path, self._frames, fps=self._fps)
        print(f"Video saved to {path} ({len(self._frames)} frames @ {self._fps} fps)")

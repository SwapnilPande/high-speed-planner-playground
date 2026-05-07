from __future__ import annotations
import numpy as np
import mujoco
import mujoco.viewer
from kinodynamic_planner.types import RobotState

N_ARM_JOINTS = 7


class Simulator:
    def __init__(self, model_path: str, control_hz: float = 100.0) -> None:
        self._model = mujoco.MjModel.from_xml_path(model_path)
        self._data = mujoco.MjData(self._model)
        self._dt = 1.0 / control_hz
        self._model.opt.timestep = self._dt
        # Use implicitfast integrator for numerical stability with stiff PD gains
        self._model.opt.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
        self._viewer = None
        self._joint_limits = self._model.jnt_range[:N_ARM_JOINTS].copy()  # (7, 2)
        self._joint_limited = self._model.jnt_limited[:N_ARM_JOINTS].astype(bool)  # (7,)
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
        q_target[self._joint_limited] = np.clip(
            q_target[self._joint_limited],
            self._joint_limits[self._joint_limited, 0],
            self._joint_limits[self._joint_limited, 1],
        )
        self._data.ctrl[:N_ARM_JOINTS] = q_target
        mujoco.mj_step(self._model, self._data)

    def step_pos(self, q_target: np.ndarray) -> None:
        """Advance simulation with a direct position setpoint (preferred for trajectory replay)."""
        clipped = q_target.copy()
        clipped[self._joint_limited] = np.clip(
            clipped[self._joint_limited],
            self._joint_limits[self._joint_limited, 0],
            self._joint_limits[self._joint_limited, 1],
        )
        self._data.ctrl[:N_ARM_JOINTS] = clipped
        mujoco.mj_step(self._model, self._data)

    def get_state(self) -> RobotState:
        return RobotState(
            t=float(self._data.time),
            q=self._data.qpos[:N_ARM_JOINTS].copy(),
            qd=self._data.qvel[:N_ARM_JOINTS].copy(),
            tau=self._data.actuator_force[:N_ARM_JOINTS].copy(),
        )

    def get_ee_position(self, q: np.ndarray, body_name: str) -> np.ndarray:
        """Return world position of `body_name` when the arm is at configuration `q`."""
        saved = self._data.qpos[:N_ARM_JOINTS].copy()
        self._data.qpos[:N_ARM_JOINTS] = q
        mujoco.mj_fwdPosition(self._model, self._data)
        body_id = self._model.body(body_name).id
        pos = self._data.xpos[body_id].copy()
        self._data.qpos[:N_ARM_JOINTS] = saved
        mujoco.mj_fwdPosition(self._model, self._data)
        return pos

    def add_sphere_marker(
        self,
        pos: np.ndarray,
        rgba: tuple[float, float, float, float],
        radius: float = 0.03,
    ) -> None:
        """Add a sphere marker to the viewer scene. No-op if viewer is not open."""
        if self._viewer is None:
            return
        scn = self._viewer.user_scn
        if scn.ngeom >= scn.maxgeom:
            return
        g = scn.geoms[scn.ngeom]
        mujoco.mjv_initGeom(
            g,
            mujoco.mjtGeom.mjGEOM_SPHERE,
            np.full(3, radius),
            np.array(pos, dtype=np.float64),
            np.eye(3).flatten(),
            np.array(rgba, dtype=np.float32),
        )
        scn.ngeom += 1

    def render(self) -> None:
        if self._viewer is None:
            self._viewer = mujoco.viewer.launch_passive(self._model, self._data)
            self._viewer.cam.distance = 3.0

    def is_viewer_open(self) -> bool:
        return self._viewer is not None and self._viewer.is_running()

    def sync_viewer(self) -> None:
        if self._viewer is not None and self._viewer.is_running():
            self._viewer.sync()

    def close(self) -> None:
        if self._viewer is not None:
            self._viewer.close()
            self._viewer = None

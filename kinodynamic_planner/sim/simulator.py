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

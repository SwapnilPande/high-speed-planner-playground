"""Thin wrapper around the Kortex API for Kinova Gen3 7-DOF hardware control."""
from __future__ import annotations
import contextlib
import math
import numpy as np

# Kortex API — assumed installed on the Jetson
from kortex_api.RouterClient import RouterClient, RouterClientSendOptions
from kortex_api.SessionManager import SessionManager
from kortex_api.TransportClientTcp import TransportClientTcp
from kortex_api.autogen.client_stubs.BaseClientRpc import BaseClient
from kortex_api.autogen.client_stubs.BaseCyclicClientRpc import BaseCyclicClient
from kortex_api.autogen.messages import Base_pb2, BaseCyclic_pb2, Common_pb2

_NJ = 7
_RAD = math.pi / 180.0   # deg → rad
_DEG = 180.0 / math.pi   # rad → deg

# Tight timeout for cyclic refresh so the RT loop never blocks long
_CYCLIC_TIMEOUT_MS = 10


class KinovaArm:
    """
    Context manager for a Kinova Gen3 arm in low-level joint-position servoing.

    Usage:
        with KinovaArm(ip="192.168.1.10") as arm:
            q, qd = arm.read_joint_state()
            arm.set_low_level_servoing()
            arm.send_joint_positions(q_cmd, qd_ff)
    """

    def __init__(
        self,
        ip: str,
        username: str = "admin",
        password: str = "admin",
        port: int = 10000,
    ) -> None:
        self._ip       = ip
        self._username = username
        self._password = password
        self._port     = port

        self._transport:    TransportClientTcp | None  = None
        self._router:       RouterClient | None        = None
        self._session_mgr:  SessionManager | None      = None
        self._base:         BaseClient | None          = None
        self._cyclic:       BaseCyclicClient | None    = None
        self._feedback:     BaseCyclic_pb2.Feedback | None = None
        self._command:      BaseCyclic_pb2.Command | None  = None
        self._cyclic_opts:  RouterClientSendOptions | None = None

    # ------------------------------------------------------------------
    # Connection lifecycle
    # ------------------------------------------------------------------

    def connect(self) -> None:
        self._transport = TransportClientTcp()
        self._transport.connect(self._ip, self._port)

        self._router = RouterClient(self._transport, error_callback=self._on_error)

        session_info = Common_pb2.CreateSessionInfo()
        session_info.username                       = self._username
        session_info.password                       = self._password
        session_info.session_inactivity_timeout     = 60_000   # ms
        session_info.connection_inactivity_timeout  = 2_000    # ms
        self._session_mgr = SessionManager(self._router)
        self._session_mgr.CreateSession(session_info)

        self._base   = BaseClient(self._router)
        self._cyclic = BaseCyclicClient(self._router)

        self._cyclic_opts = RouterClientSendOptions()
        self._cyclic_opts.timeout_ms = _CYCLIC_TIMEOUT_MS

        # Read initial feedback; allocate command with matching frame_id
        self._feedback = self._cyclic.RefreshFeedback()
        self._command  = BaseCyclic_pb2.Command()
        self._command.frame_id = self._feedback.frame_id
        for i in range(_NJ):
            a = self._command.actuators.add()
            a.flags    = 1   # POSITION mode
            a.position = self._feedback.actuators[i].position
            a.velocity = 0.0

    def disconnect(self) -> None:
        self._restore_high_level()
        with contextlib.suppress(Exception):
            if self._session_mgr:
                self._session_mgr.CloseSession()
        with contextlib.suppress(Exception):
            if self._transport:
                self._transport.disconnect()

    def __enter__(self) -> "KinovaArm":
        self.connect()
        return self

    def __exit__(self, *_) -> None:
        self.disconnect()

    # ------------------------------------------------------------------
    # Servoing mode management
    # ------------------------------------------------------------------

    def set_low_level_servoing(self) -> None:
        """Switch arm to LOW_LEVEL_SERVOING. Must be called before the RT loop."""
        self._base.SetServoingMode(Base_pb2.ServoingModeInformation(
            servoing_mode=Base_pb2.LOW_LEVEL_SERVOING
        ))

    def _restore_high_level(self) -> None:
        with contextlib.suppress(Exception):
            if self._base:
                self._base.SetServoingMode(Base_pb2.ServoingModeInformation(
                    servoing_mode=Base_pb2.SINGLE_LEVEL_SERVOING
                ))

    # ------------------------------------------------------------------
    # State and command
    # ------------------------------------------------------------------

    def read_joint_state(self) -> tuple[np.ndarray, np.ndarray]:
        """Return (q_rad, qd_rad_s) from the most recent Refresh feedback."""
        fb = self._feedback
        q  = np.array([fb.actuators[i].position for i in range(_NJ)]) * _RAD
        qd = np.array([fb.actuators[i].velocity for i in range(_NJ)]) * _RAD
        return q, qd

    def send_joint_positions(
        self,
        q_rad: np.ndarray,
        qd_rad_s: np.ndarray | None = None,
    ) -> None:
        """
        Send joint-position command and receive feedback in one RPC.
        Updates internal feedback so read_joint_state() reflects the new state.
        """
        self._command.frame_id += 1
        for i in range(_NJ):
            self._command.actuators[i].position = float(q_rad[i] * _DEG)
            self._command.actuators[i].velocity = (
                float(qd_rad_s[i] * _DEG) if qd_rad_s is not None else 0.0
            )
        self._feedback = self._cyclic.Refresh(self._command, self._cyclic_opts)

    # ------------------------------------------------------------------
    # Safety
    # ------------------------------------------------------------------

    def estop(self) -> None:
        """
        Emergency stop: drop back to high-level servoing immediately.
        The arm's internal controller will hold position.
        """
        self._restore_high_level()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _on_error(error_code) -> None:
        print(f"[KinovaArm] Router error: {error_code}")

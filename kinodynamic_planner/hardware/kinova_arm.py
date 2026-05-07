"""Thin wrapper around the Kortex API for Kinova Gen3 7-DOF hardware control."""
from __future__ import annotations
import contextlib
import math
import threading
import numpy as np

# Kortex API — assumed installed on the Jetson
from kortex_api.RouterClient import RouterClient, RouterClientSendOptions
from kortex_api.SessionManager import SessionManager
from kortex_api.TCPTransport import TCPTransport
from kortex_api.UDPTransport import UDPTransport
from kortex_api.autogen.client_stubs.ActuatorConfigClientRpc import ActuatorConfigClient
from kortex_api.autogen.client_stubs.BaseClientRpc import BaseClient
from kortex_api.autogen.client_stubs.BaseCyclicClientRpc import BaseCyclicClient
from kortex_api.autogen.messages import ActuatorConfig_pb2, Base_pb2, BaseCyclic_pb2, Common_pb2, Session_pb2

_NJ = 7
_RAD = math.pi / 180.0   # deg → rad
_DEG = 180.0 / math.pi   # rad → deg

_TCP_PORT = 10000
_UDP_PORT = 10001


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
    ) -> None:
        self._ip       = ip
        self._username = username
        self._password = password

        self._tcp:          TCPTransport | None  = None
        self._udp:          UDPTransport | None  = None
        self._tcp_router:   RouterClient | None  = None
        self._udp_router:   RouterClient | None  = None
        self._tcp_session:  SessionManager | None      = None
        self._udp_session:  SessionManager | None      = None
        self._base:         BaseClient | None          = None
        self._cyclic:       BaseCyclicClient | None    = None
        self._actuator_cfg: ActuatorConfigClient | None = None
        # Cached references and buffers so the RT loop doesn't re-traverse
        # protobuf descriptors or allocate ndarrays each cycle.
        self._cmd_actuators: list | None = None
        self._fb_actuators:  list | None = None
        self._q_buf  = np.empty(_NJ)
        self._qd_buf = np.empty(_NJ)
        self._feedback:     BaseCyclic_pb2.Feedback | None = None
        self._command:      BaseCyclic_pb2.Command | None  = None

    # ------------------------------------------------------------------
    # Connection lifecycle
    # ------------------------------------------------------------------

    def connect(self) -> None:
        # TCP for control-plane RPCs (Base, ActuatorConfig); UDP for the cyclic
        # feedback/command stream (BaseCyclic). UDP avoids TCP ack/Nagle delays
        # that otherwise inflate the 1 kHz Refresh round-trip.
        self._tcp = TCPTransport()
        self._tcp.connect(self._ip, _TCP_PORT)
        self._udp = UDPTransport()
        self._udp.connect(self._ip, _UDP_PORT)

        self._tcp_router = RouterClient(self._tcp, self._on_error)
        self._udp_router = RouterClient(self._udp, self._on_error)

        session_info = Session_pb2.CreateSessionInfo()
        session_info.username                       = self._username
        session_info.password                       = self._password
        session_info.session_inactivity_timeout     = 60_000   # ms
        session_info.connection_inactivity_timeout  = 2_000    # ms
        self._tcp_session = SessionManager(self._tcp_router)
        self._tcp_session.CreateSession(session_info)
        self._udp_session = SessionManager(self._udp_router)
        self._udp_session.CreateSession(session_info)

        self._base         = BaseClient(self._tcp_router)
        self._cyclic       = BaseCyclicClient(self._udp_router)
        self._actuator_cfg = ActuatorConfigClient(self._tcp_router)

        # Read initial feedback; allocate command with matching frame_id
        self._feedback = self._cyclic.RefreshFeedback()
        self._command  = BaseCyclic_pb2.Command()
        self._command.frame_id = self._feedback.frame_id
        for i in range(_NJ):
            a = self._command.actuators.add()
            a.flags    = 0
            a.position = self._feedback.actuators[i].position
            a.velocity = 0.0
        # Cache actuator submessage refs so the RT loop avoids one descriptor
        # traversal per field write.
        self._cmd_actuators = list(self._command.actuators)
        self._fb_actuators  = list(self._feedback.actuators)

    def disconnect(self) -> None:
        self._restore_high_level()
        for sess in (self._udp_session, self._tcp_session):
            with contextlib.suppress(Exception):
                if sess:
                    sess.CloseSession()
        for router in (self._udp_router, self._tcp_router):
            with contextlib.suppress(Exception):
                if router:
                    router.SetActivationStatus(False)
        for transport in (self._udp, self._tcp):
            with contextlib.suppress(Exception):
                if transport:
                    transport.disconnect()

    def __enter__(self) -> "KinovaArm":
        self.connect()
        return self

    def __exit__(self, *_) -> None:
        self.disconnect()

    # ------------------------------------------------------------------
    # Servoing mode management
    # ------------------------------------------------------------------

    def move_to_joints(self, q_rad: np.ndarray, timeout: float = 60.0) -> None:
        """Move to joint configuration using the high-level API. Blocks until done.

        Requires arm to be in SINGLE_LEVEL_SERVOING (the default after connect).
        Raises TimeoutError or RuntimeError if the move fails.
        """
        done = threading.Event()
        result: list[int] = []
        abort_info: list[str] = []

        def _on_notif(notif) -> None:
            if notif.action_event in (Base_pb2.ACTION_END, Base_pb2.ACTION_ABORT):
                result.append(notif.action_event)
                if notif.action_event == Base_pb2.ACTION_ABORT:
                    # Capture every field on the notification so the abort
                    # reason isn't swallowed.
                    abort_info.append(str(notif).strip())
                done.set()

        notif_handle = self._base.OnNotificationActionTopic(
            _on_notif, Base_pb2.NotificationOptions()
        )

        action = Base_pb2.Action()
        action.name = "move_to_start"
        for i in range(_NJ):
            ja = action.reach_joint_angles.joint_angles.joint_angles.add()
            ja.joint_identifier = i
            # High-level API expects angles in [0, 360°) for continuous joints
            # and the bounded joints' physical range. Wrap from canonical (-π, π].
            ja.value = float((q_rad[i] * _DEG) % 360.0)

        self._base.ExecuteAction(action)
        finished = done.wait(timeout=timeout)
        self._base.Unsubscribe(notif_handle)

        if not finished:
            raise TimeoutError(f"move_to_joints timed out after {timeout:.0f} s")
        if result and result[0] == Base_pb2.ACTION_ABORT:
            detail = abort_info[0] if abort_info else "(no detail)"
            raise RuntimeError(f"move_to_joints was aborted by the arm:\n{detail}")

    def clear_faults(self) -> None:
        """Clear any latched faults on the arm. Safe to call multiple times."""
        with contextlib.suppress(Exception):
            self._base.ClearFaults()

    def set_low_level_servoing(self) -> None:
        """Switch arm to LOW_LEVEL_SERVOING and force actuators into POSITION mode."""
        self.clear_faults()
        self._base.SetServoingMode(Base_pb2.ServoingModeInformation(
            servoing_mode=Base_pb2.LOW_LEVEL_SERVOING
        ))
        # Seed command from current feedback so the priming frame is a no-op
        self._feedback = self._cyclic.RefreshFeedback()
        self._fb_actuators = list(self._feedback.actuators)
        self._command.frame_id = self._feedback.frame_id
        cmd_acts = self._cmd_actuators
        fb_acts  = self._fb_actuators
        for i in range(_NJ):
            a = cmd_acts[i]
            a.position   = fb_acts[i].position
            a.velocity   = 0.0
            a.command_id = self._command.frame_id
        self._feedback = self._cyclic.Refresh(self._command, 0)
        self._fb_actuators = list(self._feedback.actuators)

        # Force every actuator into POSITION mode (a prior torque-control run
        # may have left them in TORQUE; SetControlMode persists across servoing-
        # mode changes). Each call is a slow TCP RPC, so pump the cyclic stream
        # between calls to keep the firmware's low-level watchdog satisfied.
        cm = ActuatorConfig_pb2.ControlModeInformation()
        cm.control_mode = ActuatorConfig_pb2.POSITION
        for idx in range(1, _NJ + 1):
            self._actuator_cfg.SetControlMode(cm, idx)
            self._pump_refresh()
        for _ in range(10):
            self._pump_refresh()

    def _pump_refresh(self) -> None:
        """Send a position-passthrough cyclic frame to keep low-level mode alive."""
        fid = (self._command.frame_id + 1) & 0xFFFF
        self._command.frame_id = fid
        cmd_acts = self._cmd_actuators
        fb_acts  = self._fb_actuators
        for i in range(_NJ):
            a = cmd_acts[i]
            a.position   = fb_acts[i].position
            a.velocity   = 0.0
            a.command_id = fid
        self._feedback = self._cyclic.Refresh(self._command, 0)
        self._fb_actuators = list(self._feedback.actuators)

    def _restore_high_level(self) -> None:
        if self._actuator_cfg is not None:
            cm = ActuatorConfig_pb2.ControlModeInformation()
            cm.control_mode = ActuatorConfig_pb2.POSITION
            for idx in range(1, _NJ + 1):
                with contextlib.suppress(Exception):
                    self._actuator_cfg.SetControlMode(cm, idx)
        with contextlib.suppress(Exception):
            if self._base:
                self._base.SetServoingMode(Base_pb2.ServoingModeInformation(
                    servoing_mode=Base_pb2.SINGLE_LEVEL_SERVOING
                ))

    # ------------------------------------------------------------------
    # State and command
    # ------------------------------------------------------------------

    def read_joint_state(self) -> tuple[np.ndarray, np.ndarray]:
        """Return (q_rad, qd_rad_s) from the most recent Refresh feedback.

        Returns the same preallocated buffers each call — copy if you need to
        retain values across the next read.
        """
        fb_acts = self._fb_actuators
        q, qd = self._q_buf, self._qd_buf
        for i in range(_NJ):
            a = fb_acts[i]
            q[i]  = a.position * _RAD
            qd[i] = a.velocity * _RAD
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
        fid = (self._command.frame_id + 1) & 0xFFFF
        self._command.frame_id = fid
        cmd_acts = self._cmd_actuators
        for i in range(_NJ):
            a = cmd_acts[i]
            a.position   = float((q_rad[i] * _DEG) % 360.0)
            a.velocity   = float(qd_rad_s[i] * _DEG) if qd_rad_s is not None else 0.0
            a.command_id = fid
        self._feedback = self._cyclic.Refresh(self._command, 0)
        self._fb_actuators = list(self._feedback.actuators)

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

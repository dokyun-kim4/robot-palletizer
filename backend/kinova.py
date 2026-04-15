import backend.utilities as utilities
from kortex_api.autogen.client_stubs.BaseClientRpc import BaseClient
from kortex_api.autogen.client_stubs.BaseCyclicClientRpc import BaseCyclicClient
from kortex_api.autogen.messages import Base_pb2, BaseCyclic_pb2, Common_pb2
import numpy as np
import threading
import queue
import sys
import time


class BaseKinova:
    # Added is_suction parameter here
    def __init__(self, is_suction=False) -> None:
        self.args = utilities.parseConnectionArguments()
        self.real_angles = np.zeros(6)
        self.gripper_position = 0.0
        self.is_suction = is_suction

        # The Action Queue System
        self.action_queue = queue.Queue()
        self._is_action_running = False

        # Threading infrastructure
        self._data_lock = threading.Lock()
        self._is_running = False
        self._thread = None

        # Admittance state tracking
        self._desired_admittance = False
        self._current_admittance = False

    def start(self):
        if not self._is_running:
            self._is_running = True
            self._thread = threading.Thread(target=self._background_loop, daemon=True)
            self._thread.start()
            time.sleep(1)

    def _background_loop(self):
        with utilities.DeviceConnection.createTcpConnection(self.args) as router:
            base = BaseClient(router)
            base_cyclic = BaseCyclicClient(router)

            while self._is_running:
                with self._data_lock:
                    desired_admittance = self._desired_admittance

                # 1. Admittance Mode Check
                if desired_admittance != self._current_admittance:
                    admittance = Base_pb2.Admittance()
                    if desired_admittance:
                        admittance.admittance_mode = Base_pb2.JOINT  # pyright: ignore
                    else:
                        admittance.admittance_mode = ( # pyright: ignore[reportAttributeAccessIssue]
                            Base_pb2.UNSPECIFIED # pyright: ignore[reportAttributeAccessIssue]
                        )  # pyright: ignore
                    try:
                        base.SetAdmittance(admittance)
                        self._current_admittance = desired_admittance
                    except Exception:
                        pass

                # 2. Process the Action Queue
                if not self._is_action_running and not self.action_queue.empty():
                    cmd = self.action_queue.get()
                    self._is_action_running = True

                    if cmd["type"] == "move":
                        threading.Thread(
                            target=self._execute_trajectory_and_grip,
                            args=(base, cmd),
                            daemon=True,
                        ).start()

                    elif cmd["type"] == "grip":
                        threading.Thread(
                            target=self._execute_standalone_grip,
                            args=(base, cmd),
                            daemon=True,
                        ).start()

                # 3. Fast Telemetry Loop
                self._update_angles(base_cyclic)
                time.sleep(0.01)

    def _check_for_end_or_abort(self, e):
        def check(notification, e=e):
            if (
                notification.action_event == Base_pb2.ACTION_END
                or notification.action_event == Base_pb2.ACTION_ABORT
            ):
                e.set()

        return check

    def _execute_trajectory_and_grip(self, base, cmd):
        action = Base_pb2.Action()
        action.name = "Setting joint angles"  # pyright: ignore
        action.application_data = ""  # pyright: ignore

        actuator_count = base.GetActuatorCount()

        for idx, joint_id in enumerate(range(actuator_count.count)):
            joint_angle = (
                action.reach_joint_angles.joint_angles.joint_angles.add() # pyright: ignore[reportAttributeAccessIssue]
            )  # pyright: ignore
            joint_angle.joint_identifier = joint_id
            joint_angle.value = np.degrees(cmd["angles"][idx])

        e = threading.Event()
        notification_handle = base.OnNotificationActionTopic(
            self._check_for_end_or_abort(e), Base_pb2.NotificationOptions()
        )

        base.ExecuteAction(action)
        finished = e.wait(20)
        
        try:
            base.Unsubscribe(notification_handle)
        except Exception:
            pass

        if self._is_running and finished and cmd["gripper"] is not None:
            self._execute_gripper_action(base, cmd["gripper"])

        cmd["event"].set()
        self._is_action_running = False

    def _execute_standalone_grip(self, base, cmd):
        self._execute_gripper_action(base, cmd["value"])
        cmd["event"].set()
        self._is_action_running = False

    def _execute_gripper_action(self, base, ratio):
        """Intelligently routes the command based on the hardware type."""

        # ROUTE A: Suction Cup (Fire-and-forget with pneumatic delay)
        if self.is_suction:
            try:
                gripper_command = Base_pb2.GripperCommand()
                gripper_command.mode = Base_pb2.GRIPPER_POSITION  # pyright: ignore
                finger = gripper_command.gripper.finger.add()  # pyright: ignore
                finger.finger_identifier = 1
                finger.value = ratio

                base.SendGripperCommand(gripper_command)

                # THE FIX: Differentiate between turning ON and turning OFF
                if ratio > 0.5:
                    # Turning ON: Give the vacuum pump 1 second to build pressure
                    time.sleep(1.0)
                else:
                    # Turning OFF: Wait 3 seconds for the vacuum seal to naturally vent!
                    # print("[BaseKinova] Venting suction cup... please wait.")
                    time.sleep(3.0)

                with self._data_lock:
                    self.gripper_position = ratio
            except Exception as e:
                print(f"\n[BaseKinova] Suction command failed: {e}")

        # ROUTE B: 2-Finger Gripper (Event-driven motor sensing)
        else:
            action = Base_pb2.Action()
            action.name = "Gripper Action"  # pyright: ignore
            action.application_data = ""  # pyright: ignore

            gripper_cmd = action.send_gripper_command  # pyright: ignore
            gripper_cmd.mode = Base_pb2.GRIPPER_POSITION  # pyright: ignore
            finger = gripper_cmd.gripper.finger.add()  # pyright: ignore
            finger.finger_identifier = 1
            finger.value = ratio

            e = threading.Event()
            notification_handle = base.OnNotificationActionTopic(
                self._check_for_end_or_abort(e), Base_pb2.NotificationOptions()
            )

            try:
                base.ExecuteAction(action)
                e.wait(5)
                with self._data_lock:
                    self.gripper_position = ratio
            except Exception as e:
                print(f"\n[BaseKinova] Gripper action failed: {e}")
            finally:
                try:
                    base.Unsubscribe(notification_handle)
                except Exception as e:
                    pass

    def _update_angles(self, base_cyclic):
        feedback = base_cyclic.RefreshFeedback()
        new_real_angles = [np.radians(a.position) for a in feedback.actuators]
        with self._data_lock:
            self.real_angles = new_real_angles

    # ------------------------------------------------------------------
    # PUBLIC API
    # ------------------------------------------------------------------

    def set_joint_angles(self, angles, gripper_percentage=None, wait=True):
        completion_event = threading.Event()
        safe_gripper = None
        if gripper_percentage is not None:
            safe_gripper = max(0.0, min(100.0, float(gripper_percentage))) / 100.0

        self.action_queue.put(
            {
                "type": "move",
                "angles": np.array(angles),
                "gripper": safe_gripper,
                "event": completion_event,
            }
        )

        if wait:
            completion_event.wait()

    def set_gripper(self, percentage: float, wait=True):
        completion_event = threading.Event()
        safe_ratio = max(0.0, min(100.0, float(percentage))) / 100.0

        self.action_queue.put(
            {"type": "grip", "value": safe_ratio, "event": completion_event}
        )

        if wait:
            completion_event.wait()

    def get_joint_angles(self):
        with self._data_lock:
            return list(self.real_angles)

    def open_gripper(self, wait=True):
        self.set_gripper(0.0, wait)

    def close_gripper(self, wait=True):
        self.set_gripper(100.0, wait)

    def set_torque(self, enable: bool):
        with self._data_lock:
            self._desired_admittance = not enable

    def stop(self):
        self._is_running = False
        try:
            with utilities.DeviceConnection.createTcpConnection(self.args) as router:
                base = BaseClient(router)
                base.Stop()
        except Exception:
            pass
        if self._thread is not None:
            self._thread.join()


class Kinova:
    # Pass the suction flag down from the main public wrapper
    def __init__(self, is_suction=False) -> None:
        self.base_kinova = BaseKinova(is_suction=is_suction)
        self.base_kinova.start()

    def set_joint_angles(self, angles, gripper_percentage=None, wait=True):
        self.base_kinova.set_joint_angles(angles, gripper_percentage, wait)

    def get_joint_angles(self):
        return self.base_kinova.get_joint_angles()

    def set_gripper(self, percentage: float, wait=True):
        self.base_kinova.set_gripper(percentage, wait)

    def open_gripper(self, wait=True):
        self.base_kinova.open_gripper(wait)

    def close_gripper(self, wait=True):
        self.base_kinova.close_gripper(wait)

    def set_torque(self, enable: bool):
        self.base_kinova.set_torque(enable)

    def stop(self):
        self.base_kinova.stop()

"""High-level orchestration for the ROS 2 data collection stack."""
from __future__ import annotations

import datetime
import logging
import os
import signal
import subprocess
import threading
import time
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Type
from zoneinfo import ZoneInfo

import serial

import rclpy
from composition_interfaces.srv import LoadNode, UnloadNode
from rclpy.parameter import Parameter as RclpyParameter
from rclpy.utilities import ok as rclpy_is_initialized  # ROS2 jazzy: is_initialized() -> ok()
from rosbag2_interfaces.srv import Pause as RecorderPause

from ament_index_python.packages import (
    PackageNotFoundError,
    get_package_share_directory,
)

from .log_aggregator import LogAggregator, SensorMetric, load_yaml, start_ros_launch
from .recorder_config import build_recorder_parameters

LOGGER = logging.getLogger(__name__)


class TeensyTrigger:
    """Simple helper to send trigger commands to the PPS Teensy."""

    def __init__(self, port: str, baudrate: int = 115200, timeout: float = 2.0):
        self.port = port
        self.baudrate = baudrate
        self.timeout = timeout
        self._break_keywords = ("started", "stopped", "status", "running", "already")

    def start(self) -> tuple[bool, str]:
        return self._send_command("START")

    def stop(self) -> tuple[bool, str]:
        return self._send_command("STOP")

    def status(self) -> tuple[bool, str]:
        return self._send_command("STATUS")

    def _send_command(self, command: str) -> tuple[bool, str]:
        try:
            with serial.Serial(self.port, self.baudrate, timeout=self.timeout) as ser:
                time.sleep(0.1)
                ser.reset_input_buffer()
                ser.write(f"{command}\n".encode())
                ser.flush()

                lines: List[str] = []
                deadline = time.time() + self.timeout
                while time.time() < deadline:
                    if ser.in_waiting:
                        raw = ser.readline().decode(errors="ignore").strip()
                        if raw:
                            lines.append(raw)
                            lowered = raw.lower()
                            if any(keyword in lowered for keyword in self._break_keywords):
                                break
                    else:
                        time.sleep(0.01)

                message = "; ".join(lines)
                return True, message if message else command
        except Exception as exc:  # pragma: no cover - hardware path
            return False, str(exc)


class ControllerState(str, Enum):
    INIT = "INIT"
    VALIDATING_SENSORS = "VALIDATING_SENSORS"
    READY = "READY"
    RECORDING = "RECORDING"
    FAILED = "FAILED"


class CollectionController:
    """Coordinate ROS launch lifecycle, bag recording, and monitoring."""

    def __init__(self, config_path: str):
        self.config_path = Path(config_path)
        self.config = load_yaml(self.config_path)
        self.state: ControllerState = ControllerState.INIT
        self.log_aggregator: Optional[LogAggregator] = None
        self.ros_process: Optional[subprocess.Popen] = None
        self.current_bag_info: Optional[Dict] = None
        self.last_failure: Optional[str] = None
        self.pps_config = self.config.get("pps", {})
        self.required_topics: List[str] = list(self.pps_config.get("required_topics", []))
        self.topic_check_interval = float(self.pps_config.get("topic_check_interval", 2.0))
        self.topic_ready_timeout = float(self.pps_config.get("topic_ready_timeout", 120.0))
        self.trigger_start_delay = float(self.pps_config.get("trigger_start_delay_s", 4.0))
        self.trigger_active = False
        self.trigger = TeensyTrigger(
            port=self.pps_config.get("serial_port", "/dev/teensy-pps"),
            baudrate=int(self.pps_config.get("serial_baudrate", 115200)),
            timeout=float(self.pps_config.get("serial_timeout", 2.0)),
        )

        recorder_config = self.config.get("recorder", {})
        self.component_container_name = recorder_config.get(
            "container_name", "sensor_container"
        )
        self.recorder_node_name = recorder_config.get("node_name", "recorder")
        self.recorder_include_hidden = bool(
            recorder_config.get("include_hidden_topics", True)
        )
        self.recorder_record_all = bool(recorder_config.get("record_all_topics", False))
        self.recorder_start_paused = bool(recorder_config.get("start_paused", False))
        self.recorder_storage_id = recorder_config.get("storage_id", "mcap")
        self.recorder_plugin = recorder_config.get(
            "plugin", "rosbag2_transport::Recorder"
        )
        self.recorder_package = recorder_config.get("package", "rosbag2_transport")
        self.recorder_component_id: Optional[int] = None
        self.component_service_timeout = float(
            self.config.get("component_service_timeout", 15.0)
        )
        self.enable_gps = self.config.get("enable_gps", True)
        self._component_service_lock = threading.Lock()

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------
    def startup_sequence(
        self,
        enable_gps: Optional[bool] = None,
        car_mode: bool = False,
    ):
        LOGGER.info("Starting collection controller sequence")
        self.state = ControllerState.VALIDATING_SENSORS
        system_cmd = self.config.get(
            "system_launch_cmd", "ros2 launch data_collect system.launch.py"
        )
        # Append enable_gps argument if specified (use parameter override or config value)
        enable_gps_value = enable_gps if enable_gps is not None else self.enable_gps
        enable_gps_str = "true" if enable_gps_value else "false"
        system_cmd = f"{system_cmd} enable_gps:={enable_gps_str}"
        # Append car_mode argument for socket CAN interface
        car_mode_str = "true" if car_mode else "false"
        system_cmd = f"{system_cmd} car_mode:={car_mode_str}"
        self.ros_process = start_ros_launch(system_cmd)

        self.log_aggregator = LogAggregator(self.config_path)
        self.log_aggregator.attach_process(self.ros_process)
        self.log_aggregator.controller_callback = self._on_sensor_failure
        self.log_aggregator.start_sensor_polling()

        sensor_ready_timeout = self.config.get("sensor_ready_timeout", 120)
        if not self._wait_for_sensors_ok(sensor_ready_timeout):
            self.state = ControllerState.FAILED
            raise RuntimeError("Sensors failed to reach expected rates")

        self.state = ControllerState.READY
        LOGGER.info("Collection controller ready")

    def shutdown(self):
        LOGGER.info("Shutting down controller")
        if self.recorder_component_id is not None:
            try:
                self.stop_recording()
            except Exception as exc:  # pragma: no cover - shutdown path
                LOGGER.error("Error stopping recording: %s", exc)

        if self.ros_process and self.ros_process.poll() is None:
            LOGGER.info("Stopping ROS system launch")
            self.ros_process.send_signal(signal.SIGINT)
            try:
                self.ros_process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.ros_process.kill()

        if self.log_aggregator:
            self.log_aggregator.stop()

        self._stop_trigger(suppress_errors=True)
        self.state = ControllerState.INIT

    # ------------------------------------------------------------------
    # Recording control
    # ------------------------------------------------------------------
    def start_recording(self, bag_type: str = "data") -> Dict:
        if self.state != ControllerState.READY:
            raise ValueError(f"Cannot start recording while in {self.state}")
        if self.recorder_component_id is not None:
            raise ValueError("Recording already in progress")
        if self.required_topics:
            LOGGER.info(
                "Waiting for required topics before recording: %s",
                ", ".join(self.required_topics),
            )
            if not self._wait_for_required_topics(self.topic_ready_timeout):
                raise RuntimeError("Timed out waiting for required topics to publish")

        bag_name = self._make_bag_filename(bag_type)
        output_dir = Path(self.config.get("bag_output_dir", "/rosbags"))
        output_dir.mkdir(parents=True, exist_ok=True)
        bag_path = str(output_dir / bag_name)

        recorder_params = build_recorder_parameters(
            bag_path,
            start_paused=self.recorder_start_paused,
            include_hidden_topics=self.recorder_include_hidden,
            record_all=self.recorder_record_all,
            storage_id=self.recorder_storage_id,
        )

        component_id = self._load_recorder_component(recorder_params)
        self.recorder_component_id = component_id

        self.current_bag_info = {
            "name": bag_name,
            "path": bag_path,
            "type": bag_type,
            "start_time": time.time(),
            "component_id": component_id,
        }
        self.state = ControllerState.RECORDING
        LOGGER.info("Recording started: %s", bag_name)

        # Wait for bag file to be created before triggering
        if not self._wait_for_bag_file(bag_path, timeout=10.0):
            LOGGER.error("Bag file not created within timeout")
            self._unload_recorder_component(suppress_errors=True)
            self.current_bag_info = None
            self.recorder_component_id = None
            self.state = ControllerState.READY
            raise RuntimeError("Bag file was not created by recorder")

        try:
            self._arm_trigger()
        except Exception:
            LOGGER.exception("Failed to start PPS trigger; stopping recording")
            self._unload_recorder_component(suppress_errors=True)
            self.current_bag_info = None
            self.recorder_component_id = None
            self.state = ControllerState.READY
            raise

        return self.current_bag_info

    def stop_recording(self) -> Dict:
        if self.recorder_component_id is None or self.current_bag_info is None:
            raise ValueError("Not currently recording")

        # Stop the PPS trigger first to stop new data from arriving
        self._stop_trigger(suppress_errors=True)

        # Pause the recorder to stop accepting new messages and flush buffers
        self._pause_recorder(suppress_errors=True)

        # Give the recorder time to flush any pending writes
        LOGGER.info("Waiting for recorder to flush pending writes...")
        time.sleep(1.0)

        # Now unload the component
        self._unload_recorder_component()

        bag_info = dict(self.current_bag_info)
        bag_info["end_time"] = time.time()
        bag_info["duration"] = bag_info["end_time"] - bag_info["start_time"]
        self.current_bag_info = None
        self.state = ControllerState.READY
        LOGGER.info("Recording stopped: %s", bag_info["name"])
        return bag_info

    # ------------------------------------------------------------------
    # Status helpers
    # ------------------------------------------------------------------
    def get_status(self) -> Dict:
        disk_info = self._get_disk_space()
        sensors = self.log_aggregator.get_status_snapshot() if self.log_aggregator else None
        status = {
            "state": self.state,
            "sensors": sensors,
            "recording": self.current_bag_info,
            "disk_space": disk_info,
            "last_failure": self.last_failure,
            "pps_trigger_active": self.trigger_active,
        }
        if self.required_topics:
            status["required_topics"] = self.required_topics
        return status

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _wait_for_sensors_ok(self, timeout: float) -> bool:
        start = time.time()
        last_log = 0.0
        while time.time() - start < timeout:
            if not self.log_aggregator:
                return False
            snapshot = self.log_aggregator.get_status_snapshot()
            sensors = snapshot.get("sensors", {})
            required_names = [
                name
                for name, sensor in self.log_aggregator.sensors.items()
                if sensor.health_required
            ]
            if not required_names:
                return True
            
            # Log sensor status every 5 seconds
            now = time.time()
            if now - last_log >= 5.0:
                ok_sensors = [
                    name for name in required_names
                    if sensors.get(name, {}).get("status") in ("ok", "warning")
                ]
                pending_sensors = []
                for name in required_names:
                    status = sensors.get(name, {}).get("status")
                    if status not in ("ok", "warning"):
                        measured = sensors.get(name, {}).get("measured")
                        if measured is not None:
                            pending_sensors.append(f"{name} ({measured:.2f})")
                        else:
                            pending_sensors.append(f"{name} (no data)")
                
                if ok_sensors:
                    LOGGER.info("Sensors meeting spec: %s", ", ".join(ok_sensors))
                if pending_sensors:
                    LOGGER.info("Sensors pending: %s", ", ".join(pending_sensors))
                last_log = now
            
            if sensors and all(
                sensors.get(name, {}).get("status") in ("ok", "warning")
                for name in required_names
            ):
                LOGGER.info("All required sensors ready")
                return True
            time.sleep(1)
        return False

    def _wait_for_required_topics(self, timeout: float) -> bool:
        if not self.required_topics:
            return True
        deadline = time.time() + timeout
        remaining: List[str] = []
        while time.time() < deadline:
            remaining = [
                topic for topic in self.required_topics if not self._topic_has_publishers(topic)
            ]
            if not remaining:
                LOGGER.info("All required topics are publishing")
                return True
            LOGGER.debug("Waiting on topics: %s", ", ".join(remaining))
            time.sleep(self.topic_check_interval)
        LOGGER.error("Timed out waiting for topics: %s", ", ".join(remaining))
        return False

    def _topic_has_publishers(self, topic: str) -> bool:
        try:
            result = subprocess.run(
                ["ros2", "topic", "info", topic],
                capture_output=True,
                text=True,
                timeout=5,
            )
        except Exception as exc:
            LOGGER.debug("Error checking topic %s: %s", topic, exc)
            return False
        if result.returncode != 0:
            return False
        for line in result.stdout.splitlines():
            stripped = line.strip()
            if stripped.lower().startswith("publisher count"):
                try:
                    count = int(stripped.split(":", 1)[1].strip())
                    return count > 0
                except (IndexError, ValueError):
                    return False
        return False

    def _arm_trigger(self):
        if self.trigger_start_delay > 0:
            LOGGER.info(
                "Waiting %.1fs before enabling PPS trigger", self.trigger_start_delay
            )
            time.sleep(self.trigger_start_delay)
        success, message = self.trigger.start()
        if not success:
            raise RuntimeError(f"Failed to start PPS trigger: {message}")
        self.trigger_active = True
        LOGGER.info("PPS trigger started: %s", message)

    def _stop_trigger(self, suppress_errors: bool = False):
        if not self.trigger_active:
            return
        success, message = self.trigger.stop()
        if success:
            LOGGER.info("PPS trigger stopped: %s", message)
            self.trigger_active = False
            return
        if suppress_errors:
            LOGGER.warning("Failed to stop PPS trigger: %s", message)
        else:
            raise RuntimeError(f"Failed to stop PPS trigger: {message}")

    # ------------------------------------------------------------------
    # Recorder helpers
    # ------------------------------------------------------------------
    def _pause_recorder(self, suppress_errors: bool = False):
        """Pause the recorder to flush buffers before unloading."""
        pause_service = f"/{self.recorder_node_name}/pause"

        # Try service-based pause first
        try:
            self._call_pause_service(pause_service)
            LOGGER.info("Recorder paused via service")
            return
        except Exception as exc:
            LOGGER.warning("Service-based pause failed: %s, trying CLI", exc)

        # Fall back to CLI-based pause
        try:
            result = subprocess.run(
                ["ros2", "service", "call", pause_service, "rosbag2_interfaces/srv/Pause"],
                capture_output=True,
                text=True,
                timeout=5.0,
            )
            if result.returncode == 0:
                LOGGER.info("Recorder paused via CLI")
                return
            if not suppress_errors:
                LOGGER.warning("CLI pause returned non-zero: %s", result.stderr.strip())
        except subprocess.TimeoutExpired:
            if not suppress_errors:
                LOGGER.warning("Timeout calling pause service via CLI")
        except Exception as exc:
            if not suppress_errors:
                LOGGER.warning("Failed to pause recorder via CLI: %s", exc)

    def _call_pause_service(self, service_name: str):
        """Call the recorder pause service via rclpy."""
        with self._component_service_lock:
            newly_initialized = False
            if not rclpy_is_initialized():
                rclpy.init(args=None)
                newly_initialized = True
            node = rclpy.create_node("collection_controller_pause_client")
            try:
                client = node.create_client(RecorderPause, service_name)
                if not client.wait_for_service(timeout_sec=2.0):
                    raise RuntimeError(f"Pause service {service_name} not available")
                request = RecorderPause.Request()
                future = client.call_async(request)
                rclpy.spin_until_future_complete(node, future, timeout_sec=2.0)
                if not future.done():
                    raise RuntimeError("Pause service call timed out")
                if future.exception() is not None:
                    raise RuntimeError(f"Pause service call failed: {future.exception()}")
            finally:
                node.destroy_node()
                if newly_initialized:
                    rclpy.shutdown()

    def _wait_for_bag_file(self, bag_path: str, timeout: float) -> bool:
        """Wait for the bag directory and metadata file to be created."""
        LOGGER.info("Waiting for bag file to be created at %s", bag_path)
        bag_dir = Path(bag_path)
        metadata_yaml = bag_dir / "metadata.yaml"
        data_suffixes = (".mcap", ".db3", ".bag", ".sqlite3")
        
        deadline = time.time() + timeout
        while time.time() < deadline:
            if bag_dir.exists() and bag_dir.is_dir():
                # Check if metadata.yaml exists (indicates recorder has initialized)
                if metadata_yaml.exists():
                    LOGGER.info("Bag file created and metadata present")
                    return True

                # Fall back to detecting data files (e.g. MCAP) which are created before metadata
                for suffix in data_suffixes:
                    data_file = next(bag_dir.glob(f"*{suffix}"), None)
                    if data_file is None:
                        continue
                    try:
                        size_bytes = data_file.stat().st_size
                    except OSError:
                        continue
                    if size_bytes > 0:
                        LOGGER.info(
                            "Detected active bag data file %s (%.1f MB); proceeding",
                            data_file.name,
                            size_bytes / (1024 * 1024),
                        )
                        return True
            time.sleep(0.5)

        LOGGER.error("Bag file not found after %.1fs", timeout)
        return False

    def _container_target(self) -> str:
        if self.component_container_name.startswith("/"):
            return self.component_container_name
        return f"/{self.component_container_name}"

    def _load_recorder_component(self, params: Dict) -> int:
        return self._load_component_via_service(params)

    def _load_component_via_service(self, params: Dict) -> int:
        request = LoadNode.Request()
        request.package_name = self.recorder_package
        request.plugin_name = self.recorder_plugin
        request.node_name = self.recorder_node_name
        if hasattr(request, "node_namespace"):
            request.node_namespace = ""
        if hasattr(request, "log_level"):
            request.log_level = ""
        if hasattr(request, "parameters"):
            request.parameters = self._build_parameter_msgs(params)
        if hasattr(request, "extra_arguments"):
            request.extra_arguments = []

        response = self._call_component_service(
            LoadNode, self._load_service_name(), request
        )
        if not getattr(response, "success", False):
            raise RuntimeError(
                f"Failed to load recorder component: {getattr(response, 'error_message', 'unknown error')}"
            )
        component_id = getattr(response, "unique_id", None)
        if component_id is None:
            raise RuntimeError("Recorder component load response missing unique_id")
        LOGGER.info(
            "Loaded recorder component %s into %s", component_id, self._container_target()
        )
        return int(component_id)

    def _unload_recorder_component(self, suppress_errors: bool = False):
        if self.recorder_component_id is None:
            return
        self._unload_component_via_service(suppress_errors)
        self.recorder_component_id = None

    def _unload_component_via_service(self, suppress_errors: bool):
        request = UnloadNode.Request()
        request.unique_id = int(self.recorder_component_id)
        response = self._call_component_service(
            UnloadNode, self._unload_service_name(), request
        )
        if not getattr(response, "success", False):
            message = getattr(response, "error_message", "unknown error")
            if suppress_errors:
                LOGGER.warning("Failed to unload recorder component: %s", message)
            else:
                raise RuntimeError(f"Failed to unload recorder component: {message}")
        else:
            LOGGER.info("Recorder component %s unloaded", self.recorder_component_id)

    def _flatten_params(self, params: Dict, prefix: str = "") -> Dict[str, object]:
        flat: Dict[str, object] = {}
        for key, value in params.items():
            name = f"{prefix}.{key}" if prefix else key
            if isinstance(value, dict):
                flat.update(self._flatten_params(value, name))
            else:
                flat[name] = value
        return flat

    def _build_parameter_msgs(self, params: Dict) -> List[object]:
        messages: List[object] = []
        for name, value in self._flatten_params(params).items():
            parameter = RclpyParameter(name=name, value=value)
            messages.append(parameter.to_parameter_msg())
        return messages

    def _load_service_name(self) -> str:
        return f"{self._container_target()}/_container/load_node"

    def _unload_service_name(self) -> str:
        return f"{self._container_target()}/_container/unload_node"

    def _call_component_service(self, service_type: Type, service_name: str, request):
        expected_type = getattr(service_type, "Request", None)
        if expected_type is not None and not isinstance(request, expected_type):
            raise TypeError("Mismatched service request type")
        with self._component_service_lock:
            newly_initialized = False
            if not rclpy_is_initialized():
                rclpy.init(args=None)
                newly_initialized = True
            node = rclpy.create_node("collection_controller_component_client")
            try:
                client = node.create_client(service_type, service_name)
                if not client.wait_for_service(timeout_sec=self.component_service_timeout):
                    raise RuntimeError(f"Service {service_name} not available")
                future = client.call_async(request)
                rclpy.spin_until_future_complete(
                    node, future, timeout_sec=self.component_service_timeout
                )
                if not future.done():
                    raise RuntimeError(
                        f"Service call to {service_name} timed out after {self.component_service_timeout}s"
                    )
                if future.exception() is not None:
                    raise RuntimeError(
                        f"Service call to {service_name} failed: {future.exception()}"
                    )
                return future.result()
            finally:
                node.destroy_node()
                if newly_initialized:
                    rclpy.shutdown()

    def _on_sensor_failure(self, sensor_name: str, sensor: SensorMetric):
        measured = sensor.measured if sensor.measured is not None else -1
        if not sensor.health_required:
            LOGGER.warning(
                "Optional sensor %s reported failure (expected %.2f, measured %.2f) -- ignoring health gate",
                sensor_name,
                sensor.expected,
                measured,
            )
            return
        LOGGER.error(
            "Sensor %s failed (expected %.2f, measured %.2f)",
            sensor_name,
            sensor.expected,
            measured,
        )
        self.last_failure = sensor_name
        if self.state == ControllerState.RECORDING:
            try:
                self.stop_recording()
            finally:
                self.state = ControllerState.FAILED

    def _make_bag_filename(self, bag_type: str) -> str:
        est = ZoneInfo("America/New_York")
        timestamp = datetime.datetime.now(est).strftime("rosbag2_%Y_%m_%d-%H_%M_%S")
        prefix_map = self.config.get(
            "bag_prefix_map", {"data": "", "cam_cal": "cam_cal", "imu_cal": "imu_cal"}
        )
        prefix = prefix_map.get(bag_type, "")
        if prefix:
            return f"{prefix}_{timestamp}"
        return timestamp

    def _get_disk_space(self) -> Dict[str, str]:
        target = self.config.get("disk_check_path", "/")
        result = subprocess.run(["df", "-h", target], capture_output=True, text=True)
        lines = result.stdout.strip().splitlines()
        if len(lines) < 2:
            return {}
        parts = lines[1].split()
        if len(parts) < 5:
            return {}
        return {
            "filesystem": parts[0],
            "total": parts[1],
            "used": parts[2],
            "available": parts[3],
            "percent": parts[4],
            "mountpoint": parts[-1],
        }


def build_controller(config_path: Optional[str] = None) -> CollectionController:
    default_path = None
    try:
        share_dir = Path(get_package_share_directory("data_collect"))
        default_path = share_dir / "config" / "sensors_config.yaml"
    except PackageNotFoundError:
        default_path = (
            Path(__file__).resolve().parent / "config" / "sensors_config.yaml"
        )
    config_path = config_path or os.environ.get("DATA_COLLECT_CONFIG", str(default_path))
    return CollectionController(config_path=config_path)

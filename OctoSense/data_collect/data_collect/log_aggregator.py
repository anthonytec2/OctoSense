"""Utilities for aggregating ROS 2 launch logs and extracting sensor metrics."""
from __future__ import annotations

import logging
import os
import re
import signal
import subprocess
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple, Union

import yaml

LOGGER = logging.getLogger(__name__)


RE_EVENT = re.compile(
    r"\[.*\]\s*\[INFO\].*\[(?P<name>event_camera_\d+)\]:\s*"
    r"bw in:\s*(?P<mbs>[\d\.]+)\s*MB/s",
)
RE_IR = re.compile(
    r"\[.*\]\s*\[INFO\].*\[(?P<serial>\d+)\]:\s*"
    r"rate \[Hz\] in\s*(?P<hz>[\d\.]+)",
)
RE_CAM_FREQ = re.compile(
    r"\[.*\]\s*\[INFO\].*\[cam_sync\]: ------ frequency:\s*"
    r"(?P<hz>[\d\.]+)\s*Hz",
)
# `ros2 topic hz` prints an "average rate: N" line for any topic when interrupted.
RE_TOPIC_RATE = re.compile(r"average rate:\s*(?P<avg>[\d\.]+)")


@dataclass
class SensorMetric:
    name: str
    metric_type: str  # 'mbps' or 'hz'
    expected: float
    measured: Optional[float] = None
    timestamp: Optional[float] = None
    out_of_spec_since: Optional[float] = None
    health_required: bool = True
    min_threshold: Optional[float] = None  # If set, consider OK if measured >= min_threshold

    def delta_pct(self) -> Optional[float]:
        if self.measured is None:
            return None
        if self.expected == 0:
            return None
        return abs(self.measured - self.expected) / self.expected

    def status(self, now: float, tolerance: float, timeout: float) -> str:
        if self.measured is None:
            return "unknown"

        # If min_threshold is set, consider OK if measured >= min_threshold
        if self.min_threshold is not None and self.measured >= self.min_threshold:
            return "ok"

        delta = self.delta_pct()
        if delta is not None and delta <= tolerance:
            return "ok"

        if self.out_of_spec_since is None:
            return "warning"

        if (now - self.out_of_spec_since) >= timeout:
            return "failed"

        return "warning"


def load_yaml(path: Union[str, Path]) -> Dict:
    with open(path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


class LogAggregator:
    """Tail ROS launch output, echo it, and parse sensor metrics."""

    def __init__(self, config_path: Union[str, Path]):
        self.config_path = Path(config_path)
        self.config = load_yaml(self.config_path)
        self.sensors: Dict[str, SensorMetric] = self._init_sensors()
        self.controller_callback: Optional[Callable[[str, SensorMetric], None]] = None
        self.process: Optional[subprocess.Popen] = None
        self._poll_threads: List[threading.Thread] = []
        self._stop_event = threading.Event()
        self.echo_ros_logs: bool = bool(self.config.get("echo_ros_logs", True))

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def attach_process(self, process: subprocess.Popen):
        self.process = process
        threading.Thread(
            target=self._tail_stream,
            args=(process.stdout, "stdout"),
            daemon=True,
        ).start()
        threading.Thread(
            target=self._tail_stream,
            args=(process.stderr, "stderr"),
            daemon=True,
        ).start()

    def stop(self):
        self._stop_event.set()
        for thread in self._poll_threads:
            if thread.is_alive():
                thread.join(timeout=1)

    def start_sensor_polling(self):
        """Start a `ros2 topic hz` polling thread for every sensor with a query_topic."""
        for name, cfg in self.config.get("sensors", {}).items():
            if not cfg.get("query_topic"):
                continue
            thread = threading.Thread(
                target=self._poll_topic_rate,
                args=(name, dict(cfg)),
                daemon=True,
            )
            self._poll_threads.append(thread)
            thread.start()

    def _run_topic_hz(self, topic, window, sample_duration, timeout_s, env_overrides):
        """Run `ros2 topic hz <topic>` for one sample window; return (stdout, stderr, return_code).

        `ros2 topic hz` runs until interrupted, so the normal path is: let it sample for
        `sample_duration`, then SIGINT it (which makes it print the average and exit),
        escalating to kill if it ignores the interrupt.
        """
        cmd = ["ros2", "topic", "hz", topic]
        if window > 0:
            cmd += ["-w", str(window)]
        cmd_env = os.environ.copy()
        cmd_env.update(env_overrides)
        process = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env=cmd_env,
        )
        try:
            stdout, stderr = process.communicate(timeout=sample_duration)
        except subprocess.TimeoutExpired:
            process.send_signal(signal.SIGINT)
            try:
                stdout, stderr = process.communicate(timeout=timeout_s)
            except subprocess.TimeoutExpired:
                process.kill()
                stdout, stderr = process.communicate()
        return_code = process.returncode
        if return_code is None:
            process.kill()
            stdout, stderr = process.communicate()
            return_code = process.returncode
        return stdout, stderr, return_code

    def _poll_topic_rate(self, name: str, cfg: Dict):
        topic = cfg["query_topic"]
        interval = float(cfg.get("query_interval", 15))
        initial_interval = float(cfg.get("initial_query_interval", 2))
        initial_poll_count = int(cfg.get("initial_poll_count", 5))
        timeout_s = float(cfg.get("query_timeout_s", 10))
        window = int(cfg.get("query_window", 5))
        sample_duration = float(cfg.get("query_sample_duration_s", 3))
        env_overrides = cfg.get("query_env", {})
        if not isinstance(env_overrides, dict):
            env_overrides = {}
        # Optional sensors (health_required: false) report 0 on a missed sample so the
        # UI shows them inactive. Required sensors keep their last reading instead, so a
        # transient `ros2 topic hz` hiccup can't trip the health gate.
        zero_on_missing = not bool(cfg.get("health_required", True))

        LOGGER.info(
            "Starting rate polling for %s on %s (initial: every %.1fs for %d polls, then every %.1fs)",
            name, topic, initial_interval, initial_poll_count, interval,
        )
        poll_count = 0
        while not self._stop_event.is_set():
            current_interval = initial_interval if poll_count < initial_poll_count else interval
            poll_count += 1
            try:
                stdout, stderr, return_code = self._run_topic_hz(
                    topic, window, sample_duration, timeout_s, env_overrides
                )
                # Exit code 2 is "no publishers"; 128+SIGINT is our own interrupt.
                if return_code not in (0, 2, 128 + signal.SIGINT):
                    LOGGER.warning(
                        "%s poll exited with %s: %s",
                        name, return_code, (stderr or "no stderr").strip(),
                    )
                match = RE_TOPIC_RATE.search(stdout)
                if match:
                    self._update_single_sensor(name, float(match.group("avg")))
                else:
                    LOGGER.debug("%s poll: no rate in output: %s", name, stdout.strip() or "<empty>")
                    if zero_on_missing:
                        self._update_single_sensor(name, 0.0)
            except Exception as exc:
                LOGGER.error("%s poll failed: %s", name, exc)
                if zero_on_missing:
                    self._update_single_sensor(name, 0.0)
            finally:
                self._stop_event.wait(current_interval)

    def get_status_snapshot(self) -> Dict:
        now = time.time()
        sensors = {}
        for name, sensor in self.sensors.items():
            sensors[name] = {
                "expected": sensor.expected,
                "measured": sensor.measured,
                "metric_type": sensor.metric_type,
                "delta_pct": sensor.delta_pct(),
                "status": sensor.status(
                    now,
                    self.config.get("tolerance_pct", 0.1),
                    self.config.get("failure_timeout_s", 80),
                ),
                "out_of_spec_duration": (
                    now - sensor.out_of_spec_since
                    if sensor.out_of_spec_since
                    else None
                ),
                "timestamp": sensor.timestamp,
                "health_required": sensor.health_required,
            }
        return {"sensors": sensors, "timestamp": now}

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _init_sensors(self) -> Dict[str, SensorMetric]:
        sensors_cfg = self.config.get("sensors", {})
        sensors: Dict[str, SensorMetric] = {}
        for name, cfg in sensors_cfg.items():
            min_threshold = cfg.get("min_threshold")
            sensors[name] = SensorMetric(
                name=name,
                metric_type=cfg.get("metric", "hz"),
                expected=float(cfg.get("expected", 0.0)),
                health_required=bool(cfg.get("health_required", True)),
                min_threshold=float(min_threshold) if min_threshold is not None else None,
            )
        return sensors

    def _tail_stream(self, stream, label: str):
        if stream is None:
            return
        for line in iter(stream.readline, ""):
            if self._stop_event.is_set():
                break
            line = line.rstrip()
            if not line:
                continue
            if self.echo_ros_logs:
                LOGGER.info("[roslaunch/%s] %s", label, line)
            for sensor_name, value in self._parse_metric(line) or []:
                self._update_single_sensor(sensor_name, value)
        LOGGER.info("%s stream closed", label)

    def _parse_metric(self, line: str) -> Optional[List[Tuple[str, float]]]:
        line = line.strip()

        match = RE_EVENT.search(line)
        if match:
            return [(match.group("name"), float(match.group("mbs")))]

        match = RE_IR.search(line)
        if match:
            serial = match.group("serial")
            for name, cfg in self.config.get("sensors", {}).items():
                pattern = cfg.get("log_pattern")
                if pattern and serial in pattern:
                    return [(name, float(match.group("hz")))]

        match = RE_CAM_FREQ.search(line)
        if match:
            hz = float(match.group("hz"))
            cam_targets = [
                name
                for name, cfg in self.config.get("sensors", {}).items()
                if cfg.get("log_pattern") == "cam_sync"
            ]
            if not cam_targets:
                cam_targets = ["cam0", "cam1"]
            return [(target, hz) for target in cam_targets]

        return None

    def _update_single_sensor(self, name: str, value: float):
        sensor = self.sensors.get(name)
        if not sensor:
            return

        now = time.time()
        prev_status = sensor.status(
            now,
            self.config.get("tolerance_pct", 0.1),
            self.config.get("failure_timeout_s", 80),
        )

        sensor.measured = value
        sensor.timestamp = now

        tolerance = self.config.get("tolerance_pct", 0.1)
        # If min_threshold is set and met, sensor is in spec
        if sensor.min_threshold is not None and sensor.measured >= sensor.min_threshold:
            sensor.out_of_spec_since = None
        elif sensor.delta_pct() is not None and sensor.delta_pct() > tolerance:
            sensor.out_of_spec_since = sensor.out_of_spec_since or now
        else:
            sensor.out_of_spec_since = None

        new_status = sensor.status(
            now,
            tolerance,
            self.config.get("failure_timeout_s", 80),
        )

        if (
            sensor.health_required
            and self.controller_callback
            and prev_status != "failed"
            and new_status == "failed"
        ):
            try:
                self.controller_callback(name, sensor)
            except Exception as exc:
                LOGGER.error("Controller callback failed: %s", exc)


def start_ros_launch(launch_cmd: str) -> subprocess.Popen:
    """Start a ros2 launch command while capturing stdout/stderr."""
    LOGGER.info("Starting ROS launch: %s", launch_cmd)
    # Source ROS environment before running the command
    ros_distro = os.environ.get("ROS_DISTRO", "jazzy")
    wrapped_cmd = f"""
        set -e
        source /opt/ros/{ros_distro}/setup.zsh
        if [ -f /catkin_ws/install/setup.zsh ]; then
            source /catkin_ws/install/setup.zsh
        fi
        {launch_cmd}
    """
    return subprocess.Popen(
        wrapped_cmd,
        shell=True,
        executable="/bin/zsh",
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,
        universal_newlines=True,
    )

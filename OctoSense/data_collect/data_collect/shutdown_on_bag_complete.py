"""
Helper node that shuts down the offline-odometry launch when RKO-LIO has
*finished processing* the bag

DRAIN-AWARE SHUTDOWN:
  1. Wait for /rko_lio/bag_progress to reach 100% (reading complete).
  2. THEN keep watching /rko_lio/odometry. While the processor drains the
     buffer it keeps publishing odom. Once no new odom arrives for
     DRAIN_IDLE_SECONDS, the buffer is empty -> processing is done -> SIGINT.
  3. MAX_DRAIN_SECONDS is an absolute safety cap so we never hang forever.
"""

import os
import signal
import time
from typing import Any

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray
from nav_msgs.msg import Odometry

# After reading hits 100%, declare processing finished once no new odometry
# message has arrived for this many seconds (the async buffer has drained).
DRAIN_IDLE_SECONDS = 10.0
# Absolute safety cap measured from the moment reading hit 100%, so a missing
# odom stream can never hang the job indefinitely.
MAX_DRAIN_SECONDS = 300.0


class BagProgressMonitor(Node):
    def __init__(self) -> None:
        super().__init__("bag_progress_monitor")
        self._sub = self.create_subscription(
            Float32MultiArray,
            "/rko_lio/bag_progress",
            self._on_progress,
            10,
        )
        
        self._odom_sub = self.create_subscription(
            Odometry,
            "/rko_lio/odometry",
            self._on_odom,
            50,
        )
        self._shutdown_sent = False
        self._reading_done = False
        self._reading_done_t: float | None = None
        self._last_odom_t: float | None = None
        self._last_logged_pct = -10.0  # ensure first message is logged
        # Poll the drain condition once a second.
        self._timer = self.create_timer(1.0, self._check_drain)
        self.get_logger().info(
            "BagProgressMonitor starts; waiting for /rko_lio/bag_progress (drain-aware)..."
        )

    def _extract_progress(self, msg: Any) -> tuple[float, float] | None:
        """Extract (percent_complete, seconds_remaining) from the 2-element message."""
        data = msg.data
        if len(data) < 2:
            self.get_logger().warn(f"Unexpected data length: {len(data)}")
            return None
        return (float(data[0]), float(data[1]))

    def _on_odom(self, _msg: Any) -> None:
        # Mark the last time the processor emitted odometry. Used to detect when
        # the async buffer has drained (odom goes silent) after reading is done.
        self._last_odom_t = time.monotonic()

    def _on_progress(self, msg: Any) -> None:
        if self._shutdown_sent or self._reading_done:
            return

        progress = self._extract_progress(msg)
        if progress is None:
            return
        pct, seconds_remaining = progress

        # Log progress every 10%
        if pct - self._last_logged_pct >= 10.0:
            minutes = int(seconds_remaining // 60)
            seconds = int(seconds_remaining % 60)
            time_str = f"{minutes}m {seconds}s" if minutes > 0 else f"{seconds}s"
            self.get_logger().info(f"Bag read progress: {pct:.1f}% (ETA: {time_str})")
            self._last_logged_pct = pct

        if pct >= 100.0:
            self._reading_done = True
            self._reading_done_t = time.monotonic()
            self.get_logger().info(
                "Bag fully READ (100%%). Waiting for RKO-LIO to finish PROCESSING "
                f"(SIGINT once /rko_lio/odometry idle for {DRAIN_IDLE_SECONDS:.0f}s, "
                f"or after {MAX_DRAIN_SECONDS:.0f}s cap)."
            )

    def _check_drain(self) -> None:
        if self._shutdown_sent or not self._reading_done:
            return
        now = time.monotonic()
        # Baseline for idle: last odom we saw, else the moment reading finished.
        baseline = self._last_odom_t if self._last_odom_t is not None else self._reading_done_t
        idle = now - baseline
        since_done = now - (self._reading_done_t or now)

        drained = idle >= DRAIN_IDLE_SECONDS
        timed_out = since_done >= MAX_DRAIN_SECONDS
        if not (drained or timed_out):
            return

        self._shutdown_sent = True
        if drained:
            reason = f"odometry idle {idle:.1f}s (buffer drained, processing complete)"
        else:
            reason = f"MAX_DRAIN_SECONDS cap ({MAX_DRAIN_SECONDS:.0f}s) reached"
        self.get_logger().info(f"Shutting down launch: {reason}. Sending SIGINT.")
        time.sleep(1)
        # SIGINT the parent (launch) for a clean shutdown of all nodes.
        os.kill(os.getppid(), signal.SIGINT)


def main(args: list[str] | None = None) -> None:
    rclpy.init(args=args)
    node = BagProgressMonitor()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        try:
            node.destroy_node()
        except Exception:
            pass
        try:
            if rclpy.ok():
                rclpy.shutdown()
        except Exception:
            pass


if __name__ == "__main__":
    main()

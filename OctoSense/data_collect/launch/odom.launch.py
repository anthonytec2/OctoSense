"""
Wrapper launch file for running RKO LIO offline odometry on a bag
and recording the resulting odometry into a new bag.

This launch file:
  1. Includes `rko_lio/odometry.launch.py`, forwarding the bag path and
     some useful defaults for offline processing.
  2. Starts a rosbag recorder that records:
       - /rko_lio/frame
       - /rko_lio/odometry
       - /rko_lio/local_map
     into a new MCAP bag under /tmp/rosbags.
  3. Starts a small helper process that subscribes to /rko_lio/bag_progress
     and, when the reported percentage reaches 100%%, sends SIGINT to the
     parent launch process to shut down all nodes.
"""

import os
import shutil
from datetime import datetime

from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    ExecuteProcess,
    IncludeLaunchDescription,
    OpaqueFunction,
    SetEnvironmentVariable,
)
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution

from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description() -> LaunchDescription:
    # -----------------------
    # Launch arguments
    # -----------------------
    bag_path = LaunchConfiguration("bag_path")
    config_file = LaunchConfiguration("config_file")
    rviz = LaunchConfiguration("rviz")
    mode = LaunchConfiguration("mode")

    # Default config file inside the installed data_collect package
    default_config = PathJoinSubstitution(
        [FindPackageShare("data_collect"), "config", "odom.yaml"]
    )

    declare_bag_path = DeclareLaunchArgument(
        "bag_path",
        description="Path to the *_lio bag produced by ci/odom/dump_lidar.py",
    )
    declare_config_file = DeclareLaunchArgument(
        "config_file",
        default_value=default_config,
        description="RKO LIO odometry config file.",
    )
    declare_rviz = DeclareLaunchArgument(
        "rviz",
        default_value="false",
        description="Whether to start RViz (true/false).",
    )
    declare_mode = DeclareLaunchArgument(
        "mode",
        default_value="offline",
        description="RKO LIO mode (online/offline).",
    )

    # -----------------------
    # Include main odometry launch
    # -----------------------
    rko_lio_share = FindPackageShare("rko_lio")
    odom_launch_path = PathJoinSubstitution(
        [rko_lio_share, "launch", "odometry.launch.py"]
    )

    odom_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(odom_launch_path),
        launch_arguments={
            "config_file": config_file,
            "rviz": rviz,
            "mode": mode,
            "bag_path": bag_path,
        }.items(),
    )

    # -----------------------
    # Rosbag recorder for odometry outputs
    # -----------------------

    def _get_processed_path(raw_bag_path: str):
        """
        Example:
            /data/rosbags/raw/car/sess8/rosbag2_2025_11_20-07_37_44
            -> /data/rosbags/processed/car/sess8/rosbag2_2025_11_20-07_37_44/
        """
        raw_bag_path = os.path.normpath(raw_bag_path)

        proc_root = os.environ.get("OCTO_PROCESSED_ROOT", "/data/rosbags/processed").rstrip("/")

        if "/raw/" in raw_bag_path:
            relative_path = raw_bag_path.split("/raw/", 1)[1]
            bag_name = os.path.basename(relative_path)
            output_path = f"{proc_root}/{relative_path}/"
        else:
            bag_name = os.path.basename(raw_bag_path)
            output_path = f"{proc_root}/{bag_name}/"

        return output_path, bag_name

    def _create_recorder(context, *args, **kwargs):
        # Resolve the actual bag path from the launch configuration
        raw_bag_path = bag_path.perform(context)
        output_path, _ = _get_processed_path(raw_bag_path)
        os.makedirs(output_path, exist_ok=True)

        # Record odometry outputs into the processed directory of this calibration bag
        output_dir = os.path.join(output_path, "odom")

        # Ensure we can record even if a previous run left this folder behind
        if os.path.exists(output_dir):
            print(f"[odom.launch] Removing existing odom bag directory: {output_dir}")
            shutil.rmtree(output_dir, ignore_errors=True)

        record_topics = [
            "/rko_lio/frame",
            "/rko_lio/odometry",
            "/rko_lio/odom_at_imu_rate",
        ]

        return [
            ExecuteProcess(
                cmd=[
                    "ros2",
                    "bag",
                    "record",
                    "-o",
                    output_dir,
                    "-s",
                    "mcap",
                ]
                + record_topics,
                output="screen",
                name="rko_lio_odom_recorder",
            )
        ]

    recorder = OpaqueFunction(function=_create_recorder)

    # -----------------------
    # Helper node: monitor /rko_lio/bag_progress and shutdown when complete
    # -----------------------
    bag_progress_monitor = Node(
        package="data_collect",
        executable="shutdown_on_bag_complete",
        name="bag_progress_monitor",
        output="screen",
    )

    zenohd = Node(
        package="rmw_zenoh_cpp",
        executable="rmw_zenohd",
        name="zenohd",
        output="screen",
    )

    return LaunchDescription(
        [
            SetEnvironmentVariable("RMW_IMPLEMENTATION", "rmw_zenoh_cpp"),
            zenohd,
            recorder,
            declare_bag_path,
            declare_config_file,
            declare_rviz,
            declare_mode,
            odom_launch,
            bag_progress_monitor,
        ]
    )

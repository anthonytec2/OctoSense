"""Launch the sensor stack with optional in-process rosbag recorder."""

import os
from datetime import datetime

import ament_index_python.packages
import yaml
from data_collect.recorder_config import build_recorder_parameters
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument as LaunchArg, IncludeLaunchDescription, LogInfo, OpaqueFunction, TimerAction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration as LaunchConfig, PathJoinSubstitution
from launch_ros.actions import ComposableNodeContainer, Node
from launch_ros.descriptions import ComposableNode
from launch_ros.substitutions import FindPackageShare
from launch.actions import SetEnvironmentVariable

# FLIR AX5 GigE camera configuration
FLIR_AX5_IP = "192.168.123.152"
FLIR_AX5_SERIAL = "73301414"
FLIR_AX5_STARTUP_DELAY = 3.0  # seconds to wait before starting camera driver

cam_parameters = {
    "debug": False,
    "compute_brightness": True,
    "dump_node_map": False,
    "adjust_timestamp": True,
    # Load UserSet0 instead of setting parameters individually
    "user_set_selector": "UserSet0",
    "user_set_load": "Yes",
}



def _resolve_bool(config_value: str) -> bool:
    return config_value.lower() in ("true", "1", "yes", "on")


def _determine_bag_path(context):
    bag_name = LaunchConfig("bag_name").perform(context).strip()
    bag_prefix = LaunchConfig("bag_prefix").perform(context).strip()
    bag_type = LaunchConfig("bag_type").perform(context).strip() or "data"
    bag_path_override = LaunchConfig("bag_path").perform(context).strip()
    bag_output_dir = LaunchConfig("bag_output_dir").perform(context).strip() or "/rosbags"

    if not bag_name:
        timestamp = datetime.utcnow().strftime("rosbag2_%Y_%m_%d-%H_%M_%S")
        prefix = bag_prefix or bag_type
        bag_name = f"{prefix + '_' if prefix else ''}{timestamp}"

    if bag_path_override:
        bag_path = bag_path_override.rstrip("/")
    else:
        bag_path = os.path.join(bag_output_dir, bag_name)

    parent = os.path.dirname(bag_path) or "."
    os.makedirs(parent, exist_ok=True)
    return bag_name, bag_path
def get_cam_params(context):
    camera_list = {
        "cam0": "24462485",
        "cam1": "25040024",
    }

    exp_ctrl_names = [cam + ".exposure_controller" for cam in camera_list.keys()]

    parameter_file = PathJoinSubstitution(
        [FindPackageShare("data_collect"), "config", "blackfly_s.yaml"]
    )
    exposure_controller_parameters = {
        "brightness_target": 60,
        "brightness_tolerance": 20,
        "max_exposure_time": 8000,
        "min_exposure_time": 1000,
        "max_gain": 40.0,
        "min_gain": 0.0,
        "gain_priority": False,
    }
    driver_parameters = {
        "cameras": list(camera_list.keys()),
        "exposure_controllers": exp_ctrl_names,
        "ffmpeg_image_transport.encoding": "hevc_nvenc",
    }
    for exp in exp_ctrl_names:
        driver_parameters.update(
            {exp + "." + k: v for k, v in exposure_controller_parameters.items()}
        )
    driver_parameters[exp_ctrl_names[0] + ".type"] = "master"
    driver_parameters[exp_ctrl_names[1] + ".type"] = "follower"
    driver_parameters[exp_ctrl_names[1] + ".master"] = exp_ctrl_names[0]

    cam_parameters["parameter_file"] = parameter_file
    for cam, serial in camera_list.items():
        cam_params = {cam + "." + k: v for k, v in cam_parameters.items()}
        cam_params[cam + ".serial_number"] = serial
        cam_params[cam + ".camerainfo_url"] = ""
        cam_params[cam + ".frame_id"] = cam
        cam_params["cam_sync." + cam + ".image_raw.ffmpeg.encoder"] = "hevc_vaapi"
        cam_params["cam_sync." + cam + ".image_raw.ffmpeg.pixel_format"] = "nv12"
        cam_params["cam_sync." + cam + ".image_raw.ffmpeg.bit_rate"] = 70000000
        cam_params["cam_sync." + cam + ".image_raw.ffmpeg.gop_size"] = 15
        cam_params["cam_sync." + cam + ".image_raw.ffmpeg.max_b_frames"] = 0
        cam_params["cam_sync." + cam + ".image_raw.ffmpeg.options"] = (
            "-bsf:v hevc_metadata=repeat_headers=1"
        )

        driver_parameters.update(cam_params)
        driver_parameters[cam + ".exposure_controller_name"] = (
            cam + ".exposure_controller"
        )

    return driver_parameters


def launch_setup(context, *args, **kwargs):
    enable_gps = _resolve_bool(LaunchConfig("enable_gps").perform(context))
    car_mode = _resolve_bool(LaunchConfig("car_mode").perform(context))
    record_on_start = _resolve_bool(LaunchConfig("record_on_start").perform(context))
    start_paused = _resolve_bool(LaunchConfig("start_paused").perform(context))
    recorder_node = None

    cam_0_str = "event_camera_0"
    cam_1_str = "event_camera_1"
    share_dir = ament_index_python.packages.get_package_share_directory(
        "metavision_driver"
    )
    bias_config = os.path.join(share_dir, "config", "silky_ev_cam.bias")

    cfg_dir = ament_index_python.packages.get_package_share_directory("data_collect")

    composable_nodes = [
        ComposableNode(
            package="spinnaker_synchronized_camera_driver",
            plugin="spinnaker_synchronized_camera_driver::SynchronizedCameraDriver",
            name="cam_sync",
            parameters=[get_cam_params(context)],
            extra_arguments=[{"use_intra_process_comms": True}],
        ),
        ComposableNode(
            package="vectornav",
            plugin="vectornav::Vectornav",
            name="vectornav",
            parameters=[
                PathJoinSubstitution(
                    [FindPackageShare("data_collect"), "config", "vectornav.yaml"]
                )
            ],
            extra_arguments=[{"use_intra_process_comms": True}],
        ),
        ComposableNode(
            package="vectornav",
            plugin="vectornav::VnSensorMsgs",
            name="vn_sensor_msgs",
            parameters=[
                PathJoinSubstitution(
                    [FindPackageShare("data_collect"), "config", "vn_sensormsgs.yaml"]
                )
            ],
            extra_arguments=[{"use_intra_process_comms": True}],
        ),
        ComposableNode(
            package="metavision_driver",
            plugin="metavision_driver::DriverROS2",
            name=cam_0_str,
            parameters=[
                {
                    "use_multithreading": True,
                    "bias_file": bias_config,
                    "camerainfo_url": "",
                    "frame_id": "cam_0",
                    "serial": "CenturyArks:silky_common_plugin:00001046",
                    "sync_mode": "primary",
                    "event_message_time_threshold": 1.0e-3,
                    "trigger_in_mode": "external",
                }
            ],
            remappings=[
                ("~/events", cam_0_str + "/events"),
                ("~/ready", cam_1_str + "/ready"),
            ],
            extra_arguments=[{"use_intra_process_comms": True}],
        ),
        ComposableNode(
            package="metavision_driver",
            plugin="metavision_driver::DriverROS2",
            name=cam_1_str,
            parameters=[
                {
                    "use_multithreading": True,
                    "bias_file": bias_config,
                    "camerainfo_url": "",
                    "frame_id": "cam_1",
                    "serial": "CenturyArks:silky_common_plugin:00001047",
                    "sync_mode": "secondary",
                    "event_message_time_threshold": 1.0e-3,
                    "trigger_in_mode": "external",
                }
            ],
            remappings=[("~/events", cam_1_str + "/events")],
            extra_arguments=[{"use_intra_process_comms": True}],
        ),
        ComposableNode(
            package="ouster_ros",
            plugin="ouster_ros::OusterSensor",
            name="os_sensor",
            parameters=[
                os.path.join(cfg_dir, "config", "ouster.yaml"),
                {"auto_start": True},
            ],
        ),
        ComposableNode(
            package="ouster_http_node",
            plugin="ouster_http_node::OusterHttpNode",
            name="ouster_http_node",
            parameters=[{"url": "http://192.168.123.29/api/v1/time/sensor"}],
        ),
    ]

    if record_on_start:
        _, bag_path = _determine_bag_path(context)
        recorder_parameters = build_recorder_parameters(
            bag_path,
            start_paused=start_paused,
            include_hidden_topics=True,
        )
        recorder_node = ComposableNode(
            package="rosbag2_transport",
            plugin="rosbag2_transport::Recorder",
            name="recorder",
            parameters=[recorder_parameters],
            extra_arguments=[{"use_intra_process_comms": True}],
        )
        composable_nodes.append(recorder_node)

    ntrip_client_launch = None
    if enable_gps:
        param_config = os.path.join(cfg_dir, "config", "zed_f9p.yaml")
        with open(param_config, "r") as f:
            gps_params = yaml.safe_load(f)["ublox_gps_node"]["ros__parameters"]
        composable_nodes.append(
            ComposableNode(
                package="ublox_gps",
                plugin="ublox_node::UbloxNode",
                name="ublox_gps_node",
                parameters=[gps_params],
                extra_arguments=[{"use_intra_process_comms": True}],
            )
        )

        ntrip_client_launch = IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                [
                    PathJoinSubstitution(
                        [FindPackageShare("data_collect"), "launch", "ntrip_client.launch.py"]
                    )
                ]
            ),
            launch_arguments={
                "host": os.environ.get("NTRIP_HOST", ""),
                "port": os.environ.get("NTRIP_PORT", "2101"),
                "mountpoint": os.environ.get("NTRIP_MOUNTPOINT", ""),
                "authenticate": "true",
                "username": os.environ.get("NTRIP_USERNAME", ""),
                "password": os.environ.get("NTRIP_PASSWORD", ""),
                "send_gga": "true",
                "gga_topic": "/nmea",
                "gga_interval": "1.0",
                "nmea_frame_id": "gps",
                "rtcm_message_package": "rtcm_msgs",
            }.items(),
        )

    socket_can_launch = None
    if car_mode:
        socket_can_launch = IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                [
                    PathJoinSubstitution(
                        [FindPackageShare("ros2_socketcan"), "launch", "socket_can_receiver.launch.py"]
                    )
                ]
            ),
            launch_arguments={
                "interface": "can0",
            }.items(),
        )

    container = ComposableNodeContainer(
        name="sensor_container",
        namespace="",
        package="rclcpp_components",
        executable="component_container",
        output="both",
        composable_node_descriptions=composable_nodes,
    )

    flir_ax5_node = Node(
        package="spinnaker_camera_driver",
        executable="camera_driver_node",
        name="flir_ax5",
        output="screen",
        parameters=[
            {
                "debug": False,
                "dump_node_map": False,
                "camera_type": "gige",
                "serial_number": FLIR_AX5_SERIAL,
                "frame_id": "flir_ax5",
                "camerainfo_url": "",
                "gige_device_ip_address": FLIR_AX5_IP,
                "parameter_file": PathJoinSubstitution(
                    [FindPackageShare("data_collect"), "config", "flir_ax5.yaml"]
                ),
                "pixel_format": "Mono8",
                "sensor_gain_mode": "HighGainMode",
                "nuc_mode": "Automatic",
                "sensor_dde_mode": "Automatic",
                "image_adjust_method": "PlateauHistogram",
                "sensor_video_standard": "PAL50HZ",
                "SyncMode": "Disabled",
                "video_orientation": "Normal",
                "flir_ax5.image_raw.ffmpeg.encoder": "hevc_vaapi",
                "flir_ax5.image_raw.ffmpeg.pixel_format": "nv12",
                "flir_ax5.image_raw.ffmpeg.bit_rate": 3000000,
                "flir_ax5.image_raw.ffmpeg.gop_size": 25,
                "flir_ax5.image_raw.ffmpeg.max_b_frames": 0,
                "flir_ax5.image_raw.ffmpeg.options": "-bsf:v hevc_metadata=repeat_headers=1",
            }
        ],
        # Respawn on crash with delay to allow camera to recover
        respawn=True,
        respawn_delay=5.0,
    )

    flir_ax5_delayed = TimerAction(
        period=FLIR_AX5_STARTUP_DELAY,
        actions=[flir_ax5_node],
    )

    zenohd = Node(
        package="rmw_zenoh_cpp",
        executable="rmw_zenohd",
        name="zenohd",
        output="screen",
    )

    nodes = [LogInfo(msg="Starting sensor stack")]
    if recorder_node is not None:
        nodes.append(LogInfo(msg="Recorder initialized inside sensor_container"))
    nodes.extend(
        [
            container,
            flir_ax5_delayed,   # Then start camera driver with delay
            zenohd,
        ]
    )
    if ntrip_client_launch is not None:
        nodes.append(ntrip_client_launch)
    if socket_can_launch is not None:
        nodes.append(socket_can_launch)

    return nodes


def generate_launch_description():
    return LaunchDescription(
        [
            SetEnvironmentVariable(
                "RMW_IMPLEMENTATION", "rmw_zenoh_cpp"
            ),
            SetEnvironmentVariable(
                "ZENOH_CONFIG_OVERRIDE",
                "transport/shared_memory/enabled=true",
            ),
            SetEnvironmentVariable(
                "ZENOH_SHM_ALLOC_SIZE",
                str(1024 * 1024 * 512),  # 512 MiB per process
            ),
            SetEnvironmentVariable(
                "ZENOH_SHM_MESSAGE_SIZE_THRESHOLD",
                "256",
            ),
            LaunchArg(
                "enable_gps",
                default_value="false",
                description="Whether to launch the optional GPS driver",
            ),
            LaunchArg(
                "car_mode",
                default_value="false",
                description="Whether to launch the socket CAN interface (ros2_socketcan on can0)",
            ),
            LaunchArg(
                "record_on_start",
                default_value="false",
                description="If true, start the rosbag recorder inside the sensor container",
            ),
            LaunchArg(
                "bag_name",
                default_value="",
                description="Manual bag folder name (overrides timestamp)",
            ),
            LaunchArg(
                "bag_prefix",
                default_value="",
                description="Prefix to prepend to bag names when auto-generating",
            ),
            LaunchArg(
                "bag_type",
                default_value="data",
                description="Logical bag type (data/cam_cal/imu_cal)",
            ),
            LaunchArg(
                "bag_path",
                default_value="",
                description="Absolute path override for bag destination",
            ),
            LaunchArg(
                "bag_output_dir",
                default_value="/rosbags",
                description="Base directory for new bags when bag_path not supplied",
            ),
            LaunchArg(
                "start_paused",
                default_value="false",
                description="Start recorder in paused mode when record_on_start is true",
            ),
            OpaqueFunction(function=launch_setup),
        ]
    )

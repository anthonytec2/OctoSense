"""Shared configuration helpers for rosbag recorder integration."""
from __future__ import annotations

from typing import Dict, List

RECORDER_TOPICS: List[str] = [
    "/ublox_gps_node/fix",
    "/ublox_gps_node/fix_velocity",
    "/ublox_gps_node/navpvt",
    "/navheading",
    "/navrelposned",
    "/timtm2",
    "/vectornav/imu",
    "/vectornav/magnetic",
    "/vectornav/imu_uncompensated",
    "/vectornav/pressure",
    "/vectornav/temperature",
    "/vectornav/time_pps",
    "/vectornav/raw/time",
    "/vectornav/time_syncin",
    "/cam_sync/cam0/image_raw/ffmpeg",
    "/cam_sync/cam1/image_raw/ffmpeg",
    "/cam_sync/cam0/meta",
    "/cam_sync/cam1/meta",
    "/cam_sync/cam0/camera_info",
    "/cam_sync/cam1/camera_info",
    "/flir_ax5/meta",
    "/flir_ax5/camera_info",
    "/flir_ax5/image_raw/ffmpeg",
    "/event_camera_0/events",
    "/event_camera_1/events",
    "/imu_packets",
    "/lidar_packets",
    "/ouster_time_status",
    "/lf/lowstate",
    "/lf/sportmodestate",
    "/lowcmd",
    "/wirelesscontroller",
    "/from_can_bus",
]


def build_recorder_parameters(
    bag_path: str,
    *,
    start_paused: bool = True,
    include_hidden_topics: bool = True,
    record_all: bool = False,
    storage_id: str = "mcap",
) -> Dict[str, Dict]:
    """Construct a rosbag2 recorder parameter dictionary."""

    return {
        "record": {
            "all": record_all,
            "include_hidden_topics": include_hidden_topics,
            "start_paused": start_paused,
            "topics": list(RECORDER_TOPICS),
        },
        "storage": {
            "uri": bag_path,
            "storage_id": storage_id,
            "max_cache_size": 2147483648,
        },
    }

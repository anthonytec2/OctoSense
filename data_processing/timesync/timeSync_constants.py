"""
Time synchronization constants.
Shared constants for time conversion, ROS2 topics, sensor parameters,
dataset roots, and CAN/radar decoding paths.
"""
import os
from pathlib import Path
from typing import Final

# ============================================================================
# Time Conversion Constants
# ============================================================================
NS_TO_S: Final[float] = 1e-9      # Nanoseconds to seconds
S_TO_NS: Final[float] = 1e9       # Seconds to nanoseconds
US_TO_NS: Final[float] = 1e3      # Microseconds to nanoseconds
US_TO_S: Final[float] = 1e-6      # Microseconds to seconds
S_TO_MS: Final[float] = 1e3       # Seconds to milliseconds


# Default event chunk size used for in-memory buffers and HDF5 chunk alignment.
EV_CHUNK: Final[int] = 50_000_000

# PPS coded-pulse block length: the 8-bit PPS counter wraps every 256 s, shared
# by all sensors driven off the same trigger.
BLOCK_DURATION: Final[float] = 256.0

# ============================================================================
# ROS2 Topic Names
# ============================================================================

# GPS Topics
GPS_TOPIC: Final[str] = '/ublox_gps_node/fix'
GPS_VEL_TOPIC: Final[str] = '/ublox_gps_node/fix_velocity'

# FLIR Camera Topics
FLIR_CAM0_TOPIC: Final[str] = "/cam_sync/cam0/image_raw/ffmpeg"
FLIR_CAM1_TOPIC: Final[str] = "/cam_sync/cam1/image_raw/ffmpeg"
FLIR_CAM0_TRIGGER_TOPIC: Final[str] = '/cam_sync/cam0/meta'
FLIR_CAM1_TRIGGER_TOPIC: Final[str] = '/cam_sync/cam1/meta'
FLIR_HZ: Final[float] = 100.0              # 100 frames in 1 second
FLIR_TOLERANCE: Final[float] = 5.0         # Allow jitter up to ±5 frames (covers 96-104 observed range)

# Event Camera Topics
EVENT_CAM0_TOPIC: Final[str] = '/event_camera_0/events'
EVENT_CAM1_TOPIC: Final[str] = '/event_camera_1/events'

# IMU Topics
IMU_TRIGGER_TOPIC: Final[str] = '/vectornav/raw/time'
IMU_TOPIC: Final[str] = "/vectornav/imu"
IMU_RAW_TOPIC: Final[str] = "/vectornav/imu_uncompensated"
IMU_HZ: Final[int] = 400 # 400 samples in 1 second
IMU_TOLERANCE: Final[float] = 5
PRESSURE_TOPIC: Final[str] = "/vectornav/pressure"
MAGNETIC_TOPIC: Final[str] = "/vectornav/magnetic"
TEMPERATURE_TOPIC: Final[str] = "/vectornav/temperature"

# Infrared Camera Topics
INFRARED_TRIGGER_TOPIC: Final[str] = '/flir_ax5/meta'
INFRARED_CAM_TOPIC: Final[str] = '/flir_ax5/image_raw/ffmpeg'

# ============================================================================
# OUSTER Camera Parameters
# ============================================================================
OUSTER_LIDAR_TOPIC: Final[str] = "/lidar_packets"
OUSTER_IMU_TOPIC: Final[str] = "/imu_packets"
OUSTER_TIME_STATUS_TOPIC: Final[str] = "/ouster_time_status"


CAN_TOPIC: Final[str] = "/from_can_bus"


# ============================================================================
# Dataset Roots & Calibration Paths
# ============================================================================
PROCESSED_ROOT: Final[str] = os.environ.get("OCTO_PROCESSED_ROOT", "/data/rosbags/processed").rstrip("/")
RAW_ROOT: Final[str] = os.environ.get("OCTO_RAW_ROOT", "/data/rosbags/raw").rstrip("/")

# Ouster metadata JSON (rev7_1 is used for all data), under <RAW_ROOT>/calibrations.
OUSTER_METADATA_REV7_1: Final[str] = os.path.join(RAW_ROOT, "calibrations", "rev7_1.json")

IMU_CAMCHAIN_BASE_DIR: Final[str] = f"{PROCESSED_ROOT}/calibrations/imu_cals"
CAM_CALS_BASE_DIR: Final[str] = f"{PROCESSED_ROOT}/calibrations/cam_cals"

# ============================================================================
# CAN / Radar Decoding
# ============================================================================
CAN_DBC_PATH: Final[str] = str(Path(__file__).parent / "mazda_2017.dbc")
CAN_RADAR_DBC_PATH: Final[str] = str(Path(__file__).parent / "mazda_radar.dbc")
# Radar CAN IDs in order — track 1..6 (RADAR_TRACK_361..366).
RADAR_TRACK_IDS: Final[tuple] = (865, 866, 867, 868, 869, 870)
OPENDBC_REF: Final[str] = os.environ.get("OPENDBC_REF", "master")
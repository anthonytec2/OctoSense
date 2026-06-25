"""
Main entry point for time synchronization processing.
Orchestrates calibration and processing of ROS2 bags.
"""
import sys
import numpy as np
import argparse
import logging
import os

from timeSyncUtil import BagTopicReader, get_processed_path, configure_file_logging
from timeSync_calibration import (
    CalibrationData,
    calibration_system_time, flir_time_offset, event_time_offset, imu_time_offset
)
from timeSync_constants import (
    FLIR_CAM0_TRIGGER_TOPIC, FLIR_CAM1_TRIGGER_TOPIC,
    EVENT_CAM0_TOPIC, EVENT_CAM1_TOPIC, IMU_TRIGGER_TOPIC,
    INFRARED_TRIGGER_TOPIC
)
from timeSync_process import process_bag

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(levelname)s:%(name)s:%(message)s')


def _restore(value):
    """Unwrap an .npz entry: 0-d object arrays (dicts / None) -> the original
    Python object; numeric arrays are returned as-is."""
    return value.item() if value.dtype == object else value


def get_time_offsets(bag_name) -> CalibrationData:
    """Extract time offset calibration from bag."""
    bag = BagTopicReader(bag_name)
    output_path, bn = get_processed_path(bag_name)
    os.makedirs(output_path, exist_ok=True)

    configure_file_logging(output_path)

    # Check if the time offset file exists
    time_offset_file = f"{output_path}/time_offset.npz"
    if os.path.exists(time_offset_file):
        # Load the saved offsets; keys match CalibrationData's fields exactly.
        data = np.load(time_offset_file, allow_pickle=True)
        fields = {k: _restore(data[k]) for k in data.files}
        logger.info("Loaded time offset objects from memory.")
        return CalibrationData(**fields)
    # Per-sensor calibration. Each spec maps the three CalibrationData fields
    # (device clock, main clock, filter) to its trigger topic, offset function,
    # and a label for the "missing topic" warning.
    calibrations = [
        ("device_clock_system", "master_clock_system", "system_filter",
         IMU_TRIGGER_TOPIC, lambda: calibration_system_time(bag),
         "IMU Topic for System Time Calibration"),
        ("device_clock_imu", "master_clock_imu", "imu_filter",
         IMU_TRIGGER_TOPIC, lambda: imu_time_offset(bag),
         "IMU Topic Messages"),
        ("device_clock_flir_0", "master_clock_flir_0", "flir_filter_0",
         FLIR_CAM0_TRIGGER_TOPIC, lambda: flir_time_offset(bag, FLIR_CAM0_TRIGGER_TOPIC),
         "FLIR 0 Topic Messages"),
        ("device_clock_flir_1", "master_clock_flir_1", "flir_filter_1",
         FLIR_CAM1_TRIGGER_TOPIC, lambda: flir_time_offset(bag, FLIR_CAM1_TRIGGER_TOPIC),
         "FLIR 1 Topic Messages"),
        ("device_clock_ev_0", "master_clock_ev_0", "event_filter_0",
         EVENT_CAM0_TOPIC, lambda: event_time_offset(bag, EVENT_CAM0_TOPIC),
         "EVENT 0 Topic Messages"),
        ("device_clock_ev_1", "master_clock_ev_1", "event_filter_1",
         EVENT_CAM1_TOPIC, lambda: event_time_offset(bag, EVENT_CAM1_TOPIC),
         "EVENT 1 Topic Messages"),
    ]

    # GPS is timestamped via the system-time calibration, so its fields stay None, legacy constraint
    fields = {"device_clock_gps": None, "master_clock_gps": None, "gps_filter": None}

    if INFRARED_TRIGGER_TOPIC in bag.topics_map:
        logger.info("Infrared camera will use system time calibration")

    for dc_key, mc_key, kf_key, topic, run_calibration, label in calibrations:
        if topic in bag.topics_map:
            fields[dc_key], fields[mc_key], fields[kf_key] = run_calibration()
        else:
            fields[dc_key] = fields[mc_key] = fields[kf_key] = None
            logger.warning("MISSING %s", label)
        logger.info('--------------------------------')

    np.savez(time_offset_file, **fields)
    logger.info(f"Saved time offset objects to {time_offset_file}")
    return CalibrationData(**fields)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process a rosbag file for time sync.")
    parser.add_argument("--bag", type=str, required=True, help="Path to the rosbag file")
    parser.add_argument(
        "--mode",
        type=str,
        choices=["calibration", "data"],
        default="calibration",
        help=(
            "Processing mode. "
            "'calibration' (default) runs on calibration bags and writes full raw sensor data. "
            "'data' runs on data-collection bags and can attach calibration metadata and use LIO bags."
        ),
    )
    parser.add_argument(
        "--cal-map",
        type=str,
        default=None,
        help="Optional path to cal_map.yaml describing which calibration bags/metadata were used (data mode).",
    )
    parser.add_argument(
        "--lio-bag",
        type=str,
        default=None,
        help=(
            "Optional path to a LIO rosbag used for Ouster data "
            "(e.g. <session>/rosbag2_XXXX_lio). "
            "If not provided in data mode, raw Ouster packets from the main bag are used (fallback)."
        ),
    )
    args = parser.parse_args()

    # Infer defaults to make CLI simpler:
    # - cal_map.yaml lives one directory up from the bag folder
    # - LIO bag lives in the processed directory with suffix `_lio`
    bag_dir = os.path.dirname(os.path.normpath(args.bag))
    default_cal_map = os.path.join(bag_dir, "cal_map.yaml")
    cal_map_path = args.cal_map or default_cal_map

    processed_path, bn = get_processed_path(args.bag)
    processed_parent = os.path.dirname(processed_path.rstrip("/"))
    default_lio_bag = os.path.join(processed_parent, f"{bn}_lio", "odom")
    lio_bag_path = args.lio_bag or default_lio_bag

    try:
        calibration_data = get_time_offsets(args.bag)

        process_bag(
            args.bag,
            calibration_data,
            mode=args.mode,
            cal_map_path=cal_map_path,
            lio_bag=lio_bag_path,
        )
    except AssertionError as e:
        logger.error(f"Assertion failed: {e}")
        sys.exit(1)
    except Exception as e:
        logger.exception(f"Error processing bag: {e}")
        sys.exit(1)

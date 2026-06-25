"""
Time synchronization processing module.
Handles parsing sensor data from ROS2 bags and writing to HDF5.
"""
import numpy as np
import logging
from event_camera_py import Decoder as ECDecoder
import h5py
import hdf5plugin
import subprocess
import multiprocessing
import time
import os
import json
import urllib.request
from pathlib import Path
from typing import Dict, Any, Optional, List
import yaml

from timeSyncUtil import BagTopicReader, map_events_to_global, get_processed_path, configure_file_logging
from ouster.sdk import client
from timeSync_constants import (
    NS_TO_S, US_TO_S, EV_CHUNK,
    GPS_TOPIC, GPS_VEL_TOPIC, FLIR_CAM0_TOPIC, FLIR_CAM1_TOPIC,
    FLIR_CAM0_TRIGGER_TOPIC, FLIR_CAM1_TRIGGER_TOPIC,
    EVENT_CAM0_TOPIC, EVENT_CAM1_TOPIC, OUSTER_LIDAR_TOPIC,
    IMU_TOPIC, IMU_RAW_TOPIC, PRESSURE_TOPIC, MAGNETIC_TOPIC, TEMPERATURE_TOPIC, IMU_TRIGGER_TOPIC,
    OUSTER_IMU_TOPIC,
    OUSTER_TIME_STATUS_TOPIC, INFRARED_CAM_TOPIC,
    CAN_TOPIC,
    PROCESSED_ROOT, RAW_ROOT, OUSTER_METADATA_REV7_1,
    IMU_CAMCHAIN_BASE_DIR, CAM_CALS_BASE_DIR,
    CAN_DBC_PATH, CAN_RADAR_DBC_PATH, RADAR_TRACK_IDS, OPENDBC_REF,
)
import cantools
import shutil
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")

os.environ.setdefault("BLOSC_NTHREADS", "8")


def _ensure_opendbc_dbc(path: str) -> None:
    """Download a CAN DBC from comma.ai/opendbc if it isn't already on disk."""
    if os.path.exists(path):
        return
    name = os.path.basename(path)
    url = f"https://raw.githubusercontent.com/commaai/opendbc/{OPENDBC_REF}/opendbc/dbc/{name}"
    logger.info(f"DBC {name} not found; downloading from opendbc@{OPENDBC_REF}")
    urllib.request.urlretrieve(url, path)


def _load_calibration_map(cal_map_path: Optional[str]) -> Dict[str, Any]:
    """Load cal_map.yaml if provided, else return empty dict.

    Expected structure (example):
        imu_cal: 'rosbag2_2025_11_20-09_02_23'
        rgb_cal: 'rosbag2_2025_11_20-09_05_55'
        lidar_cal: 'rev7_1.json'
    """
    if cal_map_path is None:
        return {}

    if not os.path.exists(cal_map_path):
        logger.warning(f"cal_map.yaml not found at {cal_map_path}, skipping calibration metadata logging.")
        return {}

    try:
        with open(cal_map_path, "r") as f:
            data = yaml.safe_load(f) or {}
        if not isinstance(data, dict):
            logger.warning(f"cal_map.yaml at {cal_map_path} did not contain a top-level mapping, got {type(data)}.")
            return {}
        return data
    except Exception as exc:
        logger.exception(f"Failed to load cal_map.yaml from {cal_map_path}: {exc}")
        return {}


# Kalibr camchain rostopic → short name / HDF5 sensor group, shared by the
# calibration-metadata writers below.
_CAM_TOPIC_TO_NAME = {
    "/flir_cam_right": "imgr",
    "/flir_cam_left": "imgl",
    "/event_camera_right": "evr",
    "/event_camera_left": "evl",
}
_CAM_TOPIC_TO_GROUP = {
    "/flir_cam_right": "img/right",
    "/flir_cam_left": "img/left",
    "/event_camera_right": "ev/right",
    "/event_camera_left": "ev/left",
}


def _load_camchain(path, label):
    """Load + validate a Kalibr camchain YAML. Returns the dict, or None (with a
    warning) if the file is missing, unreadable, or not a mapping."""
    if not os.path.exists(path):
        logger.warning(f"{label} camchain file not found at {path}; skipping calibration metadata.")
        return None
    try:
        with open(path, "r") as f:
            camchain = yaml.safe_load(f) or {}
    except Exception as exc:
        logger.exception(f"Failed to load {label} camchain from {path}: {exc}")
        return None
    if not isinstance(camchain, dict):
        logger.warning(f"Unexpected {label} camchain format in {path}: {type(camchain)}")
        return None
    return camchain


def _iter_cams(camchain):
    """Yield (idx, cam_data) for each `camN` entry (N numeric) in a camchain."""
    for cam_key, cam_data in camchain.items():
        if isinstance(cam_data, dict) and cam_key.startswith("cam") and cam_key[3:].isdigit():
            yield int(cam_key[3:]), cam_data


def _collect_t_cam_imu(camchain):
    """Map camera index -> 4x4 T_cam_imu for every `camN` entry that has one."""
    return {idx: np.asarray(cd["T_cam_imu"], dtype=np.float64)
            for idx, cd in _iter_cams(camchain) if cd.get("T_cam_imu") is not None}


def _write_camchain(h5_file, calibration_grp, params_chain, t_cam_imu_by_idx, events_file, src_label):
    """Write per-camera calibration from a camchain into HDF5.

    Camera intrinsics/distortion/resolution and the inter-camera T_cn_cnm1
    transforms come from `params_chain`; the IMU->camera transforms come from
    `t_cam_imu_by_idx` (keyed by camera index, possibly from a different file).

    For each known camera, writes into its sensor group (event cameras go to the
    sibling events file):
        <group>/intrinsics (3x3), dist_coeffs (N,), resolution (2,)
    and into the calibration group, using short-name notation:
        <name>_T_imu          (from t_cam_imu_by_idx)
        <name>_T_<prev_name>  (neighbor T_cn_cnm1)
        <last_name>_T_<name>  (chain to the last camera)
    """
    # Collect camera entries + inter-camera transforms.
    cams = []
    T_between = {}  # idx -> 4x4 T_cn_cnm1 (cam_{idx-1} -> cam_idx)
    for idx, cam_data in _iter_cams(params_chain):
        cams.append({"idx": idx, "name": _CAM_TOPIC_TO_NAME.get(cam_data.get("rostopic")), "data": cam_data})
        T_neighbor = cam_data.get("T_cn_cnm1")
        if T_neighbor is not None:
            T_between[idx] = np.asarray(T_neighbor, dtype=np.float64)

    if not cams:
        logger.warning(f"No camera entries found in {src_label}")
        return

    last_idx = max(c["idx"] for c in cams)
    last_named = [c for c in cams if c["idx"] == last_idx and c["name"]]
    if not last_named:
        logger.warning(f"Could not resolve last camera with a known rostopic in {src_label}")
        return
    last_name = last_named[0]["name"]
    idx_to_name = {c["idx"]: c["name"] for c in cams if c["name"]}

    def compute_T_last_from(i):
        """4x4 transform taking coordinates from cam_i to the last camera."""
        T = np.eye(4, dtype=np.float64)
        for j in range(i + 1, last_idx + 1):
            T_neighbor = T_between.get(j)
            if T_neighbor is None:
                logger.warning(f"Missing T_cn_cnm1 for cam index {j} in {src_label}; "
                               f"cannot compute full transform from cam{i} to cam{last_idx}.")
                break
            T = T_neighbor @ T
        return T

    for cam in cams:
        idx, name, data = cam["idx"], cam["name"], cam["data"]
        group_path = _CAM_TOPIC_TO_GROUP.get(data.get("rostopic"))
        if not group_path:
            continue

        # Event-camera intrinsics live in the sibling events.h5 so event consumers
        # are self-contained; img/* stay in the main file. Extrinsics go to /calib.
        dst_file = events_file if (group_path.startswith("ev/") and events_file is not None) else h5_file
        cam_grp = dst_file.require_group(group_path)

        intrinsics = data.get("intrinsics")
        if isinstance(intrinsics, (list, tuple)) and len(intrinsics) == 4:
            fx, fy, cx, cy = [float(v) for v in intrinsics]
            K = np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float64)
            cam_grp.create_dataset("intrinsics", data=K, dtype=np.float64)

        dist = data.get("distortion_coeffs")
        if isinstance(dist, (list, tuple)) and len(dist) > 0:
            cam_grp.create_dataset("dist_coeffs", data=np.asarray(dist, dtype=np.float64), dtype=np.float64)

        res = data.get("resolution")
        if isinstance(res, (list, tuple)) and len(res) == 2:
            cam_grp.create_dataset("resolution", data=np.asarray(res, dtype=np.int32), dtype=np.int32)

        # `name` is always set when group_path is (same rostopic keys), but guard anyway.
        if not name:
            continue

        # IMU->camera transform.
        T_cam_imu = t_cam_imu_by_idx.get(idx)
        if T_cam_imu is not None and T_cam_imu.shape == (4, 4):
            calibration_grp.create_dataset(f"{name}_T_imu", data=T_cam_imu, dtype=np.float64)

        # Neighbor extrinsic (cam_idx relative to cam_{idx-1}).
        T_neighbor = data.get("T_cn_cnm1")
        if T_neighbor is not None:
            T_neighbor_arr = np.asarray(T_neighbor, dtype=np.float64)
            prev_name = idx_to_name.get(idx - 1)
            neigh_key = f"{name}_T_{prev_name}"
            # The last cam's neighbor transform (e.g. evl_T_evr) is the same physical
            # transform as the second-to-last cam's "to-last" transform below, so guard
            # against creating it twice.
            if T_neighbor_arr.shape == (4, 4) and prev_name and neigh_key not in calibration_grp:
                calibration_grp.create_dataset(neigh_key, data=T_neighbor_arr, dtype=np.float64)

        # Transform from this camera to the last camera in the chain.
        T_last_cam = np.eye(4, dtype=np.float64) if idx == last_idx else compute_T_last_from(idx)
        last_key = f"{last_name}_T_{name}"
        if last_key not in calibration_grp:
            calibration_grp.create_dataset(last_key, data=T_last_cam, dtype=np.float64)


def _write_merged_calibration_metadata(
    h5_file: h5py.File,
    calibration_grp: h5py.Group,
    rgb_cal_id: str,
    imu_cal_id: str,
    events_file: Optional[h5py.File] = None,
) -> None:
    """Write per-camera calibration merged from two camchains: all camera params
    (intrinsics/distortion/resolution/T_cn_cnm1) come from the RGB camchain, and
    only T_cam_imu comes from the IMU camchain (matched by camera index).

        RGB camchain: <processed_root>/calibrations/cam_cals/<rgb_cal_id>/calibration-camchain.yaml
        IMU camchain: <processed_root>/calibrations/imu_cals/<imu_cal_id>/calibration-camchain-imucam.yaml
    """
    rgb_path = os.path.join(CAM_CALS_BASE_DIR, rgb_cal_id, "calibration-camchain.yaml")
    imu_path = os.path.join(IMU_CAMCHAIN_BASE_DIR, imu_cal_id, "calibration-camchain-imucam.yaml")
    rgb_camchain = _load_camchain(rgb_path, "RGB")
    if rgb_camchain is None:
        return
    imu_camchain = _load_camchain(imu_path, "IMU")
    if imu_camchain is None:
        return
    _write_camchain(h5_file, calibration_grp, rgb_camchain, _collect_t_cam_imu(imu_camchain),
                    events_file, f"RGB camchain {rgb_path}")


def _write_lidar_extrinsics(
    h5_file: h5py.File,
    calibration_grp: h5py.Group,
    rgb_cal_id: str,
) -> None:
    """Load LiDAR↔camera extrinsics from lidar_calibration_results.yaml and store under /ouster/TF.

    Expects files at:
        <processed_root>/calibrations/cam_cals/<rgb_cal_id>/lidar_calibration_results.yaml

    The YAML stores cam_T_lidar (transform from LiDAR frame into camera frame).
    In practice we always use the left FLIR camera as the reference, so we store:

        /ouster/TF/imgl_T_ouster   = cam_T_lidar

    (single direction only, LiDAR → left-camera frame).
    """
    calib_path = os.path.join(
        CAM_CALS_BASE_DIR, rgb_cal_id, "lidar_calibration_results.yaml"
    )
    if not os.path.exists(calib_path):
        logger.warning(
            f"LiDAR calibration results not found at {calib_path}; skipping LiDAR extrinsics."
        )
        return

    try:
        with open(calib_path, "r") as f:
            calib = yaml.safe_load(f) or {}
    except Exception as exc:
        logger.exception(f"Failed to load LiDAR calibration from {calib_path}: {exc}")
        return

    try:
        R = np.asarray(calib["rotation_matrix"], dtype=np.float64).reshape(3, 3)
        t = np.asarray(calib["translation_vector"], dtype=np.float64).reshape(3)
    except Exception as exc:
        logger.exception(f"Malformed LiDAR calibration file {calib_path}: {exc}")
        return

    # Build homogeneous cam_T_ouster (cam_T_lidar) from rotation + translation.
    T_cam_ouster = np.eye(4, dtype=np.float64)
    T_cam_ouster[:3, :3] = R
    T_cam_ouster[:3, 3] = t

    ouster_grp = h5_file.require_group("ouster")
    ouster_grp.create_dataset("imgl_T_ouster", data=T_cam_ouster, dtype=np.float64)


def _write_ir_calibration(
    h5_file: h5py.File,
    calibration_grp: h5py.Group,
    rgb_cal_id: str,
) -> None:
    """Load infrared intrinsics + extrinsics from ir_calib_result.json and store them.

    Expects the per-session IR calibration (from the cam-lidar calibration
    step), copied into the cam_cal dir at:
        <processed_root>/calibrations/cam_cals/<rgb_cal_id>/ir_calib_result.json

    The FLIR A35 IR camera is not part of the Kalibr camchain (thermal can't
    decode the AprilGrid tags), so it is solved separately against the left-RGB
    board pose and chained through the LiDAR↔RGB extrinsic. Mirrors the
    RGB/LiDAR writers above so downstream tooling reads IR the same way:

        /infrared/intrinsics    (3x3, native 320x256)
        /infrared/dist_coeffs   (4,)  radtan [k1, k2, 0, 0]
        /infrared/resolution    (2,)  [320, 256]
        /calib/ir_T_imgl        (4x4, maps a left-RGB point into the IR frame)
        /calib/imgl_T_ir        (4x4, inverse — IR point into left-RGB frame)
        /ouster/ir_T_ouster     (4x4, maps an Ouster point into the IR frame)
    """
    result_path = os.path.join(CAM_CALS_BASE_DIR, rgb_cal_id, "ir_calib_result.json")
    if not os.path.exists(result_path):
        logger.warning(
            f"IR calibration result not found at {result_path}; skipping IR calibration metadata."
        )
        return

    try:
        with open(result_path, "r") as f:
            res = json.load(f)
    except Exception as exc:
        logger.exception(f"Failed to load IR calibration from {result_path}: {exc}")
        return

    try:
        K = np.asarray(res["K_ir_native_320x256"], dtype=np.float64).reshape(3, 3)
        k1, k2 = (float(v) for v in res["dist_ir_k1k2"])
        IR_T_imgl = np.asarray(res["IR_T_imgl"], dtype=np.float64).reshape(4, 4)
        imgl_T_ir = np.asarray(res["imgl_T_ir"], dtype=np.float64).reshape(4, 4)
    except Exception as exc:
        logger.exception(f"Malformed IR calibration file {result_path}: {exc}")
        return

    dist = np.array([k1, k2, 0.0, 0.0], dtype=np.float64)
    resolution = np.array([320, 256], dtype=np.int32)  # FLIR A35 native (cf. K key)

    # Intrinsics into the existing /infrared sensor group.
    ir_grp = h5_file.require_group("infrared")
    for name, data, dtype in (
        ("intrinsics", K, np.float64),
        ("dist_coeffs", dist, np.float64),
        ("resolution", resolution, np.int32),
    ):
        ir_grp.create_dataset(name, data=data, dtype=dtype)

    # Extrinsics into the calibration group, using the same <name>_T_<other>
    # convention as the camchain transforms (imgl_T_imgr, etc.).
    for name, T in (("ir_T_imgl", IR_T_imgl), ("imgl_T_ir", imgl_T_ir)):
        calibration_grp.create_dataset(name, data=T, dtype=np.float64)

    # LiDAR→IR extrinsic, alongside imgl_T_ouster under /ouster. Prefer the value
    # the solver chained at calibration time; otherwise derive it from the
    # LiDAR→left-RGB extrinsic already written to /ouster/imgl_T_ouster.
    T_ir_ouster = res.get("IR_T_ouster")
    if T_ir_ouster is not None:
        T_ir_ouster = np.asarray(T_ir_ouster, dtype=np.float64).reshape(4, 4)
    else:
        ouster_grp = h5_file.require_group("ouster")
        if "imgl_T_ouster" in ouster_grp:
            T_ir_ouster = IR_T_imgl @ np.asarray(ouster_grp["imgl_T_ouster"], dtype=np.float64)
            logger.info("Derived /ouster/ir_T_ouster from IR_T_imgl @ imgl_T_ouster.")

    if T_ir_ouster is not None:
        ouster_grp = h5_file.require_group("ouster")
        ouster_grp.create_dataset("ir_T_ouster", data=T_ir_ouster, dtype=np.float64)


def _write_calibration_metadata(
    h5_file: h5py.File,
    cal_map: Dict[str, Any],
    extra_files: Optional[Dict[str, str]] = None,
    events_file: Optional[h5py.File] = None,
) -> None:
    """Persist simple calibration provenance (and merged camera-IMU calibration) into the HDF5 file.

    When both RGB and IMU calibrations are available, merges them:
    - All camera parameters from RGB calibration
    - Only T_cam_imu from IMU calibration
    - Transformation chain computed using RGB cal T_cn_cnm1 values
    """
    if not cal_map:
        return

    grp = h5_file.require_group("calib")
    dt_str = h5py.string_dtype(encoding="utf-8")
    # Store each cal_map entry as a string dataset under /calib.
    for key, value in cal_map.items():
        grp.create_dataset(key, data=str(value), dtype=dt_str)

    # Also record explicit file paths for any known calibration files used.
    files_grp = grp.require_group("cal_files")

    # Get calibration IDs
    rgb_cal_id = cal_map.get("rgb_cal")
    imu_cal_id = cal_map.get("imu_cal")

    # 1) Merged camera calibration (RGB params + IMU T_cam_imu). Every collected
    #    bag references both an RGB and an IMU cal, so we require both here.
    assert isinstance(rgb_cal_id, str) and rgb_cal_id and isinstance(imu_cal_id, str) and imu_cal_id, \
        f"expected both rgb_cal and imu_cal in cal_map, got rgb_cal={rgb_cal_id!r} imu_cal={imu_cal_id!r}"
    rgb_camchain_path = os.path.join(CAM_CALS_BASE_DIR, rgb_cal_id, "calibration-camchain.yaml")
    imu_camchain_path = os.path.join(IMU_CAMCHAIN_BASE_DIR, imu_cal_id, "calibration-camchain-imucam.yaml")
    files_grp.create_dataset("rgb_camchain", data=rgb_camchain_path, dtype=dt_str)
    files_grp.create_dataset("imu_camchain", data=imu_camchain_path, dtype=dt_str)
    _write_merged_calibration_metadata(h5_file, grp, rgb_cal_id, imu_cal_id, events_file=events_file)

    # 3) LiDAR↔camera extrinsics, if we have an RGB calibration bag ID
    #    (cam_cals/<rgb_cal_id>/lidar_calibration_results.yaml).
    if isinstance(rgb_cal_id, str) and rgb_cal_id:
        lidar_extrinsics_path = os.path.join(
            CAM_CALS_BASE_DIR, rgb_cal_id, "lidar_calibration_results.yaml"
        )
        files_grp.create_dataset("lidar_extrinsics", data=lidar_extrinsics_path, dtype=dt_str)
        _write_lidar_extrinsics(h5_file, grp, rgb_cal_id)

    # 4) Infrared intrinsics + extrinsics, if we have an RGB calibration bag ID
    #    (cam_cals/<rgb_cal_id>/ir_calib_result.json, from the cam-lidar calibration step).
    if isinstance(rgb_cal_id, str) and rgb_cal_id:
        ir_calib_path = os.path.join(
            CAM_CALS_BASE_DIR, rgb_cal_id, "ir_calib_result.json"
        )
        files_grp.create_dataset("ir_calib", data=ir_calib_path, dtype=dt_str)
        _write_ir_calibration(h5_file, grp, rgb_cal_id)

    # 5) Any additional calibration files passed in explicitly (e.g. lidar metadata JSON).
    if extra_files:
        for name, path in extra_files.items():
            if path:
                files_grp.create_dataset(name, data=str(path), dtype=dt_str)


def parse_gps(bag_name, h5_file, device_clock_system, system_filter):
    """Parse GPS data and write to HDF5.

    Reads /ublox_gps_node/fix (sensor_msgs/NavSatFix) for position + covariance + status.
    Optionally also reads /ublox_gps_node/fix_velocity (TwistWithCovarianceStamped)
    for the antenna velocity vector.

    Writes:
        gps/data            (N_fix, 7) float64   lat, lon, alt, cov_xx, cov_yy, cov_zz,
                                                 status (NavSatStatus.status: -1 NO_FIX / 0 FIX
                                                         / 1 SBAS_FIX / 2 GBAS_FIX (RTK))
        gps/t               (N_fix,)   float64   main-clock seconds
        gps/velocity_enu    (N_vel, 3) float64   antenna velocity vector m/s    (when /fix_velocity present)
        gps/velocity_t      (N_vel,)   float64   main-clock seconds           (when /fix_velocity present)
    """
    st = time.time()

    if device_clock_system is None or system_filter is None:
        logger.warning("system clock calibration is None, skipping GPS parsing")
        return

    bag = BagTopicReader(bag_name)
    fix_count = bag.get_message_count(GPS_TOPIC)
    has_vel = GPS_VEL_TOPIC in bag.topics_map
    vel_count = bag.get_message_count(GPS_VEL_TOPIC) if has_vel else 0

 
    gps_data   = np.empty((fix_count, 7), dtype=np.float64)
    fix_ts     = np.empty(fix_count, dtype=np.float64)
    gps_vel    = np.empty((vel_count, 3), dtype=np.float64) if has_vel else None
    vel_ts     = np.empty(vel_count,      dtype=np.float64) if has_vel else None

    i_fix = 0
    i_vel = 0
    topics_iter = [t for t in [GPS_TOPIC, GPS_VEL_TOPIC] if t in bag.topics_map]
    for topic, msg, _ in bag.iter_topics(topics_iter):
        if topic == GPS_TOPIC:
            fix_ts[i_fix] = msg.header.stamp.sec + msg.header.stamp.nanosec * NS_TO_S
            gps_data[i_fix] = [msg.latitude, msg.longitude, msg.altitude,
                               msg.position_covariance[0],
                               msg.position_covariance[4],
                               msg.position_covariance[8],
                               msg.status.status]   # -1 NO_FIX / 0 FIX / 1 SBAS / 2 GBAS(RTK)
            i_fix += 1
        elif topic == GPS_VEL_TOPIC:
            vel_ts[i_vel] = msg.header.stamp.sec + msg.header.stamp.nanosec * NS_TO_S
            lv = msg.twist.twist.linear
            gps_vel[i_vel] = [lv.x, lv.y, lv.z]
            i_vel += 1


    fix_ts_g = map_events_to_global(fix_ts[:i_fix], device_clock_system,
                                    system_filter['xs_smooth'])
    valid_fix = fix_ts_g > 0
    h5_file.create_dataset('gps/data', data=gps_data[:i_fix][valid_fix], dtype=np.float64)
    h5_file.create_dataset('gps/t',    data=fix_ts_g[valid_fix],         dtype=np.float64)

    if has_vel and i_vel > 0:
        vel_ts_g = map_events_to_global(vel_ts[:i_vel], device_clock_system,
                                        system_filter['xs_smooth'])
        valid_vel = vel_ts_g > 0
        h5_file.create_dataset('gps/velocity_enu', data=gps_vel[:i_vel][valid_vel], dtype=np.float64)
        h5_file.create_dataset('gps/velocity_t',   data=vel_ts_g[valid_vel],        dtype=np.float64)
        logger.info(f"parse_gps: also wrote gps/velocity_enu / gps/velocity_t "
                    f"({int(valid_vel.sum())} of {i_vel} samples)")

    duration = time.time() - st
    logger.info(f"parse_gps completed in {duration:.2f}s "
                f"(fix {int(valid_fix.sum())}/{i_fix}"
                f"{f', vel {int((vel_ts_g>0).sum())}/{i_vel}' if has_vel and i_vel>0 else ''})")

def parse_imu(bag_name, h5_file, device_clock_imu, imu_filter):
    """Parse IMU data and write to HDF5."""
    st = time.time()
    bag=BagTopicReader(bag_name)
    # Verify all IMU topics have the same message count
    imu_count = bag.get_message_count(IMU_TOPIC)
    imu_raw_count = bag.get_message_count(IMU_RAW_TOPIC)
    pressure_count = bag.get_message_count(PRESSURE_TOPIC)
    magnetic_count = bag.get_message_count(MAGNETIC_TOPIC)
    temperature_count = bag.get_message_count(TEMPERATURE_TOPIC)
    imu_trig_count = bag.get_message_count(IMU_TRIGGER_TOPIC)


    # Step 1: Get device times and map to global time to determine valid indices
    if imu_count != imu_trig_count:
        logger.warning(f"IMU topic counts don't match: IMU={imu_count}, Trigger={imu_trig_count}. Will match by time and process only messages with valid trigger matches.")
    
 
    imu_dev_time = np.empty(imu_trig_count, dtype=np.float64)
    imu_trigger_msg_time = np.empty(imu_trig_count, dtype=np.float64)
    imu_topic_msg_time = np.empty(imu_count, dtype=np.float64)
    imu_accel_all = np.empty((imu_count, 3), dtype=np.float32)
    imu_angvel_all = np.empty((imu_count, 3), dtype=np.float32)

    imu_accel_raw_all  = np.empty((imu_raw_count, 3), dtype=np.float32)
    imu_angvel_raw_all = np.empty((imu_raw_count, 3), dtype=np.float32)
    pressure_all = np.empty(pressure_count, dtype=np.float32)
    magnetic_all = np.empty((magnetic_count, 3), dtype=np.float32)
    temperature_all = np.empty(temperature_count, dtype=np.float32)

    # Per-topic counters
    i_trig = 0
    i_imu = 0
    i_imu_raw = 0
    i_pres = 0
    i_mag = 0
    i_temp = 0

    for topic, msg, timestamp in bag.iter_topics([
        IMU_TRIGGER_TOPIC, IMU_TOPIC, IMU_RAW_TOPIC,
        PRESSURE_TOPIC, MAGNETIC_TOPIC, TEMPERATURE_TOPIC,
    ]):
        if topic == IMU_TRIGGER_TOPIC:
            imu_dev_time[i_trig] = msg.timestartup
            imu_trigger_msg_time[i_trig] = msg.header.stamp.sec + msg.header.stamp.nanosec * NS_TO_S
            i_trig += 1
        elif topic == IMU_TOPIC:
            imu_topic_msg_time[i_imu] = msg.header.stamp.sec + msg.header.stamp.nanosec * NS_TO_S
            imu_accel_all[i_imu, 0] = msg.linear_acceleration.x
            imu_accel_all[i_imu, 1] = msg.linear_acceleration.y
            imu_accel_all[i_imu, 2] = msg.linear_acceleration.z
            imu_angvel_all[i_imu, 0] = msg.angular_velocity.x
            imu_angvel_all[i_imu, 1] = msg.angular_velocity.y
            imu_angvel_all[i_imu, 2] = msg.angular_velocity.z
            i_imu += 1
        elif topic == IMU_RAW_TOPIC:
            imu_accel_raw_all[i_imu_raw, 0] = msg.linear_acceleration.x
            imu_accel_raw_all[i_imu_raw, 1] = msg.linear_acceleration.y
            imu_accel_raw_all[i_imu_raw, 2] = msg.linear_acceleration.z
            imu_angvel_raw_all[i_imu_raw, 0] = msg.angular_velocity.x
            imu_angvel_raw_all[i_imu_raw, 1] = msg.angular_velocity.y
            imu_angvel_raw_all[i_imu_raw, 2] = msg.angular_velocity.z
            i_imu_raw += 1
        elif topic == PRESSURE_TOPIC:
            pressure_all[i_pres] = msg.fluid_pressure
            i_pres += 1
        elif topic == MAGNETIC_TOPIC:
            magnetic_all[i_mag, 0] = msg.magnetic_field.x
            magnetic_all[i_mag, 1] = msg.magnetic_field.y
            magnetic_all[i_mag, 2] = msg.magnetic_field.z
            i_mag += 1
        elif topic == TEMPERATURE_TOPIC:
            temperature_all[i_temp] = msg.temperature
            i_temp += 1

    imu_dev_time *= NS_TO_S

    # Step 2: Match topic messages to trigger messages by time using searchsorted
    imu_topic_to_trigger_idx = np.searchsorted(imu_topic_msg_time, imu_trigger_msg_time, side='left')
    imu_topic_to_trigger_idx = np.clip(imu_topic_to_trigger_idx, 0, len(imu_topic_msg_time) - 1)

    diff_time = (imu_topic_msg_time[imu_topic_to_trigger_idx] - imu_trigger_msg_time).sum()
    logger.info(f"IMU topic to trigger time difference: {diff_time:.2f}s")

    new_ts = map_events_to_global(imu_dev_time, device_clock_imu, imu_filter['xs_smooth'])
    valid_mask = new_ts > 0
    valid_count = np.sum(valid_mask)

    # Step 3: Create H5 datasets with final size
    accel_dset = h5_file.create_dataset('vectornav/accel', shape=(valid_count, 3), dtype=np.float32)
    ang_vel_dset = h5_file.create_dataset('vectornav/ang_vel', shape=(valid_count, 3), dtype=np.float32)
    accel_raw_dset   = h5_file.create_dataset('vectornav/accel_raw',   shape=(valid_count, 3), dtype=np.float32)
    ang_vel_raw_dset = h5_file.create_dataset('vectornav/ang_vel_raw', shape=(valid_count, 3), dtype=np.float32)
    pressure_dset = h5_file.create_dataset('vectornav/pressure', shape=(valid_count,), dtype=np.float32)
    magnetic_dset = h5_file.create_dataset('vectornav/magnetic', shape=(valid_count, 3), dtype=np.float32)
    temperature_dset = h5_file.create_dataset('vectornav/temperature', shape=(valid_count,), dtype=np.float32)
    imu_time_dset = h5_file.create_dataset('vectornav/t', shape=(valid_count,), dtype=np.float64)

    # Step 4: Bulk write all datasets using all valid trigger events
    # We use valid indices directly, even if they are duplicates (which happens if triggers are faster than IMU or if time sync is off)
    valid_indices = imu_topic_to_trigger_idx[valid_mask]
    
    accel_dset[:] = imu_accel_all[valid_indices]
    ang_vel_dset[:] = imu_angvel_all[valid_indices]
    imu_time_dset[:] = new_ts[valid_mask]
    
    # Clip indices for pressure, magnetic, temperature, raw IMU to avoid
    # potential out-of-bounds if counts differ across topics.
    pressure_indices    = np.clip(valid_indices, 0, len(pressure_all) - 1)
    magnetic_indices    = np.clip(valid_indices, 0, len(magnetic_all) - 1)
    temperature_indices = np.clip(valid_indices, 0, len(temperature_all) - 1)
    imu_raw_indices     = np.clip(valid_indices, 0, len(imu_accel_raw_all) - 1)

    pressure_dset[:]    = pressure_all[pressure_indices]
    magnetic_dset[:]    = magnetic_all[magnetic_indices]
    temperature_dset[:] = temperature_all[temperature_indices]
    accel_raw_dset[:]   = imu_accel_raw_all[imu_raw_indices]
    ang_vel_raw_dset[:] = imu_angvel_raw_all[imu_raw_indices]

    duration = time.time() - st
    logger.info(f"parse_imu completed in {duration:.2f}s")


def _ensure_capacity(dset, needed):
    """Ensure dataset has space for at least `needed` elements.
    """
    if needed > dset.shape[0]:
        current = dset.shape[0]
        # Round up to the next multiple of EV_CHUNK
        grow_to = ((needed + EV_CHUNK - 1) // EV_CHUNK) * EV_CHUNK
        # Ensure we at least grow to needed
        grow_to = max(grow_to, needed)
        dset.resize((grow_to,))

def _flush_buffer(dset, write_idx, buf, buf_len):
    """Flush buffer to HDF5 dataset."""
    if buf_len == 0:
        return write_idx
    end = write_idx + buf_len
    _ensure_capacity(dset, end)
    dset[write_idx:end] = buf[:buf_len]
    return end

def _event_camera_worker(bag_name, tmp_h5_path, event_topic, dataset_prefix,
                         device_clock, event_filter):
    """Subprocess entry-point: open a private HDF5 file, process one event
    camera, then close. The caller merges datasets into the main file."""
    with h5py.File(tmp_h5_path, 'w') as tmp_h5:
        _process_event_camera(bag_name, tmp_h5, event_topic, dataset_prefix,
                              device_clock, event_filter)

def _process_event_camera(bag_name, h5_file, event_topic, dataset_prefix,
                          device_clock, event_filter):
    """Process a single event camera (left or right)."""
    st = time.time()
    bag = BagTopicReader(bag_name)
    decoder = ECDecoder()
    
    # Create datasets using new /ev/<side> layout
    x_ds = h5_file.create_dataset(f'{dataset_prefix}/x', shape=(0,), maxshape=(None,), dtype=np.uint16, chunks=(500_000), compression=hdf5plugin.Blosc2(cname='zstd', clevel=5, filters=hdf5plugin.Blosc2.BITSHUFFLE))
    y_ds = h5_file.create_dataset(f'{dataset_prefix}/y', shape=(0,), maxshape=(None,), dtype=np.uint16, chunks=(500_000), compression=hdf5plugin.Blosc2(cname='zstd', clevel=5, filters=hdf5plugin.Blosc2.BITSHUFFLE))
    t_ds = h5_file.create_dataset(f'{dataset_prefix}/t', shape=(0,), maxshape=(None,), dtype=np.uint64, chunks=(500_000), compression=hdf5plugin.Blosc2(cname='zstd', clevel=5, filters=hdf5plugin.Blosc2.SHUFFLE))
    p_ds = h5_file.create_dataset(f'{dataset_prefix}/p', shape=(0,), maxshape=(None,), dtype=np.uint8, chunks=(500_000), compression=hdf5plugin.Blosc2(cname='zstd', clevel=5, filters=hdf5plugin.Blosc2.BITSHUFFLE))
    
    x_written = y_written = t_written = p_written = 0
    
    # Chunk buffers
    x_buf = np.empty(EV_CHUNK, dtype=np.uint16)
    y_buf = np.empty(EV_CHUNK, dtype=np.uint16)
    t_buf = np.empty(EV_CHUNK, dtype=np.uint64)
    p_buf = np.empty(EV_CHUNK, dtype=np.uint8)
    buf_len = 0

    bad_diffs=0
    bad_diffs_count=0
    
    # For ms_to_idx generation (online)
    ms_to_idx = []
    event_count = 0
    
    # Process events from topic
    for msg, timestamp in bag.iter_topic(event_topic):
        decoder.decode(msg)
        events = decoder.get_cd_events()
        if len(events['t']) == 0:
            continue
        
        # Map timestamps
        t_chunk = events['t'] * US_TO_S

        diffs = np.diff(t_chunk)
        if 0!=np.sum(diffs<-1e-7):
            bad_diffs+=1
            bad_diffs_count+=np.sum(diffs<-1e-7)
            continue
        
        mask = np.concatenate(([True], diffs != 0))
        t_unique = t_chunk[mask]
        inverse_idx = np.cumsum(mask) - 1

            
        t_mapped_unique = map_events_to_global(
            t_unique, device_clock, event_filter['xs_smooth'],
        )
        if t_mapped_unique[0] < 0:
            continue

        t_mapped_chunk = t_mapped_unique[inverse_idx]

        x_src = events['x']
        y_src = events['y']
        t_src = t_mapped_chunk
        p_src = events['p']
        src_len = len(x_src)
        src_idx = 0
        
        # Append this packet into the EV_CHUNK staging buffer, flushing whenever
        # it fills. A packet can exceed the remaining room, so copy it in pieces.
        while src_idx < src_len:
            room = EV_CHUNK - buf_len
            take = min(src_len - src_idx, room)   # fill the buffer without overrunning the packet
            end = buf_len + take
            x_buf[buf_len:end] = x_src[src_idx:src_idx+take]
            y_buf[buf_len:end] = y_src[src_idx:src_idx+take]
            np.multiply(t_src[src_idx:src_idx+take], 1e6, out=t_buf[buf_len:end], casting='unsafe')
            p_buf[buf_len:end] = p_src[src_idx:src_idx+take]
            buf_len = end
            src_idx += take
            
            # Flush when buffer is full
            if buf_len == EV_CHUNK:
                x_written = _flush_buffer(x_ds, x_written, x_buf, buf_len)
                y_written = _flush_buffer(y_ds, y_written, y_buf, buf_len)
                t_written = _flush_buffer(t_ds, t_written, t_buf, buf_len)
                p_written = _flush_buffer(p_ds, p_written, p_buf, buf_len)
                
                # Update ms_to_idx for this chunk
                ms_idx_chunk = np.floor(t_buf / 1000.0).astype(np.int64)
                if ms_idx_chunk.size > 0:
                    next_ms = len(ms_to_idx)
                    last_ms = int(ms_idx_chunk[-1])
                    if last_ms >= next_ms:
                        ms_range = np.arange(next_ms, last_ms + 1)
                        local_idx = np.searchsorted(ms_idx_chunk, ms_range, side='left')
                        ms_to_idx.extend((event_count + local_idx).tolist())
                
                event_count += buf_len
                buf_len = 0
    logger.info(f"Bad diffs: skipped {bad_diffs} event chunks ({bad_diffs_count} backward time-steps total)")
    # Flush remaining buffered events
    if buf_len > 0:
        x_written = _flush_buffer(x_ds, x_written, x_buf, buf_len)
        y_written = _flush_buffer(y_ds, y_written, y_buf, buf_len)
        t_written = _flush_buffer(t_ds, t_written, t_buf, buf_len)
        p_written = _flush_buffer(p_ds, p_written, p_buf, buf_len)
        
        # Update ms_to_idx for remaining chunk
        ms_idx_chunk = np.floor(t_buf[:buf_len] / 1000.0).astype(np.int64)
        if ms_idx_chunk.size > 0:
            next_ms = len(ms_to_idx)
            last_ms = int(ms_idx_chunk[-1])
            if last_ms >= next_ms:
                ms_range = np.arange(next_ms, last_ms + 1)
                local_idx = np.searchsorted(ms_idx_chunk, ms_range, side='left')
                ms_to_idx.extend((event_count + local_idx).tolist())
        
        event_count += buf_len
    
    # Trim datasets to final size
    x_ds.resize((x_written,))
    y_ds.resize((y_written,))
    t_ds.resize((t_written,))
    p_ds.resize((p_written,))
    
    # Write ms_to_idx
    if len(ms_to_idx) > 0:
        h5_file.create_dataset(f'{dataset_prefix}/ms_to_idx', data=np.array(ms_to_idx, dtype=np.uint64), dtype=np.uint64)
    
    
    duration = time.time() - st
    logger.info(f"{dataset_prefix.capitalize()} camera: {x_written:,} events stored in {duration:.2f}s")

def _flir_prepare(bag_name, trig_topic, cam_topic, img_prefix,
                  device_clock, flir_filter):
    """Collect timestamps, map to global time, and launch ffmpeg
    """
    st = time.time()

    bag = BagTopicReader(bag_name)
    output_path, bn = get_processed_path(bag_name)

    # Single-pass: read trigger + camera topics in one bag scan
    trig_count = bag.get_message_count(trig_topic)
    cam_count = bag.get_message_count(cam_topic)
    cam_msg_time = np.empty(trig_count, dtype=np.float64)
    timestamp = np.empty(trig_count, dtype=np.float64)
    brightness    = np.empty(trig_count, dtype=np.int16)
    exposure_time = np.empty(trig_count, dtype=np.uint32)  # microseconds
    gain          = np.empty(trig_count, dtype=np.float32) # dB
    ts_cam = np.empty(cam_count, dtype=np.float64)

    i_trig = 0
    i_cam = 0
    for topic, msg, _ in bag.iter_topics([trig_topic, cam_topic]):
        if topic == trig_topic:
            timestamp[i_trig]     = msg.camera_time
            cam_msg_time[i_trig]  = msg.header.stamp.sec + msg.header.stamp.nanosec * NS_TO_S
            brightness[i_trig]    = msg.brightness
            exposure_time[i_trig] = msg.exposure_time
            gain[i_trig]          = msg.gain
            i_trig += 1
        else:
            ts_cam[i_cam] = msg.header.stamp.sec + msg.header.stamp.nanosec * NS_TO_S
            i_cam += 1
    timestamp *= NS_TO_S

    # Find trigger index for each camera timestamp (exact matches expected)
    cam_idx = np.searchsorted(cam_msg_time, ts_cam, side='left')

    # Clamp indices to valid range (handle edge cases)
    cam_idx = np.clip(cam_idx, 0, len(cam_msg_time) - 1)

    timestamp     = timestamp[cam_idx]
    brightness    = brightness[cam_idx]
    exposure_time = exposure_time[cam_idx]
    gain          = gain[cam_idx]

    # Map timestamps to global time
    ts_new = map_events_to_global(
        timestamp, device_clock, flir_filter['xs_smooth'],
    )

    cp = np.where(ts_new >= 0)[0]
    cut_time = ts_cam[cp[0]] if len(cp) > 0 else 0

    # Use full img_prefix for unique temp filenames
    ts_filename = f"{output_path}{bn}_{img_prefix}_timestamps.txt"

    # Launch ffmpeg to convert bag to video (non-blocking)
    cmd = [
        "bash", "-c", f"""
            source /opt/ros/jazzy/setup.bash && \
            source /catkin_ws/install/setup.bash && \
            ros2 run ffmpeg_image_transport_tools bag_to_video \
                -i {bag_name} \
                -t {cam_topic} \
                -d hevc \\
                -r 100 \\
                -o {output_path}{img_prefix} \\
                -E encoder:libx265 \\
                -S {cut_time} \\
                -E preset:medium \\
                -T {ts_filename} \\
                -E crf:28 \\
                -E x265-params:rc-lookahead=0:bframes=0:pools=4
            """
    ]
    proc = subprocess.Popen(cmd)

    prep_duration = time.time() - st
    logger.info(f"_flir_prepare({img_prefix}) launched ffmpeg in {prep_duration:.2f}s")

    return {
        'proc': proc,
        'ts_new': ts_new,
        'ts_filename': ts_filename,
        'img_prefix': img_prefix,
        'output_path': output_path,
        'st': st,
        'brightness': brightness,
        'exposure_time': exposure_time,
        'gain': gain,
    }


def _flir_finalize(state, h5_file, dataset_key):
    """Wait for the ffmpeg encode started by :func:`_flir_prepare`, then
    load the timestamp file and write the final dataset to HDF5."""
    from torchcodec.decoders import VideoDecoder

    proc = state['proc']
    ts_new = state['ts_new']
    ts_filename = state['ts_filename']
    img_prefix = state['img_prefix']
    output_path = state['output_path']
    st = state['st']

    ret = proc.wait()
    if ret != 0:
        raise subprocess.CalledProcessError(ret, proc.args)

    # Load timestamps from the decoded file (column 0 = packet_number, which indexes ts_new).
    ts_file = np.loadtxt(ts_filename, dtype=np.float64)
    pts_encoded = ts_file[:, 0].astype(np.uint64)

    decoder = VideoDecoder(f"{output_path}{img_prefix}.mp4", device='cpu')
    n_encoded = len(decoder)
    if n_encoded != len(pts_encoded):
        logger.warning(
            f"_flir_finalize({img_prefix}): timestamps file has {len(pts_encoded)} entries "
            f"but video has {n_encoded} frames — trimming to {n_encoded}"
        )
        pts_encoded = pts_encoded[:n_encoded]
    ts_vid = ts_new[pts_encoded]

    # Write to HDF5 (new layout: /img_left/t, /img_right/t)
    h5_file.create_dataset(dataset_key, data=ts_vid, dtype=np.float64)

   
    group_prefix = dataset_key.rsplit("/", 1)[0]   # e.g. "img/right"
    for name, arr, dtype in (
        ("brightness",    state["brightness"],    np.int16),
        ("exposure_time", state["exposure_time"], np.uint32),
        ("gain",          state["gain"],          np.float32),
    ):
        key = f"{group_prefix}/{name}"
        if key in h5_file:
            del h5_file[key]
        h5_file.create_dataset(key, data=arr[pts_encoded], dtype=dtype)

    # Cleanup
    os.remove(ts_filename)

    duration = time.time() - st
    logger.info(f"_flir_finalize({img_prefix}) completed in {duration:.2f}s")


def _flir_worker(bag_name, tmp_h5_path, trigger_topic, cam_topic, img_prefix, dataset_key,
                 device_clock, flir_filter):
    """Subprocess: run FLIR prepare (ffmpeg launch) AND finalize (wait+write) in one go."""
    # 1. Prepare (launches ffmpeg, returns state)
    state = _flir_prepare(
        bag_name, trigger_topic, cam_topic, img_prefix,
        device_clock, flir_filter
    )
    
    # 2. Finalize (waits for ffmpeg, writes to HDF5)
    with h5py.File(tmp_h5_path, 'w') as tmp_h5:
        _flir_finalize(state, tmp_h5, dataset_key)


def _cleanup_lio_bag(lio_bag):
    """Remove a consumed LIO bag and its now-empty `_lio` wrapper dir. Non-fatal.

    Two on-disk layouts exist, so we only ever rmdir a parent that is a `_lio`
    wrapper and is empty after cleanup:
        A) <session>/<bag>_lio/odom/   parent is "<bag>_lio"  → safe to remove
        B) <session>/<bag>_lio/        parent is "<session>"  → MUST NOT remove
    """
    try:
        shutil.rmtree(lio_bag, ignore_errors=True)
        parent_dir = Path(lio_bag).parent
        if (parent_dir.exists()
                and parent_dir.name.endswith("_lio")
                and not any(parent_dir.iterdir())):
            parent_dir.rmdir()
    except Exception as exc:
        logger.warning(f"Failed to remove LIO directories for {lio_bag}: {exc}")


def _lidar_worker(bag_name, tmp_h5_path, mode, cal_map_path, lio_bag, metadata_path):
    """Subprocess: parse Lidar (Raw or LIO) into a tmp HDF5."""
    with h5py.File(tmp_h5_path, 'w') as tmp_h5:
        is_data_mode = (mode == "data")

        # Calibration mode uses the legacy raw lidar layout (2D /ouster/t).
        if mode == "calibration":
            parse_lidar_raw(bag_name, tmp_h5, metadata_path=metadata_path, is_data_mode=False)
            return

        # Data mode: prefer the LIO bag when present, otherwise fall back to raw.
        # A LIO parse failure also falls back (data still gets written from raw).
        used_lio = False
        if lio_bag is not None and os.path.exists(lio_bag):
            try:
                parse_lidar_lio(lio_bag, tmp_h5, raw_bag_path=bag_name, metadata_path=metadata_path)
                _cleanup_lio_bag(lio_bag)
                used_lio = True
            except Exception as exc:
                logger.exception(f"parse_lidar_lio failed for {lio_bag}, falling back to raw lidar: {exc}")

        if not used_lio:
            parse_lidar_raw(bag_name, tmp_h5, metadata_path=metadata_path, is_data_mode=is_data_mode)


def _can_worker(bag_name, tmp_h5_path, device_clock, system_filter):
    """Subprocess: parse CAN data."""
    with h5py.File(tmp_h5_path, 'w') as tmp_h5:
        parse_can(bag_name, tmp_h5, device_clock, system_filter)


def _misc_worker(bag_name, tmp_h5_path, calibration_data):
    """Subprocess: parse random small sensors (GPS, Infrared, IMU)."""
    with h5py.File(tmp_h5_path, 'w') as tmp_h5:
        bag = BagTopicReader(bag_name)

        if GPS_TOPIC in bag.topics_map:
            parse_gps(
                bag_name, tmp_h5,
                calibration_data.device_clock_system,
                calibration_data.system_filter,
            )

        # Infrared
        if INFRARED_CAM_TOPIC in bag.topics_map:
            process_infrared_camera(
                bag_name, tmp_h5,
                calibration_data.device_clock_system,
                calibration_data.system_filter,
            )

        # IMU
        if IMU_TOPIC in bag.topics_map:
            parse_imu(
                bag_name, tmp_h5,
                calibration_data.device_clock_imu,
                calibration_data.imu_filter,
            )


def process_infrared_camera(bag_name, h5_file, device_clock, system_filter):
    st = time.time()
    from torchcodec.decoders import VideoDecoder
    
    bag = BagTopicReader(bag_name)
    output_path, bn = get_processed_path(bag_name)
    

    # Collect camera timestamps (use bag recording time = OS clock)
    # for consistency with system_filter calibration.
    cam_count = bag.get_message_count(INFRARED_CAM_TOPIC)
    ts_cam = np.empty(cam_count, dtype=np.float64)
    for i, (msg, bag_ts) in enumerate(bag.iter_topic(INFRARED_CAM_TOPIC)):
        ts_cam[i] = bag_ts * NS_TO_S
    
    # Map timestamps to global time
    ts_new = map_events_to_global(
        ts_cam, device_clock, system_filter['xs_smooth'],
    )
    
    cp = np.where(ts_new >= 0)[0]
    cut_time = ts_cam[cp[0]] if len(cp) > 0 else 0
    
    # Run ffmpeg to convert bag to video
    cmd = [
        "bash", "-c", f"""
            source /opt/ros/jazzy/setup.bash && \
            source /catkin_ws/install/setup.bash && \
            ros2 run ffmpeg_image_transport_tools bag_to_video \
                -i {bag_name} \
                -t {INFRARED_CAM_TOPIC} \
                -d hevc \\
                -r 50 \\
                -o {output_path}img_infrared \\
                -E encoder:libx265 \\
                -S {cut_time} \\
                -E preset:medium \\
                -T {output_path}{bn}_infrared_timestamps.txt \\
                -E crf:28 \\
                -E x265-params:rc-lookahead=0:bframes=0:pools=2
            """
    ]
    subprocess.run(cmd, check=True)
    
    # Load timestamps from the decoded file (column 0 = packet_number, which indexes ts_new).
    # Trim to actual video frame count to handle encoder look-ahead dropping trailing frames.
    infrared_ts_filename = f"{output_path}{bn}_infrared_timestamps.txt"
    ts_file = np.loadtxt(infrared_ts_filename, dtype=np.float64)
    pts_encoded = ts_file[:, 0].astype(np.uint64)

    decoder = VideoDecoder(f"{output_path}img_infrared.mp4", device='cpu')
    n_encoded = len(decoder)
    if n_encoded != len(pts_encoded):
        logger.warning(
            f"process_infrared_camera: timestamps file has {len(pts_encoded)} entries "
            f"but video has {n_encoded} frames — trimming to {n_encoded}"
        )
        pts_encoded = pts_encoded[:n_encoded]
    ts_vid = ts_new[pts_encoded]

    # Write to HDF5
    h5_file.create_dataset('infrared/t', data=ts_vid, dtype=np.float64)

    # Cleanup
    os.remove(infrared_ts_filename)
    
    duration = time.time() - st
    logger.info(f"process_infrared_camera completed in {duration:.2f}s")


def parse_lidar_raw(bag_name, h5_file, *, metadata_path: Optional[str] = None, is_data_mode: bool = False):

    st = time.time()

    if metadata_path is None:
        metadata_path = OUSTER_METADATA_REV7_1

    bag = BagTopicReader(bag_name)
    output_path, bn = get_processed_path(bag_name)
    
    msg_count = bag.get_message_count(OUSTER_TIME_STATUS_TOPIC)
    timestamp = np.empty(msg_count, dtype=np.float64)
    lock_status = np.empty((msg_count), dtype=np.float64)
    count_status = np.empty((msg_count), dtype=np.float64)
    rec_ts = np.empty((msg_count), dtype=np.float64)
    for i, (msg, ts) in enumerate(bag.iter_topic(OUSTER_TIME_STATUS_TOPIC)):
        timestamp[i] = msg.timestamp_time
        lock_status[i] = msg.sync_pulse_locked
        count_status[i] = msg.diagnostics_count
        rec_ts[i]= ts
    
    start_idx=np.where(lock_status)[0][0]
    filt_time=np.round(timestamp[start_idx]*NS_TO_S)
    assert (count_status[start_idx]-count_status[start_idx-1])==1, "Count difference is not 1 btw sync pulses start and end"
    
    with open(metadata_path, 'r') as f:
        data = f.read()
   
    try:
        ouster_grp = h5_file.require_group("ouster")
        if "metadata_json" in ouster_grp:
            del ouster_grp["metadata_json"]
        dt_str = h5py.string_dtype(encoding="utf-8")
        ouster_grp.create_dataset("metadata_json", data=data, dtype=dt_str)
    except Exception as exc:
        logger.exception(f"Failed to store Ouster metadata JSON into HDF5: {exc}")
    scan_count = 0
    metadata=client.SensorInfo(data)
    xyzlut = client.XYZLut(metadata)
    batcher = client.ScanBatcher(metadata)
    scan = client.LidarScan(metadata)
    pf = client.PacketFormat.from_info(metadata)
    est_scan_count=int(((timestamp[-1]-timestamp[start_idx])*NS_TO_S)*20)
    
    # Per Ouster's RNG19_RFL8_SIG16_NIR16 profile: range = 19 bits,
    # signal/NIR = 16 bits (uint16), reflectivity = 8 bits (uint8). 
    _ouster_comp = hdf5plugin.Blosc2(cname='zstd', clevel=5, filters=hdf5plugin.Blosc2.BITSHUFFLE)
    signal_h5     = h5_file.create_dataset('ouster/signal',       shape=(est_scan_count, 64, 2048), dtype=np.uint16, chunks=(1,64,2048), compression=_ouster_comp)
    relectivity_h5= h5_file.create_dataset('ouster/reflectivity', shape=(est_scan_count, 64, 2048), dtype=np.uint8,  chunks=(1,64,2048), compression=_ouster_comp)
    ir_h5         = h5_file.create_dataset('ouster/ir',           shape=(est_scan_count, 64, 2048), dtype=np.uint16, chunks=(1,64,2048), compression=_ouster_comp)
    range_h5      = h5_file.create_dataset('ouster/range',        shape=(est_scan_count, 64, 2048), dtype=np.uint32, chunks=(1,64,2048), compression=_ouster_comp)
    ts_arr=np.empty((est_scan_count,2), dtype=np.float64)
    for i, (msg, ts) in enumerate(bag.iter_topic(OUSTER_LIDAR_TOPIC)):
        
        packet=client.LidarPacket(len(msg.buf))
        packet.buf[:]=msg.buf
        if batcher(packet, scan):
            scan_ts=scan.get_first_valid_column_timestamp()*NS_TO_S-filt_time
            
            if (scan_ts)<0:
                continue

            signal_h5[scan_count]=scan.field(client.ChanField.SIGNAL)
            relectivity_h5[scan_count]=scan.field(client.ChanField.REFLECTIVITY)
            ir_h5[scan_count]=scan.field(client.ChanField.NEAR_IR)
            range_h5[scan_count]=scan.field(client.ChanField.RANGE)
            
            ts_arr[scan_count,0]=scan_ts
            ts_arr[scan_count,1]=scan.get_last_valid_column_timestamp()*NS_TO_S-filt_time
            scan_count+=1

    signal_h5.resize((scan_count, 64, 2048))
    relectivity_h5.resize((scan_count, 64, 2048))
    ir_h5.resize((scan_count, 64, 2048))
    range_h5.resize((scan_count, 64, 2048))
    # Per-scan timestamps
    ts_scan = ts_arr[:scan_count]
    if ts_scan.size > 0:
        h5_file.create_dataset("ouster/t", data=ts_scan, dtype=np.float64)

    msg_count = bag.get_message_count(OUSTER_IMU_TOPIC)
    gyro_data=np.empty((msg_count, 3), dtype=np.float32)
    accel_data=np.empty((msg_count, 3), dtype=np.float32)
    
    gyro_ts_ar=np.empty((msg_count), dtype=np.float64)
    accel_ts_ar=np.empty((msg_count), dtype=np.float64)
    cnt=0
    

    for i, (msg, ts) in enumerate(bag.iter_topic(OUSTER_IMU_TOPIC)):
        if ts<filt_time:
            continue
            

        packet=client.ImuPacket(len(msg.buf))
        packet.buf[:]=msg.buf


        gyro_ts = pf.imu_gyro_ts(packet.buf)*NS_TO_S-filt_time
        accel_ts = pf.imu_accel_ts(packet.buf)*NS_TO_S-filt_time
        
        if accel_ts<0 or gyro_ts<0:
            continue
        
        ax = pf.imu_la_x(packet.buf)
        ay = pf.imu_la_y(packet.buf)
        az = pf.imu_la_z(packet.buf)
        
        wx = pf.imu_av_x(packet.buf)
        wy = pf.imu_av_y(packet.buf)
        wz = pf.imu_av_z(packet.buf)

        accel_data[cnt,0]=ax
        accel_data[cnt,1]=ay
        accel_data[cnt,2]=az
        gyro_data[cnt,0]=wx
        gyro_data[cnt,1]=wy
        gyro_data[cnt,2]=wz
        accel_ts_ar[cnt]=accel_ts
        gyro_ts_ar[cnt]=gyro_ts
        cnt+=1

    
    # New Ouster IMU layout (timestamps use *_t instead of *_ts)
    h5_file.create_dataset('ouster/accel_t', data=accel_ts_ar[:cnt], dtype=np.float64)
    h5_file.create_dataset('ouster/ang_t', data=gyro_ts_ar[:cnt], dtype=np.float64)
    h5_file.create_dataset('ouster/accel', data=accel_data[:cnt], dtype=np.float32)
    h5_file.create_dataset('ouster/ang_vel', data=gyro_data[:cnt], dtype=np.float32)
    duration = time.time() - st
    logger.info(f"parse_lidar_raw completed in {duration:.2f}s")


def _ouster_filt_time(bag) -> float:
    """t=0 origin for Ouster timestamps: the first PPS-locked /ouster_time_status
    """
    msg_count = bag.get_message_count(OUSTER_TIME_STATUS_TOPIC)
    if msg_count == 0:
        logger.warning("No /ouster_time_status messages; Ouster timestamps stay raw ROS time")
        return 0.0

    timestamp = np.empty(msg_count, dtype=np.float64)
    lock_status = np.empty(msg_count, dtype=np.float64)
    count_status = np.empty(msg_count, dtype=np.float64)
    for i, (msg, _ts) in enumerate(bag.iter_topic(OUSTER_TIME_STATUS_TOPIC)):
        timestamp[i] = msg.timestamp_time
        lock_status[i] = msg.sync_pulse_locked
        count_status[i] = msg.diagnostics_count

    locked_idx = np.where(lock_status)[0]
    start_idx = locked_idx[0] if locked_idx.size else 0
    if not locked_idx.size:
        logger.warning("No locked sync pulse; defaulting filt_time to first timestamp")

    if start_idx > 0 and count_status[start_idx] - count_status[start_idx - 1] != 1:
        logger.warning("Count difference between sync pulses is %s (expected 1)",
                       count_status[start_idx] - count_status[start_idx - 1])

    return np.round(timestamp[start_idx] * NS_TO_S)


def parse_lidar_lio(
    lio_bag_path: str,
    h5_file: h5py.File,
    *,
    raw_bag_path: Optional[str] = None,
    metadata_path: Optional[str] = None
) -> None:
    """Parse LIO-derived LiDAR data (frames + odometry) into a compact layout.

    Expects an MCAP bag at `lio_bag_path` with at least:
        /rko_lio/frame              (sensor_msgs/msg/PointCloud2)
        /rko_lio/odometry           (nav_msgs/msg/Odometry)
    and optionally (when launched with odom_at_imu_rate:=true):
        /rko_lio/odom_at_imu_rate   (nav_msgs/msg/Odometry, ~100 Hz)

    We store under the existing /ouster group:
        # Per-frame point clouds, each zero-padded to max_pts = 64*2048 = 131072
        # points. Frame i is range_pcl[i] / refl_pcl[i] / ...; unused tail rows
        # are zero (frames with more than max_pts points are truncated).
        /ouster/t                    (N_frames,)             per-frame timestamps (s)
        /ouster/range_pcl            (N_frames, max_pts, 3)  [x,y,z]
        /ouster/refl_pcl             (N_frames, max_pts)     reflectivity
        /ouster/sig_pcl              (N_frames, max_pts)     signal
        /ouster/nir_pcl              (N_frames, max_pts)     near IR

        # LIO odometry (time, pose, and twist) at the LiDAR rate (~10 Hz):
        /ouster/odom/t               (N_odom,)
        /ouster/odom/map_T_lidart    (N_odom, 4, 4)
        /ouster/odom/lin_vel         (N_odom, 3)     [vx,vy,vz]
        /ouster/odom/ang_vel         (N_odom, 3)     [wx,wy,wz]

        # High-frequency LIO odometry from IMU-integrated state (~100 Hz). Only present
        # when the source bag contains /rko_lio/odom_at_imu_rate. The poses snap back to
        # the LiDAR-rate values at every LiDAR scan; between scans they are dead-reckoned
        # forward from the last optimized LiDAR pose using IMU integration. Same frame
        # convention (map -> base) and same dataset shapes as /ouster/odom/*.
        /ouster/hf_odom/t            (N_hf,)
        /ouster/hf_odom/map_T_lidart (N_hf, 4, 4)
        /ouster/hf_odom/lin_vel      (N_hf, 3)
        /ouster/hf_odom/ang_vel      (N_hf, 3)

    """
    from sensor_msgs_py import point_cloud2

    st = time.time()
    logger.info(f"Parsing LIO bag at {lio_bag_path}")
    bag = BagTopicReader(lio_bag_path)
    
    if metadata_path:
        try:
            with open(metadata_path, 'r') as f:
                metadata_json = f.read()
            ouster_grp = h5_file.require_group("ouster")
            if "metadata_json" in ouster_grp:
                del ouster_grp["metadata_json"]
            dt_str = h5py.string_dtype(encoding="utf-8")
            ouster_grp.create_dataset("metadata_json", data=metadata_json, dtype=dt_str)
        except Exception as exc:
            logger.exception(f"Failed to store metadata from {metadata_path}: {exc}")

    FRAME_TOPIC = "/rko_lio/frame"
    ODOM_TOPIC = "/rko_lio/odometry"
    HF_ODOM_TOPIC = "/rko_lio/odom_at_imu_rate"  # ~100 Hz IMU-integrated odom (optional)

    has_frames = FRAME_TOPIC in bag.topics_map
    has_odom = ODOM_TOPIC in bag.topics_map
    has_hf_odom = HF_ODOM_TOPIC in bag.topics_map

    # -------------------------
    # Preallocate datasets for frames
    # -------------------------
    max_pts = 2048 * 64
    frame_idx = 0
    num_frames = 0
    range_ds = refl_ds = sig_ds = nir_ds = t_ds = None

    if has_frames:
        num_frames = bag.get_message_count(FRAME_TOPIC)
        if num_frames > 0:
            ouster_grp = h5_file.require_group("ouster")
            range_ds = ouster_grp.create_dataset("range_pcl", shape=(num_frames, max_pts, 3), chunks=(1, max_pts, 3), compression=hdf5plugin.Blosc2(cname='zstd', clevel=5, filters=hdf5plugin.Blosc2.BITSHUFFLE), dtype=np.int32)
            refl_ds = ouster_grp.create_dataset("refl_pcl", shape=(num_frames, max_pts),    chunks=(1, max_pts),    compression=hdf5plugin.Blosc2(cname='zstd', clevel=5, filters=hdf5plugin.Blosc2.BITSHUFFLE), dtype=np.uint8)
            sig_ds  = ouster_grp.create_dataset("sig_pcl",  shape=(num_frames, max_pts),    chunks=(1, max_pts),    compression=hdf5plugin.Blosc2(cname='zstd', clevel=5, filters=hdf5plugin.Blosc2.BITSHUFFLE), dtype=np.uint16)
            nir_ds  = ouster_grp.create_dataset("nir_pcl",  shape=(num_frames, max_pts),    chunks=(1, max_pts),    compression=hdf5plugin.Blosc2(cname='zstd', clevel=5, filters=hdf5plugin.Blosc2.BITSHUFFLE), dtype=np.uint16)
            t_ds = ouster_grp.create_dataset("t", shape=(num_frames,), dtype=np.float64)

    # Preallocate for odom (both lidar-rate and the optional IMU-rate stream)
    odom_msgs = []
    odom_ts = []
    hf_odom_msgs = []
    hf_odom_ts = []

    # -------------------------
    # Single-pass: read FRAME + ODOM (+ optional HF_ODOM) topics in one bag scan
    # -------------------------
    iter_list = [t for t in [FRAME_TOPIC, ODOM_TOPIC, HF_ODOM_TOPIC] if t in bag.topics_map]
    if iter_list:
        for topic, msg, ts in bag.iter_topics(iter_list):
            if topic == FRAME_TOPIC and range_ds is not None:
                pts = point_cloud2.read_points_numpy(msg,
                    field_names=("x", "y", "z", "reflectivity", "signal", "near_ir"),
                    skip_nans=True)

                if pts.size == 0:
                    logger.warning(f"Empty frame at index {frame_idx}")
                    frame_idx += 1
                    continue

                n_pts = len(pts)
     
                # Convert from meters (float32) to millimeters (int32)
                range_ds[frame_idx, :n_pts, 0] = (pts[:,0] * 1000).astype(np.int32)
                range_ds[frame_idx, :n_pts, 1] = (pts[:,1] * 1000).astype(np.int32)
                range_ds[frame_idx, :n_pts, 2] = (pts[:,2] * 1000).astype(np.int32)
                t_ds[frame_idx] = msg.header.stamp.sec + msg.header.stamp.nanosec * NS_TO_S
                refl_ds[frame_idx, :n_pts] = pts[:,3].astype(np.uint8)
                sig_ds[frame_idx, :n_pts] = pts[:,4].astype(np.uint16)
                nir_ds[frame_idx, :n_pts] = pts[:,5].astype(np.uint16)
    
                frame_idx += 1

            elif topic == ODOM_TOPIC:
                t_sec = msg.header.stamp.sec + msg.header.stamp.nanosec * NS_TO_S
                odom_ts.append(t_sec)
                odom_msgs.append(msg)

            elif topic == HF_ODOM_TOPIC:
                t_sec = msg.header.stamp.sec + msg.header.stamp.nanosec * NS_TO_S
                hf_odom_ts.append(t_sec)
                hf_odom_msgs.append(msg)

    if has_frames and num_frames > 0 and frame_idx != num_frames:
        logger.warning(
            f"Processed {frame_idx} frames but expected {num_frames} for {FRAME_TOPIC}"
        )

    # -------------------------
    # Write LIO odometry to H5
    # -------------------------
    def _write_odom_group(group_name: str, msgs, ts):
        """Pack a list of nav_msgs/Odometry into parallel datasets under /ouster/<group_name>/.
        Dataset layout (matches the existing /ouster/odom/* convention):
            t            (N,)         seconds (header.stamp)
            map_T_lidart (N, 4, 4)    SE(3) pose (msg.pose; child_frame_id is base_link from rko_lio)
            lin_vel      (N, 3)       msg.twist.linear
            ang_vel      (N, 3)       msg.twist.angular
        """
        if not msgs:
            return
        from scipy.spatial.transform import Rotation as R

        N = len(msgs)
        map_T_lidart = np.zeros((N, 4, 4), dtype=np.float64)
        lin_vel = np.empty((N, 3), dtype=np.float64)
        ang_vel = np.empty((N, 3), dtype=np.float64)
        for i, m in enumerate(msgs):
            p = m.pose.pose.position
            q = m.pose.pose.orientation
            T = np.eye(4, dtype=np.float64)
            T[0:3, 0:3] = R.from_quat([q.x, q.y, q.z, q.w]).as_matrix()
            T[0:3, 3] = [p.x, p.y, p.z]
            map_T_lidart[i] = T
            v = m.twist.twist.linear
            w = m.twist.twist.angular
            lin_vel[i] = [float(v.x), float(v.y), float(v.z)]
            ang_vel[i] = [float(w.x), float(w.y), float(w.z)]

        ouster_grp = h5_file.require_group("ouster")
        grp = ouster_grp.require_group(group_name)
        for k, data in (("t", np.asarray(ts, dtype=np.float64)),
                        ("map_T_lidart", map_T_lidart),
                        ("lin_vel", lin_vel),
                        ("ang_vel", ang_vel)):
            grp.create_dataset(k, data=data, dtype=np.float64)

    _write_odom_group("odom", odom_msgs, odom_ts)
    if hf_odom_msgs:
        logger.info(f"Writing {len(hf_odom_msgs)} HF (IMU-rate) odometry samples to /ouster/hf_odom/")
        _write_odom_group("hf_odom", hf_odom_msgs, hf_odom_ts)

    # -------------------------
    # Raw Ouster IMU samples: the LIO bag has no Ouster IMU or time-status, so
    # pull them from the original raw bag (re-deriving the same filt_time origin).
    # -------------------------
    raw_bag = None
    if raw_bag_path and os.path.exists(raw_bag_path):
        try:
            raw_bag = BagTopicReader(raw_bag_path)
        except Exception as exc:
            logger.exception(f"Failed to open raw bag {raw_bag_path} for IMU data: {exc}")
    elif raw_bag_path:
        logger.warning(f"Raw bag path {raw_bag_path} does not exist; skipping IMU fallback")

    if raw_bag and OUSTER_IMU_TOPIC in raw_bag.topics_map:
        filt_time = _ouster_filt_time(raw_bag)

        # Load metadata JSON for packet parsing
        resolved_metadata_path = metadata_path or OUSTER_METADATA_REV7_1
        metadata_text = None
        try:
            with open(resolved_metadata_path, 'r') as f:
                metadata_text = f.read()
        except Exception as exc:
            logger.exception(f"Failed to read metadata at {resolved_metadata_path}: {exc}")

        if metadata_text:
            metadata = client.SensorInfo(metadata_text)
            pf = client.PacketFormat.from_info(metadata)

            imu_msgs = raw_bag.get_message_count(OUSTER_IMU_TOPIC)
            gyro_data = np.empty((imu_msgs, 3), dtype=np.float32)
            accel_data = np.empty((imu_msgs, 3), dtype=np.float32)
            gyro_ts_ar = np.empty(imu_msgs, dtype=np.float64)
            accel_ts_ar = np.empty(imu_msgs, dtype=np.float64)
            cnt = 0

            for packet_msg, ts in raw_bag.iter_topic(OUSTER_IMU_TOPIC):
                if ts < filt_time:
                    continue

                packet = client.ImuPacket(len(packet_msg.buf))
                packet.buf[:] = packet_msg.buf

                gyro_ts = pf.imu_gyro_ts(packet.buf) * NS_TO_S - filt_time
                accel_ts = pf.imu_accel_ts(packet.buf) * NS_TO_S - filt_time

                if accel_ts < 0 or gyro_ts < 0:
                    continue

                ax = pf.imu_la_x(packet.buf)
                ay = pf.imu_la_y(packet.buf)
                az = pf.imu_la_z(packet.buf)

                wx = pf.imu_av_x(packet.buf)
                wy = pf.imu_av_y(packet.buf)
                wz = pf.imu_av_z(packet.buf)

                accel_data[cnt, 0] = ax
                accel_data[cnt, 1] = ay
                accel_data[cnt, 2] = az
                gyro_data[cnt, 0] = wx
                gyro_data[cnt, 1] = wy
                gyro_data[cnt, 2] = wz
                accel_ts_ar[cnt] = accel_ts
                gyro_ts_ar[cnt] = gyro_ts
                cnt += 1

            if cnt > 0:
                ouster_grp = h5_file.require_group("ouster")
                for ds_name in ["accel_t", "ang_t", "accel", "ang_vel"]:
                    if ds_name in ouster_grp:
                        del ouster_grp[ds_name]

                ouster_grp.create_dataset("accel_t", data=accel_ts_ar[:cnt], dtype=np.float64)
                ouster_grp.create_dataset("ang_t", data=gyro_ts_ar[:cnt], dtype=np.float64)
                ouster_grp.create_dataset("accel", data=accel_data[:cnt], dtype=np.float32)
                ouster_grp.create_dataset("ang_vel", data=gyro_data[:cnt], dtype=np.float32)
                logger.info(
                    f"Wrote {cnt} raw IMU packets from {OUSTER_IMU_TOPIC} (using metadata {resolved_metadata_path})"
                )
            else:
                logger.info("No IMU packets survived filtering; skipping IMU dataset write")
        else:
            logger.warning("Metadata unavailable; skipping raw IMU decoding for LIO output")

    duration = time.time() - st
    logger.info(f"parse_lidar_lio completed in {duration:.2f}s for {lio_bag_path}")

def parse_can(bag_name, h5_file, device_clock_system, system_filter):
    """Parse CAN data and write each signal to HDF5 under /car group.
    
    Signals parsed:
        - steer: Steering angle (CAN ID 130)
        - brake_on: Brake on/off (CAN ID 357)
        - pedal: Gas pedal position (CAN ID 514)
        - speed: Vehicle speed (CAN ID 514)
        - steer_rate: Steering angle rate (CAN ID 577)
        - wheels: Wheel speeds [FL, FR, RL, RR] (CAN ID 533)
        - vcc: Vehicle acceleration [X, Y] (CAN ID 120)
        - brake_press: Brake pressure (CAN ID 120)
    """
    st = time.time()
    
    # Check if required data is available
    if device_clock_system is None or system_filter is None:
        logger.warning("System calibration data is None, skipping CAN parsing")
        return
    
    # Load CAN databases for decoding (chassis + radar), fetching them from
    # opendbc on first use if they're not already present.
    try:
        _ensure_opendbc_dbc(CAN_DBC_PATH)
        db = cantools.database.load_file(CAN_DBC_PATH, strict=False)
    except Exception as e:
        logger.error(f"Failed to load CAN DBC file from {CAN_DBC_PATH}: {e}")
        return
    try:
        _ensure_opendbc_dbc(CAN_RADAR_DBC_PATH)
        db_radar = cantools.database.load_file(CAN_RADAR_DBC_PATH, strict=False)
    except Exception as e:
        logger.warning(f"Failed to load radar DBC file from {CAN_RADAR_DBC_PATH}: {e} "
                       f"-- radar tracks will be skipped")
        db_radar = None

    bag = BagTopicReader(bag_name)
    
    # Initialize lists for each signal and its timestamps
    # CAN ID 130: Steering angle
    steer = []
    steer_ts = []
    
    # CAN ID 357: Brake on
    brake_on = []
    brake_on_ts = []
    
    # CAN ID 514: Pedal and Speed
    pedal = []
    pedal_ts = []
    speed = []
    speed_ts = []
    
    # CAN ID 577: Steering angle rate
    steer_rate = []
    steer_rate_ts = []
    
    # CAN ID 533: Wheel speeds
    wheels = []
    wheels_ts = []
    
    # CAN ID 120: Vehicle acceleration and brake pressure
    vcc = []
    vcc_ts = []
    brake_press = []
    brake_press_ts = []

    # CAN IDs 865..870: 6 radar object tracks (RADAR_TRACK_361..366). Each
    # message carries DIST_OBJ (12-bit unsigned, sentinel 4095 = no track),
    # ANG_OBJ (12-bit signed), RELV_OBJ (11-bit signed). Values stored RAW
    # (no scale/offset)
    radar = {tid: {"t": [], "dist": [], "ang": [], "vrel": []} for tid in RADAR_TRACK_IDS}
    radar_id_set = set(RADAR_TRACK_IDS)

    # Parse all CAN messages
    # Use bag recording time (OS system clock) 
    for msg, bag_ts in bag.iter_topic(CAN_TOPIC):
        msg_time = bag_ts * NS_TO_S
        
        try:
            if msg.id == 130:
                decoded = db.decode_message(msg.id, bytes(msg.data))
                steer.append(decoded['STEER_ANGLE'])
                steer_ts.append(msg_time)
                
            elif msg.id == 357:
                decoded = db.decode_message(msg.id, bytes(msg.data))
                brake_on.append(decoded['BRAKE_ON'])
                brake_on_ts.append(msg_time)
                
            elif msg.id == 514:
                decoded = db.decode_message(msg.id, bytes(msg.data))
                pedal.append(decoded['PEDAL_GAS'])
                pedal_ts.append(msg_time)
                speed.append(decoded['SPEED'])
                speed_ts.append(msg_time)
                
            elif msg.id == 577:
                decoded = db.decode_message(msg.id, bytes(msg.data))
                steer_rate.append(decoded['STEER_ANGLE_RATE'])
                steer_rate_ts.append(msg_time)
                
            elif msg.id == 533:
                decoded = db.decode_message(msg.id, bytes(msg.data))
                wheels.append(np.array([decoded['FL'], decoded['FR'], decoded['RL'], decoded['RR']]))
                wheels_ts.append(msg_time)
                
            elif msg.id == 120:
                decoded = db.decode_message(msg.id, bytes(msg.data))
                vcc.append(np.array([decoded['VEHICLE_ACC_X'], decoded['VEHICLE_ACC_Y']]))
                vcc_ts.append(msg_time)
                brake_press.append(decoded['BRAKE_PRESSURE'])
                brake_press_ts.append(msg_time)

            elif msg.id in radar_id_set and db_radar is not None:
                decoded = db_radar.decode_message(msg.id, bytes(msg.data))
                radar[msg.id]["t"].append(msg_time)
                radar[msg.id]["dist"].append(int(decoded["DIST_OBJ"]))
                radar[msg.id]["ang"].append(int(decoded["ANG_OBJ"]))
                radar[msg.id]["vrel"].append(int(decoded["RELV_OBJ"]))
        except Exception as e:
            # Skip messages that can't be decoded
            continue
    
    # Create car group
    car_grp = h5_file.require_group('car')
    
    # Helper function to align timestamps and write to h5
    # Time is stored as the first column of each data array
    def write_signal(name, data, timestamps, scale=1.0, units=None):
        if len(data) == 0:
            logger.warning(f"No data for car/{name}, skipping")
            return

        ts_arr = np.array(timestamps, dtype=np.float64)
        ts_mapped = map_events_to_global(
            ts_arr, device_clock_system, system_filter['xs_smooth'],
        )

        valid_mask = ts_mapped > 0
        valid_ts = ts_mapped[valid_mask]

        data_arr = np.array(data, dtype=np.float64) * scale   # scale applies to data, not the time column
        valid_data = data_arr[valid_mask]
        output = np.column_stack([valid_ts, valid_data])       # [time, value(s)]

        if len(output) == 0:
            logger.warning(f"No valid timestamps for car/{name}, skipping")
            return

        ds = car_grp.create_dataset(name, data=output, dtype=np.float32)
        if units:
            ds.attrs['units'] = units
            if scale != 1.0:
                ds.attrs['note'] = f"converted from raw CAN signal (x{scale:.6f})"
        logger.info(f"car/{name}: {len(output)} samples written, shape {output.shape}")

    # Write each signal to the car group. Time is the first column.
    KMH_TO_MPH = 0.6213711922
    write_signal('steer', steer, steer_ts)
    write_signal('brake_on', brake_on, brake_on_ts)
    write_signal('pedal', pedal, pedal_ts)
    write_signal('speed', speed, speed_ts, scale=KMH_TO_MPH, units='mph')
    write_signal('steer_rate', steer_rate, steer_rate_ts)
    write_signal('wheels', wheels, wheels_ts, scale=KMH_TO_MPH, units='mph')
    write_signal('vcc', vcc, vcc_ts)
    write_signal('brake_press', brake_press, brake_press_ts)

    # ── Radar tracks: per-track subgroup under /car/radar/track_{1..6} ──
    # Layout per track:
    #   /car/radar/track_N/t     (M,)  float64  main-clock seconds
    #   /car/radar/track_N/dist  (M,)  uint16   DIST_OBJ raw (sentinel 4095)
    #   /car/radar/track_N/ang   (M,)  int16    ANG_OBJ raw (signed 12-bit)
    #   /car/radar/track_N/vrel  (M,)  int16    RELV_OBJ raw (signed 11-bit)
    # Each track is recorded for all messages received (including the no-track
    # sentinel); callers filter by `dist < 4095` to get actual detections.
    radar_grp = car_grp.require_group("radar")
    for tid in RADAR_TRACK_IDS:
        track_num = tid - 864                       # 865 -> 1, ..., 870 -> 6
        buf = radar[tid]
        if len(buf["t"]) == 0:
            continue
        ts_arr = np.asarray(buf["t"], dtype=np.float64)
        ts_mapped = map_events_to_global(
            ts_arr, device_clock_system, system_filter['xs_smooth'],
        )
        valid_mask = ts_mapped > 0
        if not valid_mask.any():
            continue
        track_grp = radar_grp.require_group(f"track_{track_num}")
        for name, dtype in (("t",    np.float64),
                            ("dist", np.uint16),
                            ("ang",  np.int16),
                            ("vrel", np.int16)):
            if name == "t":
                data = ts_mapped[valid_mask]
            else:
                data = np.asarray(buf[name])[valid_mask]
            track_grp.create_dataset(name, data=data, dtype=dtype)
        n_valid_obj = int((track_grp["dist"][:] < 4095).sum())
        logger.info(f"car/radar/track_{track_num} (CAN {tid}): "
                    f"{int(valid_mask.sum())} msgs, {n_valid_obj} with actual detection")

    duration = time.time() - st
    logger.info(f"parse_can completed in {duration:.2f}s")


def process_bag(
    bag_name: str,
    calibration_data,
    *,
    mode: str = "calibration",
    cal_map_path: Optional[str] = None,
    lio_bag: Optional[str] = None,
):
    """Main function to process a bag file.

    Parameters
    ----------
    bag_name : str
        Primary rosbag to process
    calibration_data : CalibrationData
        Time offset calibration for mapping device times to global time.
    mode : {"calibration", "data"}, optional
        - "calibration": default behavior, process full raw data (including Ouster packets).
        - "data": run on data-collection bags, optionally attaching calibration metadata
          from cal_map.yaml and using a separate LIO bag for Ouster data.
    cal_map_path : str, optional
        Path to cal_map.yaml; used only in data mode to log calibration provenance
        into the output HDF5 under /calib.
    lio_bag : str, optional
        Optional LIO rosbag containing processed Ouster data (e.g. /rko_lio/frame and
        /rko_lio/odometry). If not provided in data mode, we fall back to raw Ouster
        packets from the main bag.
    """

    is_data_mode = mode == "data"

    output_path, bn = get_processed_path(bag_name)
    os.makedirs(output_path, exist_ok=True)
    configure_file_logging(output_path)

    local_tmp_dir = f"/tmp/timeSync_{bn}_{os.getpid()}"
    os.makedirs(local_tmp_dir, exist_ok=True)

    final_h5_path = f"{output_path}/{bn}.h5"
    final_events_h5_path = f"{output_path}/{bn}_events.h5"

    f_w = h5py.File(final_h5_path, "w")
    logger.info(f"Writing to: {final_h5_path}")
    logger.info(f"Events (if any) will go to sibling file: {final_events_h5_path}")
    st = time.time()

    # Load calibration provenance for data runs (if provided). _load_calibration_map
    # returns {} for a None/missing path, so this stays empty outside data mode.
    cal_map = _load_calibration_map(cal_map_path) if is_data_mode else {}

    bag = BagTopicReader(bag_name)

    # -------------------------------------------------------------------------
    # GLOBAL PARALLELIZATION ("Grand Central Dispatch")
    # -------------------------------------------------------------------------
    # We launch all independent sensor tasks as separate processes.
    # Each process writes to its own temporary HDF5 file.
    # Finally, we join them and merge everything into f_w.
    
    procs = []
    tmp_files_to_merge = []  # List of (path, prefix=None)

    # 1. Lidar Worker
    # Check if we need to process Lidar
    if OUSTER_LIDAR_TOPIC in bag.topics_map:
        lidar_tmp_path = f"{local_tmp_dir}/lidar.h5"
        tmp_files_to_merge.append((lidar_tmp_path, None))
        
        # Ouster metadata: default to the bundled rev7_1 sensor file, unless
        # cal_map names one (relative names resolve under RAW_ROOT/calibrations).
        lidar_cal = cal_map.get("lidar_cal")
        if lidar_cal:
            metadata_path = lidar_cal if os.path.isabs(lidar_cal) \
                else os.path.join(RAW_ROOT, "calibrations", lidar_cal)
        else:
            metadata_path = OUSTER_METADATA_REV7_1


        p = multiprocessing.Process(
            target=_lidar_worker,
            args=(bag_name, lidar_tmp_path, mode, cal_map_path, lio_bag, metadata_path)
        )
        procs.append(p)
    else:
        logger.info("No Ouster Lidar topic, skipping lidar worker startup.")

    # 2. CAN Worker
    if CAN_TOPIC in bag.topics_map:
        can_tmp_path = f"{local_tmp_dir}/can.h5"
        tmp_files_to_merge.append((can_tmp_path, None))
        p = multiprocessing.Process(
            target=_can_worker,
            args=(bag_name, can_tmp_path, 
                  calibration_data.device_clock_system,
                  calibration_data.system_filter)
        )
        procs.append(p)

    # 3. Misc Sensors Worker (GPS, Infrared, IMU)
    misc_tmp_path = f"{local_tmp_dir}/misc.h5"
    tmp_files_to_merge.append((misc_tmp_path, None))
    p = multiprocessing.Process(
        target=_misc_worker,
        args=(bag_name, misc_tmp_path, calibration_data)
    )
    procs.append(p)

    # 4. Event Camera Workers
    if EVENT_CAM0_TOPIC in bag.topics_map:
        ev0_tmp_path = f"{local_tmp_dir}/ev_right.h5"
        tmp_files_to_merge.append((ev0_tmp_path, "ev/right"))
        procs.append(multiprocessing.Process(
            target=_event_camera_worker,
            args=(bag_name, ev0_tmp_path, EVENT_CAM0_TOPIC, "ev/right",
                  calibration_data.device_clock_ev_0,
                  calibration_data.event_filter_0),
        ))
    if EVENT_CAM1_TOPIC in bag.topics_map:
        ev1_tmp_path = f"{local_tmp_dir}/ev_left.h5"
        tmp_files_to_merge.append((ev1_tmp_path, "ev/left"))
        procs.append(multiprocessing.Process(
            target=_event_camera_worker,
            args=(bag_name, ev1_tmp_path, EVENT_CAM1_TOPIC, "ev/left",
                  calibration_data.device_clock_ev_1,
                  calibration_data.event_filter_1),
        ))

    # 5. FLIR Camera Workers
    if FLIR_CAM0_TRIGGER_TOPIC in bag.topics_map:
        flir0_tmp_path = f"{local_tmp_dir}/img_right.h5"
        tmp_files_to_merge.append((flir0_tmp_path, "img/right/t"))
        procs.append(multiprocessing.Process(
            target=_flir_worker,
            args=(bag_name, flir0_tmp_path,
                  FLIR_CAM0_TRIGGER_TOPIC, FLIR_CAM0_TOPIC, "img_right", "img/right/t",
                  calibration_data.device_clock_flir_0,
                  calibration_data.flir_filter_0)
        ))

    if FLIR_CAM1_TRIGGER_TOPIC in bag.topics_map:
        flir1_tmp_path = f"{local_tmp_dir}/img_left.h5"
        tmp_files_to_merge.append((flir1_tmp_path, "img/left/t"))
        procs.append(multiprocessing.Process(
            target=_flir_worker,
            args=(bag_name, flir1_tmp_path,
                  FLIR_CAM1_TRIGGER_TOPIC, FLIR_CAM1_TOPIC, "img_left", "img/left/t",
                  calibration_data.device_clock_flir_1,
                  calibration_data.flir_filter_1)
        ))

    # -------------------------------------------------------------------------
    # LAUNCH EVERYTHING
    # -------------------------------------------------------------------------
    logger.info(f"Launching {len(procs)} parallel processing workers...")
    for p in procs:
        p.start()
    
    # -------------------------------------------------------------------------
    # WAIT AND CHECK ERRORS
    # -------------------------------------------------------------------------
    for p in procs:
        p.join()
        if p.exitcode != 0:
            raise RuntimeError(f"Worker subprocess {p.name} (pid {p.pid}) exited with code {p.exitcode}")
    
    logger.info("All workers finished. Merging HDF5 files...")

    # -------------------------------------------------------------------------
    # MERGE RESULTS
    # -------------------------------------------------------------------------
    # Merge temp HDF5 datasets into the right destination:
    #   - prefix starts with "ev/"  → routed to the SIBLING <bn>_events.h5
    #     (split out so HF consumers who don't want the ~22 GB raw event stream
    #     can skip downloading it).
    #   - everything else (lidar, CAN, misc, FLIR meta) → main <bn>.h5.
    f_events = None
    for tmp_path, prefix in tmp_files_to_merge:
        if not os.path.exists(tmp_path):
            continue

        is_events = bool(prefix) and prefix.startswith("ev/")
        if is_events and f_events is None:
            f_events = h5py.File(final_events_h5_path, "w")

        dst = f_events if is_events else f_w

        with h5py.File(tmp_path, 'r') as tmp_h5:
            if prefix:
                if prefix in tmp_h5:
                    tmp_h5.copy(prefix, dst, name=prefix)
            else:
                for key in tmp_h5.keys():
                    tmp_h5.copy(key, dst, name=key)

        os.remove(tmp_path)


    # # Finally, append calibration metadata at the end of the file (data mode only).
    if is_data_mode and cal_map:
        extra_files = {}
        if "metadata_path" in locals():
            extra_files["lidar_metadata"] = metadata_path
        _write_calibration_metadata(f_w, cal_map, extra_files=extra_files, events_file=f_events)

    f_w.close()
    if f_events is not None:
        f_events.close()
        logger.info(f"Events written: {final_events_h5_path} "
                    f"({os.path.getsize(final_events_h5_path)/1e9:.2f} GB)")

    # Clean up worker scratch directory
    try:
        shutil.rmtree(local_tmp_dir, ignore_errors=True)
        logger.info(f"Cleaned up scratch directory: {local_tmp_dir}")
    except Exception as cleanup_exc:
        logger.warning(f"Failed to clean up scratch directory {local_tmp_dir}: {cleanup_exc}")

    # ── Phase-3: fuse RKO-LIO + GPS → /ref group in the final bag h5 ──
    # Runs only in data mode and only when cal_map is loaded (so we have a
    # lever to apply).
    if is_data_mode and cal_map:
        _fuse_reference_trajectory(final_h5_path, cal_map_path)

    duration = time.time() - st
    logger.info(f"Finished processing {bag_name} in {duration:.2f}s")


def _fuse_reference_trajectory(final_h5_path: str, cal_map_path: Optional[str]) -> None:
    """Invoke fuse_trajectory on the just-written bag h5. Writes the /fused_traj
    group in-place.

    Imported lazily so its heavy optional deps (gtsam, pyproj,
    projectaria_tools) aren't required unless fusion is actually run.
    """
    try:
        import fuse_trajectory
    except Exception as imp_exc:
        logger.warning(f"fuse_trajectory not importable: {imp_exc} "
                       f"-- skipping /fused_traj generation")
        return

    # IMPORTANT: fuse_trajectory needs the GLOBAL cal_map (which has the
    # platform→lever_arm map under `car:` etc)
    global_cal_map = getattr(fuse_trajectory, "DEFAULT_CAL_MAP", None)
    logger.info(f"Fusing RKO-LIO + GPS into /fused_traj on {final_h5_path}...")
    t_fuse = time.time()
    try:
        fuse_trajectory.run(
            h5=final_h5_path,
            cal_map=global_cal_map,
        )
        logger.info(f"fuse_trajectory completed in {time.time()-t_fuse:.1f}s")
    except Exception as exc:
        logger.exception(f"fuse_trajectory failed (non-fatal, h5 still written): {exc}")

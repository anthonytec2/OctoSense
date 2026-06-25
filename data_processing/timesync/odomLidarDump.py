"""
Offline LiDAR / IMU bag re-writer for Ouster packets.

This script:
  - Reads a ROS 2 bag recorded with raw Ouster packet topics
    (`/lidar_packets`, `/imu_packets`, `/ouster_time_status`).
  - Uses the Ouster SDK to decode LiDAR packets into XYZ point clouds.
  - Uses the same timestamp handling as `timeSync_process.parse_lidar`
    (i.e. uses `/ouster_time_status` and discards data before time sync lock).
  - Decodes the Ouster IMU packets to linear acceleration and angular velocity.
  - Writes a NEW ROS 2 bag (MCAP) containing:
      * A LiDAR point cloud topic with fields (x, y, z, time)
      * An IMU topic with corrected timestamps
    suitable to be consumed by odometry pipelines (e.g. rko_lio offline node).
"""

import argparse
import json
import logging
import os
import shutil
import time
from typing import Optional, Tuple

import numpy as np
import rosbag2_py
from rclpy.serialization import serialize_message
from sensor_msgs.msg import PointCloud2, PointField, Imu
from geometry_msgs.msg import TransformStamped
from tf2_msgs.msg import TFMessage
from scipy.spatial.transform import Rotation as R
from ouster.sdk import client

from timeSyncUtil import BagTopicReader
from timeSync_process import _ouster_filt_time
from timeSync_constants import (
    NS_TO_S,
    S_TO_NS,
    OUSTER_LIDAR_TOPIC,
    OUSTER_IMU_TOPIC,
)

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")


def _load_imu_extrinsic_from_metadata(
    metadata_path: str,
) -> Optional[Tuple[float, float, float, float, float, float, float]]:
    """
    Load IMU->sensor (LiDAR) extrinsic from Ouster metadata.

    Uses the `imu_intrinsics.imu_to_sensor_transform` 4x4 matrix if present.
    Returns quaternion (x, y, z, w) and translation (x, y, z) representing
    the transform from the sensor (base_frame) to the IMU frame, suitable
    for publishing as TF with parent=base_frame, child=imu_frame.
    """
    try:
        with open(metadata_path, "r") as f:
            meta_text = f.read()
        meta = json.loads(meta_text)
    except Exception as exc:
        logger.warning(
            "Failed to parse metadata at %s for IMU extrinsics (%s); "
            "falling back to identity.",
            metadata_path,
            exc,
        )
        return None

    try:
        imu_intrinsics = meta.get("imu_intrinsics", {})
        t_list = imu_intrinsics.get("imu_to_sensor_transform", None)
        if not t_list or len(t_list) != 16:
            logger.warning(
                "imu_intrinsics.imu_to_sensor_transform missing or malformed "
                "in metadata; falling back to identity."
            )
            return None

        T = np.array(t_list, dtype=float).reshape(4, 4)
        R_mat = T[:3, :3]
        t = T[:3, 3]

        # Use SciPy's Rotation to convert to quaternion (x, y, z, w)
        rot = R.from_matrix(R_mat)
        qx, qy, qz, qw = rot.as_quat()  # returns (x, y, z, w)

        return float(qx), float(qy), float(qz), float(qw), float(t[0]), float(t[1]), float(t[2])
    except Exception as exc:
        logger.warning(
            "Error extracting IMU extrinsic from metadata (%s); "
            "falling back to identity.",
            exc,
        )
        return None


def _create_bag_writer(output_uri: str) -> rosbag2_py.SequentialWriter:
    """Create and open a rosbag2 (MCAP) writer."""
    if os.path.exists(output_uri):
        logger.info("Removing existing output bag directory: %s", output_uri)
        shutil.rmtree(output_uri)

    storage_options = rosbag2_py.StorageOptions(uri=output_uri, storage_id="mcap")
    converter_options = rosbag2_py.ConverterOptions(
        input_serialization_format="cdr",
        output_serialization_format="cdr",
    )
    writer = rosbag2_py.SequentialWriter()
    writer.open(storage_options, converter_options)
    return writer


def _register_topics(
    writer: rosbag2_py.SequentialWriter,
    lidar_topic: str,
    imu_topic: str,
    add_tf: bool,
) -> None:
    """Register PointCloud2, Imu (and optionally TF) topics in the writer."""
    topic_metadata = rosbag2_py.TopicMetadata(
        id=0,
        name=lidar_topic,
        type="sensor_msgs/msg/PointCloud2",
        serialization_format="cdr",
    )
    writer.create_topic(topic_metadata)

    topic_metadata = rosbag2_py.TopicMetadata(
        id=1,
        name=imu_topic,
        type="sensor_msgs/msg/Imu",
        serialization_format="cdr",
    )
    writer.create_topic(topic_metadata)

    if add_tf:
        topic_metadata = rosbag2_py.TopicMetadata(
            id=2,
            name="/tf_static",
            type="tf2_msgs/msg/TFMessage",
            serialization_format="cdr",
        )
        writer.create_topic(topic_metadata)


def _write_static_tf(
    writer: rosbag2_py.SequentialWriter,
    parent_frame: str,
    child_frame: str,
    quat_xyzw_xyz: Tuple[float, float, float, float, float, float, float],
) -> None:
    """Write a single static transform into `/tf_static` at time 0."""
    qx, qy, qz, qw, x, y, z = quat_xyzw_xyz

    tf = TransformStamped()
    tf.header.stamp.sec = 0
    tf.header.stamp.nanosec = 0
    tf.header.frame_id = parent_frame
    tf.child_frame_id = child_frame
    tf.transform.translation.x = float(x)*1e-3
    tf.transform.translation.y = float(y)*1e-3
    tf.transform.translation.z = float(z)*1e-3
    tf.transform.rotation.x = float(qx)
    tf.transform.rotation.y = float(qy)
    tf.transform.rotation.z = float(qz)
    tf.transform.rotation.w = float(qw)

    msg = TFMessage(transforms=[tf])
    writer.write("/tf_static", serialize_message(msg), 0)


def _process_lidar(
    bag: BagTopicReader,
    writer: rosbag2_py.SequentialWriter,
    metadata_path: str,
    lidar_topic: str,
    lidar_frame: str,
    filt_time_s: float,
) -> None:
    """Decode Ouster LiDAR packets into PointCloud2 and write them to a bag."""
    if OUSTER_LIDAR_TOPIC not in bag.topics_map:
        logger.warning("No LiDAR packets on topic %s; skipping LiDAR.", OUSTER_LIDAR_TOPIC)
        return

    if not os.path.exists(metadata_path):
        raise FileNotFoundError(
            f"Ouster metadata JSON not found at {metadata_path}. "
            "Pass --metadata to override the default path."
        )

    with open(metadata_path, "r") as f:
        meta_json = f.read()

    metadata = client.SensorInfo(meta_json)
    xyzlut = client.XYZLut(metadata)
    batcher = client.ScanBatcher(metadata)
    scan = client.LidarScan(metadata)

    # Define PointCloud2 layout:
    #   x, y, z       : float32
    #   time          : float64 (seconds since filt_time_s)
    #   reflectivity  : uint8
    #   signal        : uint16
    #   near_ir       : uint16
    point_dtype = np.dtype(
        {
            "names": ["x", "y", "z", "time", "reflectivity", "signal", "near_ir"],
            "formats": ["<f4", "<f4", "<f4", "<f8", "<u1", "<u2", "<u2"],
            # Keep existing xyz/time layout and append intensity channels.
            # Bytes:
            #   0-3   : x (f32)
            #   4-7   : y (f32)
            #   8-11  : z (f32)
            #   12-19 : time (f64)
            #   20    : reflectivity (u8)
            #   21-22 : signal (u16)
            #   23-24 : near_ir (u16)
            "offsets": [0, 4, 8, 12, 20, 21, 23],
            "itemsize": 25,
        },
        align=False,
    )
    point_step = point_dtype.itemsize

    cloud_msg = PointCloud2()
    cloud_msg.header.frame_id = lidar_frame
    cloud_msg.is_bigendian = False
    cloud_msg.is_dense = True
    cloud_msg.fields = [
        PointField(name="x", offset=0, datatype=PointField.FLOAT32, count=1),
        PointField(name="y", offset=4, datatype=PointField.FLOAT32, count=1),
        PointField(name="z", offset=8, datatype=PointField.FLOAT32, count=1),
        PointField(name="time", offset=12, datatype=PointField.FLOAT64, count=1),
        PointField(name="reflectivity", offset=20, datatype=PointField.UINT8, count=1),
        PointField(name="signal", offset=21, datatype=PointField.UINT16, count=1),
        PointField(name="near_ir", offset=23, datatype=PointField.UINT16, count=1),
    ]

    scan_count = 0
    first_scan_ts_s: Optional[float] = None
    last_scan_ts_s: Optional[float] = None
    pc_first_ts_s: Optional[float] = None  # min per-point "time" across all scans
    pc_last_ts_s: Optional[float] = None   # max per-point "time" across all scans
    overall_start = time.perf_counter()
    for pkt, _ts in bag.iter_topic(OUSTER_LIDAR_TOPIC):
        # pkt is ouster_sensor_msgs/PacketMsg; we only need its buf field.
        lidar_packet = client.LidarPacket(len(pkt.buf))
        lidar_packet.buf[:] = pkt.buf

        if not batcher(lidar_packet, scan):
            continue

        # Scan is complete
        scan_ts_s = float(scan.get_first_valid_column_timestamp()) * NS_TO_S - filt_time_s
        if scan_ts_s < 0.0:
            # Discard scans before the sync time base, same as parse_lidar.
            continue

        xyz = xyzlut(scan).astype(np.float32)  # (H, W, 3)
        # Per-channel intensity images from the Ouster scan
        refl_img = scan.field(client.ChanField.REFLECTIVITY)
        sig_img = scan.field(client.ChanField.SIGNAL)
        nir_img = scan.field(client.ChanField.NEAR_IR)
        H, W, _ = xyz.shape

        # Per-column measurement timestamps from the Ouster SDK (device clock).
        # Convert to global time in seconds relative to filt_time_s, then
        # broadcast to all points in each column.
        col_ts = np.asarray(scan.timestamp, dtype=np.float64)  # shape (W,)
        col_ts_s = col_ts * NS_TO_S - filt_time_s              # seconds

        # Drop columns that are still before the sync time base. We already
        # skip whole scans whose first valid column is before filt_time_s,
        # but earlier columns in the same scan can still be negative.
        valid_cols = col_ts_s >= 0.0
        if not np.any(valid_cols):
            # Nothing in this scan is usable after time sync; skip entire scan.
            continue
        
        col_ts_s = col_ts_s[valid_cols]
        xyz = xyz[:, valid_cols, :]
        refl_img = refl_img[:, valid_cols]
        sig_img = sig_img[:, valid_cols]
        nir_img = nir_img[:, valid_cols]
        H, W, _ = xyz.shape
        

        cloud_msg.height = H
        cloud_msg.width = W
        cloud_msg.point_step = point_step
        cloud_msg.row_step = point_step * W

        # Flatten into structured array matching PointCloud2 fields:
        # x, y, z: float32
        # time   : float64 (seconds since filt_time), per column
        # reflectivity (u8), signal (u16), near_ir (u16)
        flat = np.empty(H * W, dtype=point_dtype)
        flat["x"] = xyz[..., 0].reshape(-1).astype("<f4")
        flat["y"] = xyz[..., 1].reshape(-1).astype("<f4")
        flat["z"] = xyz[..., 2].reshape(-1).astype("<f4")
        # Broadcast column times to all rows, then flatten in row-major order.
        time_flat = np.broadcast_to(col_ts_s.reshape(1, W), (H, W)).reshape(-1).astype(
            "<f8"
        )
        flat["time"] = time_flat
        # Reflectivity is specified as 8 bits; cast down to uint8.
        flat["reflectivity"] = refl_img.reshape(-1).astype("<u1")
        # Signal and Near IR are 16-bit.
        flat["signal"] = sig_img.reshape(-1).astype("<u2")
        flat["near_ir"] = nir_img.reshape(-1).astype("<u2")

        # Use the first valid column timestamp as the message stamp, same as before.
        stamp_ns = int(scan_ts_s * S_TO_NS)
        cloud_msg.header.stamp.sec = int(stamp_ns // S_TO_NS)
        cloud_msg.header.stamp.nanosec = int(stamp_ns % S_TO_NS)
        cloud_msg.data = flat.tobytes()

        writer.write(lidar_topic, serialize_message(cloud_msg), stamp_ns)
        scan_count += 1

        # Track header timestamp range (relative to filt_time_s) for debugging.
        if first_scan_ts_s is None:
            first_scan_ts_s = scan_ts_s
        last_scan_ts_s = scan_ts_s

        # Track internal per-point "time" range (relative to filt_time_s).
        col_min = float(col_ts_s.min())
        col_max = float(col_ts_s.max())
        if pc_first_ts_s is None or col_min < pc_first_ts_s:
            pc_first_ts_s = col_min
        if pc_last_ts_s is None or col_max > pc_last_ts_s:
            pc_last_ts_s = col_max

    logger.info("Wrote %d LiDAR scans to %s", scan_count, lidar_topic)
    if scan_count > 0 and first_scan_ts_s is not None and last_scan_ts_s is not None:
        logger.info(
            "LiDAR header stamp range: [%.6f, %.6f] s (relative to filt_time_s)",
            first_scan_ts_s,
            last_scan_ts_s,
        )
    if scan_count > 0 and pc_first_ts_s is not None and pc_last_ts_s is not None:
        logger.info(
            "LiDAR internal point time range: [%.6f, %.6f] s (relative to filt_time_s)",
            pc_first_ts_s,
            pc_last_ts_s,
        )
    total_elapsed = time.perf_counter() - overall_start
    logger.info("Total LiDAR processing took %.3f s", total_elapsed)


def _process_imu(
    bag: BagTopicReader,
    writer: rosbag2_py.SequentialWriter,
    metadata_path: str,
    imu_topic: str,
    imu_frame: str,
    filt_time_s: float,
) -> None:
    """Decode Ouster IMU packets into Imu messages and write them to a bag."""
    if OUSTER_IMU_TOPIC not in bag.topics_map:
        logger.warning("No IMU packets on topic %s; skipping IMU.", OUSTER_IMU_TOPIC)
        return

    if not os.path.exists(metadata_path):
        raise FileNotFoundError(
            f"Ouster metadata JSON not found at {metadata_path}. "
            "Pass --metadata to override the default path."
        )

    with open(metadata_path, "r") as f:
        meta_json = f.read()
    metadata = client.SensorInfo(meta_json)
    pf = client.PacketFormat.from_info(metadata)

    imu_msg = Imu()
    imu_msg.header.frame_id = imu_frame
    # Unknown orientation
    imu_msg.orientation_covariance[0] = -1.0

    # Ouster units -> ROS Imu: accel g -> m/s^2, angular velocity deg/s -> rad/s.
    g_to_m_s2 = 9.80665
    deg_to_rad = np.pi / 180.0

    count = 0
    for pkt, _ts in bag.iter_topic(OUSTER_IMU_TOPIC):
        imu_packet = client.ImuPacket(len(pkt.buf))
        imu_packet.buf[:] = pkt.buf

        gyro_ts_s = float(pf.imu_gyro_ts(imu_packet.buf)) * NS_TO_S - filt_time_s
        accel_ts_s = float(pf.imu_accel_ts(imu_packet.buf)) * NS_TO_S - filt_time_s
        if accel_ts_s < 0.0 or gyro_ts_s < 0.0:
            continue  # discard IMU packets before sync

        # Stamp with the later of the two timestamps.
        stamp_ns = int(max(accel_ts_s, gyro_ts_s) * S_TO_NS) # max 215us diff btw the two
        imu_msg.header.stamp.sec = int(stamp_ns // S_TO_NS)
        imu_msg.header.stamp.nanosec = int(stamp_ns % S_TO_NS)

        imu_msg.linear_acceleration.x = float(pf.imu_la_x(imu_packet.buf) * g_to_m_s2)
        imu_msg.linear_acceleration.y = float(pf.imu_la_y(imu_packet.buf) * g_to_m_s2)
        imu_msg.linear_acceleration.z = float(pf.imu_la_z(imu_packet.buf) * g_to_m_s2)
        imu_msg.angular_velocity.x = float(pf.imu_av_x(imu_packet.buf) * deg_to_rad)
        imu_msg.angular_velocity.y = float(pf.imu_av_y(imu_packet.buf) * deg_to_rad)
        imu_msg.angular_velocity.z = float(pf.imu_av_z(imu_packet.buf) * deg_to_rad)

        writer.write(imu_topic, serialize_message(imu_msg), stamp_ns)
        count += 1

    logger.info("Wrote %d IMU samples to %s", count, imu_topic)


def dump_lidar(
    bag_path: str,
    output_uri: str,
    metadata_path: str,
    lidar_topic: str,
    imu_topic: str,
    base_frame: str,
    lidar_frame: str,
    imu_frame: str,
    extrinsic_lidar2base: Tuple[float, float, float, float, float, float, float],
) -> None:
    """
    Main entry point: convert raw Ouster packet bag -> decoded LiDAR+IMU bag.
    """
    logger.info("Opening input bag: %s", bag_path)
    bag = BagTopicReader(bag_path)

    filt_time_s = _ouster_filt_time(bag)

    writer = _create_bag_writer(output_uri)
    _register_topics(writer, lidar_topic, imu_topic, add_tf=True)

    # IMU extrinsics: load from metadata if available; otherwise identity.
    imu_extrinsic = _load_imu_extrinsic_from_metadata(metadata_path)

    # Optional: publish extrinsic frames as static TF.
    if base_frame and lidar_frame:
        _write_static_tf(
            writer,
            parent_frame=base_frame,
            child_frame=lidar_frame,
            quat_xyzw_xyz=extrinsic_lidar2base,
        )
    if base_frame and imu_frame:
        quat_xyzw_xyz = imu_extrinsic or (0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0)
        _write_static_tf(
            writer,
            parent_frame=base_frame,
            child_frame=imu_frame,
            quat_xyzw_xyz=quat_xyzw_xyz,
        )

    _process_lidar(
        bag=bag,
        writer=writer,
        metadata_path=metadata_path,
        lidar_topic=lidar_topic,
        lidar_frame=lidar_frame,
        filt_time_s=filt_time_s,
    )
    _process_imu(
        bag=bag,
        writer=writer,
        metadata_path=metadata_path,
        imu_topic=imu_topic,
        imu_frame=imu_frame,
        filt_time_s=filt_time_s,
    )

    logger.info("Finished writing decoded LiDAR+IMU bag to %s", output_uri)


def _parse_extrinsic_arg(value: str) -> Tuple[float, float, float, float, float, float, float]:
    """
    Parse a 7-value string 'qx,qy,qz,qw,x,y,z' into a tuple of floats.
    """
    parts = [p for p in value.replace(" ", "").split(",") if p]
    if len(parts) != 7:
        raise argparse.ArgumentTypeError(
            "Extrinsic must have 7 comma-separated values: qx,qy,qz,qw,x,y,z"
        )
    return tuple(float(p) for p in parts)  # type: ignore[return-value]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Decode Ouster LiDAR/IMU packets to PointCloud2/Imu and write a new rosbag."
    )
    parser.add_argument(
        "--bag",
        required=True,
        type=str,
        help="Path to input ROS 2 bag directory (MCAP, as used by timeSync).",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help=(
            "Output ROS 2 bag directory to create (will be overwritten if exists). "
            "If not provided, defaults to <bag>_lio next to the input bag directory."
        ),
    )
    parser.add_argument(
        "--metadata",
        type=str,
        default="/data/rosbags/raw/calibrations/192.168.123-metadata.json",
        help="Path to Ouster metadata JSON file.",
    )
    parser.add_argument(
        "--lidar-topic",
        type=str,
        default="/rko_lio/lidar",
        help="Output LiDAR PointCloud2 topic name (default: /rko_lio/lidar).",
    )
    parser.add_argument(
        "--imu-topic",
        type=str,
        default="/rko_lio/imu",
        help="Output IMU topic name (default: /rko_lio/imu).",
    )
    parser.add_argument(
        "--base-frame",
        type=str,
        default="base_link",
        help="Base frame id (parent for TF). Default: lidar_link.",
    )
    parser.add_argument(
        "--lidar-frame",
        type=str,
        default="lidar_link",
        help="LiDAR frame id. Default: lidar_link.",
    )
    parser.add_argument(
        "--imu-frame",
        type=str,
        default="imu_link",
        help="IMU frame id. Default: imu_link.",
    )
    parser.add_argument(
        "--extrinsic-lidar2base",
        type=_parse_extrinsic_arg,
        default=(0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0),
        metavar="qx,qy,qz,qw,x,y,z",
        help=(
            "Extrinsic from lidar frame to base frame as "
            "quat_xyzw_xyz (default: identity)."
        ),
    )

    args = parser.parse_args()

    # Derive default output directory if not provided: <bag>_lio next to input.
    if args.output is None:
        bag_abs = os.path.abspath(args.bag)
        bag_dir = bag_abs.rstrip(os.sep)
        parent = os.path.dirname(bag_dir)
        bn = os.path.basename(bag_dir)
        output_uri = os.path.join(parent, f"{bn}_lio")
    else:
        output_uri = args.output

    dump_lidar(
        bag_path=args.bag,
        output_uri=output_uri,
        metadata_path=args.metadata,
        lidar_topic=args.lidar_topic,
        imu_topic=args.imu_topic,
        base_frame=args.base_frame,
        lidar_frame=args.lidar_frame,
        imu_frame=args.imu_frame,
        extrinsic_lidar2base=args.extrinsic_lidar2base,
    )


if __name__ == "__main__":
    main()



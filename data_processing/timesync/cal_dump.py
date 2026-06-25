"""Camera + IMU calibration dump.

Reconstructs event-camera frames from the per-bag HDF5 and writes them, together
with the IMU stream, into a ROS bag (`sensor_msgs/Image` + `sensor_msgs/Imu`).
That bag is the input to the camera + IMU calibration (e.g. Kalibr) used to solve
the camera intrinsics/extrinsics and the camera-IMU transform.
"""
import simple_image_recon
import h5py
import hdf5plugin
import numpy as np
from tqdm import tqdm
import os
import subprocess
import rosbag2_py
from rclpy.serialization import serialize_message
from sensor_msgs.msg import Image
from sensor_msgs.msg import Imu
from torchcodec.decoders import set_cuda_backend, VideoDecoder
import torch
import shutil
width = 640
height = 480
cutoff_event_num=45
fill_ratio=0.6
tile_size=2
dt_unit = 0.1 / 4095.0

import argparse


def write_event(h5_file, direction, max_len, img_ts, writer=None):
    f = h5py.File(h5_file + '/' + h5_file.split('/')[-1] + '.h5', "r")
    base = f'ev/{direction}'
    ms_idx = f[f'{base}/ms_to_idx']
    reconstructor = simple_image_recon.SimpleImageReconstructor()
    reconstructor.initialize(width, height, cutoff_event_num, tile_size, fill_ratio)

    # Pre-allocate message object (reuse instead of recreating)
    if writer is not None:
        msg = Image()
        msg.header.frame_id = f'recon_{direction}'
        msg.height = height
        msg.width = width
        msg.encoding = 'rgb8'
        msg.is_bigendian = 0
        msg.step = width * 3

    # Pre-compute all timestamp indices at once (vectorized)
    img_ts_ms = (img_ts / 0.001).astype(np.uint64)
    ms_start_indices = np.empty(max_len, dtype=np.uint64)
    ms_end_indices = np.empty(max_len, dtype=np.uint64)
    
    ms_start_indices[0] = ms_idx[img_ts_ms[0]]
    for i in range(max_len - 1):
        ms_end_indices[i] = ms_idx[img_ts_ms[i + 1]]
        if i < max_len - 2:
            ms_start_indices[i + 1] = ms_end_indices[i]
    
    # Pre-compute all timestamps for ROS messages
    if writer is not None:
        stamp_ns_array = (img_ts[1:max_len] * 1_000_000_000).astype(np.int64)
        sec_array = stamp_ns_array // 1_000_000_000
        nsec_array = stamp_ns_array % 1_000_000_000

    # Pre-allocate frame buffer (reuse memory)
    frame_buffer = np.empty((height, width, 3), dtype=np.uint8)
    
    # Hoist the event dataset handles out of the loop.
    x_data = f[f'{base}/x']
    y_data = f[f'{base}/y']
    p_data = f[f'{base}/p']
    t_data = f[f'{base}/t']

    for i in tqdm(range(max_len - 1)):
        ms_start_idx = ms_start_indices[i]
        ms_end_idx = ms_end_indices[i]
        
        # Read event data in one go
        x = x_data[ms_start_idx:ms_end_idx]
        y = y_data[ms_start_idx:ms_end_idx]
        p = p_data[ms_start_idx:ms_end_idx]
        t = t_data[ms_start_idx:ms_end_idx].astype(np.float64)/1e6

        t_us = t * 1e6
        reconstructor.process_events(
            t_us.astype(np.uint64),
            x.astype(np.uint32),
            y.astype(np.uint32),
            p.astype(np.uint8)
        )
        
        img = reconstructor.get_image()
        
        # Use out parameter to avoid allocation
        if img.ndim == 2:
            np.clip(img, 0, 255, out=img)
            img_u8 = img.astype(np.uint8)
            frame_buffer[:, :, 0] = img_u8
            frame_buffer[:, :, 1] = img_u8
            frame_buffer[:, :, 2] = img_u8
        else:
            np.clip(img, 0, 255, out=img)
            frame_buffer[:] = img.astype(np.uint8)

        # Write the reconstructed event frame to the ros2 (calibration) bag.
        if writer is not None:
            msg.header.stamp.sec = int(sec_array[i])
            msg.header.stamp.nanosec = int(nsec_array[i])
            msg.data = frame_buffer.tobytes()
            writer.write(f'/event_camera_{direction}', serialize_message(msg), int(stamp_ns_array[i]))


def _publish_flir_frames(decoder, indices, img_ts, writer, topic, frame_id, out_height, out_width, count):
    """Publish grayscale FLIR frames to ROS bag. 
    """

    msg = Image()
    msg.header.frame_id = frame_id
    msg.height = out_height
    msg.width = out_width
    msg.encoding = 'mono8'
    msg.is_bigendian = 0
    msg.step = out_width


    img_ts = (img_ts * 1e9).astype(np.uint64)

    sec_val=img_ts//1_000_000_000
    nsec_val=img_ts%1_000_000_000

    rgb_to_gray = torch.tensor([0.299, 0.587, 0.114], device='cuda', dtype=torch.float32).view(3, 1, 1)

    import cv2
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    for i in tqdm(range(1, count)):
        idx = indices[i]
        frame = decoder[idx]
        gray = (frame[:3].float() * rgb_to_gray).sum(dim=0).byte().cpu().numpy()

        gray = clahe.apply(gray)
        msg.header.stamp.sec = int(sec_val[idx])
        msg.header.stamp.nanosec = int(nsec_val[idx])
        msg.data = gray.tobytes()
        writer.write(topic, serialize_message(msg), int(img_ts[idx]))

def dump_image(h5_file):
    f=h5py.File(h5_file+'/'+h5_file.split('/')[-1]+'.h5', "r")

    ds_factor=5
    
    left_idx = np.arange(0, len(f['img/left/t'][:]), ds_factor)
    left_img_ts = f['img/left/t'][:]
    right_img_ts = f['img/right/t'][:]

    # For each (downsampled) left frame, find the nearest right frame by timestamp.
    target = left_img_ts[left_idx]
    idx = np.clip(np.searchsorted(right_img_ts, target), 1, len(right_img_ts) - 1)
    pick_prev = np.abs(target - right_img_ts[idx - 1]) < np.abs(target - right_img_ts[idx])
    left_to_right_idx = idx - pick_prev
    abs_diff = np.abs(right_img_ts[left_to_right_idx] - target)
    print(f"95% of the time, the difference between the left and right camera is less than: {np.quantile(abs_diff, 0.95):.6f} s")
    left_ts = f[f'ev/left/ms_to_idx']
    right_ts = f[f'ev/right/ms_to_idx']
    max_len=min(min(min(len(left_ts)//(100//ds_factor),len(right_ts)//(100//ds_factor)),len(left_to_right_idx)), len(left_idx))-1
    

    # Setup ROS2 bag writer (MCAP) if requested
    writer = None
    
    bag_uri = os.path.join(h5_file, 'cal2_bag')
    if os.path.exists(bag_uri):
        shutil.rmtree(bag_uri)  # remove old bag directory
    storage_options = rosbag2_py.StorageOptions(uri=bag_uri, storage_id="mcap")
    converter_options = rosbag2_py.ConverterOptions(
        input_serialization_format="cdr",
        output_serialization_format="cdr",
    )
    writer = rosbag2_py.SequentialWriter()
    writer.open(storage_options, converter_options)

    # Add id parameter (starting from 0)
    topic_metadata = rosbag2_py.TopicMetadata(
        id=0,
        name='/event_camera_left',
        type='sensor_msgs/msg/Image',
        serialization_format='cdr'
    )
    writer.create_topic(topic_metadata)

    topic_metadata = rosbag2_py.TopicMetadata(
        id=1,
        name='/event_camera_right',
        type='sensor_msgs/msg/Image',
        serialization_format='cdr'
    )
    writer.create_topic(topic_metadata)

    topic_metadata = rosbag2_py.TopicMetadata(
        id=2,
        name='/flir_cam_left',
        type='sensor_msgs/msg/Image',
        serialization_format='cdr'
    )
    writer.create_topic(topic_metadata)
    topic_metadata = rosbag2_py.TopicMetadata(
        id=3,
        name='/flir_cam_right',
        type='sensor_msgs/msg/Image',
        serialization_format='cdr'
    )
    writer.create_topic(topic_metadata)

    # IMU topic
    topic_metadata = rosbag2_py.TopicMetadata(
        id=4,
        name='/imu/data',
        type='sensor_msgs/msg/Imu',
        serialization_format='cdr'
    )
    writer.create_topic(topic_metadata)

    write_event(h5_file, 'left', max_len, left_img_ts[left_idx], writer)
    write_event(h5_file, 'right', max_len, left_img_ts[left_idx], writer)
    decoder = VideoDecoder(f"{h5_file}/img_left.mp4", device='cuda')

    if writer is not None:
        _publish_flir_frames(
            decoder=decoder,
            indices=left_idx,
            img_ts=left_img_ts,
            writer=writer,
            topic='/flir_cam_left',
            frame_id='recon_left',
            out_height=1456,
            out_width=1920,
            count=max_len,
        )
    decoder = VideoDecoder(f"{h5_file}/img_right.mp4", device='cuda')

    if writer is not None:
        _publish_flir_frames(
            decoder=decoder,
            indices=left_to_right_idx,
            img_ts=right_img_ts,
            writer=writer,
            topic='/flir_cam_right',
            frame_id='recon_right',
            out_height=1456,
            out_width=1920,
            count=max_len,
        )

    # Write IMU to rosbag if enabled
    if 'vectornav/accel' in f.keys() and 'vectornav/ang_vel' in f.keys() and 'vectornav/t' in f.keys() and writer is not None:
        lin_vel = f['vectornav/accel'][:]
        ang_vel = f['vectornav/ang_vel'][:]
        imu_time = f['vectornav/t'][:]
        N = min(len(imu_time), len(lin_vel), len(ang_vel))
        for i in tqdm(range(N)):
            stamp_ns = int(float(imu_time[i]) * 1_000_000_000)
            sec = stamp_ns // 1_000_000_000
            nsec = stamp_ns % 1_000_000_000
            msg = Imu()
            msg.header.stamp.sec = int(sec)
            msg.header.stamp.nanosec = int(nsec)
            msg.header.frame_id = 'imu_link'

            msg.angular_velocity.x = float(ang_vel[i,0])
            msg.angular_velocity.y = float(ang_vel[i,1])
            msg.angular_velocity.z = float(ang_vel[i,2])
            msg.linear_acceleration.x = float(lin_vel[i,0])
            msg.linear_acceleration.y = float(lin_vel[i,1])
            msg.linear_acceleration.z = float(lin_vel[i,2])
            writer.write('/imu/data', serialize_message(msg), stamp_ns)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", type=str, required=True)
    args = parser.parse_args()
    dump_image(args.dir)
    
    recon_bag_path = os.path.join(args.dir, 'cal2_bag')
    output_bag_path = os.path.join(args.dir, 'calibration.bag')
    convert_cmd = ['rosbags-convert', '--src', recon_bag_path, '--dst', output_bag_path]
    result = subprocess.run(convert_cmd, check=False)
    
    # Check if output file was created successfully, even if command returned non-zero
    if result.returncode != 0:
        if os.path.exists(output_bag_path):
            # Output file exists, conversion likely succeeded despite non-zero exit code
            print(f"Warning: rosbags-convert returned exit code {result.returncode}, but output file exists. Continuing...")
        else:
            # Output file doesn't exist, conversion actually failed
            raise subprocess.CalledProcessError(result.returncode, convert_cmd)
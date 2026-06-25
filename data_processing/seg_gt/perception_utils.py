"""
Utilities for the semantic-segmentation GT pipeline: calibration loading (left
undistort+rectify maps), LiDAR→left-image frame indexing, CLAHE, bag discovery,
the resume check, and H5 dataset creation.
"""
import os

import numpy as np
import cv2
import h5py
import hdf5plugin

# Cityscapes 19 classes
CS_CLASSES = [
    "road", "sidewalk", "building", "wall", "fence", "pole",
    "traffic light", "traffic sign", "vegetation", "terrain",
    "sky", "person", "rider", "car", "truck", "bus",
    "train", "motorcycle", "bicycle",
]


def nearest_idx(timestamps, target_t):
    """Nearest index in sorted ``timestamps`` to ``target_t`` (via searchsorted)."""
    idx = np.searchsorted(timestamps, target_t)
    if idx > 0 and idx < len(timestamps):
        if abs(timestamps[idx - 1] - target_t) < abs(timestamps[idx] - target_t):
            idx -= 1
    return int(np.clip(idx, 0, len(timestamps) - 1))


def apply_clahe(img_bgr, clip=2.0, grid=8):
    """CLAHE contrast normalization"""
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip, tileGridSize=(grid, grid))
    l = clahe.apply(l)
    return cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)


def load_calibration(h5_path):
    """Load calibration from the bag H5 and build the left undistort+rectify maps.

    Returns dict with:
        res: (W, H) image resolution
        t_lidar, t_img_l: LiDAR + left-image timestamps
        mapx_left, mapy_left: left undistort+rectify remap tables (for cv2.remap)
    """
    with h5py.File(h5_path, "r") as f:
        intr_img_left = f["img/left/intrinsics"][:]
        D_img_left = f["img/left/dist_coeffs"][:]
        intr_img_right = f["img/right/intrinsics"][:]
        D_img_right = f["img/right/dist_coeffs"][:]
        res = tuple(f["img/left/resolution"][:])           # (W, H)
        imgl_T_imgr = f["calib/imgl_T_imgr"][:]
        t_lidar = f["ouster/t"][:]
        t_img_l = f["img/left/t"][:]

    # Stereo rectification needs both cameras to fix the left rect rotation
    # (rectl_R_rawl) and projection (P_rect_left); only the left remap tables are kept.
    imgr_T_imgl = np.linalg.inv(imgl_T_imgr)
    rectify = cv2.stereoRectify(
        cameraMatrix1=intr_img_left,  cameraMatrix2=intr_img_right,
        distCoeffs1=D_img_left,       distCoeffs2=D_img_right,
        imageSize=res,
        R=imgr_T_imgl[:3, :3], T=imgr_T_imgl[:3, -1],
        flags=cv2.CALIB_ZERO_DISPARITY, alpha=0,
    )
    rectl_R_rawl, P_rect_left = rectify[0], rectify[2]   # left rect rotation + projection
    mapx_left, mapy_left = cv2.initUndistortRectifyMap(
        intr_img_left, D_img_left, rectl_R_rawl, P_rect_left, res, cv2.CV_32FC1)

    return {
        "res": res,
        "t_lidar": t_lidar, "t_img_l": t_img_l,
        "mapx_left": mapx_left, "mapy_left": mapy_left,
    }


def compute_frame_indices(calib):
    """Nearest left-image index for each LiDAR timestamp → int64 (N,)."""
    t_lidar = calib["t_lidar"]
    t_img_l = calib["t_img_l"]
    return np.array([nearest_idx(t_img_l, t) for t in t_lidar], dtype=np.int64)


def parse_bag_hour(bag_name):
    """Hour from a bag name like ``rosbag2_2026_01_04-13_51_24``, or None."""
    try:
        time_part = bag_name.replace("rosbag2_", "").split("-")[1]
        return int(time_part.split("_")[0])
    except (IndexError, ValueError):
        return None


def is_daytime(bag_name, day_start=7, day_end=17):
    """True if the bag was recorded during daytime hours (or the hour is unknown)."""
    hour = parse_bag_hour(bag_name)
    if hour is None:
        return True
    return day_start <= hour <= day_end


def get_daytime_bags(data_dir):
    """Discover daytime bags across sessions → sorted list of (bag_dir, bag_name)."""
    bags = []
    for sess in sorted(os.listdir(data_dir)):
        sess_dir = os.path.join(data_dir, sess)
        if not os.path.isdir(sess_dir) or not sess.startswith("sess"):
            continue
        for bag_name in sorted(os.listdir(sess_dir)):
            bag_dir = os.path.join(sess_dir, bag_name)
            if not os.path.isdir(bag_dir) or not is_daytime(bag_name):
                continue
            h5_path = os.path.join(bag_dir, f"{bag_name}.h5")
            left_mp4 = os.path.join(bag_dir, "img_left.mp4")
            if os.path.exists(h5_path) and os.path.exists(left_mp4):
                bags.append((bag_dir, bag_name))
    return bags


def is_bag_complete(bag_dir, expected_n):
    """True iff ``semantic.h5`` exists with the expected frame count."""
    h5_path = os.path.join(bag_dir, "semantic.h5")
    if not os.path.exists(h5_path):
        return False
    try:
        with h5py.File(h5_path, "r") as f:
            return "semantic" in f and f["semantic"].shape[0] == expected_n
    except Exception:
        return False

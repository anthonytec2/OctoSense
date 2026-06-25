from torchcodec.decoders import VideoDecoder
import cv2
import h5py
import hdf5plugin
from ouster.sdk import client
import numpy as np
import yaml
from tqdm import tqdm
import os
import sys
import shutil
import argparse

from viz_utils import (
    generate_quicklook_visualizations,
    save_lidar_detection_frame,
    create_video_from_frames,
)

sys.path.append(os.path.dirname(__file__))
from image_detection import compute_circle_center_for_frame_dict, load_kalibr_camera
from lidar_detection import detect_circle_center_3d
from robust_matching import solve_extrinsics_robust, solve_extrinsics_circle_opt

# -----------------------------------------------------------------------------
# Constants
# -----------------------------------------------------------------------------
REFLECTIVITY_THRESH = 120
# LiDAR-measured circle radius (m). ~2.5cm smaller than the physical 0.4572 m
TRUE_RADIUS = 0.432  # Measured mean from calibration_errors.yaml
FILTER_DIST = 0.010
# Circle-fit optimizer term weights (solve_extrinsics_circle_opt)
RADIUS_TERM_WEIGHT = 1.0
PLANE_TERM_WEIGHT = 1.0
CIRCLE_TERM_WEIGHT = 1.0

# Default Ouster metadata JSON (rev7_1 is used for all data), under
# <OCTO_RAW_ROOT>/calibrations. Override with --metadata-json.
DEFAULT_OUSTER_METADATA = os.path.join(
    os.environ.get("OCTO_RAW_ROOT", "/data/rosbags/raw").rstrip("/"),
    "calibrations", "rev7_1.json",
)

target_dict = {
    "grid": {
        "rows": 6,              # tagRows
        "cols": 6,              # tagCols
        "tag_size": 0.1121,     # meters (edge-to-edge)
        "tag_spacing": 0.25,    # ratio (space/tag_size)
        "marker_id_offset": 0,
    },
    "circle": {
        "radius": 0.432,        # meters 
        "x_offset": 0.4063,     # meters (circle center relative to grid origin)
        "y_offset": 0.4063,     # meters
        "z_offset": 0.0,        # circle is in-plane with grid
    },
}


def run_calibration(
    processed_dir: str,
    metadata_json_path: str,
    cam_name: str = "cam1",
    visualize: bool = False,
) -> None:
    """
    Run LiDAR-camera calibration for a given processed directory.

    processed_dir: directory containing <basename>.h5 and img_left.mp4
    """
    processed_dir = os.path.abspath(processed_dir)
    basename = os.path.basename(os.path.normpath(processed_dir))
    h5_path = os.path.join(processed_dir, f"{basename}.h5")
    video_path = os.path.join(processed_dir, "img_left.mp4")
    kalibr_file = os.path.join(processed_dir, "calibration-camchain.yaml")

    if not os.path.exists(h5_path):
        raise FileNotFoundError(f"H5 file not found at {h5_path}")
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video file not found at {video_path}")
    if not os.path.exists(metadata_json_path):
        raise FileNotFoundError(f"Metadata JSON not found at {metadata_json_path}")
    if not os.path.exists(kalibr_file):
        raise FileNotFoundError(f"Kalibr file not found at {kalibr_file}")

    # Optional visualization output directories (under processed_dir)
    frames_dir = None
    cam_frames_dir = None
    quicklook_frames_dir = None
    if visualize:
        frames_dir = os.path.join(processed_dir, "imgs")
        cam_frames_dir = os.path.join(processed_dir, "imgs_cam")
        quicklook_frames_dir = os.path.join(processed_dir, "imgs_quicklook")
        for d in (frames_dir, cam_frames_dir, quicklook_frames_dir):
            if os.path.isdir(d):
                try:
                    shutil.rmtree(d)
                except Exception:
                    pass
            os.makedirs(d, exist_ok=True)

    # -----------------------------------------------------------------------------
    # Data loading
    # -----------------------------------------------------------------------------
    f_h5 = h5py.File(h5_path)
    ouster_ts = f_h5["ouster/t"][:,1]
    reflective_img = f_h5["ouster/reflectivity"]
    range_img = f_h5["ouster/range"]
    with open(metadata_json_path, "r") as f:
        data = f.read()
    metadata = client.SensorInfo(data)

    # Find all the LiDAR centers and timestamps (no caching)
    lidar_centers = np.zeros((len(range_img), 3))
    lidar_radius = np.zeros((len(range_img)))
    u = np.zeros((len(range_img), 3))
    v = np.zeros((len(range_img), 3))
    oust_ts=np.zeros((len(range_img)))

    # Keep full-resolution inliers for each LiDAR frame for downstream metrics
    lidar_inliers_all = []
    detect_pts=0
    cnt = -1
    for i in tqdm(range(len(range_img))):
        try:
            center_3d, radius, plane, inliers, out_pt, uu, vv = detect_circle_center_3d(
                metadata,
                range_img[i],
                reflective_img[i],
                refl_thresh=REFLECTIVITY_THRESH,
            )
            lidar_centers[detect_pts] = center_3d
            lidar_radius[detect_pts] = radius
            u[detect_pts] = uu
            v[detect_pts] = vv
            oust_ts[detect_pts] = ouster_ts[i]
            lidar_inliers_all.append(out_pt)
            detect_pts += 1

            # Save per-frame LiDAR detection visualization
            if visualize and frames_dir is not None:
                try:
                    if np.abs(radius - TRUE_RADIUS) < FILTER_DIST:
                        cnt += 1
                        save_lidar_detection_frame(
                            center_3d=center_3d,
                            plane_model=plane,
                            outer_pts_3d=out_pt,
                            u=uu,
                            v=vv,
                            frame_idx=cnt,
                            out_dir=frames_dir,
                        )
                except Exception:
                    pass
        except Exception:
            pass
    
    lidar_centers=lidar_centers[:detect_pts]
    lidar_radius=lidar_radius[:detect_pts]
    u=u[:detect_pts]
    v=v[:detect_pts]
    oust_ts=oust_ts[:detect_pts]


    act_lidar_idx = np.abs(lidar_radius - TRUE_RADIUS) < FILTER_DIST
    lidar_centers_all = lidar_centers.copy()
    act_lidar_ts = oust_ts[act_lidar_idx]  # Using end of scan timestamp

    left_img = VideoDecoder(video_path)
    # New image timestamp dataset: /img/left/t
    left_img_ts = f_h5["img/left/t"]
    # Choose nearest camera frame (left or right neighbor) for each LiDAR timestamp
    idx = np.searchsorted(left_img_ts, act_lidar_ts, side="left")
    idx = np.clip(idx, 1, len(left_img_ts) - 1)
    left_neighbors = left_img_ts[idx - 1]
    right_neighbors = left_img_ts[idx]
    choose_left = np.abs(act_lidar_ts - left_neighbors) < np.abs(act_lidar_ts - right_neighbors)
    lidar_2_img = idx - choose_left

    # Camera detections (AprilTag + PnP) without caching
    center_img = np.zeros((len(lidar_2_img), 3))
    cam_R_board_img = np.zeros((len(lidar_2_img), 3, 3))
    cam_T_board_img = np.zeros((len(lidar_2_img), 3))
    cnt = 0
    for i in tqdm(range(len(lidar_2_img))):
        img = left_img[lidar_2_img[i]].cpu().numpy().transpose(1, 2, 0)
        center_cam, cam_R_board, cam_T_board, vis_img = compute_circle_center_for_frame_dict(
            kalibr_file,
            target_dict,
            img,
            cam_name=cam_name,
            visualize=visualize,
        )
        cam_R_board_img[i] = cam_R_board
        cam_T_board_img[i] = cam_T_board
        center_img[i] = center_cam
        # Save camera overlay frame if available
        if visualize and cam_frames_dir is not None and vis_img is not None:
            try:
                out_path = os.path.join(cam_frames_dir, f"{cnt:05d}.jpg")
                cv2.imwrite(out_path, vis_img)
                cnt += 1
            except Exception:
                pass

    act_cam_idx = ~np.isnan(center_img[:, 0])

    lidar_centers_use = lidar_centers[act_lidar_idx][act_cam_idx]  # N x 3
    cam_centers = center_img[act_cam_idx]  # N x 3 Optimize for (RP+T) for (R,T)

    cam_R_lidar, cam_T_lidar, inliers = solve_extrinsics_robust(
        lidar_centers_use,
        cam_centers,
        ransac_threshold=0.04,
    )

    # Refine extrinsics using circle center + radius + plane residuals across observations
    observations = []
    circle_cfg = target_dict["circle"]
    obj_center = np.array(
        [circle_cfg["x_offset"], circle_cfg["y_offset"], circle_cfg.get("z_offset", 0.0)],
        dtype=float,
    )
    cam_idx_use = lidar_2_img[act_cam_idx]
    lidar_idx_use = np.where(act_lidar_idx)[0][act_cam_idx]
    for j in range(len(cam_idx_use)):
        li = int(lidar_idx_use[j])
        rim_pts_L = lidar_inliers_all[li]
        if rim_pts_L is None or len(rim_pts_L) == 0:
            continue
        observations.append(
            {
                "rim_points_L": np.asarray(rim_pts_L, dtype=float),
                "circle_center_L": lidar_centers_all[li],
                "cam_R_board": cam_R_board_img[act_cam_idx][j],
                "cam_T_board": cam_T_board_img[act_cam_idx][j],
            }
        )

    print("cam_R_lidar (RANSAC init):")
    print(cam_R_lidar)
    print("cam_T_lidar (RANSAC init):")
    print(cam_T_lidar)
    print(len(observations))
    # Refine if we have rim observations; otherwise keep the RANSAC extrinsics.
    if len(observations) > 0:
        cam_R_lidar, cam_T_lidar, _ = solve_extrinsics_circle_opt(
            observations,
            cam_R_lidar,
            cam_T_lidar,
            circle_center_O=obj_center,
            circle_radius=TRUE_RADIUS,
            plane_normal_O=np.array([0.0, 0.0, 1.0]),
            center_weight=CIRCLE_TERM_WEIGHT,
            radius_weight=RADIUS_TERM_WEIGHT,
            plane_weight=PLANE_TERM_WEIGHT,
        )

    K, dist = load_kalibr_camera(kalibr_file, cam_name)

    quicklook_video_out = None
    if visualize and quicklook_frames_dir is not None:
        quick_cnt = 0
        # Select up to 10 random indices from the full set of aligned frames
        num_frames = len(cam_idx_use)
        if num_frames > 0:
            num_samples = min(100, num_frames)
            sample_indices = np.random.choice(num_frames, size=num_samples, replace=False)
        else:
            sample_indices = []

        for j in tqdm(sample_indices):
            try:
                # Save "all points colored by range" images directly to quicklook frames:
                quick_frame_path = os.path.join(quicklook_frames_dir, f"{quick_cnt:05d}.jpg")

                # With calibrated extrinsics
                generate_quicklook_visualizations(
                    j,
                    cam_R_lidar,
                    cam_T_lidar,
                    metadata,
                    range_img,
                    left_img,
                    cam_idx_use,
                    lidar_idx_use,
                    K,
                    dist,
                    quick_frame_path,
                )
                quick_cnt += 1
            except Exception:
                continue

        # Create quicklook video in processed directory
        quicklook_video_out = os.path.join(processed_dir, "quicklook.mp4")
        create_video_from_frames(quicklook_frames_dir, quicklook_video_out, fps=20)

    # Save calibration results for interactive viewer (lidar_ prefix, in processed_dir)
    calib_results = {
        "rotation_matrix": cam_R_lidar.tolist(),
        "translation_vector": cam_T_lidar.tolist(),
        "config": {
            "h5_path": h5_path,
            "video_path": video_path,
            "metadata_json_path": metadata_json_path,
            "kalibr_file": kalibr_file,
            "cam_name": cam_name,
            "true_radius": TRUE_RADIUS,
            "filter_dist": FILTER_DIST,
        },
    }
    calib_results_path = os.path.join(processed_dir, "lidar_calibration_results.yaml")
    with open(calib_results_path, "w") as f:
        yaml.safe_dump(calib_results, f, sort_keys=False)

    print("=" * 70)
    print("CALIBRATION COMPLETE")
    print("=" * 70)
    print("Rotation matrix:")
    print(cam_R_lidar)
    print("\nTranslation vector:")
    print(cam_T_lidar)
    print(f"\n✓ Results saved to: {calib_results_path}")
    if quicklook_video_out is not None:
        print(f"✓ Quicklook video saved to: {quicklook_video_out}")


def parse_args():
    parser = argparse.ArgumentParser(description="LiDAR-camera calibration from processed directory.")
    parser.add_argument(
        "--processed-dir",
        type=str,
        required=True,
        help="Processed directory containing <basename>.h5 and img_left.mp4 (basename = directory name).",
    )
    parser.add_argument(
        "--metadata-json",
        type=str,
        default=None,
        help=(
            "Path to Ouster metadata JSON file. Defaults to the bundled rev7_1.json "
            "under <OCTO_RAW_ROOT>/calibrations."
        ),
    )
    parser.add_argument(
        "--cam-name",
        type=str,
        default="cam1",
        help="Camera name in the Kalibr file (default: cam1) for Left Camera",
    )
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Enable visualization outputs (frames, quicklook video).",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    metadata_json_path = args.metadata_json or DEFAULT_OUSTER_METADATA
    run_calibration(
        processed_dir=args.processed_dir,
        metadata_json_path=metadata_json_path,
        cam_name=args.cam_name,
        visualize=args.visualize,
    )

 

import os
import matplotlib.pyplot as plt
import cv2
from ouster.sdk import client
import numpy as np
import subprocess


def undistort_image(img, K, dist, alpha=0.0):
    h, w = img.shape[:2]
    new_K, _ = cv2.getOptimalNewCameraMatrix(K, dist, (w, h), alpha=alpha)
    undistorted = cv2.undistort(img, K, dist, None, new_K)
    return undistorted.astype(np.uint8), new_K


def plot_all_points_projection(img, pts_2d, ranges, out_path):
    fig = plt.figure(figsize=(14, 10))
    plt.imshow(img)
    sc = plt.scatter(pts_2d[:, 0], pts_2d[:, 1],
                     c=ranges, s=3, cmap="jet", alpha=0.8, vmax=10)
    plt.colorbar(sc, label="LiDAR range (meters)")
    plt.title("LiDAR → Camera Projection (All Points, Colored by Range)")
    plt.axis("off")
    fig.savefig(out_path, bbox_inches="tight", dpi=200)
    plt.close(fig)
    return out_path

def _ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

def save_lidar_detection_frame(
    center_3d,
    plane_model,
    outer_pts_3d,
    u, v,
    frame_idx: int,
    out_dir: str = None,
):
    """
    Save a compact 2D visualization of the detected circle (LiDAR-only) by
    projecting the fitted circle inliers onto the estimated plane basis (u, v).
    """
    if out_dir is None:
        out_dir = os.path.join(os.path.dirname(__file__), "imgs")
    _ensure_dir(out_dir)

    a, b, c, d = plane_model
    normal = np.array([a, b, c], dtype=float)
    normal /= (np.linalg.norm(normal) + 1e-12)
    plane_origin = -d * normal

    center_local = (center_3d - plane_origin)
    cx = float(center_local @ u)
    cy = float(center_local @ v)

    fig = plt.figure(figsize=(6, 6))
    # Plot the fitted circle inliers in the plane basis
    pts_local = outer_pts_3d - plane_origin
    X = pts_local @ u
    Y = pts_local @ v
    plt.scatter(X, Y, s=10, c="white", edgecolors="none", label="Circle inliers")

    # Draw estimated center
    plt.scatter([cx], [cy], c="red", s=30)
    plt.gca().set_aspect("equal", adjustable="box")
    plt.title(f"LiDAR Circle Detection (frame {frame_idx})")
    plt.axis("off")

    outfile = os.path.join(out_dir, f"{frame_idx:05d}.jpg")
    fig.savefig(outfile, bbox_inches="tight", dpi=150, facecolor="black")
    plt.close(fig)
    return outfile

def create_video_from_frames(
    frames_dir: str,
    output_path: str,
    fps: int = 20
):
    """
    Use ffmpeg to stitch frames_dir/%05d.jpg into a movie at output_path.
    Requires ffmpeg to be installed.
    """
    _ensure_dir(os.path.dirname(output_path))
    pattern = os.path.join(frames_dir, "%05d.jpg")
    cmd = [
        "ffmpeg", "-y",
        "-framerate", str(fps),
        "-i", pattern,
        "-pix_fmt", "yuv420p",
        output_path
    ]
    subprocess.run(cmd, check=True)
    return output_path


def generate_quicklook_visualizations(
    idx,
    R, t,
    metadata,
    range_img,
    left_img,
    cam_idx_use,
    lidar_idx_use,
    K,
    dist,
    all_points_out_path,
):
    """
    Save a quicklook of all LiDAR points projected into the (undistorted) camera
    image, colored by range, to `all_points_out_path`.
    Intrinsics (K, dist) must be provided by caller to avoid cross-module imports.
    """
    raw_img = left_img[cam_idx_use[idx]].cpu().numpy().transpose(1, 2, 0)
    img_rect, K_rect = undistort_image(raw_img, K, dist, alpha=0.0)
    H, W = img_rect.shape[:2]

    rimg = range_img[lidar_idx_use[idx]]
    xyzlut = client.XYZLut(metadata)
    pts_all = xyzlut(rimg).reshape(-1, 3)
    cam_pts_tf = (R @ pts_all.T + t[:, None]).T.astype(np.float32)

    def _project(points_cam):
        pts_cv = points_cam.reshape(-1, 1, 3)
        proj, _ = cv2.projectPoints(
            objectPoints=pts_cv,
            rvec=np.zeros(3),
            tvec=np.zeros(3),
            cameraMatrix=K_rect,
            distCoeffs=None
        )
        return proj.reshape(-1, 2)

    # All points colored by range
    ranges = np.linalg.norm(pts_all, axis=1)
    valid_depth = cam_pts_tf[:, 2] > 0.1
    cam_pts_depth = cam_pts_tf[valid_depth]
    ranges = ranges[valid_depth]
    pts_2d = _project(cam_pts_depth)
    mask_img = (
        (pts_2d[:, 0] >= 0) & (pts_2d[:, 0] < W) &
        (pts_2d[:, 1] >= 0) & (pts_2d[:, 1] < H)
    )
    pts_2d = pts_2d[mask_img]
    ranges = ranges[mask_img]
    plot_all_points_projection(img_rect, pts_2d, ranges, all_points_out_path)
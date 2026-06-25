"""
Camera/LiDAR geometry for depth-GT generation: rectified intrinsics + the
rectification grid, RGB-frame snapping, scan loading, camera-viewport culling,
the map-consistency voxel filter, and projection/z-buffer into the rectified
left camera.
"""
import cv2
import h5py
import numpy as np
import torch


def nearest_rgb_indices(raw_img_t: np.ndarray, target_t: np.ndarray) -> np.ndarray:
    """For each ``target_t``, return the raw-RGB frame index nearest it."""
    pos = np.searchsorted(raw_img_t, target_t)
    pos = np.clip(pos, 1, len(raw_img_t) - 1)
    lo = pos - 1
    lo_diff = np.abs(raw_img_t[lo] - target_t)
    hi_diff = np.abs(raw_img_t[pos] - target_t)
    return np.where(lo_diff < hi_diff, lo, pos).astype(np.int64)


def build_rectified_intrinsics(raw_h5: h5py.File):
    """Return (K_rect, rectl_R_rawl, (W, H)). ``K_rect`` is the zero-distortion
    rectified left-camera intrinsics at full (1920, 1456). ``rectl_R_rawl`` is the
    OpenCV rectification rotation mapping the raw left camera into the rectified
    left frame (P_rect = rectl_R_rawl @ P_raw). 
    """
    intr_img_left = raw_h5["img/left/intrinsics"][()]
    D_img_left = raw_h5["img/left/dist_coeffs"][()]
    intr_img_right = raw_h5["img/right/intrinsics"][()]
    D_img_right = raw_h5["img/right/dist_coeffs"][()]
    res = raw_h5["img/left/resolution"][()]
    W, H = int(res[0]), int(res[1])

    imgr_T_imgl = np.linalg.inv(raw_h5["calib/imgl_T_imgr"][()])
    rectify = cv2.stereoRectify(
        cameraMatrix1=intr_img_left,  cameraMatrix2=intr_img_right,
        distCoeffs1=D_img_left,       distCoeffs2=D_img_right,
        imageSize=(W, H),
        R=imgr_T_imgl[:3, :3], T=imgr_T_imgl[:3, -1],
        flags=cv2.CALIB_ZERO_DISPARITY, alpha=0,
    )
    rectl_R_rawl, P_rect_left = rectify[0], rectify[2]   # left rect rotation + projection
    K_rect = P_rect_left[:3, :3].astype(np.float64).copy()
    return K_rect, rectl_R_rawl.astype(np.float64), (W, H)


def build_rectify_grid(raw_h5: h5py.File, rectl_R_rawl: np.ndarray, K_rect: np.ndarray,
                       hw, device: str) -> torch.Tensor:
    """Return a (1, H, W, 2) ``grid_sample`` grid that turns a raw (distorted)
    RGB tensor into the zero-distortion rectified image.
    """
    intr_img_left = raw_h5["img/left/intrinsics"][()]
    D_img_left = raw_h5["img/left/dist_coeffs"][()]
    H, W = hw
    P_rect_left = np.eye(3, 4, dtype=np.float64)
    P_rect_left[:3, :3] = K_rect
    mapx_left, mapy_left = cv2.initUndistortRectifyMap(
        intr_img_left.astype(np.float64), D_img_left.astype(np.float64),
        rectl_R_rawl.astype(np.float64), P_rect_left, (W, H), cv2.CV_32FC1,
    )
    gx = torch.from_numpy(mapx_left).float() / (W - 1) * 2.0 - 1.0
    gy = torch.from_numpy(mapy_left).float() / (H - 1) * 2.0 - 1.0
    return torch.stack([gx, gy], dim=-1).unsqueeze(0).to(device).contiguous()


def load_scan_xyz(f: h5py.File, idx: int):
    raw = f["ouster/range_pcl"][idx]
    mask = (raw != 0).any(axis=1)
    return raw[mask].astype(np.float32) * 1e-3, mask


def cull_to_cam_viewport(pts_l: np.ndarray, T_cam_lidar: np.ndarray,
                         R3: np.ndarray, K_rect: np.ndarray, hw,
                         near: float = 0.2, far: float = 200.0) -> np.ndarray:
    """Keep only LiDAR-body points (N, 3) that fall inside the (rectified) left
    camera's viewport at this scan's own pose: ``near < Z_cam < far`` AND the
    projected pixel ``(u, v)`` is in ``[0, W) x [0, H)``.

    """
    H, W = hw
    n = pts_l.shape[0]
    if n == 0:
        return pts_l
    h = np.concatenate([pts_l, np.ones((n, 1), np.float32)], axis=1).T   # (4, N)
    cam_raw = (T_cam_lidar @ h)[:3]                                      # (3, N) raw cam
    rect = R3 @ cam_raw                                                  # (3, N) rectified
    z = rect[2]
    z_safe = np.where(z > 1e-6, z, 1e-6)
    uv = (K_rect @ rect) / z_safe                                        # (3, N), uv[2] == 1
    in_front = z > near
    in_range = z < far
    in_image = (uv[0] >= 0) & (uv[0] < W) & (uv[1] >= 0) & (uv[1] < H)
    in_image &= np.isfinite(uv[0]) & np.isfinite(uv[1])
    return pts_l[in_front & in_range & in_image]


def map_consistency_filter_gpu(pts_map: torch.Tensor, scan_tag: torch.Tensor,
                               voxel: float, min_scans: int) -> torch.Tensor:
    """ Packs (voxel, scan) into one int64 key for a single ``torch.unique`` instead
    of ``np.unique(axis=0)``. Returns a bool mask over ``pts_map``."""
    N = pts_map.shape[0]
    if N == 0:
        return torch.zeros(0, dtype=torch.bool, device=pts_map.device)
    vox = torch.floor(pts_map / voxel).to(torch.int64)
    vox -= vox.min(dim=0, keepdim=True).values
    vmax = vox.max(dim=0).values + 1
    key = vox[:, 0] * (vmax[1] * vmax[2]) + vox[:, 1] * vmax[2] + vox[:, 2]

    # Pack: upper bits = voxel key, lower 16 bits = scan_tag (max ~2^16 scans).
    assert scan_tag.max().item() < (1 << 16), "scan_tag overflows 16 bits"
    packed = (key << 16) | scan_tag.to(torch.int64)
    uniq_packed = torch.unique(packed, sorted=True)       # dedup (voxel, scan) pairs
    # Project back to voxel key; it's now non-decreasing because of the packing.
    uniq_key_flat = uniq_packed >> 16
    uniq_vox, counts = torch.unique_consecutive(uniq_key_flat, return_counts=True)

    pos = torch.searchsorted(uniq_vox, key)
    pos_c = pos.clamp(max=uniq_vox.numel() - 1)
    match = uniq_vox[pos_c] == key
    counts_per_point = torch.where(match, counts[pos_c], torch.zeros_like(counts[pos_c]))
    return counts_per_point >= min_scans


def project_to_rect_gpu(pts_lidar_t: torch.Tensor, K_rect, R3, T_cam_lidar, hw):
    """Project LiDAR-frame points (N, 3) into the rectified left cam at ``hw``.
    Returns (uv int64 M×2, depth float32 M) for points in front of and inside
    the image."""
    if pts_lidar_t.shape[0] == 0:
        return (torch.zeros((0, 2), dtype=torch.int64, device=pts_lidar_t.device),
                torch.zeros((0,), dtype=torch.float32, device=pts_lidar_t.device))
    H, W = hw
    dev = pts_lidar_t.device
    T_cam_lidar_t = torch.from_numpy(T_cam_lidar.astype(np.float32)).to(dev)
    R3_t = torch.from_numpy(R3.astype(np.float32)).to(dev)
    K_rect_t = torch.from_numpy(K_rect.astype(np.float32)).to(dev)

    # Column-vector pipeline: each column of pts_h is one homogeneous point.
    ones = torch.ones((pts_lidar_t.shape[0], 1), dtype=torch.float32, device=dev)
    pts_h = torch.cat([pts_lidar_t, ones], dim=1).T                           # (4, N)
    pts_cam_raw = (T_cam_lidar_t @ pts_h)[:3]                                 # (3, N) raw cam
    pts_rect = R3_t @ pts_cam_raw                                             # (3, N) rectified

    # Keep only in-front points, then project with K.
    front = pts_rect[2] > 0.2
    if not front.any():
        return (torch.zeros((0, 2), dtype=torch.int64, device=dev),
                torch.zeros((0,), dtype=torch.float32, device=dev))
    pts_rect = pts_rect[:, front]                                             # (3, M)
    Z = pts_rect[2]
    uv_h = K_rect_t @ pts_rect                                                # (3, M)
    uv_h = uv_h / uv_h[2]                                                     # row 2 == 1
    u, v = uv_h[0], uv_h[1]
    inside = (u >= 0) & (u < W) & (v >= 0) & (v < H) & torch.isfinite(u) & torch.isfinite(v)
    return torch.stack([u[inside].long(), v[inside].long()], dim=1), Z[inside]


def zbuffer_gpu(uv: torch.Tensor, depth: torch.Tensor, hw):
    H, W = hw
    out = torch.full((H * W,), float("inf"), dtype=torch.float32, device=depth.device)
    if uv.shape[0]:
        flat = uv[:, 1] * W + uv[:, 0]
        out.scatter_reduce_(0, flat, depth, reduce="amin", include_self=True)
    out = torch.where(torch.isinf(out), torch.zeros_like(out), out)
    return out.reshape(H, W)

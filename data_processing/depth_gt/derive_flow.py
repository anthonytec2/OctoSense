"""
Derive sparse optical-flow GT from stored depth + camera motion, on demand.

The flow is purely camera-motion-induced (the reprojection of the depth point cloud
under the rigid camera motion), so it's an exact function of the depth + poses and is
recomputed here instead of stored:

    flow[r] = f( depth_cm[r], K_rect, rect_T_lidar, poses[r], poses[r+gap] )

This is written for the RECTIFIED LEFT camera (K_rect = its intrinsics,
rect_T_lidar = its pose w.r.t. the LiDAR). The same machinery applies to the EVENT or
INFRARED cameras unchanged — just pass that camera's intrinsics as ``K`` and its
``cam_T_lidar`` extrinsics
"""
import numpy as np
import torch


# Flow GT is int16 fixed-point: stored = round(flow_px * FLOW_SCALE). A pixel is valid
# iff both channels != FLOW_INVALID (INT16_MIN); clamped real flow never reaches it.
FLOW_SCALE = 8.0
FLOW_INVALID = -32768  # INT16_MIN


def compute_flow_gpu(depth_hr_t: torch.Tensor, K_rect_t: torch.Tensor,
                     T_delta_rect: torch.Tensor, hw):
    """Compute pixel-space optical flow from rect frame r → rect frame r+gap
    for every pixel that has a valid depth.

    For each valid pixel ``(u, v)`` with depth ``Z`` at scan r:
      1. Back-project to 3D in rect_r frame
      2. Transform to rect_{r+gap} using ``T_delta_rect``.
      3. Project back through ``K_rect`` → pixel ``(u', v')`` at scan r+gap.
      4. Flow is ``(u' - u, v' - v)``.

    This is the rigid / static-scene flow induced by camera motion. Depth has already
    been masked to 0 on moving objects, so those pixels pass through as invalid.

    Args:
        depth_hr_t: (H, W) float32 metres, 0 = invalid.
        K_rect_t:   (3, 3) float32 rectified intrinsics.
        T_delta_rect: (4, 4) float32, rect_{r+gap}_T_rect_r.
        hw: (H, W) tuple.

    Returns:
        flow:  (2, H, W) float32 — (du, dv) in pixels.
        valid: (H, W) bool     — True where flow is physically meaningful.
    """
    H, W = hw
    dev = depth_hr_t.device
    valid_depth = depth_hr_t > 0
    flow = torch.zeros((2, H, W), dtype=torch.float32, device=dev)
    valid_out = torch.zeros((H, W), dtype=torch.bool, device=dev)
    if not valid_depth.any():
        return flow, valid_out

    vs, us = torch.nonzero(valid_depth, as_tuple=True)                        # (M,), (M,)
    Z = depth_hr_t[vs, us]                                                    # (M,)
    u_f = us.float()
    v_f = vs.float()

    # Back-project: P_rect_r = Z * K⁻¹ @ [u, v, 1]
    K_inv = torch.linalg.inv(K_rect_t)
    pix_h = torch.stack([u_f, v_f, torch.ones_like(Z)], dim=0)                # (3, M)
    pts_xyz_r = (K_inv @ pix_h) * Z                                           # (3, M) in rect_r cam frame
    pts_r = torch.cat([pts_xyz_r, torch.ones_like(Z).unsqueeze(0)], dim=0)    # (4, M) homogeneous

    # Transform to rect_{r+gap}
    pts_r2 = T_delta_rect @ pts_r                                             # (4, M)

    # Project with K_rect: uv_h = K @ P, then normalize by Z' (row 2).
    uv_h = K_rect_t @ pts_r2[:3]                                              # (3, M)
    uv_h = uv_h / uv_h[2]                                                     # row 2 == 1
    u2, v2 = uv_h[0], uv_h[1]

    # Flow is valid as long as the world point is still in front of the camera at
    # r+gap. We do NOT require the re-projected pixel to stay inside the image
    Z2 = pts_r2[2]
    ok = (Z2 > 0.2) & torch.isfinite(u2) & torch.isfinite(v2)
    if not ok.any():
        return flow, valid_out

    vs_ok = vs[ok]; us_ok = us[ok]
    flow[0, vs_ok, us_ok] = u2[ok] - u_f[ok]
    flow[1, vs_ok, us_ok] = v2[ok] - v_f[ok]
    valid_out[vs_ok, us_ok] = True
    return flow, valid_out


def encode_flow_i16(flow_t: torch.Tensor, valid_t: torch.Tensor,
                    scale: float = FLOW_SCALE) -> np.ndarray:
    """Quantise signed pixel flow → ``(2, H, W) int16``.

    stored = round(flow_px * scale) ∈ [-32767, +32767]. Invalid pixels (and any flow
    with |u| or |v| > max representable) are written as INT16_MIN (-32768); a pixel is
    valid iff BOTH channels are != INT16_MIN.
    """
    max_abs = 32767.0
    max_px = max_abs / scale
    in_range = (flow_t[0].abs() < max_px) & (flow_t[1].abs() < max_px)
    ok = valid_t & in_range
    u_q = (flow_t[0] * scale).round().clamp(-max_abs, max_abs).to(torch.int32)
    v_q = (flow_t[1] * scale).round().clamp(-max_abs, max_abs).to(torch.int32)
    inv = torch.full_like(u_q, FLOW_INVALID)
    u_q = torch.where(ok, u_q, inv)
    v_q = torch.where(ok, v_q, inv)
    return torch.stack([u_q, v_q], dim=0).cpu().numpy().astype(np.int16)


def build_rect_T_lidar(R3: np.ndarray, T_cam_lidar: np.ndarray) -> np.ndarray:
    """
    rect_T_lidar: maps LiDAR(ouster)-frame points -> rectified left-camera frame.
    Static (calibration only) = R3 (raw-left -> rectified-left, padded to 4x4)
    composed with imgl_T_ouster (LiDAR -> raw-left).
    """
    R3_pad = np.eye(4, dtype=np.float64)
    R3_pad[:3, :3] = R3
    return (R3_pad @ T_cam_lidar).astype(np.float64)


def derive_flow_i16(depth_cm: np.ndarray, r: int, K_rect: np.ndarray,
                    rect_T_lidar: np.ndarray, poses: np.ndarray, gap: int,
                    scale: float = FLOW_SCALE, device: str = "cpu") -> np.ndarray:
    """Derive the flow_uv (2,H,W int16) for scan ``r`` from depth + camera motion.

    ``poses[i]`` is map_T_lidar at scan i. Last ``gap`` frames have no future pose
    → all-invalid.
    """
    H, W = depth_cm.shape
    N = poses.shape[0]
    if r + gap >= N:
        return np.full((2, H, W), FLOW_INVALID, np.int16)
    dev = torch.device(device)
    depth_hr = torch.from_numpy((depth_cm.astype(np.float32)) / 100.0).to(dev)   # metres, 0=invalid
    K_rect_t = torch.from_numpy(K_rect.astype(np.float32)).to(dev)
    rect_T_lidar_t = torch.from_numpy(rect_T_lidar.astype(np.float32)).to(dev)
    lidar_T_rect_t = torch.linalg.inv(rect_T_lidar_t)
    map_T_lidar_r = torch.from_numpy(poses[r].astype(np.float32)).to(dev)
    map_T_lidar_rgap = torch.from_numpy(poses[r + gap].astype(np.float32)).to(dev)
    # Camera motion r -> r+gap in the rectified-camera frame. Conjugate the body-frame
    # relative motion (lidar_rgap <- map <- lidar_r) into the rect-cam frame:
    #   rectgap_T_rect = rect_T_lidar @ inv(map_T_lidar_rgap) @ map_T_lidar_r @ lidar_T_rect
    rectgap_T_rect = (rect_T_lidar_t @ torch.linalg.inv(map_T_lidar_rgap)
                      @ map_T_lidar_r @ lidar_T_rect_t)
    flow, valid = compute_flow_gpu(depth_hr, K_rect_t, rectgap_T_rect, (H, W))
    return encode_flow_i16(flow, valid, scale)


def derive_flow_from_h5(h5path: str, r: int, poses: np.ndarray = None,
                        device: str = "cpu") -> np.ndarray:
    """Convenience: pull metadata from a depth GT h5 and derive flow_uv[r]. ``poses``
    may be passed in (e.g. from the source bag h5 ``ouster/odom/map_T_lidart``) for
    files that don't store them."""
    import h5py, hdf5plugin  # noqa: F401
    with h5py.File(h5path, "r") as f:
        depth_cm = f["depth_cm"][r]
        K_rect = f["K_rect"][()]
        R3 = f["R3"][()]
        T_cam_lidar = f["imgl_T_ouster"][()]
        gap = int(f.attrs.get("flow_gap", 2))
        scale = float(f.attrs.get("flow_scale", FLOW_SCALE))
        if poses is None:
            poses = f["poses"][()]
    rect_T_lidar = build_rect_T_lidar(R3, T_cam_lidar)
    return derive_flow_i16(depth_cm, r, K_rect, rect_T_lidar, poses, gap, scale, device)


def derive_flow_to_h5(depth_h5_path: str, out_path: str = None,
                      poses: np.ndarray = None, device: str = "cpu",
                      clevel: int = 5) -> str:
    """Derive flow for every scan in a depth GT h5 and write it to a flow.h5.

    flow_uv is (N, 2, H, W) int16, index-aligned with depth_cm: scans [0, N-gap)
    carry real flow; the last ``gap`` (no future pose) read as FLOW_INVALID. ``poses``
    may be passed in for files that don't store them. Returns the output path
    (defaults to ``flow.h5`` next to the depth h5).
    """
    import os
    import h5py
    import hdf5plugin
    from tqdm import tqdm

    with h5py.File(depth_h5_path, "r") as f:
        depth = f["depth_cm"]
        N, H, W = depth.shape
        K_rect = f["K_rect"][()]
        rect_T_lidar = build_rect_T_lidar(f["R3"][()], f["imgl_T_ouster"][()])
        gap = int(f.attrs.get("flow_gap", 2))
        scale = float(f.attrs.get("flow_scale", FLOW_SCALE))
        if poses is None:
            poses = f["poses"][()]

        if out_path is None:
            out_path = os.path.join(os.path.dirname(os.path.abspath(depth_h5_path)), "flow.h5")
        blosc = hdf5plugin.Blosc2(cname="zstd", clevel=clevel,
                                  filters=hdf5plugin.Blosc2.BITSHUFFLE)
        with h5py.File(out_path, "w") as g:
            flow_uv = g.create_dataset(
                "flow_uv", shape=(N, 2, H, W), dtype=np.int16,
                chunks=(1, 2, H, W), fillvalue=FLOW_INVALID, **blosc,
            )
            for r in tqdm(range(N - gap), desc="derive flow"):
                flow_uv[r] = derive_flow_i16(depth[r], r, K_rect, rect_T_lidar,
                                             poses, gap, scale, device)
            g.attrs["flow_gap"] = gap
            g.attrs["flow_scale"] = scale
            g.attrs["flow_invalid"] = FLOW_INVALID
            g.attrs["n_scans"] = N
    return out_path


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(description="Derive flow.h5 from a depth GT h5.")
    p.add_argument("--depth_h5", required=True, help="Path to the depth GT h5 (depth_cm + calib + poses).")
    p.add_argument("--out", default=None, help="Output flow.h5 (default: flow.h5 next to --depth_h5).")
    p.add_argument("--device", default="cpu", help="'cpu' (default) or 'cuda'.")
    args = p.parse_args()
    out = derive_flow_to_h5(args.depth_h5, out_path=args.out, device=args.device)
    print(f"Wrote {out}")

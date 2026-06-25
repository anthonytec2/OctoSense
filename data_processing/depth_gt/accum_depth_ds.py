"""
Dense depth labels in the rectified/undistorted left-camera frame.

For each requested LiDAR scan ``i``:
  * Use ``map_T_lidart[i]`` directly as the body-frame pose.
  * Accumulate ±W Ouster scans in map frame with scan-idx tags.
  * Map-consistency filter: keep only world-frame voxels observed by ≥ K
    distinct scans. Static surfaces pile up in the same voxel from every scan;
    moving (or co-moving) vehicles leave single-scan trails and are rejected.
  * Transform remaining points into the rectified left-camera frame
  * z-buffer keeps the nearest depth per pixel.
  * Save uint16 cm depth at the full rectified resolution (1920×1456).
"""
import argparse
import os
import pathlib
import time

import h5py
import hdf5plugin
import numpy as np
import torch

from geometry import (
    nearest_rgb_indices, build_rectified_intrinsics, build_rectify_grid,
    load_scan_xyz, cull_to_cam_viewport, map_consistency_filter_gpu,
    project_to_rect_gpu, zbuffer_gpu,
)
from masking import DynamicMasker


class ScanCache:
    """Rolling cache of map-frame scan points on the GPU, keyed by scan index.

    Each cached entry is a ``(N_j, 3)`` float32 tensor in the world ``map``
    frame. As the window slides scan-by-scan, only one new HDF5 read and one
    GPU upload happens per frame
    When a ``DynamicMasker`` is supplied, each scan's points are first stripped
    of dynamic-object returns (cars / peds / etc.) before being cached.
    """

    def __init__(self, device: str = "cuda", masker=None, cam_viewport_cfg=None):
        """``cam_viewport_cfg`` (if given) carries the static geometry needed
        to apply per-scan camera-viewport culling: a dict with keys
        ``T_cam_lidar`` (4x4), ``R3`` (3x3), ``K_rect`` (3x3), ``hw`` ((H,W)),
        ``near`` (m, default 0.2), ``far`` (m, default 200). 
        """
        self.device = device
        self.masker = masker
        self.cam_viewport_cfg = cam_viewport_cfg
        self.store: dict[int, torch.Tensor] = {}

    def evict_outside(self, lo: int, hi: int) -> None:
        drop = [k for k in self.store if k < lo or k >= hi]
        for k in drop:
            del self.store[k]
            if self.masker is not None:
                self.masker.evict(k)

    def ensure(self, f: h5py.File, scan_idx: int, poses: np.ndarray,
               max_range_m: float, cull_viewport: bool = False) -> None:
        if scan_idx in self.store:
            return
        pts_j, _ = load_scan_xyz(f, scan_idx)
        if max_range_m is not None:
            pts_j = pts_j[np.linalg.norm(pts_j, axis=1) < max_range_m]
        if cull_viewport and self.cam_viewport_cfg is not None and len(pts_j) > 0:
            c = self.cam_viewport_cfg
            pts_j = cull_to_cam_viewport(
                pts_j, c["T_cam_lidar"], c["R3"], c["K_rect"], c["hw"],
                near=c.get("near", 0.2), far=c.get("far", 200.0),
            )
        if self.masker is not None and len(pts_j) > 0:
            pts_j = self.masker.filter(scan_idx, pts_j)
        if len(pts_j) == 0:
            self.store[scan_idx] = torch.zeros((0, 3), dtype=torch.float32, device=self.device)
            return
        map_T_j = poses[scan_idx]
        h = np.concatenate([pts_j, np.ones((len(pts_j), 1), np.float32)], axis=1)
        pts_map = (map_T_j @ h.T).T[:, :3].astype(np.float32)
        self.store[scan_idx] = torch.from_numpy(pts_map).to(self.device, non_blocking=True)

    def bulk_gather(self, lo: int, hi: int):
        """Return (pts_map_cat (N, 3) float32 cuda, scan_tag_cat (N,) int32 cuda)."""
        tensors, tags = [], []
        for j in range(lo, hi):
            pts = self.store.get(j)
            if pts is None or pts.shape[0] == 0:
                continue
            tensors.append(pts)
            tags.append(torch.full((pts.shape[0],), j, dtype=torch.int32, device=self.device))
        if not tensors:
            return (torch.zeros((0, 3), dtype=torch.float32, device=self.device),
                    torch.zeros((0,), dtype=torch.int32, device=self.device))
        return torch.cat(tensors, dim=0), torch.cat(tags, dim=0)


def accumulate_cached(
    f_raw, ref_scan: int, poses: np.ndarray,
    window: int, max_range_m: float,
    map_filter: bool, map_voxel: float, map_min_scans: int,
    include_ref_unfiltered: bool,
    cache: ScanCache,
    cam_viewport_neighbors: bool = False,
):
    """Accumulate +/- ``window`` LiDAR scans around ``ref_scan`` in the world map
    frame (rolling GPU cache), apply the map-consistency filter, then transform
    survivors into the LiDAR body frame at ``ref_scan``.

    When ``cam_viewport_neighbors`` is set, every scan contributes only points
    inside its OWN (rectified) left-camera viewport (``near < Z_cam < far`` AND
    ``uv in [0,W) x [0,H)``).
    """
    n_scans = poses.shape[0]
    lo = max(0, ref_scan - window)
    hi = min(n_scans, ref_scan + window + 1)

    cache.evict_outside(lo, hi)
    for j in range(lo, hi):
        cache.ensure(f_raw, j, poses, max_range_m, cull_viewport=cam_viewport_neighbors)

    pts_map, scan_tag = cache.bulk_gather(lo, hi)
    total_in = int(pts_map.shape[0])

    if map_filter and total_in > 0:
        ok = map_consistency_filter_gpu(pts_map, scan_tag, map_voxel, map_min_scans)
        pts_map = pts_map[ok]

    if include_ref_unfiltered and ref_scan in cache.store:
        ref_pts = cache.store[ref_scan]
        if ref_pts.shape[0] > 0:
            pts_map = torch.cat([pts_map, ref_pts], dim=0)

    kept = int(pts_map.shape[0])
    if kept == 0:
        return (torch.zeros((0, 3), dtype=torch.float32, device=cache.device),
                total_in, kept, (lo, hi))

    R_T_map = np.linalg.inv(poses[ref_scan]).astype(np.float32)
    R_T_map_t = torch.from_numpy(R_T_map).to(cache.device)
    ones = torch.ones((kept, 1), dtype=torch.float32, device=cache.device)
    pts_h = torch.cat([pts_map, ones], dim=1).T
    pts_R = (R_T_map_t @ pts_h)[:3].T.contiguous()
    return pts_R, total_in, kept, (lo, hi)






# ---------------------------------- Main ------------------------------------

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--raw_h5", required=True,
                   help="Pre-conversion h5 (range_pcl, poses, calib, raw img_left.mp4 next to it)")
    p.add_argument("--frames", default="all",
                   help="Indices into LiDAR scans to process (comma-separated). "
                        "Default 'all' iterates every scan.")
    p.add_argument("--window", type=int, default=30)
    p.add_argument("--max_range", type=float, default=200.0)
    p.add_argument("--no-map_filter", dest="map_filter", action="store_false", default=True,
                   help="Disable the map-consistency voxel filter (on by default).")
    p.add_argument("--map_voxel", type=float, default=0.3)
    p.add_argument("--map_min_scans", type=int, default=4,
                   help="Voxel-consistency filter threshold: keep only world voxels "
                        "seen by >= this many distinct scans. Higher suppresses ")
    p.add_argument("--include_ref_unfiltered", action="store_true", default=False,
                   help="Re-add the ref scan's own points after filtering. ")
    p.add_argument("--no-cam_viewport_neighbors", dest="cam_viewport_neighbors",
                   action="store_false", default=True,
                   help="Disable per-neighbor camera-viewport culling (on by default). "
                        "When on, each neighbor scan j != ref drops LiDAR points outside "
                        "scan-j's own camera viewport (near<Z<far AND uv in image)")
    p.add_argument("--no-yolo_mask", dest="yolo_mask", action="store_false", default=True,
                   help="Disable YOLO dynamic-object masking (on by default). When on, YOLO "
                        "seg on each scan's RGB drops LiDAR points on moving objects "
                        "(person/bike/car/moto/bus/truck) at source.")
    p.add_argument("--yolo_weights", default="yolo26m-seg.pt")
    p.add_argument("--yolo_conf", type=float, default=0.25)
    p.add_argument("--yolo_imgsz", type=int, default=1280,
                   help="YOLO inference image size (letterboxed)")
    p.add_argument("--yolo_dilate_px", type=int, default=16,
                   help="Pad each YOLO mask outward by this many pixels (binary "
                        "dilation via GPU max-pool). Catches the vehicle rim "
                        "(mirrors, antennas, bumpers, side-view edge pixels) that "
                        "0 = disable.")
    p.add_argument("--out", default="/tmp/accum_depth_ds")
    p.add_argument("--out_h5", default=None,
                   help="Filename of the packed per-bag h5 (depth_cm + metadata) "
                        "inside --out. Defaults to '<bag_stem>_gt.h5'.")
    p.add_argument("--device", default=None,
                   help="Force 'cuda' or 'cpu'. Default = auto.")
    args = p.parse_args()

    out_dir = pathlib.Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    with h5py.File(args.raw_h5, "r") as f_raw:
        K_rect, R3, (W, H) = build_rectified_intrinsics(f_raw)
        T_cam_lidar = f_raw["ouster/imgl_T_ouster"][()]
        ouster_t = f_raw["ouster/t"][()].astype(np.float64).ravel()
        poses_raw = f_raw["ouster/odom/map_T_lidart"][()]
        img_t_raw = f_raw["img/left/t"][()].astype(np.float64)

        pose_t = f_raw["ouster/odom/t"][()].astype(np.float64).ravel()
        idx = np.searchsorted(pose_t, ouster_t)
        idx = np.clip(idx, 1, len(pose_t) - 1)
        pick_prev = (ouster_t - pose_t[idx - 1]) < (pose_t[idx] - ouster_t)
        pose_idx = np.where(pick_prev, idx - 1, idx).astype(np.int64)
        poses = poses_raw[pose_idx]
        max_dt_us = float(np.max(np.abs(ouster_t - pose_t[pose_idx])) * 1e6)
        print(f"  [info] aligned {len(ouster_t)} poses to scans by odom/t "
              f"(max |Δ| = {max_dt_us:.3f} us)")

        # range_pcl must cover every scan we'll iterate. If it's shorter,
        # abort
        range_pcl_len = f_raw["ouster/range_pcl"].shape[0]
        if range_pcl_len < len(ouster_t):
            raise RuntimeError(
                f"range_pcl has {range_pcl_len} scans but ouster/t has "
                f"{len(ouster_t)} — bag is corrupt, re-run conversion.")
        assert len(poses) == len(ouster_t), \
            f"internal: pose alignment failed ({len(poses)} vs {len(ouster_t)})"
        N_scans = len(ouster_t)

        print(f"LiDAR scans: {N_scans}  rectified cam: {W}×{H}")
        print(f"K_rect:\n{K_rect}")

        if args.frames.strip().lower() == "all":
            frame_ids = list(range(N_scans))
        else:
            frame_ids = [int(x) for x in args.frames.split(",") if x.strip()]

        # Device resolution. --device forces cuda or cpu; else auto-detect.
        if args.device:
            device = args.device
        else:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        # For each target scan, which raw-camera frame is the nearest exposure?
        nearest_rgb_by_scan = nearest_rgb_indices(img_t_raw, ouster_t)

   
        decoder = grid = None
        masker = None
        if args.yolo_mask:
            from torchcodec.decoders import VideoDecoder
            raw_mp4 = pathlib.Path(args.raw_h5).parent / "img_left.mp4"
            if not raw_mp4.exists():
                raise FileNotFoundError(raw_mp4)
            decoder = VideoDecoder(str(raw_mp4), device=device)
            grid = build_rectify_grid(f_raw, R3, K_rect, (H, W), device)
            masker = DynamicMasker(
                yolo_path=args.yolo_weights,
                decoder=decoder, rgb_indices=nearest_rgb_by_scan,
                grid=grid, T_cam_lidar=T_cam_lidar, R3=R3, K_rect=K_rect,
                hw=(H, W), device=device,
                conf=args.yolo_conf, imgsz=args.yolo_imgsz,
                dilate_px=args.yolo_dilate_px,
            )
            print(f"YOLO mask: {args.yolo_weights}  classes={masker.classes}  "
                  f"imgsz={masker.imgsz}  dilate_px={masker.dilate_px}")

        cam_viewport_cfg = {
            "T_cam_lidar": T_cam_lidar,
            "R3": R3,
            "K_rect": K_rect,
            "hw": (H, W),
            "near": 0.2,
            "far": float(args.max_range),
        }
        cache = ScanCache(device=device, masker=masker, cam_viewport_cfg=cam_viewport_cfg)
        if args.cam_viewport_neighbors:
            print(f"  [cam_viewport_neighbors] non-ref scans culled to cam viewport "
                  f"({W}x{H}, near={cam_viewport_cfg['near']}m, far={cam_viewport_cfg['far']}m)")

        # Open the bag-level h5 output for depth + calib metadata.
        import hdf5plugin as _h5p
        bag_stem = pathlib.Path(args.raw_h5).stem
        out_h5_name = args.out_h5 or f"{bag_stem}_gt.h5"
        out_h5 = out_dir / out_h5_name
        fw_h5 = h5py.File(str(out_h5), "w")
        _clevel = int(os.environ.get("DEPTH_CLEVEL", "5"))
        blosc = _h5p.Blosc2(cname="zstd", clevel=_clevel, filters=_h5p.Blosc2.BITSHUFFLE)
        depth_dset = fw_h5.create_dataset(
            "depth_cm", shape=(N_scans, H, W), dtype=np.uint16,
            chunks=(1, H, W), **blosc,
        )
        # Calibration + metadata embedded in the same h5 for self-containment.
        fw_h5.create_dataset("K_rect", data=K_rect.astype(np.float64))
        fw_h5.create_dataset("R3", data=R3.astype(np.float64))
        fw_h5.create_dataset("imgl_T_ouster", data=T_cam_lidar.astype(np.float64))
        fw_h5.create_dataset("raw_res", data=np.asarray([W, H], np.int32))
        fw_h5.create_dataset("poses", data=poses.astype(np.float32),
                             chunks=True, **blosc)
        fw_h5.create_dataset("timestamps", data=ouster_t.astype(np.float64))      # master-clock s
        fw_h5.create_dataset("lidar_indices", data=np.arange(N_scans, dtype=np.int64))
        _pos = np.clip(np.searchsorted(img_t_raw, ouster_t), 1, len(img_t_raw) - 1)
        _pick_prev = (ouster_t - img_t_raw[_pos - 1]) <= (img_t_raw[_pos] - ouster_t)
        fw_h5.create_dataset("left_img_indices",
                             data=np.where(_pick_prev, _pos - 1, _pos).astype(np.int64))
        fw_h5.attrs["depth_scale_cm"] = 100.0
        fw_h5.attrs["n_scans"] = int(N_scans)
        print(f"Writing GT → {out_h5}  depth_cm{(N_scans, H, W)}")

        t0 = time.time()

        # Prefill YOLO masks ahead of the per-frame loop -- batched decode +
        # rectify + YOLO inference over the entire scan list. Each scan in
        # frame_ids will be referenced O(window) times in the rolling cache,
        # and the masker also runs at projection-time on the ref scan
        if masker is not None and len(frame_ids) > 0:
            lo = max(0, min(frame_ids) - args.window)
            hi = min(N_scans, max(frame_ids) + args.window + 1)
            scan_list = list(range(lo, hi))
            print(f"Prefilling YOLO masks for {len(scan_list)} scans "
                  f"[{lo}..{hi-1}]  batch_size=32 ...", flush=True)
            masker.prefill_masks(scan_list, batch_size=32)

        for fi_idx, fi in enumerate(frame_ids):
            if fi < 0 or fi >= N_scans:
                print(f"skip {fi}: out of range")
                continue

            pts_R_t, total_in, kept_final, (lo, hi) = accumulate_cached(
                f_raw, fi, poses,
                args.window, args.max_range,
                args.map_filter, args.map_voxel, args.map_min_scans,
                args.include_ref_unfiltered,
                cache,
                cam_viewport_neighbors=args.cam_viewport_neighbors,
            )
            uv_t, depth_t = project_to_rect_gpu(pts_R_t, K_rect, R3, T_cam_lidar, (H, W))
            depth_hr_t = zbuffer_gpu(uv_t, depth_t, (H, W))

            # Projection-time dynamic mask:
            if masker is not None:
                dyn_mask = masker.mask_for_scan(fi)
                depth_hr_t = torch.where(dyn_mask, torch.zeros_like(depth_hr_t), depth_hr_t)

            depth_hr = depth_hr_t.detach().cpu().numpy()
            valid = depth_hr > 0
            cov = valid.mean() * 100.0 if valid.any() else 0.0

            # Pack depth into the bag h5.
            depth_cm = np.clip(depth_hr * 100.0, 0, 65535).astype(np.uint16)
            depth_dset[fi] = depth_cm

            if fi_idx % 50 == 0 or fi_idx == len(frame_ids) - 1:
                elapsed = time.time() - t0
                rate = (fi_idx + 1) / max(elapsed, 1e-6)
                eta = (len(frame_ids) - fi_idx - 1) / max(rate, 1e-6)
                print(f"[{fi_idx+1:5d}/{len(frame_ids)}] scan {fi}  "
                      f"depth_cov={cov:4.1f}%  {rate:5.2f} fps  ETA {eta/60:5.1f} min")

        fw_h5.close()
        print(f"\nGT written → {out_h5}")

    print(f"\nOutputs → {out_dir}")


if __name__ == "__main__":
    main()

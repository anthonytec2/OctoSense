"""
YOLO-seg dynamic-object masking. Per-scan instance segmentation of moving
objects (person/bike/car/motorcycle/bus/truck) in the rectified left camera,
used to drop LiDAR points that land on dynamic objects before accumulation.
"""
import cv2
import numpy as np
import torch
import torch.nn.functional as Fn


class DynamicMasker:
    """Per-scan YOLO-seg dynamic-object mask, applied to LiDAR points before
    accumulation.

    For each scan ``j`` we decode the corresponding raw RGB frame, rectify it
    on the GPU, run YOLO instance segmentation for a fixed set of dynamic
    classes (person/bicycle/car/motorcycle/bus/truck by default), up-sample the
    segmentation masks to the rectified-frame resolution, and project the
    scan's LiDAR points into the same rectified camera. Points whose pixel
    lands inside any dynamic mask are dropped at source.
    """

    DEFAULT_CLASSES = (0, 1, 2, 3, 5, 7)

    def __init__(self, yolo_path, decoder, rgb_indices: np.ndarray,
                 grid: torch.Tensor, T_cam_lidar: np.ndarray, R3: np.ndarray,
                 K_rect: np.ndarray, hw, device: str,
                 conf: float = 0.25, imgsz: int = 640,
                 classes=DEFAULT_CLASSES, dilate_px: int = 0):
        from ultralytics import YOLO
        self.yolo = YOLO(yolo_path)
        self.decoder = decoder
        self.rgb_indices = rgb_indices
        self.grid = grid
        self.T_cam_lidar = torch.from_numpy(T_cam_lidar.astype(np.float32)).to(device)
        self.R3 = torch.from_numpy(R3.astype(np.float32)).to(device)
        self.K_rect = torch.from_numpy(K_rect.astype(np.float32)).to(device)
        self.H, self.W = hw
        self.device = device
        self.conf = conf
        self.imgsz = imgsz
        self.classes = list(classes)
        # Dilation pad on each YOLO mask before applying. Catches vehicle rim
        # that YOLO shaves off (mirrors / antennas / bumpers off the seg
        # boundary) which would otherwise leak rim returns through the voxel
        # filter as moving-object trails.
        self.dilate_px = int(dilate_px)
        self._mask_cache: dict[int, torch.Tensor] = {}

    def evict(self, scan_idx: int) -> None:
        self._mask_cache.pop(scan_idx, None)

    def prefill_masks(self, scan_indices, batch_size: int = 32) -> None:
        """Compute + cache masks for the whole scan list ahead of the accumulation
        loop, in batches (amortizes the ultralytics dispatch over the bag)."""
        import time
        t0 = time.time()
        n = len(scan_indices)
        for i_b in range(0, n, batch_size):
            self._compute_masks(scan_indices[i_b:i_b + batch_size])
            done = min(i_b + batch_size, n)
            if (done % 500 == 0) or done == n:
                elapsed = time.time() - t0
                print(f"  [yolo prefill] {done}/{n}  {done / max(elapsed, 1e-6):.1f} fps  "
                      f"elapsed={elapsed:.1f}s", flush=True)

    def mask_for_scan(self, scan_idx: int) -> torch.Tensor:
        """(H, W) bool mask on GPU: True where YOLO sees a dynamic object in
        scan_idx's rectified RGB frame. Cached; computed on demand if not
        already prefilled."""
        if scan_idx not in self._mask_cache:
            self._compute_masks([scan_idx])
        return self._mask_cache[scan_idx]

    @torch.inference_mode()
    def _compute_masks(self, scan_indices) -> None:
        """Decode -> rectify -> batched YOLO-seg -> dilate, caching the (H, W) bool
        dynamic mask for every scan in ``scan_indices``. 
        """
        rgb_idx = [int(self.rgb_indices[s]) for s in scan_indices]
        frames = self.decoder.get_frames_at(indices=rgb_idx)
        imgs = frames.data.float() / 255.0                # (B, 3, H_raw, W_raw)
        grid = self.grid.expand(imgs.shape[0], -1, -1, -1)
        rect = Fn.grid_sample(imgs, grid, mode="bilinear",
                              padding_mode="zeros", align_corners=False)
        rect_u8 = (rect.clamp(0, 1) * 255).byte()         # (B, 3, H, W)
        # ultralytics accepts a list of BGR arrays; cv2 conversion is host-side
        # per image but cheap relative to the conv stack.
        bgr_list = [
            cv2.cvtColor(rect_u8[b].permute(1, 2, 0).contiguous().cpu().numpy(),
                         cv2.COLOR_RGB2BGR)
            for b in range(rect_u8.shape[0])
        ]
        results = self.yolo(bgr_list, verbose=False, imgsz=self.imgsz, device=0,
                            classes=self.classes, conf=self.conf, retina_masks=True)
        for s, r in zip(scan_indices, results):
            self._mask_cache[int(s)] = self._result_to_mask(r)

    def _result_to_mask(self, r) -> torch.Tensor:
        """One ultralytics result -> (H, W) bool mask (union of instance masks),
        dilated by dilate_px."""
        if r.masks is None or r.masks.data.shape[0] == 0:
            return torch.zeros((self.H, self.W), dtype=torch.bool, device=self.device)
        assert tuple(r.masks.data.shape[-2:]) == (self.H, self.W), (
            f"YOLO masks at {tuple(r.masks.data.shape[-2:])}, expected "
            f"{(self.H, self.W)} — retina_masks letterbox regression?")
        m = r.masks.data.unsqueeze(1).float()
        m = Fn.interpolate(m, size=(self.H, self.W), mode="nearest").squeeze(1)
        mask = m.any(dim=0) > 0.5
        if self.dilate_px > 0 and mask.any():
            # GPU binary dilation: max_pool2d (stride 1, symmetric pad) grows the
            # mask outward by dilate_px. Kernel side = 2*dilate_px + 1.
            k = 2 * self.dilate_px + 1
            dilated = Fn.max_pool2d(mask.float()[None, None], kernel_size=k,
                                    stride=1, padding=self.dilate_px)[0, 0]
            mask = dilated > 0.5
        return mask

    def filter(self, scan_idx: int, pts_lidar: np.ndarray) -> np.ndarray:
        """Return pts_lidar with dynamic-object-masked points removed."""
        if len(pts_lidar) == 0:
            return pts_lidar
        mask_hw = self.mask_for_scan(scan_idx)

        # Column-vector pipeline: each column of pts_h is one homogeneous point.
        pts = torch.from_numpy(pts_lidar).to(self.device)                     # (N, 3)
        ones = torch.ones((pts.shape[0], 1), dtype=torch.float32, device=self.device)
        pts_h = torch.cat([pts, ones], dim=1).T                               # (4, N)

        pts_cam_raw = (self.T_cam_lidar @ pts_h)[:3]                          # (3, N) raw cam
        pts_rect = self.R3 @ pts_cam_raw                                      # (3, N) rectified

        # Project: u_h = K @ P, then normalize by Z. Rows of ``uv_h`` are [u, v, 1].
        uv_h = self.K_rect @ pts_rect                                         # (3, N)
        uv_h = uv_h / uv_h[2]                                                 # (3, N), row 2 == 1
        u, v = uv_h[0].long(), uv_h[1].long()
        front = pts_rect[2] > 0.2
        inside = front & (u >= 0) & (u < self.W) & (v >= 0) & (v < self.H)
        is_dyn = torch.zeros_like(front)
        u_c = u.clamp(0, self.W - 1)
        v_c = v.clamp(0, self.H - 1)
        is_dyn[inside] = mask_hw[v_c[inside], u_c[inside]]
        keep = ~is_dyn
        return pts[keep].cpu().numpy()

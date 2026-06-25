#!/usr/bin/env python3
"""
Fuse RKO-LIO odometry + GPS into a georeferenced 6-DOF reference trajectory.

Pipeline:
  1. Load LIO odom + GPS + Vectornav IMU from the per-bag HDF5.
  2. Drop garbage GPS fixes (sigma_h > adaptive threshold + ±5s guard).
  3. Project GPS lat/lon to local UTM metres (origin = first valid fix).
  4. Compute adaptive LIO sigma_t (~2x bag's median GPS sigma) so the LIO
     stiffness matches GPS quality — prevents drift+snap kinks on noisy bags.
  5. Trim LIO cold-start/shutdown transients; remove odom glitches.
  6. Build keyframes (one per LIO sample ∪ GPS times).
  7. Flag bad_lio_speed keyframes (LIO disagrees with GPS speed → cold-start
     failure). Inflate their LIO between-factor sigma to ~disable them.
  8. Horn-align (good-LIO subset of) LIO trajectory → world for init.
     Override positions with GPS-interp for ALL kfs (smoother init).
  9. GTSAM batch solve, two-pass:
       pass 1: GPS factors target antenna directly (lever absorbed in rotation).
       pass 2: GPS factors re-target lidar = antenna - R_i * lever, using
               pass-1 rotations. Outliers rejected between passes by
               residual / receiver-sigma ratio.
 10. Trim non-physical endpoint kinks + kfs outside GPS span.
 11. Write the fused trajectory into the per-bag HDF5 under /fused_traj.

Inputs (per-bag HDF5):
  ouster/odom/map_T_lidart    [L,4,4]  lidar-rate (~10 Hz) RKO-LIO pose
  ouster/odom/t               [L]      lidar-rate timestamps
  gps/data                    [N,7]    lat, lon, alt, cov_xx, cov_yy, cov_zz, status
  gps/t                       [N]      GPS timestamps (s, main clock)
  vectornav/{t,accel,ang_vel} [M,...]  Vectornav IMU (gyro → rotation between-factors)
  calib/imgl_T_imu            [4,4]    camera-IMU calibration (Kalibr)
  ouster/imgl_T_ouster        [4,4]    camera-lidar calibration
  → composed R_lidar_imu = inv(imgl_T_ouster) @ imgl_T_imu (used as-is)

Lever (lidar -> antenna, lidar body frame, metres) is physically measured and
loaded from cal_map.yaml. Fixed in the graph — not solved for.
"""
import logging
import os
import numpy as np
import h5py
import pyproj
import yaml
import gtsam
from gtsam import Pose3, Point3
from scipy.spatial.transform import Rotation as R
from scipy.linalg import orthogonal_procrustes
from projectaria_tools.core.sophus import SE3, interpolate as se3_interp

logger = logging.getLogger(__name__)

DEFAULT_CAL_MAP = os.path.join(
    os.environ.get("OCTO_RAW_ROOT", "/data/rosbags/raw").rstrip("/"),
    "calibrations", "cal_map.yaml",
)

# GTSAM symbol shorthand: X(i) = i'th Pose3 in the trajectory.
X = lambda i: gtsam.symbol("x", i)
GPS_SIGMA_VALID_MAX_M = 5.0   # fixes with sigma_h above this are dropped.


# =============================================================================
# 1. DATA LOADING (HDF5 + cal_map)
# =============================================================================

def load_bag(h5_path):
    """Read lidar-rate odom + GPS from the per-bag HDF5.

    Returns:
      odom_T  [K,4,4]   lidar-rate pose (from /ouster/odom/map_T_lidart)
      odom_t  [K]       odom timestamps
      gps     [N,7]     lat, lon, alt, cov_xx, cov_yy, cov_zz, status (NO_FIX filtered out)
      gps_t   [N]       GPS timestamps
    """
    with h5py.File(h5_path, "r") as f:
        # New grouped paths: /gps/data + /gps/t.
        missing = [k for k in ("gps/data", "gps/t",
                               "ouster/odom/map_T_lidart", "ouster/odom/t")
                   if k not in f]
        if missing:
            logger.info(f"missing h5 keys: {missing}")
            return None
        odom_T  = f["ouster/odom/map_T_lidart"][:]
        odom_t  = f["ouster/odom/t"][:]
        gps     = f["gps/data"][:]
        gps_t   = f["gps/t"][:]
    
    valid = gps[:, 6].astype(np.int8) >= 0

    # Optional covariance gate (env GPS_COV_MAX, m^2): drop fixes whose horizontal
    # covariance exceeds the threshold. 
    cov_max = float(os.environ.get("GPS_COV_MAX", "inf"))
    if np.isfinite(cov_max) and gps.shape[1] >= 5:
        cov_ok = np.maximum(gps[:, 3], gps[:, 4]) < cov_max
        valid = valid & cov_ok
    return odom_T, odom_t, gps[valid], gps_t[valid]


def load_imu(h5_path):
    """Load Vectornav gyro + accel + lidar<-imu extrinsic (full SE3).

    lidar_T_imu = inv(imgl_T_ouster) @ imgl_T_imu, derived from h5 extrinsics:
    """
    with h5py.File(h5_path, "r") as f:
        if "vectornav/t" not in f:
            return None
        imu_t           = f["vectornav/t"][:]
        gyro            = f["vectornav/ang_vel"][:].astype(np.float64)
        accel           = f["vectornav/accel"][:].astype(np.float64)
        imgl_T_imu      = f["calib/imgl_T_imu"][:]
        imgl_T_ouster   = f["ouster/imgl_T_ouster"][:]
    lidar_T_imu = np.linalg.inv(imgl_T_ouster) @ imgl_T_imu
    return imu_t, gyro, accel, lidar_T_imu


def imu_relative_rotations(imu_t, gyro_lidar, kf_t):
    """For each adjacent (kf_i, kf_{i+1}) pair, integrate Vectornav gyro
    samples between them and return the relative rotation matrix.

    Midpoint rule per IMU step, composed multiplicatively:
        R_total = exp(ω_0 dt_0) · exp(ω_1 dt_1) · ...

    Each segment window is extended to span EXACTLY [kf_t[i], kf_t[i+1]] by
    prepending/appending the gyro linearly-interpolated at the kf boundary
    times.

    Output shape (N-1, 3, 3)."""
    n_kf  = len(kf_t)
    rel_R = np.tile(np.eye(3), (n_kf - 1, 1, 1))
    # Pre-interp gyro at every kf time (clipped to imu_t range by np.interp).
    g_kf  = np.column_stack([np.interp(kf_t, imu_t, gyro_lidar[:, k]) for k in range(3)])
    idx_lo = np.searchsorted(imu_t, kf_t[:-1])
    idx_hi = np.searchsorted(imu_t, kf_t[1:])
    for i in range(n_kf - 1):
        lo, hi = idx_lo[i], idx_hi[i]
        # Sandwich the imu samples between the kf-time interpolated values:
        #   t_in = [kf_t[i],    imu_t[lo], ..., imu_t[hi-1],    kf_t[i+1]]
        #   g_in = [g_kf[i],    gyro_lidar[lo:hi],              g_kf[i+1]]
        g_in  = np.vstack([g_kf[i:i+1], gyro_lidar[lo:hi], g_kf[i+1:i+2]])
        t_in  = np.concatenate([[kf_t[i]], imu_t[lo:hi], [kf_t[i+1]]])
        dt    = np.diff(t_in)
        g_mid = 0.5 * (g_in[:-1] + g_in[1:])
        steps = R.from_rotvec(g_mid * dt[:, None])   # exp(ω_k * dt_k) per step
        R_seg = R.identity()
        for s in steps:
            R_seg = R_seg * s
        rel_R[i] = R_seg.as_matrix()
    return rel_R


def load_lever(cal_map_path):
    """Car lever-arm (lidar->antenna, lidar frame, m) from cal_map. Falls back to
    [0,0,0] with a warning if absent (treats GPS as a lidar-origin measurement)."""
    arm = None
    if os.path.exists(cal_map_path):
        with open(cal_map_path) as f:
            cal = yaml.safe_load(f) or {}
        arm = (cal.get("car") or {}).get("lever_arm_default_xyz_m")
    if arm is None:
        logger.warning(f"no car lever in {cal_map_path}; using [0,0,0] "
                       f"(treats GPS as lidar-origin measurement).")
        return np.zeros(3)
    return np.asarray(arm, dtype=float)



def gps_to_local(gps):
    """GPS lat/lon/alt -> local UTM metric (E-E0, N-N0, alt-alt0).
    Origin = first fix (caller must pre-filter no-fix sentinels in main).
    UTM zone hardcoded to 18N (EPSG:32618) — covers Philadelphia + New York,
    where all our data is collected."""
    lat, lon, alt = gps[:, 0], gps[:, 1], gps[:, 2]
    tf = pyproj.Transformer.from_crs("EPSG:4326", "EPSG:32618", always_xy=True)
    east, nrth = tf.transform(lon, lat)
    e0, n0, a0 = east[0], nrth[0], alt[0]
    xyz  = np.column_stack([east - e0, nrth - n0, alt - a0])
    meta = {"epsg": 32618, "utm_zone": 18, "hemisphere": "N",
            "origin_easting": float(e0), "origin_northing": float(n0),
            "origin_alt": float(a0)}
    return xyz, meta


# =============================================================================
# 2. ODOMETRY HEALTH (trim/clean before fusion)
# =============================================================================

def trim_odom_transients(odom_t, odom_T, accel_g=1.5, win_s=2.0):
    """Trim LIO cold-start / shutdown ends, defined as non-physical acceleration
    (>accel_g·g) within `win_s` of either bag boundary.

    A LIO cold-start typically shows a *calm* unconverged lead-in (estimator
    thinks the vehicle is stationary) followed by a 'catch-up' jump once it
    locks on. We trim everything up to and including the LAST high-accel step
    near the start, and from the FIRST one near the end."""
    velocity     = np.gradient(odom_T[:, :3, 3], odom_t, axis=0)
    acceleration = np.linalg.norm(np.gradient(velocity, odom_t, axis=0), axis=1)
    is_bad       = acceleration > accel_g * 9.81
    t0, t1, n    = odom_t[0], odom_t[-1], len(odom_t)
    lead  = np.where(is_bad & (odom_t - t0 <= win_s))[0]
    trail = np.where(is_bad & (t1 - odom_t <= win_s))[0]
    lo = (lead.max() + 1) if len(lead) else 0
    hi = (trail.min() - 1) if len(trail) else n - 1
    return odom_t[lo:hi + 1], odom_T[lo:hi + 1]


def clean_odom_glitches(odom_t, odom_T, neighbour_win=5, k=8.0, floor_m=0.30):
    """Remove isolated LIO pose glitches (scan-match slips) that deviate from
    a local quadratic fit of their temporal neighbours and snap back.

    Left in, a single bad pose corrupts between-factors and interpolation,
    and the fusion amplifies it into a large kink. Threshold is adaptive:
    `max(k * median-deviation, floor_m)` so genuine hard motion is kept."""
    pos = odom_T[:, :3, 3]
    n   = len(odom_t)
    dev = np.zeros(n)
    for i in range(n):
        lo = max(0, i - neighbour_win)
        hi = min(n, i + neighbour_win + 1)
        neigh = [j for j in range(lo, hi) if j != i]
        if len(neigh) < 5:
            continue
        dt   = odom_t[neigh] - odom_t[i]
        pred = [np.polyval(np.polyfit(dt, pos[neigh, c], 2), 0.0) for c in range(3)]
        dev[i] = np.linalg.norm(pos[i] - pred)
    median_dev = np.median(dev[dev > 0]) if np.any(dev > 0) else 0.0
    threshold  = max(k * median_dev, floor_m)
    is_glitch  = dev > threshold
    return odom_t[~is_glitch], odom_T[~is_glitch]


# =============================================================================
# 3. GPS HEALTH (sigma, glitch detection, quality tier)
# =============================================================================

def sigma_from_cov(cov_row, floor=(0.01, 0.01, 0.02)):
    return np.maximum(np.sqrt(np.maximum(cov_row, 1e-6)), floor)


def detect_gps_glitches(gps_t, gps_xyz, neighbour_win=4, k=6.0, floor_m=0.15):
    """Flag GPS fixes inconsistent with a local quadratic fit of their temporal
    neighbours -- catches multipath/cycle-slip glitches that the receiver
    mislabels as RTK-fixed (their reported covariance misses them).
    """
    n   = len(gps_t)
    dev = np.zeros(n)
    for i in range(n):
        lo = max(0, i - neighbour_win)
        hi = min(n, i + neighbour_win + 1)
        neigh = [j for j in range(lo, hi) if j != i]
        if len(neigh) < 4:
            continue
        dt   = gps_t[neigh] - gps_t[i]
        pred = [np.polyval(np.polyfit(dt, gps_xyz[neigh, c], 2), 0.0) for c in range(2)]
        dev[i] = np.hypot(gps_xyz[i, 0] - pred[0], gps_xyz[i, 1] - pred[1])
    median_dev = np.median(dev[dev > 0]) if np.any(dev > 0) else 0.0
    return dev > max(k * median_dev, floor_m)


def classify_gps_quality(gps_cov):
    """Categorise each fix as RTK-fixed (sig_h<5cm), float (<50cm), or single.
    sig_h is max(sqrt(cov_xx), sqrt(cov_yy)) — worst of East/North to be
    robust to asymmetric multipath.
    Returns (rtk_pct, float_pct, single_pct, tier_string)."""
    sig_h = np.sqrt(np.maximum(gps_cov[:, 0], gps_cov[:, 1]))
    rtk   = float((sig_h < 0.05).mean() * 100)
    flt   = float(((sig_h >= 0.05) & (sig_h < 0.5)).mean() * 100)
    sng   = float((sig_h >= 0.5).mean() * 100)
    tier  = "RTK_fixed_cm" if rtk > 80 else ("float_dm" if rtk + flt > 60 else "single_m")
    return rtk, flt, sng, tier


# =============================================================================
# 4. SE(3) MATH (interp, Horn alignment, snap-tick distance)
# =============================================================================

def interp_se3(src_t, src_T, query_t):
    """Geodesic SE(3) interp at query times via projectaria_tools sophus —
    constant-twist between adjacent samples.
    """
    poses = [SE3.from_matrix3x4(src_T[i, :3, :]) for i in range(len(src_T))]
    qt    = np.clip(np.asarray(query_t, dtype=float), src_t[0], src_t[-1])
    j     = np.searchsorted(src_t, qt).clip(1, len(src_t) - 1)
    alpha = (qt - src_t[j - 1]) / (src_t[j] - src_t[j - 1])
    Tg    = [se3_interp(poses[jj - 1], poses[jj], float(a)) for jj, a in zip(j, alpha)]
    out = np.tile(np.eye(4), (len(qt), 1, 1))
    out[:, :3, :3] = np.stack([T.rotation().to_matrix().squeeze()     for T in Tg])
    out[:, :3, 3]  = np.stack([np.asarray(T.translation()).reshape(3) for T in Tg])
    return out


def horn_align(src, dst):
    """Rigid  Umeyama src->dst. Returns 4x4 T_dst_src.
    """
    cs, cd  = src.mean(0), dst.mean(0)
   
    R_row, _ = orthogonal_procrustes(src - cs, dst - cd)
    Rm = R_row.T
    if np.linalg.det(Rm) < 0:
        Rm[:, -1] *= -1     # flip a column → make it a proper rotation
    T = np.eye(4); T[:3, :3] = Rm; T[:3, 3] = cd - Rm @ cs
    return T


def flag_lio_speed_disagreement(kf_t, kf_T_in_map, gps_t, gps_xyz,
                                rel_tol=0.30, abs_tol_mps=1.5):
    """Per-keyframe mask: True where LIO speed disagrees with GPS speed.

    Returned as a mask preserving the bag duration. The solver inflates the
    LIO between-factor sigma in the masked region so GPS factors drive those
    poses — trajectory still has values at those kfs, just GPS-driven not
    LIO-driven.

    LIO cold-start failures show as long contiguous True runs at the start
    of the bag (LIO reports ~0 m/s while GPS shows the car moving)."""
    lio_pos     = kf_T_in_map[:, :3, 3]
    lio_speed   = np.linalg.norm(np.gradient(lio_pos, kf_t, axis=0), axis=1)
    gps_speed   = np.linalg.norm(np.gradient(gps_xyz[:, :2], gps_t, axis=0), axis=1)
    gps_at_kf   = np.interp(kf_t, gps_t, gps_speed)
    return np.abs(lio_speed - gps_at_kf) > np.maximum(rel_tol * gps_at_kf, abs_tol_mps)


# =============================================================================
# 5. KEYFRAME SELECTION + GRAPH SOLVE
# =============================================================================

def select_keyframes(odom_t, gps_t, min_dt_s=0.005):
    """Keyframes = every IMU-rate odom sample UNION GPS times.
    `min_dt_s` collapses any pair closer than that (avoids degenerate edges)."""
    gps_in = gps_t[(gps_t >= odom_t[0]) & (gps_t <= odom_t[-1])]
    kf_t   = np.unique(np.concatenate([odom_t, gps_in]))
    keep   = np.concatenate([[True], np.diff(kf_t) >= min_dt_s])
    return kf_t[keep]


def associate_gps_to_keyframes(kf_t, gps_t):
    """Map each in-span GPS fix to its nearest keyframe index.
    Returns (kf_idx_per_fix [M], in_span [N])."""
    in_span = (gps_t >= kf_t[0]) & (gps_t <= kf_t[-1])
    fix_t   = gps_t[in_span]
    kf_idx  = np.searchsorted(kf_t, fix_t).clip(0, len(kf_t) - 1)
    # nearest of (kf[idx-1], kf[idx])
    left    = (kf_idx - 1).clip(0)
    pick    = np.abs(kf_t[left] - fix_t) < np.abs(kf_t[kf_idx] - fix_t)
    kf_idx[pick] = left[pick]
    return kf_idx, in_span


def solve_pose_graph(kf_t, kf_T_init, kf_T_lio, kf_idx_per_fix, gps_meas, gps_cov,
                     odom_sigma_t, odom_sigma_r_deg,
                     huber_k, exclude_mask, extra_bad_kf=None,
                     imu_rel_R=None, imu_gyro_sigma=6.1e-5):
    """Build + solve one batch pose graph.

    Variables:
      X(i) = Pose3 = T_world_lidar at keyframe i.

    Factors:
      - Loose prior on X(0).
      - BetweenFactorPose3 between every adjacent keyframe pair, with the
        relative pose interpolated from odom and a sigma that:
          * scales as sqrt(gap/0.1) — larger gaps trust LIO less
          * inflates 10000x if either endpoint sits at a non-physical accel
            spike or a bad_lio_speed flagged keyframe
      - BetweenFactorPose3 from Vectornav gyro (rotation-only, translation
        sigma 100m), independent of LIO.
      - GPSFactor at every kf_idx_per_fix entry (unless masked out), Huber-wrapped.

    Returns (result, graph, error_before, error_after, n_gps_added).
    """
    n_kf  = len(kf_t)
    graph = gtsam.NonlinearFactorGraph()
    init  = gtsam.Values()
    for i in range(n_kf):
        init.insert(X(i), Pose3(kf_T_init[i]))

    # Loose prior on X(0)
    prior_sigma = np.array([1., 1., 1., 10., 10., 10.])    # rot rad, trans m
    graph.add(gtsam.PriorFactorPose3(
        X(0), Pose3(kf_T_init[0]),
        gtsam.noiseModel.Diagonal.Sigmas(prior_sigma)))

    # Between-factors: LIO-derived relative poses at the keyframe pairs.
    # kf_T_lio is the LIO-frame pose at each kf (precomputed in main).
    # bad_kf is the union of all "don't trust this kf's LIO" flags (speed
    # disagreement + accel spike), also computed in main. At a flagged edge
    # the sigma is inflated 10000x to effectively disable the LIO factor.
    bad_kf   = np.asarray(extra_bad_kf, bool) if extra_bad_kf is not None \
               else np.zeros(n_kf, bool)
    base_sig = np.array([np.radians(odom_sigma_r_deg)] * 3 + [odom_sigma_t] * 3)
    for i in range(n_kf - 1):
        between   = Pose3(kf_T_lio[i]).between(Pose3(kf_T_lio[i + 1]))
        gap_scale = np.sqrt(max(kf_t[i + 1] - kf_t[i], 1e-3) / 0.1)
        scale     = gap_scale
        if bad_kf[i] or bad_kf[i + 1]:
            scale *= 10000.0     # effectively disable LIO between-factor at this edge
        graph.add(gtsam.BetweenFactorPose3(
            X(i), X(i + 1), between,
            gtsam.noiseModel.Diagonal.Sigmas(base_sig * scale)))

    # Independent IMU rotation between-factors (Vectornav, NOT the Ouster IMU
    # that LIO is integrating). Rotation sigma comes from gyro noise density;
    # translation sigma is set to 100m so the factor effectively does nothing
    # to position — orientation only. Independent evidence rescues cold-start
    # regions where LIO is broken.
    if imu_rel_R is not None:
        for i in range(n_kf - 1):
            gap = max(kf_t[i + 1] - kf_t[i], 1e-3)
            # Per-axis rotation sigma over the gap: noise_density * sqrt(gap).
            # Floor at 1e-4 rad so unmodeled bias drift doesn't get over-trusted.
            rot_sig = max(imu_gyro_sigma * np.sqrt(gap), 1e-4)
            sig_imu = np.array([rot_sig, rot_sig, rot_sig, 100., 100., 100.])
            T_rel = np.eye(4); T_rel[:3, :3] = imu_rel_R[i]
            graph.add(gtsam.BetweenFactorPose3(
                X(i), X(i + 1), Pose3(T_rel),
                gtsam.noiseModel.Diagonal.Sigmas(sig_imu)))

    # GPS factors: one per non-excluded fix, Huber-wrapped on its own sigma.
    n_gps = 0
    for j, kf_i in enumerate(kf_idx_per_fix):
        if exclude_mask[j]:
            continue
        sig   = sigma_from_cov(gps_cov[j])
        noise = gtsam.noiseModel.Robust.Create(
            gtsam.noiseModel.mEstimator.Huber(huber_k),
            gtsam.noiseModel.Diagonal.Sigmas(sig))
        graph.add(gtsam.GPSFactor(X(kf_i), Point3(*gps_meas[j]), noise))
        n_gps += 1

    params = gtsam.LevenbergMarquardtParams()
    params.setMaxIterations(100)
    err_before = graph.error(init)
    result     = gtsam.LevenbergMarquardtOptimizer(graph, init, params).optimize()
    return result, graph, err_before, graph.error(result), n_gps


def reject_outliers(result, kf_t, kf_idx_per_fix, gps_meas_antenna, gps_cov,
                    already_excluded, k=5.0, floor_m=0.20, dilate_s=0.6):
    """Drop GPS fixes whose pass-1 horizontal residual exceeds both:
      - absolute floor (catches gross glitches)
      - k * its own reported sigma_h (catches RTK-grade fixes that lied)

    Glitches cluster (cycle-slips last several fixes), so dilate in time:
    any fix within `dilate_s` of a detected outlier is also dropped, which
    lets smooth odom bridge the gap.

    Returns (new_exclude_mask, n_newly_rejected)."""
    pred_pos = np.array([result.atPose3(X(kf_idx_per_fix[j])).translation()
                         for j in range(len(kf_idx_per_fix))])
    horiz_resid = np.linalg.norm(gps_meas_antenna[:, :2] - pred_pos[:, :2], axis=1)
    sig_h       = np.sqrt(np.maximum(np.maximum(gps_cov[:, 0], gps_cov[:, 1]), 1e-6))
    is_outlier  = horiz_resid > np.maximum(k * sig_h, floor_m)
    fix_t       = kf_t[kf_idx_per_fix]
    if is_outlier.any():
        outlier_t  = fix_t[is_outlier]
        is_outlier = is_outlier | (np.abs(fix_t[:, None] - outlier_t[None, :]).min(1) <= dilate_s)
    n_newly_rejected = int((is_outlier & ~already_excluded).sum())
    return already_excluded | is_outlier, n_newly_rejected


def solve_two_pass(kf_t, kf_T_init, kf_T_lio,
                   gps_t, gps_xyz, gps_cov, lever,
                   odom_sigma_t, odom_sigma_r_deg,
                   huber_k, glitch_mask, reject_k, reject_floor,
                   extra_bad_kf=None, imu_rel_R=None):
    """Two-pass GTSAM solve with FIXED lever:
      pass 1 — GPS factors target the antenna position directly (lever
               absorbed into the rotation freedom). Use this to get an
               orientation estimate good enough for the lever correction.
      pass 2 — re-target each GPS factor to the LIDAR origin via
               meas_lidar = meas_antenna - R_i * lever  (R_i from pass 1).
               Solve again, excluding outliers detected after pass 1.

    Returns (result, graph, e0, e1, n_gps, n_outliers, kf_idx_per_fix, in_span).
    """
    kf_idx, in_span = associate_gps_to_keyframes(kf_t, gps_t)
    gps_meas_antenna = gps_xyz[in_span]
    cov              = gps_cov[in_span]
    init_excluded    = np.asarray(glitch_mask)[in_span]

    # PASS 1 — lever absorbed (GPS sees the antenna).
    result1, _, _, _, _ = solve_pose_graph(
        kf_t, kf_T_init, kf_T_lio, kf_idx, gps_meas_antenna, cov,
        odom_sigma_t, odom_sigma_r_deg, huber_k,
        exclude_mask=init_excluded, extra_bad_kf=extra_bad_kf,
        imu_rel_R=imu_rel_R)

    # Outlier rejection using pass-1 residuals on the antenna target.
    # Glitches also get caught by the Huber kernel + the upfront
    # detect_gps_glitches self-consistency pass.
    excluded, n_outliers = reject_outliers(
        result1, kf_t, kf_idx, gps_meas_antenna, cov,
        init_excluded, k=reject_k, floor_m=reject_floor)

    # PASS 2 — re-target GPS to lidar origin (antenna - R_i*lever).
    rot_at_fix = [result1.atPose3(X(kf_idx[j])).rotation().matrix()
                  for j in range(len(kf_idx))]
    gps_meas_lidar = np.array([gps_meas_antenna[j] - rot_at_fix[j] @ lever
                               for j in range(len(kf_idx))])
    kf_T_pass1 = np.array([result1.atPose3(X(i)).matrix() for i in range(len(kf_t))])
    result2, graph2, e0, e1, n_gps = solve_pose_graph(
        kf_t, kf_T_pass1, kf_T_lio, kf_idx, gps_meas_lidar, cov,
        odom_sigma_t, odom_sigma_r_deg, huber_k,
        exclude_mask=excluded, extra_bad_kf=extra_bad_kf,
        imu_rel_R=imu_rel_R)
    return result2, graph2, e0, e1, n_gps, n_outliers, kf_idx, in_span


def trim_endpoint_kinks(kf_t, kf_T, accel_thresh=30.0, max_trim=15):
    """Drop poorly-constrained endpoint keyframes whose placement is
    non-physical (under-anchored first/last nodes). Only the first/last few
    are eligible; the interior is never touched. Returns (lo, hi) slice."""
    n = len(kf_t)
    v = np.gradient(kf_T[:, :3, 3], kf_t, axis=0)
    a = np.linalg.norm(np.gradient(v, kf_t, axis=0), axis=1)
    lo = 0
    while lo < max_trim and lo < n - 2 and a[lo] > accel_thresh:
        lo += 1
    hi = n
    while (n - hi) < max_trim and hi > lo + 2 and a[hi - 1] > accel_thresh:
        hi -= 1
    return lo, hi


# =============================================================================
# 6. POST-SOLVE METRICS
# =============================================================================

def gps_residuals(kf_t, kf_T, gps_t, gps_xyz, lever):
    """Per-fix antenna-position residuals (m). Array-based (decoupled from the
    gtsam result) so it survives endpoint trimming."""
    out = []
    for j in range(len(gps_t)):
        if gps_t[j] < kf_t[0] or gps_t[j] > kf_t[-1]:
            continue
        i = int(np.argmin(np.abs(kf_t - gps_t[j])))
        pred_antenna = kf_T[i, :3, 3] + kf_T[i, :3, :3] @ lever
        out.append(gps_xyz[j] - pred_antenna)
    return np.array(out)


# =============================================================================
# 7. OUTPUT + ORCHESTRATION (run() is invoked from timeSync_process)
# =============================================================================

def write_h5_trajectory(h5_path, kf_t, kf_T, utm_meta, lever, gps_cov,
                        residuals, e0, e1, n_gps_factors, n_outliers):
    """Persist the fused reference trajectory into the per-bag h5 under /fused_traj.

    Layout:
      /fused_traj/t              (N,)     float64  main-clock timestamps
      /fused_traj/T_world_lidar  (N,4,4)  float64  SE(3) pose, lidar in world (UTM-rel)
      /fused_traj attrs:         provenance, UTM origin, lever, residual stats, tier.

    """
    _, _, _, tier = classify_gps_quality(gps_cov)
    horiz     = np.linalg.norm(residuals[:, :2], axis=1)
    rms_horiz = float(np.sqrt(np.mean(horiz**2)))
    p90       = float(np.percentile(horiz, 90))

    with h5py.File(h5_path, "a") as f:
        if "fused_traj" in f:
            del f["fused_traj"]
        grp = f.create_group("fused_traj")
        grp.create_dataset("t",             data=np.asarray(kf_t, np.float64))
        grp.create_dataset("T_world_lidar", data=np.asarray(kf_T, np.float64))
        # Provenance + frame
        grp.attrs["frame"]               = "world (UTM-relative; origin = first valid GPS fix)"
        grp.attrs["reference_point"]     = "lidar" if not np.allclose(lever, 0) else "antenna"
        grp.attrs["lever_xyz_lidar_m"]   = np.asarray(lever, np.float64)
        # UTM origin (so a consumer can reconstruct global UTM by adding origin_*)
        grp.attrs["epsg"]                = int(utm_meta["epsg"])
        grp.attrs["utm_zone"]            = int(utm_meta["utm_zone"])
        grp.attrs["hemisphere"]          = str(utm_meta["hemisphere"])
        grp.attrs["origin_easting_m"]    = float(utm_meta["origin_easting"])
        grp.attrs["origin_northing_m"]   = float(utm_meta["origin_northing"])
        grp.attrs["origin_alt_m"]        = float(utm_meta["origin_alt"])
        # Fusion quality (compact: horizontal residual RMS + p90 + tier).
        grp.attrs["gps_residual_rms_horiz_m"] = rms_horiz
        grp.attrs["gps_residual_horiz_p90_m"] = p90
        grp.attrs["confidence_tier"]          = str(tier)


def run(h5, cal_map=DEFAULT_CAL_MAP,
        odom_sigma_t=0.20, odom_sigma_r_deg=0.5,
        huber_k=10.0, reject_k=5.0, reject_floor=0.20):
    """Fuse LIO + GPS into /fused_traj for a single bag h5, writing the
    /fused_traj group into `h5` in-place. Invoked from timeSync_process."""
    # ---- 1. LOAD --------------------------------------------------------------
    result = load_bag(h5)
    if result is None:
        logger.info(f"SKIP {os.path.basename(h5)}: required h5 datasets absent "
                    f"(GPS inactive or RKO-LIO odom missing from /ouster/odom)")
        return
    odom_T, odom_t, gps, gps_t = result
    if len(gps_t) < 20:
        logger.info(f"SKIP {os.path.basename(h5)}: only {len(gps_t)} GPS fixes")
        return

    # Drop high-sigma fixes (multipath in tunnels, no-fix sentinels) PLUS a guard
    # window of +/-5s around each. Threshold is ADAPTIVE: RTK-dominated bags
    # (sub-cm normal sigma) drop sigma > 0.5m; single-pt bags (50cm-2m normal)
    # drop sigma > 5m only. Decision based on the bag's 25th-percentile sigma.
    sig_h_raw = np.sqrt(np.maximum(np.maximum(gps[:, 3], gps[:, 4]), 0))
    sig_p25   = float(np.percentile(sig_h_raw, 25))
    threshold = 0.5 if sig_p25 < 0.05 else GPS_SIGMA_VALID_MAX_M   # 5cm = RTK
    bad = sig_h_raw >= threshold
    if bad.any():
        guard = np.zeros(len(gps_t), bool)
        for tb in gps_t[bad]:
            guard |= (np.abs(gps_t - tb) <= 5.0)
        valid = ~guard
    else:
        valid = np.ones(len(gps_t), bool)
    if valid.sum() < 20:
        logger.info(f"SKIP {os.path.basename(h5)}: only {int(valid.sum())} valid GPS "
                    f"fixes (sigma_h < {threshold}m + 5s guard), need >= 20")
        return
    gps   = gps[valid]
    gps_t = gps_t[valid]

    lever = load_lever(cal_map)
    gps_xyz, utm_meta = gps_to_local(gps)
    gps_cov = gps[:, 3:6]

    # Adaptive LIO sigma: loosen LIO on noisy-GPS bags so the trajectory tracks
    # GPS instead of fighting it. LIO sigma_t = max(user, 2 * GPS median sigma_h).
    gps_sig_median = float(np.median(np.sqrt(np.maximum(gps_cov[:, 0], gps_cov[:, 1]))))
    odom_sigma_t = max(odom_sigma_t, 2.0 * gps_sig_median)

    # ---- 2. ODOMETRY HEALTH ---------------------------------------------------
    odom_t, odom_T = trim_odom_transients(odom_t, odom_T)
    odom_t, odom_T = clean_odom_glitches(odom_t, odom_T)

    # ---- 3. KEYFRAMES + WORLD-FRAME INIT --------------------------------------
    kf_t        = select_keyframes(odom_t, gps_t)
    kf_T_in_map = interp_se3(odom_t, odom_T, kf_t)

    # Flag keyframes where LIO is unreliable: (a) speed disagreement with GPS
    # (cold-start failure), (b) accel spike > 1.5g (isolated bad frame). The
    # solver inflates the LIO between-factor sigma at flagged edges.
    bad_lio_speed = flag_lio_speed_disagreement(kf_t, kf_T_in_map, gps_t, gps_xyz)
    kf_accel = np.linalg.norm(np.gradient(np.gradient(kf_T_in_map[:, :3, 3],
                                                      kf_t, axis=0), kf_t, axis=0), axis=1)
    bad_lio = bad_lio_speed | (kf_accel > 1.5 * 9.81)

    # Independent Vectornav gyro evidence -- anchors rotation locally without
    # depending on LIO. R_lidar_imu (Kalibr o cam-lidar calibration) used as-is.
    imu_rel_R = None
    loaded = load_imu(h5)
    if loaded is None:
        logger.warning("vectornav/* not in h5; continuing without IMU rotation factors")
    else:
        imu_t, gyro, accel, lidar_T_imu = loaded
        R_lidar_imu = lidar_T_imu[:3, :3]
        gyro_lidar  = (R_lidar_imu @ gyro.T).T
        imu_rel_R = imu_relative_rotations(imu_t, gyro_lidar, kf_t)

    # Align (map -> world) using GPS positions at keyframes, restricted to
    # well-LIO-tracked correspondences 
    kf_idx_gps, gps_in_span = associate_gps_to_keyframes(kf_t, gps_t)
    if kf_idx_gps.size < 10:
        logger.info(f"SKIP {os.path.basename(h5)}: only {kf_idx_gps.size} GPS fixes "
                    f"overlap the odom span")
        return
    horn_keep = ~bad_lio[kf_idx_gps]
    if horn_keep.sum() < 10:
        logger.warning(f"only {int(horn_keep.sum())} good-LIO GPS samples for Horn -- falling back to all")
        horn_keep[:] = True
    kf_idx_hk   = kf_idx_gps[horn_keep]
    src_lidar   = kf_T_in_map[kf_idx_hk, :3, 3]
    src_R       = kf_T_in_map[kf_idx_hk, :3, :3]
    src_antenna = src_lidar + (src_R @ lever)
    T_world_from_map = horn_align(src_antenna, gps_xyz[gps_in_span][horn_keep])
    kf_T_init = T_world_from_map @ kf_T_in_map

    # Override init TRANSLATION with GPS-interp (smooth world-correct seed);
    # rotation stays from Horn-aligned LIO.
    gps_pos_at_kf = np.column_stack(
        [np.interp(kf_t, gps_t, gps_xyz[:, k]) for k in range(3)])
    kf_T_init[:, :3, 3] = gps_pos_at_kf

    # ---- 4. PRE-SOLVE GLITCH FLAGGING -----------------------------------------
    glitch_mask = detect_gps_glitches(gps_t, gps_xyz)

    # ---- 5. BATCH SOLVE (2-pass, fixed lever) ---------------------------------
    result, graph, e0, e1, n_gps, n_outliers, kf_idx, in_span = solve_two_pass(
        kf_t, kf_T_init, kf_T_in_map,
        gps_t, gps_xyz, gps_cov, lever,
        odom_sigma_t, odom_sigma_r_deg,
        huber_k, glitch_mask,
        reject_k, reject_floor, extra_bad_kf=bad_lio,
        imu_rel_R=imu_rel_R)

    # ---- 6. EXTRACT + ENDPOINT TRIM -------------------------------------------
    kf_T = np.array([result.atPose3(X(i)).matrix() for i in range(len(kf_t))])
    lo, hi = trim_endpoint_kinks(kf_t, kf_T)
    # Also trim keyframes outside the GPS time span (no anchor there -> drift).
    gps_lo, gps_hi = float(gps_t[0]), float(gps_t[-1])
    while lo < hi - 1 and kf_t[lo] < gps_lo: lo += 1
    while hi > lo + 1 and kf_t[hi - 1] > gps_hi: hi -= 1
    kf_t, kf_T = kf_t[lo:hi], kf_T[lo:hi]

    # ---- 7. WRITE /fused_traj TO THE BAG H5 -----------------------------------
    # GPS residuals feed the trajectory's quality attrs (RMS horiz / p90 / tier).
    resid = gps_residuals(kf_t, kf_T, gps_t, gps_xyz, lever)
    write_h5_trajectory(h5, kf_t, kf_T, utm_meta, lever, gps_cov,
                        resid, e0, e1, n_gps, n_outliers)

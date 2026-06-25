import numpy as np
import cv2
from scipy.optimize import least_squares
from typing import Tuple


def umeyama_solve(A: np.ndarray, B: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Solve rigid transform B = R*A + t for A,B (Nx3).
    Returns R (3x3), t (3,)
    """
    mu_A = A.mean(axis=0)
    mu_B = B.mean(axis=0)
    X = A - mu_A
    Y = B - mu_B
    H = Y.T @ X                
    U, S, Vt = np.linalg.svd(H)
    R = U @ Vt
    if np.linalg.det(R) < 0:     # ensure a proper rotation, not a reflection
        Vt[-1, :] *= -1
        R = U @ Vt
    t = mu_B - R @ mu_A
    return R, t


def robust_ransac_extrinsics(lidar_pts: np.ndarray,
                             cam_pts: np.ndarray,
                             min_samples: int = 3,
                             threshold: float = 0.05,
                             max_iters: int = 500):
    N = lidar_pts.shape[0]
    best_inliers = []

    # RANSAC: find the inlier set;
    # we re-fit on all inliers below.
    for _ in range(max_iters):
        idx = np.random.choice(N, size=min_samples, replace=False)
        try:
            cam_R_lidar, cam_T_lidar = umeyama_solve(lidar_pts[idx], cam_pts[idx])
        except Exception:
            continue

        pred = (cam_R_lidar @ lidar_pts.T).T + cam_T_lidar
        err = np.linalg.norm(pred - cam_pts, axis=1)
        inliers = np.where(err < threshold)[0]
        if len(inliers) > len(best_inliers):
            best_inliers = inliers

    if len(best_inliers) < min_samples:
        raise RuntimeError("RANSAC failed to find a valid model")

    # Final estimate: re-fit on the full inlier consensus set.
    best_cam_R_lidar, best_cam_T_lidar = umeyama_solve(
        lidar_pts[best_inliers], cam_pts[best_inliers]
    )
    return best_cam_R_lidar, best_cam_T_lidar, best_inliers


def refine_extrinsics(cam_R_lidar: np.ndarray, cam_T_lidar: np.ndarray,
                      lidar_pts: np.ndarray, cam_pts: np.ndarray):
    rvec, _ = cv2.Rodrigues(cam_R_lidar)

    def residuals(params):
        r = params[:3]
        t_ = params[3:]
        Rm, _ = cv2.Rodrigues(r)
        pred = (Rm @ lidar_pts.T).T + t_
        return (pred - cam_pts).ravel()

    x0 = np.hstack([rvec.flatten(), cam_T_lidar])
    result = least_squares(residuals, x0, method='lm')
    cam_R_lidar_opt, _ = cv2.Rodrigues(result.x[:3])
    cam_T_lidar_opt = result.x[3:]
    return cam_R_lidar_opt, cam_T_lidar_opt


def solve_extrinsics_robust(lidar_pts: np.ndarray, cam_pts: np.ndarray,
                            ransac_threshold: float = 0.05):
    cam_R_lidar, cam_T_lidar, inliers = robust_ransac_extrinsics(
        lidar_pts, cam_pts, threshold=ransac_threshold
    )
    cam_R_lidar, cam_T_lidar = refine_extrinsics(
        cam_R_lidar, cam_T_lidar, lidar_pts[inliers], cam_pts[inliers]
    )
    return cam_R_lidar, cam_T_lidar, inliers


def solve_extrinsics_circle_opt(
    observations,
    cam_R_lidar_init: np.ndarray,
    cam_T_lidar_init: np.ndarray,
    circle_center_O: np.ndarray,
    circle_radius: float,
    plane_normal_O: np.ndarray = np.array([0.0, 0.0, 1.0]),
    center_weight: float = 1.0,
    radius_weight: float = 1.0,
    plane_weight: float = 1.0,
):
    """
    Refine the LiDAR->camera extrinsics (cam_R_lidar) using circle center,
    radius, and plane residuals.
    observations: list of dicts with keys:
      - rim_points_L: (N_i, 3) LiDAR rim points in LiDAR frame
      - circle_center_L: (3,) measured LiDAR circle center in LiDAR frame
      - cam_R_board: (3,3) board->camera rotation from PnP
      - cam_T_board: (3,)  board->camera translation from PnP
    cam_R_lidar_init, cam_T_lidar_init: initial LiDAR->camera extrinsics.
    Returns refined (cam_R_lidar, cam_T_lidar, result).
    """
    circle_center_O = np.asarray(circle_center_O, dtype=float).reshape(3)
    plane_normal_O = np.asarray(plane_normal_O, dtype=float).reshape(3)
    plane_normal_O = plane_normal_O / (np.linalg.norm(plane_normal_O) + 1e-12)

    rvec_init, _ = cv2.Rodrigues(cam_R_lidar_init)
    x0 = np.hstack([rvec_init.flatten(), cam_T_lidar_init.reshape(3)])

    # Use sqrt of weights so that weight behaves like a variance scaling in least squares
    w_center = float(np.sqrt(max(center_weight, 0.0)))
    w_radius = float(np.sqrt(max(radius_weight, 0.0)))
    w_plane = float(np.sqrt(max(plane_weight, 0.0)))

    def residuals(x):
        rvec = x[:3]
        cam_T_lidar = x[3:]
        cam_R_lidar, _ = cv2.Rodrigues(rvec)
        res = []
        for obs in observations:
            rim_L = np.asarray(obs["rim_points_L"], dtype=float)
            if rim_L.size == 0:
                continue
            c_L = np.asarray(obs["circle_center_L"], dtype=float).reshape(3)
            cam_R_board = np.asarray(obs["cam_R_board"], dtype=float).reshape(3, 3)
            cam_T_board = np.asarray(obs["cam_T_board"], dtype=float).reshape(3)

            # camera -> board (inverse of the PnP board->camera pose)
            board_R_cam = cam_R_board.T
            board_T_cam = -cam_R_board.T @ cam_T_board
            # Center residual (3-vector in board/object frame)
            c_C = cam_R_lidar @ c_L + cam_T_lidar # predict cc from Lidar in cam frame
            c_O_pred = board_R_cam @ c_C + board_T_cam # transform them in board frame
            res_center = (c_O_pred - circle_center_O) * w_center
            res.extend(res_center.tolist()) # Center Residuals
            # Per-point residuals
            p_C = (cam_R_lidar @ rim_L.T + cam_T_lidar[:, None]).T # rim points TF to board frame
            p_O = (board_R_cam @ p_C.T + board_T_cam.reshape(3, 1)).T
            delta_O = p_O - circle_center_O.reshape(1, 3) # center the circle, GT
            # radius residuals
            r = (np.linalg.norm(delta_O[:, :2], axis=1) - float(circle_radius)) * w_radius
            # plane residuals
            pl = (delta_O @ plane_normal_O) * w_plane
            res.extend(r.tolist())
            res.extend(pl.tolist())
        return np.asarray(res, dtype=float)

    result = least_squares(residuals, x0, method="lm")
    cam_R_lidar, _ = cv2.Rodrigues(result.x[:3])
    cam_T_lidar = result.x[3:]
    return cam_R_lidar, cam_T_lidar, result


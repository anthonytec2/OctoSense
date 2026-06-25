import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.linear_model import RANSACRegressor
import open3d as o3d
from ouster.sdk import client


class CircleModel(BaseEstimator, RegressorMixin):
    """Algebraic circle fit used inside RANSAC."""
    def fit(self, X, y=None):
        x = X[:, 0]
        y_ = X[:, 1]
        A = np.column_stack([x, y_, np.ones_like(x)])
        b = -(x**2 + y_**2)
        C, *_ = np.linalg.lstsq(A, b, rcond=None)

        D, E, F = C
        self.center_ = np.array([-D/2, -E/2])
        self.radius_ = np.sqrt((D**2 + E**2)/4 - F)
        return self

    def predict(self, X):
        d = np.linalg.norm(X - self.center_, axis=1)
        return np.abs(d - self.radius_)


def detect_circle_center_3d(
    metadata,
    range_img,
    refl_img,
    refl_thresh=180,
    plane_dist=0.025,
    circle_dist=0.025,
    max_trials=2000,
):
    """
    Detects the reflective target's OUTER 3D circle center.
    Returns:
      center_3d, final_radius, plane_model, inlier_mask, outer_pts_3d, u, v
    """
    mask = refl_img > refl_thresh
    xyzlut = client.XYZLut(metadata)
    pts_all = xyzlut(range_img)
    circle_points = pts_all[mask]

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(circle_points)

    plane_model, inliers = pcd.segment_plane(
        distance_threshold=plane_dist,
        ransac_n=5,
        num_iterations=4000
    )

    a, b, c, d = plane_model
    normal = np.array([a, b, c])
    normal /= np.linalg.norm(normal)

    inlier_mask = np.zeros(len(circle_points), dtype=bool)
    inlier_mask[inliers] = True
    pts = circle_points[inlier_mask]

    dist = (pts @ normal) + d
    pts_proj = pts - dist[:, None] * normal

    tmp = np.array([1, 0, 0]) if abs(normal[0]) < 0.9 else np.array([0, 1, 0])
    u = np.cross(normal, tmp)
    u /= np.linalg.norm(u)
    v = np.cross(normal, u)
    v /= np.linalg.norm(v)

    plane_origin = -d * normal

    X = (pts_proj - plane_origin) @ u
    Y = (pts_proj - plane_origin) @ v
    pts_2d = np.column_stack([X, Y])

    ransac = RANSACRegressor(
        estimator=CircleModel(),  # use all ring points; robust to outliers
        min_samples=3,
        residual_threshold=circle_dist,
        max_trials=max_trials
    )
    ransac.fit(pts_2d, np.zeros(len(pts_2d)))
    inlier_mask = getattr(ransac, "inlier_mask_", None)
    if inlier_mask is None:
        residuals = np.abs(ransac.estimator_.predict(pts_2d))
        inlier_mask = residuals <= circle_dist

    final_model = CircleModel().fit(pts_2d[inlier_mask])
    final_center_2d = final_model.center_
    final_radius = final_model.radius_

    center_3d = (
        plane_origin
        + final_center_2d[0] * u
        + final_center_2d[1] * v
    )
    # 3D points corresponding to inliers used in the fit
    outer_pts_3d = pts_proj[inlier_mask]

    return center_3d, final_radius, plane_model, inlier_mask, outer_pts_3d, u, v



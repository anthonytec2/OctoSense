import numpy as np
import yaml
import cv2
from pupil_apriltags import Detector


def load_kalibr_camera(calib_file: str, cam_name: str):
    """
    Load camera intrinsics from a Kalibr camchain YAML file (top-level keys
    cam0/cam1/..., each with `intrinsics` [fx,fy,cx,cy] and `distortion_coeffs`).
    Returns K (3x3) and dist (N,) distortion coefficients.
    """
    with open(calib_file, "r") as f:
        cam = yaml.safe_load(f)[cam_name]

    fx, fy, cx, cy = cam["intrinsics"][:4]
    K = np.array([[fx, 0, cx],
                  [0, fy, cy],
                  [0,  0,  1]], dtype=float)
    dist = np.array(cam["distortion_coeffs"], dtype=float)
    return K, dist




# ---------------------------------------------------------
# 3. Build AprilGrid object points
# ---------------------------------------------------------
def aprilgrid_obj_points_from_dict(grid):
    rows = grid["rows"]
    cols = grid["cols"]
    s = grid["tag_size"] # size of the interior tag
    spacing = grid["tag_spacing"] # ratios of size of tag spacing /  size of interior tag
    id0 = grid.get("marker_id_offset", 0)
    step = s * (1 + spacing) # s*spacing +s distance btw tag starts

    obj = {}
    for r in range(rows):
        for c in range(cols):
            tag_id = id0 + r * cols + c
            x0 = c * step
            y0 = r * step
            # 4 corners in object frame
            # Order of April Grid Points: https://github.com/huangqinjin/apriltag3/blob/f211d9011ee474e5e24bac9c2419176968c93e48/apriltag.c#L920-L924
            # https://github.com/pupil-labs/apriltags/blob/f5334c6e007dc7256386e30e948d63fef5dbc264/src/pupil_apriltags/bindings.py#L208
            obj[tag_id] = np.array([ # Counter Clockwise ordering 
                [x0,   y0,   0],
                [x0+s, y0,   0],
                [x0+s, y0+s, 0],
                [x0,   y0+s, 0]
            ], dtype=float)
    return obj


# ---------------------------------------------------------
# 4. PnP correspondences
# ---------------------------------------------------------
def assemble_pnp_points(detections, obj_points_map):
    obj_pts = []
    img_pts = []
    for det in detections:
        tid = det.tag_id
        if tid in obj_points_map:
            obj_pts.append(obj_points_map[tid])
            img_pts.append(det.corners)
    if len(obj_pts) == 0:
        return None, None
    obj_pts = np.vstack(obj_pts).astype(np.float32)
    img_pts = np.vstack(img_pts).astype(np.float32)
    return obj_pts, img_pts


# ---------------------------------------------------------
# 5. Solve PnP (grid pose in camera frame)
# ---------------------------------------------------------
def solve_grid_pose(obj_pts, img_pts, K, dist):
    #https://docs.opencv.org/4.13.0/d9/d0c/group__calib3d.html#ga549c2075fac14829ff4a58bc931c033d
    ok, rvec, tvec = cv2.solvePnP(
        obj_pts, img_pts, K, dist,
        flags=cv2.SOLVEPNP_ITERATIVE
    )
    if not ok:
        return None, None
    R, _ = cv2.Rodrigues(rvec)
    return R, tvec.reshape(3)


# ---------------------------------------------------------
# 6. AprilTag detector (pupil_apriltags)
# ---------------------------------------------------------
detector = Detector(
    families="tag36h11",
    nthreads=4,
    quad_decimate=1.0,
    refine_edges=True
)


# ---------------------------------------------------------
# 7. MAIN FUNCTION: compute circle center in camera frame
# ---------------------------------------------------------
def detect_circle_center_camera_from_dict(
    img,
    K, dist,
    target_dict, visualize=False
):
    # Extract grid + circle info from the dict
    grid_config = target_dict["grid"]
    circle = target_dict["circle"]

    # Circle center in object frame
    circle_center_obj = np.array([
        circle["x_offset"],
        circle["y_offset"],
        circle.get("z_offset", 0.0)
    ], dtype=float)

    # Undistort
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Detect AprilTags
    detections = detector.detect(gray)


    if len(detections) < 4:
        return None, None, None, None

    # Build AprilGrid object points
    obj_map = aprilgrid_obj_points_from_dict(grid_config)


    obj_pts, img_pts = assemble_pnp_points(detections, obj_map)
    if obj_pts is None:
        return None, None, None, None
    


    cam_R_board, cam_T_board = solve_grid_pose(obj_pts, img_pts, K, dist)
    if cam_R_board is None:
        return None, None, None, None

    # Transform circle center into camera frame
    circle_center_cam = cam_R_board @ circle_center_obj + cam_T_board

    vis = None
    if visualize:
        vis = visualize_aprilgrid_and_circle(
            img,
            detections,
            K,
            dist,
            circle_center_cam,
            R=cam_R_board,
            t=cam_T_board,
            circle_radius_m=circle["radius"]
        )

    return circle_center_cam, cam_R_board, cam_T_board, vis


# ---------------------------------------------------------
# 8. Helper function for your workflow
# ---------------------------------------------------------
def compute_circle_center_for_frame_dict(
    kalibr_file: str,
    target_dict: dict,
    img: np.ndarray,
    cam_name: str = "cam1",
    visualize: bool = False,
):
    K, dist = load_kalibr_camera(kalibr_file, cam_name)
    return detect_circle_center_camera_from_dict(
        img,
        K, dist,
        target_dict, visualize=visualize
    )


def visualize_aprilgrid_and_circle(
    img,
    detections,
    K,
    dist,
    circle_center_cam,
    R=None,
    t=None,
    thickness=2,
    circle_radius_m=None,
    num_circle_samples: int = 180
):
    """Draw AprilTag detections + projected circle center."""
    vis = img.copy()

    # Draw AprilTag corners & IDs
    for det in detections:
        pts = det.corners.astype(int)
        cv2.polylines(
            vis,
            [pts],
            isClosed=True,
            color=(0, 255, 0),
            thickness=thickness
        )
        cX, cY = det.center.astype(int)
        cv2.putText(
            vis,
            f"{det.tag_id}",
            (cX - 10, cY + 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2,
            cv2.LINE_AA
        )
        cv2.circle(vis, (cX, cY), 5, (0, 255, 0), -1)

    # Project the 3D circle center into the image
    if circle_center_cam is not None and circle_center_cam.shape[0] == 3:
        x, y, z = circle_center_cam
        if z != 0:
            pt, _ = cv2.projectPoints(
                circle_center_cam.reshape(1, 1, 3),
                np.zeros(3),
                np.zeros(3),
                K,
                dist,
            )
            u, v = pt.reshape(2)
            u_i, v_i = int(round(float(u))), int(round(float(v)))
            cv2.circle(vis, (u_i, v_i), 5, (0, 0, 255), -1)
            cv2.putText(
                vis,
                "circle_center",
                (u_i + 8, v_i - 8),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 0, 255),
                2,
                cv2.LINE_AA
            )

    # Draw the estimated circle (projected as an ellipse) if radius is provided
    if (
        circle_center_cam is not None
        and R is not None and t is not None
        and circle_radius_m is not None
        and num_circle_samples > 3
    ):
        # Plane basis in camera frame from grid x/y axes
        u_cam = R @ np.array([1.0, 0.0, 0.0])
        v_cam = R @ np.array([0.0, 1.0, 0.0])
        thetas = np.linspace(0, 2 * np.pi, num_circle_samples, endpoint=True)
        pts_cam = (
            circle_center_cam.reshape(3, 1)
            + circle_radius_m * (u_cam.reshape(3, 1) * np.cos(thetas)
                                 + v_cam.reshape(3, 1) * np.sin(thetas))
        ).T.astype(np.float32)
        pts_cv = pts_cam.reshape(-1, 1, 3)
        pts_2d, _ = cv2.projectPoints(
            pts_cv,
            np.zeros(3),
            np.zeros(3),
            K,
            dist
        )
        poly = pts_2d.reshape(-1, 2).astype(int)
        cv2.polylines(vis, [poly], isClosed=True, color=(0, 0, 255), thickness=thickness)

    # Project grid axes (optional)
    if R is not None and t is not None:
        axis = np.float32([
            [0, 0, 0],
            [0.2, 0, 0],
            [0, 0.2, 0],
            [0, 0, 0.2]
        ])
        rvec, _ = cv2.Rodrigues(R)
        pts, _ = cv2.projectPoints(
            axis,
            rvec,
            t.reshape(3,1),
            K,
            dist
        )
        pts = pts.reshape(-1,2).astype(int)
        origin = tuple(pts[0])
        cv2.line(vis, origin, tuple(pts[1]), (255,0,0), thickness)   # X
        cv2.line(vis, origin, tuple(pts[2]), (0,255,0), thickness)   # Y
        cv2.line(vis, origin, tuple(pts[3]), (0,0,255), thickness)   # Z

    return vis



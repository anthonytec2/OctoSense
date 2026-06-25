"""
Ego-motion / time-of-day enrichment for captioning and hybrid search.

We HAVE the fused trajectory (RKO-LIO + GPS), we summarize the ego trajectory OVER THE WINDOW INTERVAL plus
the time-of-day and hand the VLM that text (build_ego_motion_line -> the prompt's
{ego_motion_line} slot). The same per-window values are persisted as structured
TELEM_FIELDS alongside captions so search can filter on them (night + turning + speed).
"""
import math
import re

import numpy as np

# Structured per-window telemetry persisted alongside captions so search can do
# HYBRID (structured + semantic) filtering — e.g. night + turning + speed range.
TELEM_FIELDS = [("speed_mps", "f4"), ("turn_deg", "f4"), ("dist_m", "f4"), ("is_night", "i1")]


def _load_ego_traj(hf):
    """(poses (N,4,4), t (N,), source) — prefer the GPS-fused trajectory (/fused_traj),
    else raw ouster/odom (LIO-only, no GPS); None if neither."""
    if "fused_traj/T_world_lidar" in hf and "fused_traj/t" in hf:
        return hf["fused_traj/T_world_lidar"][()], hf["fused_traj/t"][()].astype(float).ravel(), "fused_traj"
    if "ouster/odom/map_T_lidart" in hf and "ouster/odom/t" in hf:
        return hf["ouster/odom/map_T_lidart"][()], hf["ouster/odom/t"][()].astype(float).ravel(), "odom"
    return None, None, None


def _traj_summary(poses, t, ts, half):
    """Summarize ego motion over [ts-half, ts+half]. Returns dict or None (too few samples)."""
    lo = max(0, int(np.searchsorted(t, ts - half)) - 1)
    hi = min(len(t), int(np.searchsorted(t, ts + half)) + 1)
    if hi - lo < 2:
        return None
    P = poses[lo:hi]; tt = t[lo:hi]
    pos = P[:, :3, 3].astype(float)
    dist = float(np.linalg.norm(np.diff(pos, axis=0), axis=1).sum())
    dur = float(tt[-1] - tt[0]) or 1e-3
    n = len(tt)
    def _spd(i0, i1):
        dt = float(tt[i1] - tt[i0])
        return float(np.linalg.norm(pos[i1] - pos[i0]) / dt) if dt > 0 else 0.0
    v0 = _spd(0, min(2, n - 1)); v1 = _spd(max(0, n - 3), n - 1)
    yaw0 = math.atan2(P[0, 1, 0], P[0, 0, 0]); yaw1 = math.atan2(P[-1, 1, 0], P[-1, 0, 0])
    dyaw = (yaw1 - yaw0 + math.pi) % (2 * math.pi) - math.pi
    return dict(dist_m=dist, mean_mps=dist / dur, v0=v0, v1=v1, turn_deg=math.degrees(dyaw))


def build_ego_motion_line(summary=None, time_ctx="") -> str:
    """One telemetry sentence from the interval summary + local time context, e.g.
    'Ego vehicle telemetry: decelerating from 24 to 12 m/s while turning left ~25°,
     covering ~90 m; local time 2026-01-21 22:35 (nighttime).' '' if nothing is known."""
    motion = None
    if summary is not None:
        mean, v0, v1, turn = summary["mean_mps"], summary["v0"], summary["v1"], summary["turn_deg"]
        if mean < 0.5:
            spd = "stationary"
        elif v1 - v0 > 1.5:
            spd = f"accelerating from {v0:.0f} to {v1:.0f} m/s"
        elif v0 - v1 > 1.5:
            spd = f"decelerating from {v0:.0f} to {v1:.0f} m/s"
        else:
            spd = f"~{mean:.0f} m/s forward"
        clause = [spd]
        if abs(turn) >= 5 and mean >= 0.5:
            side = "left" if turn > 0 else "right"          # +yaw (z-up) = left
            qual = "slightly " if abs(turn) < 20 else ""
            clause.append(f"turning {qual}{side} ~{abs(turn):.0f}°")
        if mean >= 0.5 and summary["dist_m"] >= 1:
            clause.append(f"covering ~{summary['dist_m']:.0f} m")
        motion = ", ".join(clause).replace(", turning", " while turning")
    seg = []
    if motion:
        seg.append(f"Ego vehicle telemetry: {motion}")
    if time_ctx:
        seg.append(f"local time {time_ctx}")
    if not seg:
        return ""
    return "; ".join(seg).rstrip(".") + ". "


def _tod_from_bag_id(video_id: str):
    """Local time-of-day from the recording datetime in the bag id
    (rosbag2_YYYY_MM_DD-HH_MM_SS). Coarse buckets; the bag clock is local."""
    m = re.search(r"(\d{4})_(\d{2})_(\d{2})-(\d{2})_(\d{2})_(\d{2})", video_id or "")
    if not m:
        return None
    h = int(m.group(4))
    if h < 6 or h >= 20:
        return "nighttime"
    if h < 8:
        return "dawn"
    if h >= 18:
        return "dusk"
    return "daytime"


def _local_datetime_from_bag_id(video_id: str):
    """(human local datetime, tod) from the recording datetime in the bag id, e.g.
    ('2026-01-21 22:35 (nighttime)', 'nighttime'). Bag clock is local. (None, None) if unparseable."""
    m = re.search(r"(\d{4})_(\d{2})_(\d{2})-(\d{2})_(\d{2})_(\d{2})", video_id or "")
    if not m:
        return None, None
    y, mo, d, h, mi, _s = m.groups()
    tod = _tod_from_bag_id(video_id)
    return f"{y}-{mo}-{d} {h}:{mi} ({tod})", tod


def _telem_tuple(w):
    """Telemetry values for a window as a tuple in TELEM_FIELDS order (NaN/0 if absent)."""
    t = getattr(w, "telem", None) or {}
    return (float(t.get("speed_mps", math.nan)), float(t.get("turn_deg", math.nan)),
            float(t.get("dist_m", math.nan)), int(t.get("is_night", 0)))


def _telem_from_row(meta_row):
    """Recover the telemetry dict from a structured-array metadata row (only fields present)."""
    names = meta_row.dtype.names or ()
    out = {}
    for name, _ in TELEM_FIELDS:
        if name in names:
            out[name] = int(meta_row[name]) if name == "is_night" else float(meta_row[name])
    return out or None

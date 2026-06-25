"""
Visualize a processed OctoSense sequence in Rerun.

Reads the final processed layout for one bag directory::

    <dir>/data.h5      # cameras (calib + timestamps), ouster, gps, imu, car
    <dir>/events.h5    # ev/{left,right} event streams
    <dir>/captions.h5  # per-window scene captions
    <dir>/img_{left,right}.mp4, img_infrared.mp4   # encoded camera video

and writes ``<dir>/rerun_data.rrd``.

Every stream is on one common clock, logged to a single timeline ("time", in
seconds from sequence start). The full ~10-min sequence logs the entire ~131k-point
LiDAR grid every frame (~12 GB .rrd) and is slow, so by default only a short window
is rendered (``--start``/``--end``, default the first 60 s); pass ``--full`` for the
whole sequence.

Within the window: videos are logged as-is (already HEVC — no re-encode), only their
per-frame references are restricted to the window; the LiDAR / scalar streams are
sliced by timestamp; and the event streams (no pre-encoded video) have just the
windowed frames reconstructed on the fly (rendered in parallel, piped to one ffmpeg).
"""
import argparse
import os
import subprocess
from multiprocessing import cpu_count

import h5py
import hdf5plugin
import numpy as np
import rerun as rr
from joblib import Parallel, delayed

os.environ.setdefault("BLOSC_NTHREADS", "8")

# Event camera native resolution (fixed hardware) — used for the reconstruction canvas.
EVENT_W, EVENT_H = 640, 480

# Default time window (seconds). Visualizing the full ~10-min sequence is slow and makes
# a ~12 GB .rrd, so the default is a short clip from the start; pass --full for everything.
DEFAULT_WINDOW_S = 60.0

# Saved viewer layout bundled into every .rrd. Its application_id is "octosense" (matching
# rr.init below), so the viewer auto-applies it on open. Edit the layout in the viewer and
# re-export over this file to change the default views.
BLUEPRINT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "octosense.rbl")


# ---------------------------------------------------------------------------
# small helpers
# ---------------------------------------------------------------------------
def _colormap_lut(name, n=256):
    """256x3 uint8 RGB lookup table for a matplotlib colormap (falls back to turbo)."""
    import matplotlib

    try:
        cmap = matplotlib.colormaps[name]
    except KeyError:
        cmap = matplotlib.colormaps["turbo"]
    return (cmap(np.linspace(0, 1, n))[:, :3] * 255).astype(np.uint8)


def _index_range(ts, t_start, t_end):
    """[start, end) index window into a sorted timestamp array (seconds)."""
    ts = np.asarray(ts)
    n = ts.shape[0]
    start = int(np.searchsorted(ts, t_start, "left")) if t_start is not None else 0
    end = int(np.searchsorted(ts, t_end, "right")) if t_end is not None else n
    start = max(0, min(start, n))
    end = max(start, min(end, n))
    return start, end


def _time_mask(ts, t_start, t_end):
    """Boolean mask selecting samples inside [t_start, t_end] (either bound optional)."""
    m = np.ones(len(ts), dtype=bool)
    if t_start is not None:
        m &= ts >= t_start
    if t_end is not None:
        m &= ts <= t_end
    return m


# ---------------------------------------------------------------------------
# Ouster point clouds (+ top-down trajectory overlay)
# ---------------------------------------------------------------------------
def log_ouster_pointclouds(f, t_start=None, t_end=None, radius=0.03,
                           colormap="turbo", range_clip=(1.0, 80.0)):
    """Log deskewed Ouster clouds from ``ouster/range_pcl`` (N, P, 3) int32 mm,
    colored by per-point range. If ``ouster/odom/map_T_lidart`` is present, also
    draw the accumulating top-down trajectory in the ``ouster/trajectory_points``
    2D view."""
    if "ouster/range_pcl" not in f:
        print("[viz] no ouster/range_pcl, skipping point clouds")
        return
    ts = f["ouster/t"][:]
    range_pcl = f["ouster/range_pcl"]
    idxs = np.where(_time_mask(ts, t_start, t_end))[0]
    if idxs.size == 0:
        print("[viz] no Ouster frames inside time window")
        return

    positions = colors_ts = None
    if "ouster/odom/map_T_lidart" in f:
        odom = f["ouster/odom/map_T_lidart"]
        if odom.shape[0] > 1:
            positions = np.column_stack([odom[:, 0, -1], -odom[:, 1, -1]])
            tn = (ts - ts.min()) / (ts.max() - ts.min() + 1e-9)
            colors_ts = _colormap_lut("jet")[(tn * 255).astype(np.uint8)]

    lut = _colormap_lut(colormap)
    rmin, rmax = range_clip
    for k, i in enumerate(idxs):
        xyz = range_pcl[i].astype(np.float32) / 1000.0
        if xyz.size == 0:
            continue
        r = np.linalg.norm(xyz, axis=1)
        r_norm = np.clip((r - rmin) / (rmax - rmin + 1e-9), 0.0, 1.0)
        colors = lut[(r_norm * 255).astype(np.uint8)]

        rr.set_time("time", timestamp=ts[i])
        rr.log("ouster/points", rr.Points3D(xyz, colors=colors, radii=radius))
        if positions is not None:
            sel = idxs[: k + 1]
            rr.log("ouster/trajectory_points",
                   rr.Points2D(positions[sel], colors=colors_ts[sel], radii=2.0))


# ---------------------------------------------------------------------------
# Video logging (cameras + reconstructed event video)
# ---------------------------------------------------------------------------
def log_video(stream, mp4_path, timestamps, t_start, t_end):
    """Log an mp4 as a Rerun video, referencing only frames inside the time window.
    """
    if not os.path.exists(mp4_path):
        print(f"[viz] {mp4_path} not found, skipping {stream}")
        return
    timestamps = np.asarray(timestamps)

    asset = rr.AssetVideo(path=mp4_path)
    rr.log(stream, asset, static=True)
    frame_ts = asset.read_frame_timestamps_nanos()
    n = min(len(timestamps), len(frame_ts))
    if len(timestamps) != len(frame_ts):
        print(f"[viz] {stream}: {len(timestamps)} timestamps vs {len(frame_ts)} frames, using {n}")

    s, e = _index_range(timestamps[:n], t_start, t_end)
    if e <= s:
        print(f"[viz] no {stream} frames inside time window")
        return
    rr.send_columns(
        stream,
        indexes=[rr.TimeColumn("time", timestamp=timestamps[s:e])],
        columns=rr.VideoFrameReference.columns_nanos(frame_ts[s:e]),
    )


# ---------------------------------------------------------------------------
# Event reconstruction (parallel render -> raw pipe -> single ffmpeg encode)
# ---------------------------------------------------------------------------
EVENT_CHUNK = 32  # frames per worker task (bounds per-worker memory and IPC payload)


def _render_event_chunk(events_h5, side, i0, i1):
    """Worker: render event frames [i0, i1) and return their raw BGR bytes, in order.

    Frame ``k`` spans [10k, 10k+10) ms; ``ms_to_idx`` maps a millisecond to its first
    event index. The whole chunk's events are read in one slice (one open + one
    decompress pass per chunk, not per frame), then split back into frames."""
    import hdf5plugin  # noqa: F401  (each worker process re-registers the filter)
    with h5py.File(events_h5, "r") as f:
        base = f"ev/{side}"
        bounds = f[f"{base}/ms_to_idx"][i0 * 10:i1 * 10 + 1:10].astype(np.int64)
        lo, hi = int(bounds[0]), int(bounds[-1])
        x = f[f"{base}/x"][lo:hi].astype(np.uint32)
        y = f[f"{base}/y"][lo:hi].astype(np.uint32)
        p = f[f"{base}/p"][lo:hi]
    bounds -= lo
    buf = bytearray()
    for k in range(i1 - i0):
        a, b = bounds[k], bounds[k + 1]
        img = np.zeros((EVENT_H, EVENT_W, 3), dtype=np.uint8)  # BGR (rawvideo bgr24)
        on = p[a:b] == 1
        xa, ya = x[a:b], y[a:b]
        img[ya[on], xa[on]] = (255, 191, 0)     # deepskyblue (ON)  RGB(0,191,255)
        img[ya[~on], xa[~on]] = (71, 99, 255)   # tomato      (OFF) RGB(255,99,71)
        buf += img.tobytes()
    return bytes(buf)


def _render_event_video(events_h5, side, mp4_path, i0, i1, fps=100):
    """Reconstruct event frames [i0, i1) to an HEVC mp4.

    Workers render chunks of frames in parallel; ``return_as="generator"`` yields the
    chunks back *in order* as they finish, and we stream the raw bytes straight into one
    ffmpeg over a pipe. Render is parallel across workers,
    encode is parallel inside ffmpeg's libx265 thread pool
    """
    proc = subprocess.Popen(
        ["ffmpeg", "-y", "-loglevel", "error",
         "-f", "rawvideo", "-pixel_format", "bgr24",
         "-video_size", f"{EVENT_W}x{EVENT_H}", "-framerate", str(fps), "-i", "pipe:0",
         "-c:v", "libx265", "-preset", "medium", "-crf", "25", "-pix_fmt", "yuv420p",
         mp4_path],
        stdin=subprocess.PIPE,
    )
    chunks = [(i, min(i + EVENT_CHUNK, i1)) for i in range(i0, i1, EVENT_CHUNK)]
    results = Parallel(n_jobs=max(1, min(cpu_count(), 16)), prefer="processes",
                       return_as="generator")(
        delayed(_render_event_chunk)(events_h5, side, a, b) for a, b in chunks)
    try:
        for data in results:
            proc.stdin.write(data)
    finally:
        proc.stdin.close()
    if proc.wait() != 0:
        print(f"[viz] ffmpeg failed encoding {side} events")


# ---------------------------------------------------------------------------
# Scalar / geo streams
# ---------------------------------------------------------------------------
def log_gps(f, t_start, t_end):
    """Raw GPS fixes as Rerun GeoPoints (lat/lon), colored red."""
    if "gps/data" not in f or "gps/t" not in f:
        print("[viz] no GPS data, skipping")
        return
    lat = f["gps/data"][:, 0].astype(np.float64)
    lon = f["gps/data"][:, 1].astype(np.float64)
    t = f["gps/t"][:]
    m = (t > 0) & _time_mask(t, t_start, t_end)
    n = int(m.sum())
    if n == 0:
        print("[viz] no GPS points inside time window")
        return
    colors = np.full(n, (255 << 24) | 255, dtype=np.uint32)  # opaque red (RGBA)
    rr.send_columns(
        "gps",
        indexes=[rr.TimeColumn("time", timestamp=t[m])],
        columns=rr.GeoPoints.columns(
            positions=np.stack([lat[m], lon[m]], axis=1),
            radii=rr.Radius.ui_points(np.full(n, 10.0)),
            colors=colors,
        ),
    )


def log_imu(f, t_start, t_end):
    """VectorNav angular velocity, acceleration, magnetometer as per-axis scalars."""
    keys = ("vectornav/ang_vel", "vectornav/accel", "vectornav/magnetic", "vectornav/t")
    if not all(k in f for k in keys):
        print("[viz] no IMU data, skipping")
        return
    t = f["vectornav/t"][:]
    m = _time_mask(t, t_start, t_end)
    if not m.any():
        print("[viz] no IMU samples inside time window")
        return
    ts = t[m]
    streams = {"omega": f["vectornav/ang_vel"][:][m],
               "lin_vel": f["vectornav/accel"][:][m],
               "mag": f["vectornav/magnetic"][:][m]}
    for j, axis in enumerate("XYZ"):
        for name, arr in streams.items():
            rr.send_columns(
                f"{name}_{axis}",
                indexes=[rr.TimeColumn("time", timestamp=ts)],
                columns=rr.Scalars.columns(scalars=arr[:, j]),
            )


def log_car(f, t_start, t_end):
    """Car OBD/CAN signals (time in column 0, value(s) after) as scalars."""
    if "car" not in f:
        print("[viz] no car data, skipping")
        return
    car = f["car"]
    multi_labels = {"wheels": ["FL", "FR", "RL", "RR"], "vcc": ["acc_x", "acc_y"]}

    def log_signal(name, n_cols=1):
        if name not in car:
            return
        data = car[name][:]
        if data.size == 0:
            return
        m = _time_mask(data[:, 0], t_start, t_end)
        if not m.any():
            return
        ts = data[m, 0]
        for c in range(n_cols):
            label = name if n_cols == 1 else f"{name}_{multi_labels[name][c]}"
            rr.send_columns(
                f"car/{label}",
                indexes=[rr.TimeColumn("time", timestamp=ts)],
                columns=rr.Scalars.columns(scalars=data[m, 1 + c]),
            )

    for sig in ("steer", "brake_on", "pedal", "speed", "steer_rate", "brake_press"):
        log_signal(sig)
    log_signal("wheels", 4)
    log_signal("vcc", 2)


def log_captions(dir_read, t_start, t_end):
    """Per-window scene captions (captions.h5) shown as a text document on the timeline.

    Each window's caption is logged at its ``metadata['timestamp']`` so the text view
    tracks the current scene as the time cursor moves."""
    cap_path = os.path.join(dir_read, "captions.h5")
    if not os.path.exists(cap_path):
        print("[viz] no captions.h5, skipping captions")
        return
    with h5py.File(cap_path, "r") as f:
        captions = f["captions"][:]
        ts = f["metadata"]["timestamp"].astype(np.float64)
    m = _time_mask(ts, t_start, t_end)
    # Captions are sparse (~one per 5 s window), so a short window can contain none.
    # Also include the caption active when the window opens (last one at/before t_start).
    if t_start is not None:
        before = np.where(ts <= t_start)[0]
        if before.size:
            m[before[-1]] = True
    if not m.any():
        print("[viz] no captions inside time window")
        return
    for cap, t in zip(captions[m], ts[m]):
        text = cap.decode("utf-8") if isinstance(cap, (bytes, bytearray)) else str(cap)
        rr.set_time("time", timestamp=float(t))
        rr.log("caption", rr.TextDocument(text))


# ---------------------------------------------------------------------------
# entry point
# ---------------------------------------------------------------------------
def visualize(dir_read, t_start=None, t_end=None):
    dir_read = os.path.normpath(dir_read)
    data_path = os.path.join(dir_read, "data.h5")
    events_path = os.path.join(dir_read, "events.h5")

    rrd_path = os.path.join(dir_read, "rerun_data.rrd")
    if os.path.exists(rrd_path):
        os.remove(rrd_path)
    rr.init("octosense", spawn=False)
    rr.save(rrd_path)

    # Bundle the saved viewer layout into the .rrd (same application_id, so it auto-applies).
    if os.path.exists(BLUEPRINT_PATH):
        rr.log_file_from_path(BLUEPRINT_PATH)

    f = h5py.File(data_path, "r")
    f_ev = h5py.File(events_path, "r") if os.path.exists(events_path) else None

    log_ouster_pointclouds(f, t_start, t_end)

    log_video("img_left", os.path.join(dir_read, "img_left.mp4"),
              f["img/left/t"][:], t_start, t_end)
    log_video("img_right", os.path.join(dir_read, "img_right.mp4"),
              f["img/right/t"][:], t_start, t_end)
    if "infrared/t" in f:
        log_video("img_infrared", os.path.join(dir_read, "img_infrared.mp4"),
                  f["infrared/t"][:], t_start, t_end)

    log_gps(f, t_start, t_end)
    log_imu(f, t_start, t_end)
    log_car(f, t_start, t_end)
    log_captions(dir_read, t_start, t_end)

    # Event cameras: reconstruct only the windowed frame range to a temp mp4, log it,
    # then delete it. Frame i is at i * 0.01 s on the same common clock as everything
    # else; rendering only the window (not the full 10-min stream) keeps spot-checks fast.
    if f_ev is not None:
        for side in ("left", "right"):
            if f"ev/{side}/ms_to_idx" not in f_ev:
                continue
            n_total = (len(f_ev[f"ev/{side}/ms_to_idx"]) - 1) // 10
            if n_total <= 0:
                continue
            ts_full = np.arange(n_total) * 0.01
            f0, f1 = _index_range(ts_full, t_start, t_end)
            if f1 <= f0:
                print(f"[viz] no events_{side} frames inside time window")
                continue
            tmp = os.path.join(dir_read, f"_events_{side}.mp4")
            _render_event_video(events_path, side, tmp, f0, f1)
            if os.path.exists(tmp):
                log_video(f"events_{side}", tmp, ts_full[f0:f1], None, None)
                os.remove(tmp)

    print(f"[viz] wrote {rrd_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize a processed OctoSense bag in Rerun.")
    parser.add_argument("--dir", type=str, required=True,
                        help="Bag directory containing data.h5 / events.h5 / *.mp4")
    parser.add_argument("--start", type=float, default=0.0,
                        help="Window start time in seconds (default: 0).")
    parser.add_argument("--end", type=float, default=None,
                        help=f"Window end time in seconds (default: start + {DEFAULT_WINDOW_S:.0f}s).")
    parser.add_argument("--full", action="store_true",
                        help="Visualize the entire sequence (slow; ~12 GB .rrd).")
    args = parser.parse_args()
    if args.full:
        t_start, t_end = None, None
    else:
        t_start = args.start
        t_end = args.end if args.end is not None else args.start + DEFAULT_WINDOW_S
    visualize(args.dir, t_start=t_start, t_end=t_end)

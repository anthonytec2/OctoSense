import numpy as np
from numba import njit
from pykalman.standard import _filter_correct, _smooth
import os
import logging

import rosbag2_py
from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message

from timeSync_constants import PROCESSED_ROOT


def get_processed_path(bag_path):
    """
    Convert raw bag path to processed output path, mirroring directory structure.
    
    Example:
        <raw_root>/session1/rosbag2_2025_11_20-07_37_44
        -> <processed_root>/session1/rosbag2_2025_11_20-07_37_44/
    """
    # Normalize the path
    bag_path = os.path.normpath(bag_path)

    proc_root = PROCESSED_ROOT

    # Find the position of 'raw' in the path
    if '/raw/' in bag_path:
        # Split at '/raw/' and get everything after it
        relative_path = bag_path.split('/raw/', 1)[1]
        # Get the bag name (last component)
        bag_name = os.path.basename(relative_path)
        # Construct processed path
        output_path = f'{proc_root}/{relative_path}/'
    else:
        # Fallback: just use the bag name
        bag_name = os.path.basename(bag_path)
        output_path = f'{proc_root}/{bag_name}/'

    return output_path, bag_name


def configure_file_logging(output_dir: str, filename: str = "timeSync.log") -> None:
    """
    Attach a file handler to the root logger that writes to output_dir/filename.
    Safe to call multiple times; will not duplicate handlers for the same file.
    """
    try:
        os.makedirs(output_dir, exist_ok=True)
    except Exception:
        # If directory cannot be created, skip file logging
        return
    log_file_path = os.path.join(output_dir, filename)
    root_logger = logging.getLogger()
    for handler in root_logger.handlers:
        if isinstance(handler, logging.FileHandler):
            try:
                if os.path.abspath(handler.baseFilename) == os.path.abspath(log_file_path):
                    return
            except Exception:
                # Some handlers may not have baseFilename (e.g., stream handlers)
                pass
    file_handler = logging.FileHandler(log_file_path)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(logging.Formatter('%(asctime)s %(levelname)s:%(name)s:%(message)s'))
    root_logger.addHandler(file_handler)


class BagTopicReader:
    def __init__(self, bag_uri: str):
        self.bag_uri = bag_uri
        self._reader = self._open_reader()

        # Get available topics and types
        topics = self._reader.get_all_topics_and_types()
        self.topics_map = {t.name: t.type for t in topics}

        # Get message counts from bag metadata
        metadata = self._reader.get_metadata()
        self.message_counts = {}
        for topic_metadata in metadata.topics_with_message_count:
            self.message_counts[topic_metadata.topic_metadata.name] = topic_metadata.message_count

    def _open_reader(self) -> rosbag2_py.SequentialReader:
        """Open a fresh SequentialReader on this bag (mcap storage, cdr)."""
        reader = rosbag2_py.SequentialReader()
        reader.open(
            rosbag2_py.StorageOptions(uri=self.bag_uri, storage_id="mcap"),
            rosbag2_py.ConverterOptions(
                input_serialization_format="cdr",
                output_serialization_format="cdr",
            ),
        )
        return reader

    def print_topics(self):
        print("Available topics:")
        for topic, typ in self.topics_map.items():
            print(f"- {topic} [{typ}]")
    
    def get_message_count(self, topic_name: str) -> int:
        """Get the number of messages for a given topic."""
        return self.message_counts.get(topic_name, 0)

    def iter_topic(self, topic_name: str):
        """Yield (msg, timestamp) for a single topic (one pass over the bag)."""
        for _topic, msg, timestamp in self.iter_topics([topic_name]):
            yield msg, timestamp

    def iter_topics(self, topic_names):
        """Iterate the bag once, yielding (topic, msg, timestamp) for all requested topics.
        """
        # Resolve message types for each requested topic
        msg_types = {}
        for t in topic_names:
            if t not in self.topics_map:
                raise ValueError(
                    f"Topic '{t}' not found in bag {self.bag_uri}.\n"
                    f"Available topics: {list(self.topics_map.keys())}"
                )
            msg_types[t] = get_message(self.topics_map[t])

        topic_set = set(topic_names)

        # Single sequential read
        reader = self._open_reader()
        while reader.has_next():
            topic, data, timestamp = reader.read_next()
            if topic not in topic_set:
                continue
            msg = deserialize_message(data, msg_types[topic])
            yield topic, msg, timestamp


def kalman_filter_and_rts(
    tD, tG,
    Q_diag=None,
    R=None,
    mahal_gate=5.0,
):
    """
    Run Kalman filter + RTS smoother on matched PPS pairs (tD, tG).
    Inputs:
      tD: (N,) device timestamps at PPS (seconds, unwrapped)
      tG: (N,) global PPS counters (1,2,3,...)
      Q_diag: iterable length-2 for diagonal Q per-second variances [var_b_per_s, var_a_per_s].
              Q_k = diag(Q_diag) * Δt between samples. If None, defaults will be used.
      R: measurement noise variance (scalar). If None, default used.
      mahal_gate: Mahalanobis distance threshold for outlier rejection. Measurements
                  whose normalized innovation |y|/sqrt(S) exceeds this are treated as
                  prediction-only steps (no update). Set to None to disable gating.
    Returns dict with:
      xs_filt, Ps_filt, xs_pred, Ps_pred, xs_smooth, Ps_smooth, innov, S, Qk_list, R, dt, rejected
    """
    tD = np.asarray(tD, dtype=float)
    tG = np.asarray(tG, dtype=float)
    N = len(tD)
    assert N == len(tG), "tD and tG must be same length"

    # Center device clock for numerical stability.
    # When tD contains large values (e.g. Unix timestamps ~1.77e9),
    # H = [1, tD] makes S_k ≈ tD² · P[1,1] astronomically large,
    # breaking the Kalman gain and Mahalanobis gate.
    # We store tD_ref so map_events_to_global can apply the same centering.
    tD_ref = tD[0]
    tD = tD - tD_ref

    # Defaults
    if R is None:
        # default measurement variance: assumes PPS jitter ~10 us -> var ~ (10e-6)^2
        R = (10e-6)**2

    if Q_diag is None:
        # default per-second process variances (tunable)
        # var_b_per_s: how much offset b (s) can diffuse per second
        # var_a_per_s: how much skew a (unitless) can diffuse per second
        Q_diag = [1e-10, 1e-13]  # small defaults; tune for your hardware
    Q_base = np.diag(Q_diag).astype(float)

    # Δt between successive PPS pulses (from the main-clock counts)
    dt = np.empty(N)
    dt[0] = 1.0
    dt[1:] = np.diff(tG)

    # Initialize (using centered tD)
    # Initial state [b, a]: slope a=1 (device ≈ main rate), offset b=tG[0]
    # (tD is centered so tD[0]=0).
    x0 = np.array([tG[0], 1.0], dtype=float)
    # Initial covariance: loose on offset b, tight on slope a.
    P0 = np.diag([1.0, 1e-6]).astype(float)

    # Allocate per-step buffers (state = [b, a]: offset b, skew a)
    xs_pred = np.zeros((N, 2))         # predicted state at each step
    Ps_pred = np.zeros((N, 2, 2))      # predicted covariance at each step
    xs_filt = np.zeros((N, 2))         # filtered (post-update) state
    Ps_filt = np.zeros((N, 2, 2))      # filtered (post-update) covariance
    innov = np.zeros(N)                # innovation y = z - H x_pred
    S = np.zeros(N)                    # innovation variance
    Qk_list = np.zeros((N, 2, 2))      # per-step process noise Q_k
    rejected = np.zeros(N, dtype=bool)  # outlier rejection mask (Mahalanobis gate)

    x_prev = x0.copy()
    P_prev = P0.copy()

    # Forward filter pass: predict + (optionally gated) update.
    # Update step delegated to pykalman's _filter_correct, which uses
    # scipy.linalg.pinv natively and so handles near-singular S_k without our
    # legacy try/except dance. The predict step is trivial (F=I) so we keep it
    # inline; outlier gating must happen between predict and update
    
    R_mat = np.array([[float(R)]])
    obs_offset = np.zeros(1)
    n_rejected = 0
    for k in range(N):
        # Predict (identity state-transition / random walk: state carries over,
        # covariance just grows by the process noise)
        x_pred = x_prev.copy()
        Qk = Q_base * dt[k]
        P_pred = P_prev + Qk

        # Time-varying measurement matrix and innovation/Mahalanobis gate
        H = np.array([1.0, tD[k]]).reshape(1, 2)
        z = tG[k]
        y = z - float((H @ x_pred).item())
        # S_k = H P_pred H^T + R is always > 0 (R > 0 floors it), so no guard needed.
        S_k = float((H @ P_pred @ H.T + R_mat).item())
        mahal_dist = abs(y) / np.sqrt(S_k)

        if mahal_gate is not None and k > 30 and mahal_dist > mahal_gate:
            x_upd, P_upd = x_pred.copy(), P_pred.copy()
            rejected[k] = True
            n_rejected += 1
        else:
            _, x_upd, P_upd = _filter_correct(
                H, R_mat, obs_offset, x_pred, P_pred, np.array([z])
            )

        xs_pred[k], Ps_pred[k] = x_pred, P_pred
        xs_filt[k], Ps_filt[k] = x_upd, P_upd
        innov[k], S[k] = y, S_k
        Qk_list[k] = Qk
        x_prev, P_prev = x_upd, P_upd

    if n_rejected > 0:
        logging.getLogger(__name__).warning(
            f"KF outlier rejection: {n_rejected}/{N} measurements rejected "
            f"(Mahalanobis gate={mahal_gate})"
        )

    # RTS smoother (backward) — pykalman runs the full backward recursion.
    xs_smooth, Ps_smooth, _ = _smooth(np.eye(2), xs_filt, Ps_filt, xs_pred, Ps_pred)

    return {
        'xs_pred': xs_pred,
        'Ps_pred': Ps_pred,
        'xs_filt': xs_filt,
        'Ps_filt': Ps_filt,
        'xs_smooth': xs_smooth,
        'Ps_smooth': Ps_smooth,
        'innov': innov,
        'S': S,
        'Qk_list': Qk_list,
        'R': R,
        'dt': dt,
        'rejected': rejected,
        'tD_ref': tD_ref,
    }

# ----------------------------
# Map events to smoothed timeline
# ----------------------------
@njit(fastmath=True, cache=True)
def _map_events_core(events_tD, pps_tD, b_arr, a_arr):
    """
    Linearly interpolates the smoothed [b, a] state between the bracketing PPS
    pulses, then evaluates tG = a * tD + b for each event.
    """
    N = len(pps_tD)
    M = len(events_tD)

    # Find bracketing indices
    idxs = np.searchsorted(pps_tD, events_tD) - 1
    idxs = np.clip(idxs, 0, N-1)

    # Initialize output arrays
    a_evt = np.empty(M, dtype=np.float64)
    b_evt = np.empty(M, dtype=np.float64)

    for i in range(M):
        idx = idxs[i]
        next_idx = min(idx + 1, N - 1)

        t0 = pps_tD[idx]
        t1 = pps_tD[next_idx]

        if idx < N - 1 and t1 > t0:
            # Linear interpolation between bracketing PPS states
            tau = (events_tD[i] - t0) / (t1 - t0)
            a_evt[i] = a_arr[idx] + tau * (a_arr[next_idx] - a_arr[idx])
            b_evt[i] = b_arr[idx] + tau * (b_arr[next_idx] - b_arr[idx])
        else:
            # Edge case (last interval / degenerate gap): use bracketing state as-is
            a_evt[i] = a_arr[idx]
            b_evt[i] = b_arr[idx]

    # Compute mapped times
    mapped_tG = a_evt * events_tD + b_evt

    return mapped_tG

def map_events_to_global(events_tD, pps_tD, xs_smooth):
    """
    Map arbitrary device events (events_tD) to global time using smoothed states.
    Inputs:
      events_tD: (M,) device timestamps for events (seconds, unwrapped)
      pps_tD: (N,) device timestamps at PPS (seconds, unwrapped)
      xs_smooth: (N,2) smoothed states [b,a] from kalman/RTS
    Returns:
      mapped_tG: (M,) mapped global times.
    """
    events_tD = np.asarray(events_tD, dtype=np.float64)
    pps_tD = np.asarray(pps_tD, dtype=np.float64)

    # Center device clocks using the same reference as the KF.
    # The KF states [b, a] are in centered coordinates: tG = b + a * (tD - tD_ref)
    # We must apply the same centering here so the mapping is consistent.
    tD_ref = pps_tD[0]
    events_tD_c = events_tD - tD_ref
    pps_tD_c = pps_tD - tD_ref

    # Extract b and a arrays
    b_arr = xs_smooth[:, 0].astype(np.float64)
    a_arr = xs_smooth[:, 1].astype(np.float64)

    # Use JIT-compiled core function (with centered device clocks)
    mapped_tG = _map_events_core(events_tD_c, pps_tD_c, b_arr, a_arr)

    return mapped_tG

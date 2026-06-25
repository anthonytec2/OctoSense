"""
Time synchronization calibration.

Recovers, for each sensor, the affine map between its own device clock and a
shared absolute "main" clock, so that all sensors can later be placed on one
timeline (see ``map_events_to_global`` in timeSyncUtil).

THE TRIGGER
-----------
A hardware trigger (Teensy `pps_clock`, disciplined by a DS3231 1 Hz RTC) emits
one pulse per second to every sensor, phase-aligned to the RTC edge. Each
sensor reports those pulses in its own data stream; we detect them, decode an
absolute second for each, and fit (device_time -> main_second) with a
Kalman filter + RTS smoother (``kalman_filter_and_rts``).

UID CODED PULSES (how absolute time is encoded)
-----------------------------------------------
A plain 1 Hz tick only says "another second passed" — not *which* second. So at
the start of every 256 s window the trigger emits a short run of *coded* pulses
that carry the second index ``i``. 256 s is the period because the trigger's
pulse counter is 8-bit; the decoders therefore work in 256 s "blocks":

    main_second = block_idx * 256 + offset

``block_idx`` is dead-reckoned from the running time estimate; ``offset`` is decoded
from the coded pulse. Index ``i=0`` is the block-wrap marker (offset -1); the
first regular coded seconds are ``i=1..6`` (offsets 0..5). Between coded runs
the trigger sends plain pulses and the decoder just advances +1 s each pulse
(or +N for a missed pulse). There are two physical encodings:

  * SPLIT-PULSE (spacing-coded) — IMU, system, FLIR:
        Two edges per coded second; the gap between them encodes ``i``.
  * WIDE-PULSE (width-coded) — event cameras:
        One pulse per coded second; its ON-duration encodes ``i``.

PER-SENSOR DECODING
-------------------
* system / IMU (`calibration_system_time`, `imu_time_offset` via
  `_decode_imu_pulses`): VectorNav `/vectornav/raw/time`. Pulses are found as
  `timesyncin` resets; the coded first-interval is measured in 400 Hz IMU
  samples and decoded as ``offset = (interval - 16) / 4`` (i=0 -> interval 12 ->
  offset -1 at the wrap). 

* FLIR cameras (`flir_time_offset`): edges are rising transitions of GPIO
  line-status bit 2; intervals are counted in 100 Hz frames; the coded
  first-interval (< 20 frames) decodes as ``offset = interval - 4`` (1 frame/s).

* event cameras (`event_time_offset`): uses the ext-trigger ON/OFF events.
  Here the second index is in the pulse WIDTH (ON->OFF): width > 15 ms is a
  coded pulse, ``index = round((width_ms - 20) / 10)`` and
  ``main = block*256 + index - 1``; ~10 ms is a plain pulse.

"""
import numpy as np
import logging
from dataclasses import dataclass
from event_camera_py import Decoder as ECDecoder
from timeSyncUtil import BagTopicReader, kalman_filter_and_rts
from typing import Any, Tuple, Optional
from timeSync_constants import (
    NS_TO_S, US_TO_NS, US_TO_S, BLOCK_DURATION,
    IMU_TRIGGER_TOPIC,
    IMU_HZ, IMU_TOLERANCE,
    FLIR_HZ, FLIR_TOLERANCE,
)

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(levelname)s:%(name)s:%(message)s')



@dataclass(eq=False)
class CalibrationData:
    """Per-sensor clock calibration: each sensor has a device clock, main clock,
    and Kalman filter dict. GPS uses the system-time calibration, so its fields
    stay None. Values are numpy arrays / filter dicts / None."""
    device_clock_gps: Any = None
    master_clock_gps: Any = None
    gps_filter: Any = None
    device_clock_flir_0: Any = None
    master_clock_flir_0: Any = None
    flir_filter_0: Any = None
    device_clock_flir_1: Any = None
    master_clock_flir_1: Any = None
    flir_filter_1: Any = None
    device_clock_ev_0: Any = None
    master_clock_ev_0: Any = None
    event_filter_0: Any = None
    device_clock_ev_1: Any = None
    master_clock_ev_1: Any = None
    event_filter_1: Any = None
    device_clock_imu: Any = None
    master_clock_imu: Any = None
    imu_filter: Any = None
    device_clock_system: Any = None
    master_clock_system: Any = None
    system_filter: Any = None


def _decode_imu_pulses(bag, dev_clock_of, R, Q, label):
    """Recover absolute time from the VN SyncIn coded-pulse stream.

    The trigger drives SyncIn with a coded split-pulse sequence; edges are the
    ``timesyncin`` resets. Each pulse is decoded to an absolute main-clock
    second and paired with a device-clock timestamp from ``dev_clock_of(msg,
    timestamp)`` (seconds). Shared by the system (bag OS clock) and IMU (VN
    ``timestartup`` crystal) calibrations
    """
    msg_count = bag.get_message_count(IMU_TRIGGER_TOPIC)
    sin_time = np.empty(msg_count, dtype=np.float64)
    dev_clock = np.empty(msg_count, dtype=np.float64)
    for i, (msg, timestamp) in enumerate(bag.iter_topic(IMU_TRIGGER_TOPIC)):
        sin_time[i] = msg.timesyncin
        dev_clock[i] = dev_clock_of(msg, timestamp)

    # Edges are the timesyncin resets (counter ramps up, drops to ~0 at each pulse).
    edge_indices = np.where(np.diff(sin_time) < 0)[0]
    if len(edge_indices) == 0:
        logger.warning(f"{label}: no valid IMU trigger edges found")
        return None, None, None

    raw_intervals = np.diff(edge_indices) # Intervals btw UID/PPS pulses
    aligned_dev_times = []
    aligned_main_times = []
    current_time = -1.0
    last_anchor_block = -1  # which 256 s block we last anchored in
    i = 0
    while i < len(raw_intervals):
        interval = raw_intervals[i]

        # --- A. For our UID sequence we have a set of two pulses
        # this code detects is this a UID sequence or just a PPS
        # for UID, we have two pulses next to each other that add to 1s int
        # this code detects this by adding the two sequential intervals
        is_split = False
        if i + 1 < len(raw_intervals):
            pair_sum = interval + raw_intervals[i + 1]
            if abs(pair_sum - IMU_HZ) < IMU_TOLERANCE and interval < 100:
                is_split = True

        if is_split:
            est_time = max(0.0, current_time) # Our UID sequence repeats every 256s, so we need to figure
            block_idx = round(est_time / BLOCK_DURATION) # out which 256s block we are in through our past timestamps

            base_id = 16.0
            offset = (interval - base_id) / 4.0
            # Absolute main second = block start + second-within-block.
            ground_truth = (block_idx * BLOCK_DURATION) + offset
            # Re-anchor only on the FIRST split pulse of each block.
            if current_time < 0 or block_idx != last_anchor_block:
                current_time = ground_truth
                last_anchor_block = block_idx
            aligned_dev_times.append(float(dev_clock[edge_indices[i]]))
            aligned_main_times.append(float(current_time))
            current_time += 1.0  # a split pulse spans 2 intervals but represents 1 s
            i += 2
            continue

        # --- B. Standard / missed pulse: device-clock gap rounds to whole seconds ---
        dev_dt_s = dev_clock[edge_indices[i + 1]] - dev_clock[edge_indices[i]]
        n_seconds = max(1, round(dev_dt_s))
        if n_seconds >= 1 and abs(dev_dt_s - n_seconds) < 0.1:  # within 100 ms of integer s
            if current_time < 0:
                i += 1  # skip standard pulses until first absolute lock
                continue
            aligned_dev_times.append(float(dev_clock[edge_indices[i]]))  # save before advancing
            aligned_main_times.append(float(current_time))
            current_time += float(n_seconds)
            i += 1
            continue

        # --- B2. False edge: a spurious timesyncin reset split a real gap in two ---
        if interval > 100 and i + 1 < len(raw_intervals):
            combined_dt = dev_clock[edge_indices[i + 2]] - dev_clock[edge_indices[i]]
            combined_n = max(1, round(combined_dt))
            if combined_n >= 1 and abs(combined_dt - combined_n) < 0.1:
                if current_time < 0:
                    i += 2
                    continue
                aligned_dev_times.append(float(dev_clock[edge_indices[i]]))
                aligned_main_times.append(float(current_time))
                current_time += float(combined_n)
                logger.info(f"{label} false edge at i={i}: {dev_dt_s:.3f}s + next -> {combined_dt:.3f}s")
                i += 2
                continue

        # --- C. Noise ---
        i += 1

    dev = np.array(aligned_dev_times)
    mas = np.array(aligned_main_times)
    if len(dev) == 0:
        logger.warning(f"{label}: no aligned points found")
        return None, None, None

    filt = kalman_filter_and_rts(dev, mas, Q_diag=Q, R=R)

    logger.info(f"{label} Total Pulses: {len(dev)}")
    if len(dev) > 1:
        logger.info(f"{label} Time Range: {mas[0]} to {mas[-1]}")
        logger.info(f"{label} Message Duration: {dev[-1] - dev[0]:.1f}s")
        logger.info(f"{label} Dropped Packets: {int(np.sum(np.round(np.diff(mas)) > 1))}")
        dur_mas = mas[-1] - mas[0]
        if dur_mas > 0:
            drift_ppm = ((dev[-1] - dev[0]) - dur_mas) / dur_mas * 1e6
            logger.info(f"{label} Clock Drift: {drift_ppm:.2f} ppm")
    return dev, mas, filt


def calibration_system_time(bag) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
    """System-clock calibration: decode the SyncIn coded pulses using the bag
    recording timestamp (OS clock) as the device clock.

    Bag timestamps carry OS scheduling jitter (~25 ms std), so R is set wide and
    Q allows the model to track OS-clock drift.
    """
    return _decode_imu_pulses(
        bag,
        dev_clock_of=lambda msg, timestamp: timestamp * NS_TO_S,
        R=(25e-3) ** 2,
        Q=[1e-6, 1e-9],
        label="Sys",
    )


def imu_time_offset(bag) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
    """IMU-clock calibration: same coded-pulse decode, but using the VectorNav's
    raw ``timestartup`` crystal counter as the device clock.

    R reflects 400 Hz edge quantization (~1 ms)
    """
    return _decode_imu_pulses(
        bag,
        dev_clock_of=lambda msg, timestamp: msg.timestartup * NS_TO_S,
        R=(1e-3) ** 2,
        Q=[1e-8, 1e-10],
        label="IMU",
    )


def flir_time_offset(bag, topic) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
    """Extract FLIR camera time offset using robust Split-Pulse parsing."""

    # 1. Load Data
    msg_count = bag.get_message_count(topic)
    flir_cam_time = np.empty(msg_count, dtype=np.float64)
    flir_msg_linestatus = np.empty(msg_count, dtype=np.uint32)
    msg_time = np.empty(msg_count, dtype=np.float64)

    for i, (msg, timestamp) in enumerate(bag.iter_topic(topic)):
        # Device Time (in ns) -> converted to Seconds
        flir_cam_time[i] = msg.camera_time - msg.exposure_time * US_TO_NS # start of capture timestamp
        flir_msg_linestatus[i] = msg.line_status
        msg_time[i] = msg.header.stamp.sec + msg.header.stamp.nanosec * NS_TO_S

    # 2. Robust Edge Detection
    # Extract Bit 2 (Trigger). 
    # We want the INDEX where the signal flips from 0 to 1 (Rising Edge).
    binary_signal = (flir_msg_linestatus.astype(np.uint8) >> 2) & 1   
    edge_indices = np.where(np.diff(binary_signal, prepend=0) == 1)[0]
    
    if len(edge_indices) == 0:
        logger.warning(f"No valid trigger edges found in {topic}")
        return None, None, None

    # 3. Calculate Intervals (Frames between pulses)
    raw_intervals = np.diff(edge_indices)

    aligned_dev_times = []
    aligned_main_times = []
    
    current_time = -1.0
    last_anchor_block = -1  # Track which block we last anchored in
    i = 0

    # 4. Parsing Loop
    while i < len(raw_intervals):
        interval = raw_intervals[i]
       
        # --- A. Check for Coded Sequence (Split Pulse) ---
        # Logic: Current + Next sums to ~100 (1s), and Current is a small ID (< 20)
        is_split = False
        if i + 1 < len(raw_intervals):
            pair_sum = interval + raw_intervals[i+1]
            if abs(pair_sum - FLIR_HZ) < FLIR_TOLERANCE and interval < 20:
                is_split = True

        if is_split:
            # 1. Determine Block (0-255s, 256-511s, etc.)
            est_time = max(0.0, current_time)
            block_idx = round(est_time / BLOCK_DURATION)
            
            # 2. Determine Base ID. FLIR encodes the second index in whole frames
            # (1 frame/second), so base_id=4 is the index-0 reference:
            #   Normal: ID=4 -> offset 0 ;  Wrap: ID=3 -> offset -1 (255->0 fix).
            base_id = 4.0

            # 3. Decode & Anchor
            # Only re-anchor on the FIRST split pulse of each block.
            offset = (interval - base_id)
            ground_truth = (block_idx * BLOCK_DURATION) + offset
            
            # On the first lock, reject an impossible offset<0 pulse — it would
            # anchor the whole FLIR clock a second low; wait for the first legal one.
            if current_time < 0 and offset < 0:
                logger.warning(
                    f"FLIR {topic}: rejecting impossible first coded pulse "
                    f"(interval={interval}, offset={offset:+.0f} < 0); waiting for first legal pulse"
                )
                i += 2
                continue

            # Re-anchor on first lock or when entering a new block; otherwise coast
            if current_time < 0 or block_idx != last_anchor_block:
                current_time = ground_truth
                last_anchor_block = block_idx
            elif abs(ground_truth - current_time) > 1.5:
                logger.warning(
                    f"FLIR split pulse: ground_truth={ground_truth:.0f} vs expected={current_time:.0f}, "
                    f"staying with expected (interval={interval}, offset={offset})"
                )
            
            # 4. Append Data
            # Convert Device Time to Seconds
            dev_t = flir_cam_time[edge_indices[i]] * NS_TO_S
            
            aligned_dev_times.append(dev_t)
            aligned_main_times.append(float(current_time))
            
            # 5. Advance (Split pulse consumes 2 intervals, represents 1 second)
            current_time += 1.0
            i += 2
            continue

        # --- B. Check for Standard or Missed Pulse ---
        # Use camera hardware timestamp to determine n_seconds.
        dev_dt_s = (flir_cam_time[edge_indices[i+1]] - flir_cam_time[edge_indices[i]]) * NS_TO_S
        n_seconds = max(1, round(dev_dt_s))
        
        if n_seconds >= 1 and abs(dev_dt_s - n_seconds) < 0.1:  # within 100ms of integer seconds
            
            # GUARD: Wait for lock-on
            if current_time < 0:
                i += 1
                continue

            # 1. Append Data (SAVE BEFORE INCREMENTING)
            dev_t = flir_cam_time[edge_indices[i]] * NS_TO_S
            
            aligned_dev_times.append(dev_t)
            aligned_main_times.append(float(current_time)) 
            
            # 2. Advance Time
            current_time += float(n_seconds)
            
            i += 1
            continue

        # --- C. Noise ---
        i += 1

    # 5. Format Output
    device_clock_flir = np.array(aligned_dev_times)
    main_clock_flir = np.array(aligned_main_times)

    if len(device_clock_flir) == 0:
        logger.warning("FLIR Time Offset: No aligned points found.")
        return None, None, None

    # 6. Filter / Regression
    # FLIR trigger edge is detected at frame granularity (10 ms at 100 Hz).
    # Typical jitter is 1-4 frames, so R ~ (5 ms)^2 is appropriate.
    flir_R = (5e-3) ** 2  # 5 ms measurement noise std
    flir_filter = kalman_filter_and_rts(
        device_clock_flir, main_clock_flir,
        Q_diag=None,
        R=flir_R,
    )
    
    # Logging
    logger.info(f"FLIR Total Pulses: {len(device_clock_flir)}")
    if len(device_clock_flir) > 1:
        logger.info(f"FLIR Time Range: {main_clock_flir[0]} to {main_clock_flir[-1]}")
        logger.info(f"FLIR Message Duration: {msg_time[edge_indices[-1]] - msg_time[edge_indices[0]]:.1f}")
        # Calculate Dropped Packets
        # A drop is when the difference in Main Time > 1.0 (indicating we skipped a second)
        drops = np.sum(np.round(np.diff(main_clock_flir)) > 1)
        logger.info(f"FLIR Dropped/Missed Syncs: {drops}")
        
        # Calculate Slope (Crystal Error)
        duration_dev = device_clock_flir[-1] - device_clock_flir[0]
        duration_mas = main_clock_flir[-1] - main_clock_flir[0]
        drift_ppm = ((duration_dev - duration_mas) / duration_mas) * 1e6
        logger.info(f"FLIR Clock Drift: {drift_ppm:.2f} ppm")
        logger.info(f"FLIR First Pulse Duration: {raw_intervals[0]:.2f}ms")
        logger.info(f"FLIR Second Pulse Duration: {raw_intervals[2]:.2f}ms")
        
        n_rejected = np.sum(flir_filter.get('rejected', np.zeros(0, dtype=bool)))
        if n_rejected > 0:
            logger.warning(f"FLIR KF rejected {n_rejected} outlier measurements")
    
    return device_clock_flir, main_clock_flir, flir_filter


def event_time_offset(bag, topic) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
    """Extract event camera time offset from trigger messages."""
    decoder = ECDecoder()
    ev_trigon_times = []
    ev_trigoff_times = []
    ts_range=[]

    # 1. Extract Raw Edges
    last_added_on = -1
    last_added_off = -1

    # 1. Extract Raw Edges with Deduplication
    for msg, timestamp in bag.iter_topic(topic):
        decoder.decode(msg)
        trig_events = decoder.get_ext_trig_events()
        
        for event in trig_events:
            p, t = event[0], event[1]
            
            if p == 0:  # On Edge
                # DEDUPLICATION GUARD
                if t != last_added_on:
                    ev_trigon_times.append(t)
                    ts_range.append(timestamp)
                    last_added_on = t
            else:       # Off Edge
                # DEDUPLICATION GUARD
                if t != last_added_off:
                    ev_trigoff_times.append(t)
                    last_added_off = t

    ev_trigon_times = np.array(ev_trigon_times)
    ev_trigoff_times = np.array(ev_trigoff_times)
    ts_range = np.array(ts_range)
    if len(ev_trigon_times) == 0:
        logger.warning("No event triggers found")
        return None, None, None

    # 2. Pair ON/OFF to get Pulse Widths (Vectorized)
    # We use searchsorted to find the nearest OFF after each ON
    off_indices = np.searchsorted(ev_trigoff_times, ev_trigon_times, side='left')
    
    valid_on_times = []
    valid_durations_ms = []

    for i, on_t in enumerate(ev_trigon_times):
        off_idx = off_indices[i]
        if off_idx < len(ev_trigoff_times):
            off_t = ev_trigoff_times[off_idx]
            duration_us = off_t - on_t
            
            # Filter valid pulses (10ms to 80ms, allowing some tolerance)
            # 8000us (8ms) to 90000us (90ms)
            if 8000 < duration_us < 90000:
                valid_on_times.append(on_t)
                valid_durations_ms.append(duration_us / 1000.0)

    valid_on_times = np.array(valid_on_times) * US_TO_S  # Convert to Seconds
    valid_durations_ms = np.array(valid_durations_ms)
    

    # 3. Decode Time
    aligned_dev_times = []
    aligned_main_times = []

    current_main_time = -1.0
    last_on_time = -1.0

    for i in range(len(valid_durations_ms)):
        dur = valid_durations_ms[i]
        on_time = valid_on_times[i]

        # --- A. Coded Pulse (>= 15ms) ---
        if dur > 15.0:
            # Formula: Index = (Duration - 20) / 10  (20ms -> 0, 30ms -> 1, ... 80ms -> 6)
            pulse_index = round((dur - 20.0) / 10.0)

            # Determine Block. If locked on, dead-reckon the block from elapsed device time.
            if current_main_time < 0:
                est_time = 0.0  # Force Block 0 at start
            else:
                dt = on_time - last_on_time
                est_time = current_main_time + dt

            block_idx = round(est_time / 256.0)

            # Absolute second = block start + (pulse_index - 1). The "- 1" makes
            # index 1 the first second of a block (offset 0), and index 0 the
            # block-wrap marker (offset -1 = the previous block's last second).
            #   e.g. block 0, 30 ms -> index 1 -> second 0
            #        block 1, 20 ms -> index 0 -> second 255 (wrap)
            ground_truth = (block_idx * 256.0) + pulse_index - 1.0

            # Event coded pulses use pulse WIDTH (not interval jitter) → no backward-jump risk;
            # always anchor to ground truth.
            current_main_time = ground_truth
            aligned_dev_times.append(on_time)
            aligned_main_times.append(current_main_time)
            last_on_time = on_time

        # --- B. Standard Pulse (~10ms) ---
        elif abs(dur - 10.0) < 1.4:
            if current_main_time < 0:
                continue  # wait for lock

 
            dt = on_time - last_on_time
            steps = max(1, round(dt))
            current_main_time += steps
            aligned_dev_times.append(on_time)
            aligned_main_times.append(current_main_time)
            last_on_time = on_time

    # 4. Filter & Return
    dev_clock = np.array(aligned_dev_times)
    mas_clock = np.array(aligned_main_times)
    
    # Optional: Kalman Filter
    event_filter = None
    if len(dev_clock) > 10:
        try:
             event_filter = kalman_filter_and_rts(
                dev_clock, mas_clock,
                Q_diag=None, R=None,
            )
        except Exception as e:
            logger.warning(f"KF failed: {e}")
    
    if len(dev_clock) > 1:
        # Calculate stats
        avg_rate = 1.0 / np.mean(np.diff(dev_clock))
        dropped = np.sum(np.round(np.diff(dev_clock)) > 1)
        
        logger.info(f"{topic} Total Pulses: {len(dev_clock)}")
        logging.info(f"{topic} Time Range: {mas_clock[0]} to {mas_clock[-1]}")
        logger.info(f"{topic} Message Duration: {(ts_range[-1] - ts_range[0])*NS_TO_S:.1f}s")
        logging.info(f'{topic} Dropped Packets: {dropped}')
        logger.info(f"{topic} Trig Publishing Rate: {avg_rate:.2f}Hz")
        
        # Log first few durations to verify decoding (using our ms array)
        if len(valid_durations_ms) > 1:
            logger.info(f"{topic} First Pulse Duration: {valid_durations_ms[0]:.2f}ms")
            logger.info(f"{topic} Second Pulse Duration: {valid_durations_ms[1]:.2f}ms")
    return dev_clock, mas_clock, event_filter
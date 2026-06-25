# OctoSense — `data_processing`

Turns raw OctoSense ROS 2 bags into the released multi-modal dataset: time-synchronized
per-bag HDF5 + camera video, plus the depth / segmentation ground-truth, the
semantic-search index, and visualization tooling.

```
data_processing/
├── timesync/      # raw ROS 2 bag → time-synced per-bag HDF5 (data + events) + camera video
├── lidar_cal/     # LiDAR ↔ camera extrinsic calibration
├── depth_gt/      # LiDAR depth + camera-motion optical-flow ground truth
├── seg_gt/        # EoMT semantic-segmentation ground truth
├── sem_search/    # caption + embed + FAISS/BM25 natural-language search index
└── viz/           # Rerun visualization of a processed sequence
```

Most stages run in the data-processing container image; the ROS 2 conversion in
`timesync/` additionally needs ROS 2 Jazzy + the `data_collect` workspace. `timesync` (GPU-accelerated
camera-video encoding), the GT stages, and **building** the search index (`sem_search ingest-all`)
need a GPU; **querying/serving** a built index runs on CPU.

## Setup

Every stage except the ROS 2 conversion in `timesync/` runs from the repo's Python environment,
installed once from the repo root (see the [main README](../README.md#installation)):

```bash
conda env create -f environment.yml && conda activate octosense   # bundles FFmpeg
# or:  pip install -r requirements.txt   (install FFmpeg yourself)
```

This single environment covers all of the stages below, including the GT generators
(`depth_gt`, `seg_gt`) and the semantic-search index. Notes:

- **`seg_gt`** downloads the EoMT (Cityscapes) weights via `transformers` on first run.
- **`sem_search ingest-all`** also needs a Gemma-4 VLM served at an OpenAI-compatible endpoint
  (passed via `--vllm-api-base`); querying/serving a built index needs neither. Start the captioner
  with vLLM (served under the name the client expects, `gemma4-captioner`):

  ```bash
  vllm serve google/gemma-4-31B-it --served-model-name gemma4-captioner --port 8000
  # then point ingest-all at it:
  python -m sem_search.main ingest-all --data-dir <root> --output-dir <index> \
      --vllm-api-base http://localhost:8000/v1
  ```
- **`timesync/`** runs in the ROS 2 (Jazzy) data-processing container — see [`../OctoSense/`](../OctoSense/).

All commands below are run from inside the relevant stage directory.

> **Internal vs. released file naming.** `timesync/` writes the per-bag HDF5 as
> `<bag_id>.h5` (e.g. `rosbag2_2026_01_04-13_51_24.h5`), and the downstream stages
> that consume it (**`lidar_cal/`, `depth_gt/`, `seg_gt/`**). In the **released dataset on Hugging Face that  same file is published as `data.h5`**. So to run these stages on downloaded sequences, point them at a bag dir where the main HDF5 is named `<bag_id>.h5` (rename or symlink
> `data.h5` to `<bag_id>.h5`); otherwise discovery finds no bags.

### Camera / IMU calibration (Kalibr)

The camera intrinsics + camera↔camera/IMU extrinsics that `timeSync` / `lidar_cal` consume are
solved with [Kalibr](https://github.com/ethz-asl/kalibr), run from a separate Docker image built
from our fork [anthonytec2/kalibr](https://github.com/anthonytec2/kalibr). The fork sets the
AprilGrid default `blackTagBorder` to 1: our tags use a 1-bit black border, and stock Kalibr
defaults to 2 and fails to detect them.

```bash
# 1. Build the Kalibr image
git clone https://github.com/anthonytec2/kalibr.git && cd kalibr
docker build -t kalibr -f Dockerfile_ros1_20_04 .

# 2. Prep a calibration bag (cam_cals/ bag for cam-cam, imu_cals/ bag for imu-cam):
python timesync/timeSync.py --bag <raw cal bag> --mode calibration   # -> processed cal h5
python timesync/cal_dump.py --dir <processed cal-bag dir>            # -> <dir>/calibration.bag

# 3. Solve a calibration (run whichever one you need):

# camera-to-camera (all 4 cams: stereo RGB + stereo event)
docker run --rm -v /data:/data kalibr bash -c \
  "source /catkin_ws/devel/setup.bash && rosrun kalibr kalibr_calibrate_cameras \
     --bag <dir>/calibration.bag --target calibrations/aprilgrid.yaml \
     --models pinhole-radtan pinhole-radtan pinhole-radtan pinhole-radtan \
     --topics /flir_cam_right /flir_cam_left /event_camera_right /event_camera_left \
     --dont-show-report"

# imu-to-camera (RGB cams only; --cams = a camchain trimmed to cam0/cam1)
docker run --rm -v /data:/data kalibr bash -c \
  "source /catkin_ws/devel/setup.bash && rosrun kalibr kalibr_calibrate_imu_camera \
     --bag <dir>/calibration.bag --target calibrations/aprilgrid.yaml \
     --cams <rgb-only camchain.yaml> --imu calibrations/imu.yaml"
```

`-v /data:/data` mounts the host data root (where the raw + processed bags live) into the
container; change the host side to wherever your data is stored.

---

## `timesync/` — raw bag → time-synchronized dataset

The core conversion. PPS-locked time-synchronizes every sensor (FLIR stereo RGB, event
cameras, infrared, Ouster LiDAR, IMU, GPS, CAN/radar) onto one common clock and writes the
per-bag HDF5 + encoded camera video. Runs as a 3-step pipeline per bag:

1. **`odomLidarDump.py`**: decode Ouster LiDAR/IMU packets into a `<bag>_lio` ROS 2 bag for odometry.
2. **RKO-LIO odometry** (ROS 2 launch, in the `data_collect` package): offline LiDAR-inertial odometry, recorded next to the processed bag.
3. **`timeSync.py --mode data`**: time-sync all sensors, attach calibration, fuse the LiDAR-odometry + GPS trajectory, and write `<bag>.h5`, `<bag>_events.h5`, `img_{left,right,infrared}.mp4`, `time_offset.npz`.

`python timeSync.py --bag <raw_bag> --mode {data|calibration} [--cal-map cal_map.yaml] [--lio-bag <odom>]`

| file | role |
|---|---|
| `timeSync.py` | entry point (`--mode data` for collection bags, `calibration` for calib bags) |
| `timeSync_process.py` | the per-sensor workers (event/RGB/IR/LiDAR/IMU/GPS/CAN) + HDF5 merge |
| `timeSyncUtil.py` | bag reading (`BagTopicReader`), processed-path resolution, logging |
| `timeSync_calibration.py` | PPS triger parsing and Kalman filter time sync |
| `timeSync_constants.py` | topic names, sensor rates, dataset roots, CAN/radar config |
| `odomLidarDump.py` | Ouster packet → PointCloud2/Imu decode for the odometry step |
| `cal_dump.py` | reconstruct event frames + dump a ROS 2 bag for camera/IMU calibration |
| `fuse_trajectory.py` | fuse RKO-LIO + GPS into the `/fused_traj` reference trajectory |

## `lidar_cal/` — LiDAR ↔ camera extrinsic calibration

Solves the LiDAR-to-left-camera extrinsics from a calibration-target bag: detects the
circular target center in both the image (AprilTag + PnP) and the LiDAR point cloud, then
robustly fits the rigid transform. Output `lidar_calibration_results.yaml` feeds the
calibration metadata used by `timesync`.

`python lidarCal.py --processed-dir <cal_bag_dir> --cam-name cam1`

| file | role |
|---|---|
| `lidarCal.py` | entry point: orchestrates detection + extrinsic solve, writes the results yaml |
| `image_detection.py` | circle-center detection in the camera image (Kalibr intrinsics, AprilTag/PnP) |
| `lidar_detection.py` | circle-center detection in the LiDAR cloud (plane fit + circle fit) |
| `robust_matching.py` | robust / circle-optimized extrinsic solve |
| `viz_utils.py` | detection-overlay frames + quicklook video |

## `depth_gt/` — LiDAR depth (and optical-flow) ground truth

Accumulates deskewed Ouster scans in the map frame, voxel-filters for map consistency,
masks dynamic objects with YOLO segmentation, and z-buffers into the rectified-left camera
to produce a dense per-frame depth map (uint16 cm). Optical flow is then re-derived from
depth + camera motion.

```
python accum_depth_ds.py --raw_h5 <bag>.h5 --out <dir>        # → <bag>_gt.h5  (depth_cm)
python derive_flow.py    --depth_h5 <bag>_gt.h5 --out flow.h5 # → flow_uv (int16 fixed-point)
```

| file | role |
|---|---|
| `accum_depth_ds.py` | entry point: scan accumulation → map filter → dynamic mask → z-buffer → depth H5 |
| `geometry.py` | rectified intrinsics, rectify grid, viewport culling, map-consistency filter, GPU projection/z-buffer |
| `masking.py` | `DynamicMasker` — YOLO-seg dynamic-object masks (person/vehicle) |
| `derive_flow.py` | camera-motion-induced optical flow from depth + poses (rectified-left; swappable to event/IR) |

## `seg_gt/` — semantic-segmentation ground truth

Runs EoMT (Cityscapes DINOv2-L-1024, 19 classes) on the rectified-left camera at the LiDAR
10 Hz frame times, writing a per-bag `semantic.h5` (uint8 class maps) with CLAHE
preprocessing.

`python run_seg.py --data_dir <root> [--bags sess/<bag> …] [--output_dir <out>]`

| file | role |
|---|---|
| `run_seg.py` | entry point: bag discovery, frame indexing, EoMT inference, `semantic.h5` writer |
| `perception_utils.py` | calibration loading (left undistort+rectify maps), CLAHE, bag discovery, resume check |
| `models/seg_eomt.py` | EoMT HuggingFace wrapper (`tue-mps/cityscapes_semantic_eomt_large_1024`) |

## `sem_search/` — natural-language video search

Captions each window (Gemma via a vLLM server), embeds the captions (Qwen3-Embedding-8B),
and builds a hybrid FAISS + BM25 index, served behind a FastAPI app for natural-language
search over the dataset.

```
# ingest-all needs the captioner VLM running first (served as gemma4-captioner):
vllm serve google/gemma-4-31B-it --served-model-name gemma4-captioner --port 8000

python -m sem_search.main ingest-all --data-dir <root> --output-dir <index> --vllm-api-base http://localhost:8000/v1
DATA_BASE_PATH=<dataset_root>/car python -m sem_search.main serve --index-dir <index> --port 8000   # FastAPI web UI
python -m sem_search.main query "a car turning left at dusk" --index-dir <index>   # CLI search, no server
```

> **`DATA_BASE_PATH` (clip playback).** Search/`query` need no video. The web UI's inline player and
> thumbnails, however, fetch each result's clip from `$DATA_BASE_PATH/<session>/<bag>/img_left.mp4`
> (and read `data.h5` there for frame timing). Set `DATA_BASE_PATH` to the dataset root that contains
> the `<session>/<bag>/` sequence dirs (defaults to `/data/rosbags/hf_staging_octosense`). A result
> only plays if its sequence's `img_left.mp4` exists under that root; missing ones return 404 while the
> text results still render.

| path | role |
|---|---|
| `main.py` | CLI: `ingest-all` / `serve` / `query` |
| `text_encoder.py` | Qwen3 query/caption text embedding |
| `processing/ingestion.py` | scan dataset → windows → captions → embeddings → index |
| `processing/embedder.py` | caption + embedding generation (VLM prompts, telemetry) |
| `processing/prompts.py`, `processing/telemetry.py`, `processing/metadata.py` | prompt templates, run telemetry, window metadata |
| `serving/search_api.py` | FastAPI app: hybrid retrieval + clip extraction endpoints |
| `serving/diversity.py` | per-sequence diversity reranking |
| `serving/window_merger.py` | merge adjacent result windows into clips |
| `serving/clip_extractor.py` | stream-copy H.265 clip extraction with ffmpeg |
| `serving/webui/` | static web UI |

## `viz/` — Rerun visualization

Renders a processed bag (LiDAR clouds, camera + reconstructed event video, GPS, IMU, car
signals, captions) into a single `.rrd` on a shared timeline, with a bundled viewer layout.
Windowed by default for speed; `--full` for the whole ~10-min sequence.

`python rerun_viz.py --dir <processed_bag_dir> [--start S --end E | --full]`

| file | role |
|---|---|
| `rerun_viz.py` | entry point: logs all streams to `rerun_data.rrd`, bundles the blueprint |
| `octosense.rbl` | saved Rerun viewer layout |

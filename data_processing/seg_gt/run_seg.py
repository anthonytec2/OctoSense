"""
Semantic-segmentation ground truth.

Runs EoMT (Cityscapes DINOv2-L, 19 driving classes) over the rectified left
camera at the LiDAR 10 Hz frame times, writing a per-bag ``semantic.h5``
"""
import argparse
import os
import logging
import threading
import queue

import numpy as np
import cv2
import h5py
import hdf5plugin
import torch
from tqdm import tqdm
from torchcodec.decoders import VideoDecoder

from perception_utils import (
    load_calibration, compute_frame_indices, apply_clahe,
    get_daytime_bags, is_bag_complete, CS_CLASSES,
)
from models.seg_eomt import load_model as load_eomt, run_inference as run_eomt


def setup_logging(data_dir):
    """Log to both a file in data_dir and the console."""
    log_path = os.path.join(data_dir, "seg_log.txt")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(message)s",
        datefmt="%H:%M:%S",
        handlers=[logging.FileHandler(log_path, mode="a"), logging.StreamHandler()],
    )
    return logging.getLogger(__name__)


def decode_frame_bgr(decoder, idx):
    """Decode frame ``idx`` as a BGR numpy array."""
    frame = decoder[int(idx)]                                       # (C, H, W) uint8 RGB
    return frame.permute(1, 2, 0).cpu().numpy()[:, :, ::-1].copy()  # RGB→BGR


def process_seg(bag_dir, bag_name, calib, left_indices, overwrite, logger, out_dir=None):
    """Run EoMT semantic segmentation on one bag with background prefetch.

    Each frame is decoded, rectified with the calibration's undistort/rectify
    maps, and CLAHE-equalized before inference, so the labels live in the
    rectified-left frame and align 1:1 with the rectified RGB at training time.
    """
    if out_dir is None:
        out_dir = bag_dir
    out_path = os.path.join(out_dir, "semantic.h5")
    n_frames = len(calib["t_lidar"])
    H, W = calib["res"][1], calib["res"][0]

    if not overwrite and is_bag_complete(out_dir, n_frames):
        logger.info(f"[{bag_name}] semantic.h5 complete, skipping")
        return
    if os.path.exists(out_path):
        os.remove(out_path)

    logger.info(f"[{bag_name}] Seg: {n_frames} frames")
    model, processor = load_eomt()
    dec_l = VideoDecoder(os.path.join(bag_dir, "img_left.mp4"), device="cpu")

    out_h5 = h5py.File(out_path, "w")
    ds_sem = out_h5.create_dataset(
        "semantic", shape=(n_frames, H, W), dtype=np.uint8,
        chunks=(1, H, W), **hdf5plugin.Zstd(clevel=5),
    )
    out_h5.create_dataset("timestamps", data=calib["t_lidar"])
    out_h5.create_dataset("lidar_indices", data=np.arange(n_frames, dtype=np.int64))
    out_h5.create_dataset("left_img_indices", data=left_indices)
    out_h5.attrs["model"] = "EoMT-Cityscapes-DINOv2-L-1024"
    out_h5.attrs["num_classes"] = 19
    out_h5.attrs["classes"] = CS_CLASSES
    out_h5.attrs["resolution"] = f"{W}x{H}"
    out_h5.attrs["preprocessing"] = "rectify+CLAHE"
    out_h5.attrs["coordinate_frame"] = "rectified_left"

    # Prefetch decode + rectify + CLAHE in a background thread.
    prefetch_q = queue.Queue(maxsize=4)
    mapx_left, mapy_left = calib["mapx_left"], calib["mapy_left"]

    def prefetch_worker():
        for i in range(n_frames):
            bgr = decode_frame_bgr(dec_l, left_indices[i])
            bgr = cv2.remap(bgr, mapx_left, mapy_left, cv2.INTER_LINEAR)
            bgr = apply_clahe(bgr)
            prefetch_q.put((i, bgr))
        prefetch_q.put(None)

    thread = threading.Thread(target=prefetch_worker, daemon=True)
    thread.start()

    pbar = tqdm(total=n_frames, desc=f"{bag_name} seg")
    while True:
        item = prefetch_q.get()
        if item is None:
            break
        i, frame_bgr = item
        ds_sem[i] = run_eomt(model, processor, frame_bgr)
        pbar.update(1)

    pbar.close()
    thread.join()
    out_h5.close()
    del model
    torch.cuda.empty_cache()
    logger.info(f"[{bag_name}] Seg complete: {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Semantic-segmentation GT (EoMT)")
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--bags", nargs="+", default=None,
                        help="Specific bags to process (e.g. sess7/rosbag2_...). "
                             "Default: all daytime bags under --data_dir.")
    parser.add_argument("--output_dir", default=None,
                        help="Write H5 outputs here instead of the bag dir "
                             "(mirrors the data_dir structure).")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    logger = setup_logging(args.data_dir)

    # Discover bags.
    if args.bags:
        all_bags = []
        for b in args.bags:
            bag_dir = os.path.join(args.data_dir, b)
            if os.path.isdir(bag_dir):
                all_bags.append((bag_dir, os.path.basename(bag_dir)))
            else:
                logger.warning(f"Bag not found: {bag_dir}")
    else:
        all_bags = get_daytime_bags(args.data_dir)
    logger.info(f"Total bags: {len(all_bags)}")

    completed = skipped = failed = 0
    for bag_dir, bag_name in all_bags:
        try:
            calib = load_calibration(os.path.join(bag_dir, f"{bag_name}.h5"))
            n_frames = len(calib["t_lidar"])

            if args.output_dir:
                out_dir = os.path.join(args.output_dir, os.path.relpath(bag_dir, args.data_dir))
                os.makedirs(out_dir, exist_ok=True)
            else:
                out_dir = bag_dir

            if not args.overwrite and is_bag_complete(out_dir, n_frames):
                skipped += 1
                continue

            left_indices = compute_frame_indices(calib)
            process_seg(bag_dir, bag_name, calib, left_indices, args.overwrite, logger, out_dir=out_dir)
            completed += 1
        except Exception as e:
            failed += 1
            logger.error(f"[{bag_name}] FAILED: {e}")
            import traceback
            traceback.print_exc()

    logger.info(f"Done: {completed} completed, {skipped} skipped, {failed} failed")


if __name__ == "__main__":
    main()

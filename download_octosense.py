#!/usr/bin/env python3
"""Download the OctoSense dataset (or a subset) from Hugging Face.

Examples
--------
  python download_octosense.py --platform car --no-events     # all car, skip raw events
  python download_octosense.py --platform car --modalities data,depth,seg
  python download_octosense.py --sequence rosbag2_2026_01_04-13_51_24
  python download_octosense.py --all                          # the whole 8+ TB dataset

"""
import argparse
import sys
from huggingface_hub import snapshot_download

REPO = "anthonytec2/OctoSense"

# modality name -> file(s) inside <platform>/<session>/<bag>/
MODALITY_FILES = {
    "data":     ["data.h5"],                    # LiDAR / IMU / GPS / CAN, time-synced
    "events":   ["events.h5"],                  # raw async event streams (~78% of a sequence)
    "captions": ["captions.h5"],                # caption + embedding index
    "depth":    ["rgb_left_rect_depth.h5"],     # depth ground truth (car daytime)
    "seg":      ["rgb_left_rect_semantic.h5"],  # segmentation ground truth (car daytime)
    "rgb":      ["img_left.mp4", "img_right.mp4"],
    "ir":       ["img_infrared.mp4"],
}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--platform", choices=["car", "boat", "unitree"],
                    help="download one platform (default: all)")
    ap.add_argument("--sequence", help="download a single bag_id, e.g. rosbag2_2026_01_04-13_51_24")
    ap.add_argument("--modalities",
                    help=f"comma list of {list(MODALITY_FILES)}; default: all")
    ap.add_argument("--no-events", action="store_true",
                    help="shortcut to skip events.h5 (~78%% of each sequence)")
    ap.add_argument("--all", action="store_true", help="download the entire 8+ TB dataset")
    ap.add_argument("--out", default="./octosense", help="output directory (default ./octosense)")
    args = ap.parse_args()

    if not (args.platform or args.sequence or args.all):
        sys.exit("Pass --platform, --sequence, or --all (the full dataset is 8+ TB).")

    # which files within each sequence
    mods = [m.strip() for m in args.modalities.split(",")] if args.modalities else list(MODALITY_FILES)
    bad = [m for m in mods if m not in MODALITY_FILES]
    if bad:
        sys.exit(f"Unknown modalities {bad}. Choose from {list(MODALITY_FILES)}.")
    if args.no_events and "events" in mods:
        mods.remove("events")
    pick_files = bool(args.modalities) or args.no_events
    files = [f for m in mods for f in MODALITY_FILES[m]]

    # build allow_patterns from (scope) x (files)
    if not pick_files:                       # all files of the scope
        if args.sequence:   allow = [f"**/{args.sequence}/*"]
        elif args.platform: allow = [f"{args.platform}/**"]
        else:               allow = None
    else:                                    # only the chosen files
        if args.sequence:   allow = [f"**/{args.sequence}/{f}" for f in files]
        elif args.platform: allow = [f"{args.platform}/*/{f}" for f in files]
        else:               allow = [f"**/{f}" for f in files]

    snapshot_download(REPO, repo_type="dataset", local_dir=args.out,
                      allow_patterns=allow, max_workers=8)
    print(f"\nDone -> {args.out}")


if __name__ == "__main__":
    main()

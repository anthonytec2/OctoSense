"""Entry point for running the collection controller + web server."""
from __future__ import annotations

import argparse
import logging
import sys

from .collection_controller import build_controller
from .web_server import start_server


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="ROS data collection controller")
    parser.add_argument(
        "--config",
        default=None,
        help="Path to sensors_config.yaml (defaults to package config)",
    )
    parser.add_argument("--host", default="0.0.0.0", help="Web server host")
    parser.add_argument("--port", type=int, default=8080, help="Web server port")
    parser.add_argument(
        "--enable-gps",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Enable or disable GPS driver in the launch file. "
            "If not specified, uses the value from config file (default: true)."
        ),
    )
    parser.add_argument(
        "--car-mode",
        action=argparse.BooleanOptionalAction,
        default=True,
        dest="car_mode",
        help=(
            "Enable car mode which launches the socket CAN interface "
            "(ros2_socketcan on can0). Default: enabled."
        ),
    )
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    host = args.host
    logging.info(f"Starting web server on {host}:{args.port}")

    controller = build_controller(args.config)
    try:
        controller.startup_sequence(
            enable_gps=args.enable_gps,
            car_mode=args.car_mode,
        )
        start_server(controller, host=host, port=args.port)
    except KeyboardInterrupt:
        logging.info("Received interrupt, shutting down controller")
    finally:
        controller.shutdown()


if __name__ == "__main__":
    sys.exit(main())

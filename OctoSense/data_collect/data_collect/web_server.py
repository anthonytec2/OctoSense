"""FastAPI web server exposing control APIs and live logs."""
from __future__ import annotations

import asyncio
import logging
import os
import signal
from pathlib import Path
from typing import Dict, Optional

from ament_index_python.packages import (
    PackageNotFoundError,
    get_package_share_directory,
)
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse

from .collection_controller import CollectionController

LOGGER = logging.getLogger(__name__)

app = FastAPI(title="Data Collection Controller")
controller: Optional[CollectionController] = None
shutdown_task: Optional[asyncio.Task] = None


def _schedule_process_shutdown(loop: asyncio.AbstractEventLoop, delay: float = 0.25):
    global shutdown_task

    if shutdown_task:
        return shutdown_task

    async def _shutdown_worker():
        await asyncio.sleep(delay)
        LOGGER.info("Shutdown requested via API; stopping controller and exiting process")
        if controller:
            controller.shutdown()
        os.kill(os.getpid(), signal.SIGINT)

    shutdown_task = loop.create_task(_shutdown_worker())
    return shutdown_task


@app.get("/api/status")
async def get_status():
    if controller is None:
        raise HTTPException(status_code=503, detail="Controller not initialized")
    status = controller.get_status()
    return status


@app.post("/api/start")
async def start_recording(request: Dict):
    if controller is None:
        raise HTTPException(status_code=503, detail="Controller not initialized")
    bag_type = request.get("bag_type", "data")
    try:
        bag_info = controller.start_recording(bag_type)
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    return JSONResponse({"status": "started", "bag": bag_info})


@app.post("/api/stop")
async def stop_recording():
    if controller is None:
        raise HTTPException(status_code=503, detail="Controller not initialized")
    try:
        bag_info = controller.stop_recording()
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    return JSONResponse({"status": "stopped", "bag": bag_info})


@app.post("/api/system/shutdown")
async def shutdown_service():
    if controller is None:
        raise HTTPException(status_code=503, detail="Controller not initialized")

    loop = asyncio.get_running_loop()
    _schedule_process_shutdown(loop)
    return JSONResponse({"status": "shutting_down"}, status_code=202)


def _resolve_static_dir() -> Optional[Path]:
    try:
        share_dir = Path(get_package_share_directory("data_collect"))
        static_dir = share_dir / "static"
        if static_dir.exists():
            return static_dir
    except PackageNotFoundError:
        pass

    fallback = Path(__file__).resolve().parent / "static"
    if fallback.exists():
        return fallback
    return None


# Resolve static directory at module level
STATIC_DIR = _resolve_static_dir()

if STATIC_DIR:
    from fastapi.responses import FileResponse

    @app.get("/")
    async def serve_root():
        return FileResponse(STATIC_DIR / "index.html")

    @app.get("/{file_path:path}")
    async def serve_static(file_path: str):
        full_path = STATIC_DIR / file_path
        if full_path.exists() and full_path.is_file():
            return FileResponse(full_path)
        raise HTTPException(status_code=404, detail="File not found")


def start_server(controller_instance: CollectionController, host: str = "0.0.0.0", port: int = 8080):
    global controller
    controller = controller_instance

    import uvicorn

    def handle_shutdown(signum, frame):
        if controller:
            controller.shutdown()
        raise KeyboardInterrupt

    signal.signal(signal.SIGINT, handle_shutdown)
    signal.signal(signal.SIGTERM, handle_shutdown)

    LOGGER.info(f"Binding web server to {host}:{port}")
    try:
        uvicorn.run(app, host=host, port=port)
    finally:
        if controller:
            controller.shutdown()

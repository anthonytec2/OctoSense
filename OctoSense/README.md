# OctoSense — Data Collection

The data-collection stage of OctoSense: a ROS 2 (Jazzy) node that brings up a
multi-sensor rig, hardware-synchronizes it with a Teensy PPS trigger, and records
[rosbag2](https://github.com/ros2/rosbag2) (MCAP) datasets. It ships with a
small web UI for monitoring sensor health and starting/stopping recordings.

Sensors recorded here include event cameras, FLIR vision cameras, a FLIR
thermal camera, an Ouster LiDAR, a VectorNav IMU, u-blox GPS, and the vehicle CAN bus.

## Layout

```
OctoSense/
├── Dockerfile            # ROS 2 Jazzy + source-built FFmpeg image
├── run.sh                # build/run the container
├── entrypoint.zsh        # in-container: build the ws, then launch the controller in tmux
├── cfg/                  # udev rules (serial devices, CAN) + shell/tmux dotfiles
├── teensy/
│   └── pps_clock/        # Teensy MCU firmware: the PPS clock that hardware-syncs all sensors
├── SyncBoard/            # KiCad PCB (schematic + board + .pdf): distributes the Teensy PPS trigger to each sensor
├── CAD/                  # SolidWorks  parts for the sensor rig (camera bar, plates, cases)
└── data_collect/         # the ROS 2 data collection package
    ├── config/           # per-sensor configs + sensors_config.yaml (rates, health gates)
    ├── launch/           # system.launch.py brings up the whole sensor stack
    └── data_collect/
        ├── main.py                  # entry point: build controller + start web server
        ├── collection_controller.py # state machine: launch → validate → record
        ├── log_aggregator.py        # tails ROS output, polls topic rates, gates health
        ├── recorder_config.py       # the rosbag2 recorder topic list + parameters
        ├── web_server.py            # FastAPI control API (start/stop/shutdown/status)
        └── static/                  # the web UI (sensor table + recording controls)
```

## How it works

1. `main.py` builds a `CollectionController` and starts the FastAPI web server (`:8080`).
2. The controller runs `system.launch.py`, which brings up every sensor driver.
3. `LogAggregator` polls each sensor's topic rate (`ros2 topic hz`) and parses driver logs
   to track health. The controller waits until all *required* sensors hit spec, then reports
   `READY`.
4. On **Start Recording**, the rosbag2 `Recorder` is loaded into the container as a
   composable node, the bag file is confirmed on disk, and the Teensy PPS trigger is armed
   so all sensors capture on a common clock. **Stop** disarms the trigger, flushes, and
   unloads the recorder.

## Build and run

The data-collection container runs on the rig's host machine, where it can reach the physical sensors
(cameras, LiDAR, IMU, GPS, CAN).

**Prerequisites (host):** Docker, the udev rules from [`cfg/`](cfg/) installed (serial/CAN device
names), and, for the event cameras, the CenturyArks SilkyEvCam HAL plugin (see below).

```bash
# 1. Build the image (ROS 2 Jazzy + source-built FFmpeg + the capture workspace).
#    Run from this directory — the build context is OctoSense/ (it COPYs data_collect/, cfg/,
#    entrypoint.zsh). This is a large build (CUDA base + ~25 ROS repos + two colcon builds).
docker build -t octosense .

# 2. Run it. run.sh takes the image name.
ROSBAGS_DIR=$HOME/rosbags ./run.sh octosense
```

On start, `entrypoint.zsh` builds the `data_collect` package, then launches the collection
controller in tmux. Open the web UI at **`http://<host>:8080`** to watch per-sensor health and
start/stop recording. Recorded bags land in `ROSBAGS_DIR` on the host (mounted to `/rosbags`;
the recorder's path comes from `bag_output_dir` in `data_collect/config/sensors_config.yaml`).

**Runtime options** are passed through `run.sh` as `KEY=VALUE` args (it forwards them to the
container as `-e`):

```bash
./run.sh octosense ENABLE_GPS=false CAR_MODE=false PORT=8080
```

| variable | default | effect |
|---|---|---|
| `ENABLE_GPS` | `true` | `false` brings the stack up without GPS (`--no-enable-gps`) |
| `CAR_MODE` | `true` | `false` for non-vehicle rigs, e.g. boat/quadruped (`--no-car-mode`) |
| `HOST` | `0.0.0.0` | web-server bind address |
| `PORT` | controller default | web-server port |
| `CONFIG` | `sensors_config.yaml` | alternate sensor config path (in the container) |

## Hardware (build the rig)

The physical platform is fully open: the sync firmware, the trigger PCB, and the mechanical CAD.

<table align="center">
  <tr>
    <td align="center"><img src="../assets/hw_cad.png" alt="OctoSense sensor rig (CAD)" height="230"/></td>
    <td align="center"><img src="../assets/hw_syncboard.png" alt="SyncBoard with the Teensy PPS clock" height="230"/></td>
  </tr>
  <tr>
    <td align="center"><sub>OctoSense Sensor Rig (CAD)</sub></td>
    <td align="center"><sub>Time Synchronization Custom PCB</sub></td>
  </tr>
</table>

**Teensy PPS clock** ([`teensy/pps_clock/`](teensy/pps_clock/)) is the firmware that generates the
pulse-per-second trigger distributed to every sensor. It targets a **Teensy 4.1** and uses the
Adafruit **RTClib** library. Compile `pps_clock.ino` to a `.hex` with the Teensy toolchain (the
Arduino IDE + [Teensyduino](https://www.pjrc.com/teensy/td_download.html), or `arduino-cli` with the
Teensy core), then flash it with [`teensy_loader_cli`](https://www.pjrc.com/teensy/loader_cli.html):

```bash
# compile to a .hex (arduino-cli with the Teensy core + Adafruit RTClib installed)
arduino-cli compile --fqbn teensy:avr:teensy41 --export-binaries teensy/pps_clock

# flash the exported .hex to the Teensy 4.1
teensy_loader_cli --mcu=TEENSY41 -v -w pps_clock.ino.hex
```

**SyncBoard** ([`SyncBoard/`](SyncBoard/)) is a KiCad PCB that fans the Teensy's PPS trigger out to
each sensor connector. Open `SyncBoard.kicad_pro` in KiCad to edit.

**CAD** ([`CAD/`](CAD/)) holds the SolidWorks parts and assemblies for the sensor rig (camera bar,
plates, cases).

## SilkyEvCam event cameras — proprietary HAL plugin

The stereo event cameras (CenturyArks **SilkyEvCam**) require CenturyArks'
**proprietary HAL plugin**, which is **not bundled in
this repo or image**. The open-source event-camera driver stack (`metavision_driver`,
`openeb_vendor`, `event_camera_*`) *is* built into the image, but the SilkyEvCam won't
enumerate until you install the plugin yourself. Download the installer from CenturyArks and
run it on the host, it installs the HAL plugin, tools, and udev rules:
- **https://centuryarks.com/en/download/** — e.g. `SilkyEvCam_Plugin_Installer_for_ubuntu_v5.2.0.zip`


> **Hardware-specific values** (camera serials, sensor IP addresses, network interface
> names, the CAN/serial udev rules in `cfg/`) are tuned to our rig. Adjust them in the
> configs and `cfg/` rules for your own hardware.

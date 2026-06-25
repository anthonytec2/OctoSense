#!/bin/bash

IMG="$1"
shift  # Remove image name from arguments

# Resolve repo location so the data_collect package is bind-mounted from the
# checkout, and let host paths be overridden via environment variables.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROSBAGS_DIR="${ROSBAGS_DIR:-$HOME/rosbags}"

# Collect environment variable arguments (format: KEY=VALUE)
ENV_ARGS=()
while [ $# -gt 0 ]; do
    if [[ "$1" == *"="* ]]; then
        # Argument contains =, treat as environment variable
        ENV_ARGS+=("-e" "$1")
    else
        # Not an env var, could be other docker flags - pass through
        ENV_ARGS+=("$1")
    fi
    shift
done

echo "Running docker..."
docker run \
  -it \
  --privileged \
  --network host \
  -v /run/udev/data:/run/udev/data:ro \
  -v /dev/bus/usb:/dev/bus/usb \
  --device /dev/snd \
  -v "/etc/localtime:/etc/localtime:ro" \
  -v "${ROSBAGS_DIR}:/rosbags" \
  -v "/dev:/dev" \
  -v "${SCRIPT_DIR}/data_collect:/catkin_ws/src/data_collect" \
  -h "$HOSTNAME" \
  --rm \
  --security-opt seccomp=unconfined \
  --group-add=dialout \
  "${ENV_ARGS[@]}" \
  $IMG

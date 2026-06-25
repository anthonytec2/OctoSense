#!/usr/bin/env zsh
set -e

# --------------------------------------------
# 1. Source ROS + workspace
# --------------------------------------------
if [ -f /opt/ros/${ROS_DISTRO}/setup.zsh ]; then
    source /opt/ros/${ROS_DISTRO}/setup.zsh
    source /root/octosense/bin/activate
fi

cd /catkin_ws

# --------------------------------------------
# 2. Build only data_collect package
# --------------------------------------------
echo "Building data_collect..."
colcon build --packages-select data_collect --symlink-install

# --------------------------------------------
# 3. Source the overlay workspace
# --------------------------------------------
if [ -f /catkin_ws/install/setup.zsh ]; then
    source /catkin_ws/install/setup.zsh
fi

# --------------------------------------------
# 4. Build command-line arguments from environment variables
# --------------------------------------------
ARGS=()

# Boolean flags (use --flag or --no-flag based on env var)
if [ "${ENABLE_GPS:-true}" = "false" ]; then
    ARGS+=("--no-enable-gps")
elif [ "${ENABLE_GPS:-true}" = "true" ]; then
    ARGS+=("--enable-gps")
fi

if [ "${CAR_MODE:-true}" = "false" ]; then
    ARGS+=("--no-car-mode")
elif [ "${CAR_MODE:-true}" = "true" ]; then
    ARGS+=("--car-mode")
fi

# Value flags
# Bind to all interfaces by default; override with HOST if provided.
HOST="${HOST:-0.0.0.0}"
ARGS+=("--host" "${HOST}")
[ -n "${PORT:-}" ] && ARGS+=("--port" "${PORT}")
[ -n "${CONFIG:-}" ] && ARGS+=("--config" "${CONFIG}")

# Build the command with proper quoting
CMD="source /opt/ros/${ROS_DISTRO}/setup.zsh; source /catkin_ws/install/setup.zsh; ros2 run data_collect collection_controller"
if [ ${#ARGS[@]} -gt 0 ]; then
    # Properly quote each argument to handle spaces in paths
    for arg in "${ARGS[@]}"; do
        CMD="${CMD} ${(q)arg}"
    done
fi

# --------------------------------------------
# 5. Start tmux with 2 panes
# --------------------------------------------
SESSION="startup"

# Create new session detached with a shell
tmux new-session -d -s $SESSION -n main

# # Pane 0 (left) - send commands to the default pane
tmux send-keys -t $SESSION:main.0 "${CMD}" C-m

# # --------------------------------------------
# # 5. Attach to tmuxd
# # --------------------------------------------
tmux attach -t $SESSION
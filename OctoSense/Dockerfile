# =============================================================================
# Multi-sensor data-collection workstation
# CUDA-13 / ROS 2 Jazzy / FFmpeg (source-built w/ NVENC)
# Python venv: /root/octosense (torch, gtsam, h5py, rerun, etc.)
# =============================================================================
FROM nvidia/cuda:13.0.1-devel-ubuntu24.04

SHELL ["/bin/bash", "-o", "pipefail", "-c"]

ENV DEBIAN_FRONTEND=noninteractive \
    ROS_DISTRO=jazzy \
    LANG=en_US.UTF-8 \
    LC_ALL=en_US.UTF-8


# -----------------------------------------------------------------------------
# 1. APT — base system, dev tools, GStreamer, ROS prerequisites
# -----------------------------------------------------------------------------
RUN set -eux; \
    apt-get update; \
    apt-get install -y --no-install-recommends \
        locales tzdata ca-certificates software-properties-common \
        curl wget git unzip openssh-server vim zsh tmux htop \
        build-essential cmake pkg-config \
        python3 python3-dev python3-pip python3.12-venv nlohmann-json3-dev \
        libopencv-dev libboost-all-dev libeigen3-dev libjsoncpp-dev \
        libspdlog-dev libcurl4-openssl-dev libprotobuf-dev protobuf-compiler \
        libusb-1.0-0-dev libhdf5-dev libcanberra-gtk-module \
        iputils-arping iputils-ping iproute2 net-tools \
        gstreamer1.0-plugins-bad gstreamer1.0-plugins-ugly gstreamer1.0-libav \
        libgtest-dev libglew-dev libglfw3-dev \
        libomp-dev clang-tidy inotify-tools h5utils can-utils; \
    locale-gen en_US.UTF-8; \
    update-locale LANG=en_US.UTF-8 LC_ALL=en_US.UTF-8; \
    rm -rf /var/lib/apt/lists/*


# -----------------------------------------------------------------------------
# 2. APT — ROS 2 Jazzy + ROS dev tools
# -----------------------------------------------------------------------------
RUN set -eux; \
    curl -fsSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key \
         -o /usr/share/keyrings/ros-archive-keyring.gpg; \
    echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] \
http://packages.ros.org/ros2/ubuntu $(. /etc/os-release && echo $UBUNTU_CODENAME) main" \
        | tee /etc/apt/sources.list.d/ros2.list >/dev/null; \
    apt-get update; \
    apt-get install -y --no-install-recommends \
        ros-dev-tools \
        python3-colcon-common-extensions \
        ros-$ROS_DISTRO-desktop \
        ros-$ROS_DISTRO-pcl-ros \
        ros-$ROS_DISTRO-tf2-eigen \
        ros-$ROS_DISTRO-rviz2 \
        ros-$ROS_DISTRO-image-transport \
        ros-$ROS_DISTRO-compressed-image-transport \
        ros-$ROS_DISTRO-image-transport-plugins \
        ros-$ROS_DISTRO-ament-cmake-clang-format \
        ros-$ROS_DISTRO-ament-cmake-vendor-package \
        ros-$ROS_DISTRO-camera-info-manager \
        ros-$ROS_DISTRO-diagnostic-updater \
        ros-$ROS_DISTRO-nmea-msgs \
        ros-$ROS_DISTRO-rtcm-msgs \
        ros-$ROS_DISTRO-rosbag2-storage-mcap \
        ros-$ROS_DISTRO-rmw-zenoh-cpp \
        ros-$ROS_DISTRO-ament-cmake-black \
        ros-$ROS_DISTRO-rmw-cyclonedds-cpp \
        ros-$ROS_DISTRO-rosidl-generator-dds-idl \
        ros-$ROS_DISTRO-ros2-socketcan \
        libasio-dev; \
    rm -rf /var/lib/apt/lists/*


# -----------------------------------------------------------------------------
# 3. APT — FFmpeg build-time deps + GPU video stack (built from source below)
# -----------------------------------------------------------------------------
RUN set -eux; \
    apt-get update; \
    apt-get install -y --no-install-recommends \
        autoconf automake libtool pkgconf yasm nasm \
        libass-dev libfreetype6-dev libgnutls28-dev libmp3lame-dev libsdl2-dev \
        libtheora-dev libvorbis-dev libvpx-dev libx264-dev libx265-dev libxvidcore-dev \
        libva-dev libvdpau-dev zlib1g-dev libfdk-aac-dev libopus-dev \
        libfontconfig1-dev libfribidi-dev libharfbuzz-dev libjansson-dev libsoxr-dev \
        libspeex-dev libsrt-gnutls-dev libbluray-dev libdrm-dev libxml2-dev libzmq3-dev \
        ocl-icd-opencl-dev opencl-headers \
        vainfo libva2 libva-drm2 libva-x11-2 mesa-va-drivers \
        libvulkan1 vulkan-tools mesa-vulkan-drivers rtklib \
        libdrm2 libdrm-amdgpu1 \
        usbutils strace; \
    rm -rf /var/lib/apt/lists/*


# -----------------------------------------------------------------------------
# 4. Source-built native libs: NVIDIA codec headers (for NVENC)
# -----------------------------------------------------------------------------
ENV PKG_CONFIG_PATH=/usr/local/lib/pkgconfig

RUN set -eux; \
    cd /tmp; \
    git clone --depth 1 --branch n13.0.19.0 https://git.videolan.org/git/ffmpeg/nv-codec-headers.git; \
    cd nv-codec-headers; \
    make install; \
    rm -rf /tmp/nv-codec-headers


# -----------------------------------------------------------------------------
# 5. FFmpeg from source — GPL + CUDA/NVENC
#    Codec-rich build installed to /usr/local; used by the ffmpeg_* ROS packages.
# -----------------------------------------------------------------------------
RUN set -eux; \
    cd /tmp; \
    git clone --depth 1 --branch n8.0.1 https://git.ffmpeg.org/ffmpeg.git ffmpeg; \
    cd ffmpeg; \
    ./configure \
        --prefix=/usr/local \
        --enable-gpl --enable-version3 --enable-nonfree \
        --enable-shared --disable-static \
        --enable-libass --enable-libfdk-aac --enable-libfreetype \
        --enable-libmp3lame --enable-libopus --enable-libtheora \
        --enable-libvorbis --enable-libvpx --enable-libx264 --enable-libx265 \
        --enable-libxvid --enable-libwebp \
        --enable-libsoxr --enable-libspeex --enable-libsrt \
        --enable-libbluray --enable-libdrm --enable-libzmq \
        --enable-opencl --enable-opengl --enable-vaapi --enable-vdpau \
        --enable-cuda-nvcc \
        --extra-cflags='-I/usr/local/cuda/include -fPIC' \
        --extra-ldflags='-L/usr/local/cuda/lib64' \
        --nvccflags='-Xcompiler -fPIC'; \
    make -j"$(nproc)"; make install; ldconfig


# -----------------------------------------------------------------------------
# 6. Source-built C++ utility lib used by event-camera pipeline
# -----------------------------------------------------------------------------
WORKDIR /tmp
RUN set -eux; \
    git clone -b python_bindings https://github.com/anthonytec2/simple_image_recon_lib.git; \
    cd simple_image_recon_lib; mkdir build; cd build; \
    cmake .. -DPYBIND11_TEST=OFF; \
    cmake --build . -j"$(nproc)"; \
    cmake --install .


# -----------------------------------------------------------------------------
# 7. Shell environment: zsh / oh-my-zsh / powerlevel10k / tmux-themepack / fzf
# -----------------------------------------------------------------------------
RUN set -eux; \
    sh -c "$(curl -fsSL https://raw.githubusercontent.com/ohmyzsh/ohmyzsh/master/tools/install.sh)" "" --unattended; \
    git clone --depth 1 https://github.com/junegunn/fzf.git ~/.fzf; ~/.fzf/install --all; \
    git clone https://github.com/zsh-users/zsh-syntax-highlighting.git ${ZSH_CUSTOM:-~/.oh-my-zsh/custom}/plugins/zsh-syntax-highlighting; \
    git clone https://github.com/zsh-users/zsh-autosuggestions      ${ZSH_CUSTOM:-~/.oh-my-zsh/custom}/plugins/zsh-autosuggestions; \
    git clone https://github.com/zsh-users/zsh-history-substring-search ${ZSH_CUSTOM:-~/.oh-my-zsh/custom}/plugins/zsh-history-substring-search; \
    git clone --depth=1 https://github.com/romkatv/powerlevel10k.git ${ZSH_CUSTOM:-$HOME/.oh-my-zsh/custom}/themes/powerlevel10k; \
    git clone https://github.com/jimeh/tmux-themepack.git ~/.tmux-themepack; \
    chsh -s /bin/zsh; \
    mkdir -p /root/hist


# -----------------------------------------------------------------------------
# 8. Python venv (/root/octosense) — torch, ML/IO libs, h5py + hdf5plugin,
#    pykalman (timeSync KF), gtsam (fuse_trajectory), projectaria-tools, etc.
# -----------------------------------------------------------------------------
RUN set -eux; \
    cd /root && \
    python3 -m venv octosense && \
    /root/octosense/bin/pip install --upgrade pip && \
    /root/octosense/bin/pip install --no-cache-dir \
        'torch==2.12.*' 'torchvision==0.27.*' \
        --index-url https://download.pytorch.org/whl/cu130 && \
    /root/octosense/bin/pip install --no-cache-dir \
        rosbags==0.11.0 \
        opencv-python \
        matplotlib ruff joblib empy numba catkin_pkg \
        pybind11[global] rerun-sdk ipython numpy \
        einops ninja jupyterlab zarr h5py hdf5plugin \
        pytest tqdm scipy psutil zmq && \
    /root/octosense/bin/pip install --no-cache-dir 'torchcodec>=0.14' \
        --index-url=https://download.pytorch.org/whl/cu130

# Extras + repo-relative tools that need the venv to exist first
RUN set -eux; \
    /root/octosense/bin/pip install --no-cache-dir \
        ouster-sdk==0.15.0 pyyaml pyserial setuptools wheel \
        colcon-common-extensions pip \
        rosbags geopandas contextily cantools pupil-apriltags \
        scikit-learn open3d \
        fastapi uvicorn[standard] pyzt gtsam \
        sentence-transformers==5.5.1 transformers==5.11.0 faiss-cpu==1.14.2 rank-bm25==0.2.2 \
        mcap-ros2-support projectaria-tools pykalman && \
    cd /tmp/simple_image_recon_lib/ && /root/octosense/bin/python setup.py install && \
    cd /tmp && git clone https://github.com/mlaiacker/rosbag2video


# -----------------------------------------------------------------------------
# 9. ROS workspace: clone all source repos, install deps, build with colcon
#    (FFmpeg from layer 5 must already be present so ffmpeg_* packages compile)
# -----------------------------------------------------------------------------
WORKDIR /catkin_ws/src

RUN set -ex; \
    . /opt/ros/$ROS_DISTRO/setup.bash; \
    git clone -b ros2 --recurse-submodules https://github.com/ouster-lidar/ouster-ros.git; \
    git clone https://github.com/dawonn/vectornav.git -b ros2; \
    git clone https://github.com/ros-event-camera/openeb_vendor.git; \
    git clone https://github.com/ros-event-camera/event_camera_msgs.git; \
    git clone https://github.com/ros-event-camera/metavision_driver.git; \
    git clone https://github.com/ros-event-camera/event_camera_viewer.git; \
    cd event_camera_viewer && git checkout 2ed8d28c78f1831f877fe1e30b874953efa949ee && cd ..; \
    git clone https://github.com/ros-event-camera/event_camera_codecs.git; \
    cd event_camera_codecs && \
        git checkout 3a74561674170be99c6b1f22bc72a722fb6a241e && \
        sed -i '286,296c\    constexpr timestamp_t ROLLOVER_HIGH_THRESHOLD = 4000;\n    constexpr timestamp_t ROLLOVER_LOW_THRESHOLD = 100;\n    if (lastHigh > ROLLOVER_HIGH_THRESHOLD \&\& t < ROLLOVER_LOW_THRESHOLD) {\n      timeHigh += (1 << 24);\n    }' include/event_camera_codecs/evt3_decoder.h && \
        cd /catkin_ws/src; \
    git clone https://github.com/ros-event-camera/dvs_msgs.git; \
    git clone https://github.com/ros-event-camera/prophesee_event_msgs.git; \
    git clone https://github.com/berndpfrommer/sync_event_frames.git; \
    git clone https://github.com/berndpfrommer/simple_image_recon.git; \
    git clone https://github.com/ros-event-camera/event_camera_py.git; \
    cd event_camera_py && git checkout d961f02bcb691c3be2549dd784f7b3fa8eeeb643 && cd ..; \
    git clone https://github.com/ros-event-camera/event_camera_tools.git; \
    cd event_camera_tools && git checkout a2ff722a8d187e863f8dddc668a513c28f1f8902 && cd ..; \
    git clone -b zed_9fp_tim https://github.com/anthonytec2/ublox.git; \
    git clone https://github.com/anthonytec2/ouster_http_node.git; \
    git clone -b line_status https://github.com/anthonytec2/flir_camera_driver.git; \
    git clone -b use_lastest_pybind11 https://github.com/ros-event-camera/pybind11_vendor.git; \
    git clone -b imu-odom https://github.com/anthonytec2/rko_lio.git; \
    git clone -b ros2 https://github.com/anthonytec2/ntrip_client.git; \
    git clone https://github.com/ros-misc-utilities/ffmpeg_image_transport.git; \
    git clone -b pts_arr https://github.com/anthonytec2/ffmpeg_encoder_decoder.git; \
    git clone -b transcoder_pts https://github.com/anthonytec2/ffmpeg_image_transport_tools.git; \
    git clone https://github.com/ros-misc-utilities/ffmpeg_image_transport_msgs.git; \
    git clone https://github.com/unitreerobotics/unitree_ros2.git; \
    cd /catkin_ws/src/unitree_ros2/cyclonedds_ws/src; \
        git clone https://github.com/eclipse-cyclonedds/cyclonedds -b releases/0.10.x; \
        git clone https://github.com/ros2/rmw_cyclonedds -b jazzy

# Install all ROS package deps (rosdep) in one pass.
WORKDIR /catkin_ws
RUN set -eo pipefail; \
    . "/opt/ros/${ROS_DISTRO}/setup.sh"; \
    apt-get update; \
    rosdep init || true; \
    rosdep update; \
    rosdep install --from-paths src --ignore-src -r -y; \
    rm -rf /var/lib/apt/lists/*

# First colcon: everything except the ffmpeg-dependent + unitree subset
RUN set -eo pipefail; \
    . "/opt/ros/${ROS_DISTRO}/setup.sh"; \
    colcon build --symlink-install --cmake-args -DCMAKE_BUILD_TYPE=Release

# Second colcon: ffmpeg-dependent + unitree pkgs (need our /usr/local FFmpeg)
RUN set -eo pipefail; \
    . "/opt/ros/${ROS_DISTRO}/setup.sh"; \
    colcon build --symlink-install \
        --cmake-args --no-warn-unused-cli -DFFMPEG_PKGCONFIG=/usr/local/lib/pkgconfig -DCMAKE_BUILD_TYPE=RelWithDebInfo \
        --packages-select ffmpeg_image_transport_tools ffmpeg_image_transport_msgs \
                          ffmpeg_encoder_decoder ffmpeg_image_transport \
                          cyclonedds unitree_go unitree_api


# -----------------------------------------------------------------------------
# 9b. data_collect — the repo's own ROS 2 package
# -----------------------------------------------------------------------------
COPY data_collect /catkin_ws/src/data_collect
RUN set -eo pipefail; \
    . "/opt/ros/${ROS_DISTRO}/setup.sh"; \
    . /catkin_ws/install/setup.sh; \
    colcon build --symlink-install --packages-select data_collect \
        --cmake-args -DCMAKE_BUILD_TYPE=Release


# -----------------------------------------------------------------------------
# 10. Shell dotfiles.
#     NOTE: SilkyEvCam event cameras require CenturyArks' proprietary HAL plugin,
#     which is NOT bundled in this repo. To enable SilkyEvCam support, download
#     the plugin installer from CenturyArks and run it in the container (or on
#     the host) -- it ships the HAL plugin, tools, and udev rules:
#       https://centuryarks.com/en/download/
#       (e.g. SilkyEvCam_Plugin_Installer_for_ubuntu_v5.2.0.zip)
# -----------------------------------------------------------------------------
COPY cfg/ /tmp/cfg/

RUN set -eux; \
    cp -f /tmp/cfg/.tmux.conf /tmp/cfg/.zshrc /root/; \
    rm -rf /tmp/cfg


# -----------------------------------------------------------------------------
# 11. Final shell configuration (.zshrc auto-sources ROS + octosense venv + CycloneDDS)
# -----------------------------------------------------------------------------
RUN set -eux; \
    { \
        printf '\nsource /opt/ros/$ROS_DISTRO/setup.zsh\n';                                                    \
        printf 'source /catkin_ws/install/setup.zsh\n';                                                        \
        printf 'export RMW_IMPLEMENTATION=rmw_zenoh_cpp\n';                                                    \
        printf '\nsource /root/octosense/bin/activate\n';                                                            \
    } >> /root/.zshrc

ENV HISTFILE=/root/hist/.zsh_history


# -----------------------------------------------------------------------------
# 12. Entrypoint script
# -----------------------------------------------------------------------------
COPY entrypoint.zsh /usr/local/bin/entrypoint.zsh
RUN chmod +x /usr/local/bin/entrypoint.zsh

ENTRYPOINT ["/usr/local/bin/entrypoint.zsh"]

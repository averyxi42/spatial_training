# syntax=docker/dockerfile:1.7
# LongNav-R1 reproduction image.
#
# The hard part of this project is habitat-sim: the conda package for 0.3.3 is
# only published against python 3.9, so we build it from source in a python
# 3.10.16 conda env using --headless --with-cuda --bullet.
#
# Two conda envs exist because the code requires them: config_schema.py defaults
# vlm_conda_env="longnav" / habitat_conda_env="vln", and factories.py hands those
# to Ray as runtime_env={"conda": ...}. Ray re-execs each actor inside the named
# env, so both must exist under those exact names and both must carry a matching
# ray version.
#
# Build:
#   docker compose build                       # -> target: runtime
#   docker build --target habitat -t longnav-base:habitat0.3.3 -f docker/Dockerfile .
#
# Stages are ordered by volatility: editing anything in `runtime` can never
# invalidate the ~40 minute habitat-sim compile in `habitat`.
#
# LAYERING RULES, please keep them:
#   1. Declare an ARG immediately before the RUN that consumes it, never at the
#      top of a stage. An ARG/ENV change invalidates every layer below it.
#   2. Every RUN that writes into /opt/conda ends with `chmod -R a+rwX` on what
#      it touched, in the SAME RUN. A trailing chmod in its own layer would
#      force an overlayfs copy-up of the whole ~15GB conda tree.
#
# The CUDA *devel* image is required: --with-cuda needs nvcc and the CUDA
# headers at build time.
ARG CUDA_IMAGE=nvidia/cuda:12.9.2-cudnn-devel-ubuntu22.04


# =============================================================================
FROM ${CUDA_IMAGE} AS system
# System packages, GL stack, conda itself, and policy that never changes.
# =============================================================================

ENV DEBIAN_FRONTEND=noninteractive \
    TZ=Etc/UTC \
    CONDA_DIR=/opt/conda \
    PYTHONUNBUFFERED=1

# habitat-sim's CMake and any later CUDA extension build look for the toolkit
# under several different variable names; set them all.
ENV CUDA_HOME=/usr/local/cuda \
    CUDAToolkit_ROOT=/usr/local/cuda \
    CUDA_PATH=/usr/local/cuda \
    PATH=/usr/local/cuda/bin:${PATH} \
    LD_LIBRARY_PATH=/usr/local/cuda/lib64:${LD_LIBRARY_PATH}

# No GPU is visible during `docker build`, so CUDA architectures cannot be
# autodetected — they must be listed explicitly or the build targets nothing.
# Volta through Blackwell; trim the list to shorten build time.
ENV TORCH_CUDA_ARCH_LIST="7.0;7.5;8.0;8.6;8.9;9.0;10.0;12.0" \
    CMAKE_CUDA_ARCHITECTURES="70;75;80;86;89;90;100;120"

# Headless EGL rendering needs the graphics capability, not just compute.
ENV NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics

# Note: deliberately NOT a login shell (`-lc`). Ubuntu's /etc/profile resets
# PATH unconditionally for uid 0, which would wipe the conda and CUDA entries
# set above.
SHELL ["/bin/bash", "-o", "pipefail", "-c"]

# habitat-sim's documented build dependencies, plus the EGL/OSMesa runtime bits
# needed for headless GPU rendering.
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        ca-certificates \
        cmake \
        ninja-build \
        pkg-config \
        git \
        git-lfs \
        curl \
        wget \
        unzip \
        zip \
        bzip2 \
        vim \
        less \
        tmux \
        htop \
        rsync \
        sudo \
        ffmpeg \
        libjpeg-dev \
        libpng-dev \
        libglm-dev \
        libomp-dev \
        libhdf5-dev \
        libglvnd-dev \
        libegl1 \
        libegl1-mesa-dev \
        libgles2 \
        libgl1-mesa-dev \
        libgl1-mesa-dri \
        libglu1-mesa \
        mesa-common-dev \
        mesa-utils \
        libosmesa6-dev \
        libx11-dev \
        libxrandr-dev \
        libxinerama-dev \
        libxcursor-dev \
        libxi-dev \
        xorg-dev \
        freeglut3-dev \
        libglfw3-dev \
        libbullet-dev \
    && rm -rf /var/lib/apt/lists/*

# Software-GL fallback: without this a headless container with no usable EGL
# device fails to find swrast instead of degrading to it.
RUN mkdir -p /usr/lib/dri \
    && ln -sf /usr/lib/x86_64-linux-gnu/dri/swrast_dri.so /usr/lib/dri/ 2>/dev/null || true

# --- policy ------------------------------------------------------------------
# System-level git config applies to every uid regardless of $HOME, which a
# --global config cannot do once the container runs as an arbitrary user.
#   safe.directory '*' : the repo is bind-mounted from the host and will always
#     have a "wrong" owner inside the container. '*' also covers the verl and
#     ovon submodules and their .git/modules/* gitdirs.
#   url.insteadOf      : .gitmodules declares both submodules over SSH
#     (git@github.com:...), and a container has no SSH key. Both repos are
#     public, so rewriting to https lets `git clone --recursive` work with no
#     credential story at all.
RUN git config --system --add safe.directory '*' \
    && git config --system url."https://github.com/".insteadOf "git@github.com:" \
    && echo "ALL ALL=(ALL) NOPASSWD: ALL" >> /etc/sudoers

# Miniforge (conda-forge by default) avoids the Anaconda default-channel terms
# of service prompt that breaks unattended builds.
RUN curl -fsSL -o /tmp/miniforge.sh \
        "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh" \
    && bash /tmp/miniforge.sh -b -p ${CONDA_DIR} \
    && rm /tmp/miniforge.sh \
    && ${CONDA_DIR}/bin/conda clean -afy \
    && chmod -R a+rwX ${CONDA_DIR}

ENV PATH=${CONDA_DIR}/bin:${PATH}

# --- arbitrary-uid support ---------------------------------------------------
# The image is deliberately uid-agnostic: no USER_UID/USER_GID build args exist,
# so changing the runtime user can never invalidate a single cached layer.
# `longnav` (uid 1000) is only the default when compose does not pass `user:`.
# /etc/passwd and /etc/group are world-writable so the entrypoint can synthesize
# an entry for whatever uid it is actually given — sudo, `id -gn` and git all
# misbehave for a uid absent from the passwd database.
RUN useradd -m -u 1000 -s /bin/bash longnav \
    && chmod 777 /home/longnav \
    && chmod 666 /etc/passwd /etc/group \
    && mkdir -p /workspace \
    && chmod 777 /workspace


# =============================================================================
FROM system AS habitat
# The expensive stage. ~40 minutes, dominated by the habitat-sim compile.
# Nothing below this line in `runtime` can invalidate it.
# =============================================================================

ARG PYTHON_VERSION=3.10.16
ARG HABITAT_CONDA_ENV=vln
ENV HABITAT_CONDA_ENV=${HABITAT_CONDA_ENV}

# python 3.10.16 is what the project pins; the stock habitat conda package would
# drag the env back to 3.9, which is the whole reason for building from source.
#
# The libstdc++ dance: the conda env ships an older libstdc++ than Ubuntu 22.04,
# and magnum plus several CUDA extensions load the system copy and then die on
# `GLIBCXX_3.4.30 not found`.
#
# numpy is pinned to exactly 1.26.4 image-wide: habitat-sim's magnum bindings
# compile against the numpy C API present at build time, so any later move
# silently breaks `import habitat_sim`. Re-asserted at the end of the build.
RUN conda create -y -n ${HABITAT_CONDA_ENV} python=${PYTHON_VERSION} cmake=3.27 \
    && conda install -y -n ${HABITAT_CONDA_ENV} -c conda-forge \
        "libgcc-ng>=12" "libstdcxx-ng>=12" "libgomp>=12" \
    && conda run -n ${HABITAT_CONDA_ENV} python -m pip install --upgrade pip setuptools wheel \
    && conda run -n ${HABITAT_CONDA_ENV} pip install --upgrade urllib3 \
    && conda run -n ${HABITAT_CONDA_ENV} pip install "numpy==1.26.4" \
    && if [ -e ${CONDA_DIR}/envs/${HABITAT_CONDA_ENV}/lib/libstdc++.so.6 ]; then \
           mv ${CONDA_DIR}/envs/${HABITAT_CONDA_ENV}/lib/libstdc++.so.6 \
              ${CONDA_DIR}/envs/${HABITAT_CONDA_ENV}/lib/libstdc++.so.6.old ; \
           ln -s /usr/lib/x86_64-linux-gnu/libstdc++.so.6 \
                 ${CONDA_DIR}/envs/${HABITAT_CONDA_ENV}/lib/libstdc++.so.6 ; \
       fi \
    && ldconfig \
    && conda clean -afy \
    && chmod -R a+rwX ${CONDA_DIR}/envs/${HABITAT_CONDA_ENV}

ARG HABITAT_VERSION=v0.3.3
# Long and memory hungry; lower this if the build host OOMs.
ARG CMAKE_BUILD_PARALLEL_LEVEL=8
ENV CMAKE_BUILD_PARALLEL_LEVEL=${CMAKE_BUILD_PARALLEL_LEVEL}
RUN git clone --branch ${HABITAT_VERSION} --depth 1 --recursive \
        https://github.com/facebookresearch/habitat-sim.git /opt/habitat-sim \
    && cd /opt/habitat-sim \
    && conda run -n ${HABITAT_CONDA_ENV} pip install -r requirements.txt \
    && conda run --no-capture-output -n ${HABITAT_CONDA_ENV} \
        python setup.py install --headless --with-cuda --bullet \
    && rm -rf /opt/habitat-sim/build \
    && chmod -R a+rwX ${CONDA_DIR}/envs/${HABITAT_CONDA_ENV}

RUN git clone --branch ${HABITAT_VERSION} --depth 1 \
        https://github.com/facebookresearch/habitat-lab.git /opt/habitat-lab \
    && cd /opt/habitat-lab \
    && conda run -n ${HABITAT_CONDA_ENV} pip install -e habitat-lab \
    && chmod -R a+rwX ${CONDA_DIR}/envs/${HABITAT_CONDA_ENV} /opt/habitat-lab


# =============================================================================
FROM habitat AS runtime
# Everything volatile. Edit freely: the habitat stage above stays cached.
# =============================================================================

ARG PYTHON_VERSION=3.10.16
ARG VLM_CONDA_ENV=longnav
ENV VLM_CONDA_ENV=${VLM_CONDA_ENV}

RUN conda create -y -n ${VLM_CONDA_ENV} python=${PYTHON_VERSION} \
    && conda install -y -n ${VLM_CONDA_ENV} -c conda-forge \
        "libgcc-ng>=12" "libstdcxx-ng>=12" "libgomp>=12" \
    && conda run -n ${VLM_CONDA_ENV} python -m pip install --upgrade pip setuptools wheel \
    && conda run -n ${VLM_CONDA_ENV} pip install --upgrade urllib3 \
    && conda clean -afy \
    && chmod -R a+rwX ${CONDA_DIR}/envs/${VLM_CONDA_ENV}

# Fail the build loudly rather than shipping an image whose habitat_sim imports
# against the wrong numpy ABI.
RUN conda run -n ${HABITAT_CONDA_ENV} python -c \
        "import numpy; assert numpy.__version__ == '1.26.4', numpy.__version__" \
    && conda run -n ${HABITAT_CONDA_ENV} python -c \
        "import habitat_sim, habitat; print('habitat_sim', habitat_sim.__version__, \
'cuda', habitat_sim.cuda_enabled, 'bullet', habitat_sim.built_with_bullet)"

# flash-attn is deliberately not built here. Its setup.py imports torch, and no
# python packages are installed at image build time, so it belongs to
# docker/install.sh alongside the [vlm] extra that provides torch:
#   INSTALL_FLASH_ATTN=1 docker/install.sh

# No VOLUME declarations on purpose: anonymous volumes would shadow the bind
# mounts. Mount points are /workspace (repo), /workspace/data (datasets) and
# /home/longnav/.cache (HF/torch caches) — see docker-compose.yml.
# Put the VLM env first on PATH. `docker compose exec` does NOT run the
# ENTRYPOINT, so without this a plain `exec longnav python ...` would land on
# the conda *base* interpreter, where nothing is installed. With it, exec
# behaves the same as an interactive shell. Ray still reaches the habitat env by
# name, and `conda run -n vln ...` still works for one-off habitat commands.
ENV PATH=${CONDA_DIR}/envs/${VLM_CONDA_ENV}/bin:${PATH}

ENV HF_HOME=/home/longnav/.cache/huggingface \
    TORCH_HOME=/home/longnav/.cache/torch \
    HISTFILE=/home/longnav/.cache/bash_history \
    MAGNUM_LOG=quiet \
    HABITAT_SIM_LOG=quiet \
    GLOG_minloglevel=2

# So that `docker compose exec ... bash` lands in an activated env.
RUN conda init bash \
    && echo "conda activate \${LONGNAV_ENV:-${VLM_CONDA_ENV}}" >> /etc/skel/.bashrc \
    && echo "conda activate \${LONGNAV_ENV:-${VLM_CONDA_ENV}}" >> /home/longnav/.bashrc \
    && echo "conda activate \${LONGNAV_ENV:-${VLM_CONDA_ENV}}" >> /root/.bashrc \
    && mkdir -p /home/longnav/.cache \
    && chmod -R 777 /home/longnav

COPY docker/entrypoint.sh /usr/local/bin/entrypoint.sh
RUN chmod +x /usr/local/bin/entrypoint.sh

# Python dependencies are not installed here. The project is pip-installable and
# the repo is bind-mounted rather than baked in, so `docker/install.sh` runs once
# against the mount after the image is built:
#
#   docker compose up -d
#   docker compose exec longnav bash docker/install.sh
#
# It must run in a container you keep: the installs land in that container's
# writable layer, so `docker compose run --rm` would discard them on exit.
# Shipped into the image as `longnav-install` so it can also be run from a
# container with no repo mounted yet.
COPY docker/install.sh /usr/local/bin/longnav-install
RUN chmod +x /usr/local/bin/longnav-install

ARG install_packages=false
RUN if [ "${install_packages}" = "true" ]; then \
        /usr/local/bin/longnav-install --repo /workspace; \
    fi
# Only the default when compose does not pass `user:`. Overriding it costs a
# container restart, never a rebuild.
USER longnav
WORKDIR /workspace

ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
CMD ["bash"]

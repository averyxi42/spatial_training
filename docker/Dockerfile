# LongNav-R1 reproduction image.
#
# The hard part of this project is habitat-sim: the conda package for 0.3.3 is
# only published against python 3.9, so we build it from source inside a
# python 3.10.16 conda env ("vln") using --headless --with-cuda --bullet.
#
# Build:
#   DOCKER_BUILDKIT=1 docker build -f docker/Dockerfile \
#     --build-arg USER_UID=$(id -u) --build-arg USER_GID=$(id -g) \
#     -t longnav:dev .
#
# The CUDA *devel* image is required: --with-cuda needs nvcc and the CUDA
# headers at build time.
ARG CUDA_IMAGE=docker.io/library/nvidia/cuda:12.9.2-cudnn-devel-ubuntu22.04
FROM ${CUDA_IMAGE}

ARG PYTHON_VERSION=3.10.16
ARG CONDA_ENV=vln
ARG HABITAT_VERSION=v0.3.3
# Match the host user so files written into bind-mounted volumes stay writable.
ARG USERNAME=longnav
ARG USER_UID=1000
ARG USER_GID=1000

ENV DEBIAN_FRONTEND=noninteractive \
    TZ=Etc/UTC \
    CONDA_DIR=/opt/conda \
    CONDA_ENV=${CONDA_ENV} \
    CUDA_HOME=/usr/local/cuda \
    PYTHONUNBUFFERED=1

# --- system deps -------------------------------------------------------------
# The list below is habitat-sim's documented build dependency set plus the
# usual EGL/OpenMP bits needed for headless GPU rendering.
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
        ffmpeg \
        libjpeg-dev \
        libpng-dev \
        libglm-dev \
        libomp-dev \
        libhdf5-dev \
        libegl1-mesa-dev \
        libgl1-mesa-dev \
        libgles2-mesa-dev \
        libx11-dev \
        xorg-dev \
        freeglut3-dev \
        libglfw3-dev \
        libbullet-dev \
    && rm -rf /var/lib/apt/lists/*

# Headless EGL rendering needs the graphics capability, not just compute.
ENV NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics

# --- conda -------------------------------------------------------------------
# Miniforge (conda-forge channel by default) avoids the Anaconda default-channel
# terms-of-service prompt that breaks unattended builds.
RUN curl -fsSL -o /tmp/miniforge.sh \
        "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh" \
    && bash /tmp/miniforge.sh -b -p ${CONDA_DIR} \
    && rm /tmp/miniforge.sh \
    && ${CONDA_DIR}/bin/conda clean -afy

ENV PATH=${CONDA_DIR}/bin:${PATH}

# python 3.10.16 is the version the rest of the project pins; the stock habitat
# conda package would drag the env back to 3.9.
RUN conda create -y -n ${CONDA_ENV} python=${PYTHON_VERSION} cmake=3.27 \
    && conda run -n ${CONDA_ENV} python -m pip install --upgrade pip setuptools wheel \
    && conda clean -afy

# Make `vln` the env for every shell, interactive or not.
ENV PATH=${CONDA_DIR}/envs/${CONDA_ENV}/bin:${PATH} \
    CONDA_DEFAULT_ENV=${CONDA_ENV}

# --- habitat-sim (source build) ---------------------------------------------
# numpy is pinned below 2 for the whole image: habitat-sim's magnum bindings are
# compiled against the numpy C API present at build time, and letting a later
# pip install pull numpy 2.x in silently breaks `import habitat_sim`.
RUN conda run -n ${CONDA_ENV} pip install "numpy<2"

WORKDIR /opt
RUN git clone --branch ${HABITAT_VERSION} --depth 1 --recursive \
        https://github.com/facebookresearch/habitat-sim.git /opt/habitat-sim

# Build is long (~20-40 min) and memory hungry; cap parallelism if the build
# host OOMs by overriding CMAKE_BUILD_PARALLEL_LEVEL.
ARG CMAKE_BUILD_PARALLEL_LEVEL=8
ENV CMAKE_BUILD_PARALLEL_LEVEL=${CMAKE_BUILD_PARALLEL_LEVEL}
RUN cd /opt/habitat-sim \
    && conda run -n ${CONDA_ENV} pip install -r requirements.txt \
    && conda run --no-capture-output -n ${CONDA_ENV} \
        python setup.py install --headless --with-cuda --bullet \
    && rm -rf /opt/habitat-sim/build

# --- habitat-lab -------------------------------------------------------------
RUN git clone --branch ${HABITAT_VERSION} --depth 1 \
        https://github.com/facebookresearch/habitat-lab.git /opt/habitat-lab \
    && cd /opt/habitat-lab \
    && conda run -n ${CONDA_ENV} pip install -e habitat-lab

# --- torch + project deps ----------------------------------------------------
# Installed after habitat so the habitat build never sees a half-resolved env.
# Kept in sync with pyproject.toml (torch 2.8.0 / cu128).
RUN conda run -n ${CONDA_ENV} pip install \
        torch==2.8.0 torchvision torchaudio \
        --index-url https://download.pytorch.org/whl/cu128 \
    && conda run -n ${CONDA_ENV} pip install \
        "numpy<2" \
        "transformers<5.0" \
        "ray[default]==2.53.0" \
        datasets \
        hydra-core \
        "fastapi[standard]" \
        pillow \
        opencv-python-headless \
        peft \
        wandb \
        tensordict \
        huggingface-hub \
        codetiming \
    && conda run -n ${CONDA_ENV} pip cache purge || true

# flash-attn is a separate, very slow build; leave it opt-in.
ARG INSTALL_FLASH_ATTN=false
RUN if [ "${INSTALL_FLASH_ATTN}" = "true" ]; then \
        conda run --no-capture-output -n ${CONDA_ENV} \
            pip install flash-attn --no-build-isolation ; \
    fi

# --- user + mount points -----------------------------------------------------
# Reuse the group/uid if the base image already claims it (e.g. uid 1000), and
# rename that account to ${USERNAME} so the paths below stay predictable.
RUN if ! getent group ${USER_GID} >/dev/null; then groupadd -g ${USER_GID} ${USERNAME}; fi \
    && if getent passwd ${USER_UID} >/dev/null; then \
           usermod -l ${USERNAME} -d /home/${USERNAME} -m -s /bin/bash \
               "$(getent passwd ${USER_UID} | cut -d: -f1)"; \
       else \
           useradd -m -u ${USER_UID} -g ${USER_GID} -s /bin/bash ${USERNAME}; \
       fi \
    && mkdir -p /workspace /workspace/data /home/${USERNAME}/.cache \
    && chown -R ${USER_UID}:${USER_GID} /workspace /home/${USERNAME} /opt/habitat-lab \
    && chown -R ${USER_UID}:${USER_GID} ${CONDA_DIR}/envs/${CONDA_ENV}

# No VOLUME declarations on purpose: anonymous volumes would shadow the bind
# mounts. Mount points are /workspace (repo), /workspace/data (datasets) and
# /home/longnav/.cache (HF/torch caches) — see docker-compose.yml.

ENV HF_HOME=/home/${USERNAME}/.cache/huggingface \
    TORCH_HOME=/home/${USERNAME}/.cache/torch \
    MAGNUM_LOG=quiet \
    HABITAT_SIM_LOG=quiet \
    GLOG_minloglevel=2 \
    PYTHONPATH=/workspace/src

# conda init so `docker exec -it ... bash` lands in the vln env.
RUN conda init bash \
    && echo "conda activate ${CONDA_ENV}" >> /etc/skel/.bashrc \
    && echo "conda activate ${CONDA_ENV}" >> /home/${USERNAME}/.bashrc \
    && echo "conda activate ${CONDA_ENV}" >> /root/.bashrc

COPY --chown=${USER_UID}:${USER_GID} docker/entrypoint.sh /usr/local/bin/entrypoint.sh
RUN chmod +x /usr/local/bin/entrypoint.sh

USER ${USERNAME}
WORKDIR /workspace

ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
CMD ["bash"]

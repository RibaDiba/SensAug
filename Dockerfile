# sensaug - Sensitivity-Informed Augmentation for Robust Segmentation
#
# Matches the environment documented in README.md / CLAUDE.md:
#   Python 3.10, PyTorch 2.0.0, CUDA 11.7, MMSegmentation latest, mmcv 2.1.x.
#
# Build:    docker build -t sensaug:latest .
# Run:      docker run --gpus all -it --rm \
#               -v "$PWD":/workspace \
#               -v /path/to/data:/workspace/data \
#               sensaug bash
#
# NOTE: CUDA images require an NVIDIA GPU host (Linux). They cannot run natively
# on an Apple-Silicon Mac; build there only for pushing to a Linux host/registry.

ARG TORCH_VERSION=2.0.0
ARG CUDA_VERSION=11.7
FROM pytorch/pytorch:${TORCH_VERSION}-cuda${CUDA_VERSION}-cudnn8-devel

USER root

# System deps:
#  - build tools for any on-the-fly compiled mmcv/extension code
#  - libgl1/libglib2.0-0 for opencv-python's runtime libs (libGL.so.1)
#  - imagemagick + libmagickwand-dev: sensaug/dataset/imagenet_c.py imports
#    `wand` at module level, and Wand hard-fails at import without MagickWand.
RUN apt-get update && apt-get install -y --no-install-recommends \
        git \
        ninja-build \
        build-essential \
        curl \
        wget \
        libgl1 \
        libglib2.0-0 \
        libsm6 \
        libxext6 \
        libxrender1 \
        imagemagick \
        libmagickwand-dev \
    && rm -rf /var/lib/apt/lists/*

# mmcv with prebuilt CUDA ops for torch 2.0 + cu117 (avoids a source build).
# The `-f` index is mmcv's official prebuilt-wheel repo; the matching
# {cuda}/{torch} URL is what pins the wheel to this image's toolchain.
RUN python -m pip install --no-cache-dir \
        mmcv==2.1.0 \
        -f https://download.openmmlab.com/mmcv/dist/cu117/torch2.0/index.html

# MMSeg core stack (latest, as on Della).
RUN python -m pip install --no-cache-dir \
        mmengine \
        mmsegmentation \
        mmpretrain

WORKDIR /workspace

# Project requirements first so source edits do not invalidate the dependency
# layer. torch/torchvision lines in requirements.txt are already satisfied by
# the base image (2.0.0) and are left untouched.
COPY requirements.txt .
RUN python -m pip install --no-cache-dir -r requirements.txt

# The package itself (setup.py installs nothing; only the source import matters).
COPY . .
RUN python -m pip install --no-cache-dir -e .

ENV PYTHONUNBUFFERED=1 \
    MPLBACKEND=Agg

# Default CMD: drop into a shell so training is launched the same way as the
# sbatch scripts (torchrun train.py ...).
CMD ["bash"]
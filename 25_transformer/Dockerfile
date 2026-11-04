# syntax=docker/dockerfile:1.7
#
# 25_transformer — two-stage neural TTS.
#
#   stage 1: VITS/GAN Generator + RVQ spec codec + RVQ pitch codec (1_*.py)
#   stage 2: CodeTransformer over the frozen stage-1 codes     (2_*.py)
#
# Image target: vast.ai.  It extends the official `vastai/pytorch` image
# instead of starting from bare `nvidia/cuda`, because that base already
# provides — and vast.ai hosts already have cached —
#
#   * torch / torchaudio built for the CUDA line of the tag.  This project
#     needs torch >= 2.9: `2_training.py` trains with `torch.optim.Muon`.
#   * the primary python venv at /venv/main (python 3.12).
#   * sshd + ssh-key propagation, the Instance Portal, Jupyter (8080) and
#     TensorBoard (6006), so no start-up plumbing is needed here.
#
# Only the extra system/Python dependencies and this repository are added, so
# a pull on vast.ai is a few GB instead of ~10 GB.
#
# ---------------------------------------------------------------------------
# Build (run from THIS directory — build context == this project):
#
#   docker build -t <dockerhub-user>/raraai-25-transformer:latest .
#
#   # bake the Qwen3-ASR + forced-aligner weights too (+~2.5 GB, only useful
#   # if the instance will run 1_preprocess.py / 2_preprocess.py itself):
#   docker build --build-arg PREFETCH_QWEN=1 -t <user>/raraai-25-transformer:latest .
#
#   # host driver older than 580 (CUDA 13)?  use the CUDA 12 line instead:
#   docker build --build-arg VAST_BASE=vastai/pytorch:cuda-12.9.2-auto -t ... .
#
#   docker push <dockerhub-user>/raraai-25-transformer:latest
#
# ---------------------------------------------------------------------------
# On vast.ai: select the image, then
#
#   cd /workspace/25_transformer
#   python 2_training.py --config configs/transformer.yaml
#
# See DOCKER.md for the vast.ai specifics (volumes, ports, env vars).
# ---------------------------------------------------------------------------

# CUDA 13.0 line: matches the dev environment (torch 2.x + cu130, python 3.12).
# `-auto` = "latest torch published for this CUDA version", so the torch build
# and its companion packages (torchvision/torchaudio/triton/...) always agree.
ARG VAST_BASE=vastai/pytorch:cuda-13.0.3-auto
FROM ${VAST_BASE}

LABEL org.opencontainers.image.title="raraai 25_transformer" \
      org.opencontainers.image.description="Two-stage neural TTS: VITS/RVQ stage-1 Generator + stage-2 CodeTransformer" \
      org.opencontainers.image.source="https://github.com/effusiveperiscope/raraai"

SHELL ["/bin/bash", "-o", "pipefail", "-c"]
ARG DEBIAN_FRONTEND=noninteractive

# --------------------------------------------------------------------------
# Environment
# --------------------------------------------------------------------------
# HF_HOME/NLTK_DATA/TORCH_HOME live outside /workspace so the (large) model
# caches are not copied into the workspace volume on first boot.
# TENSORBOARD_LOG_DIR is honoured by the base image's TensorBoard service.
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    TOKENIZERS_PARALLELISM=false \
    HF_HOME=/opt/hf-cache \
    NLTK_DATA=/opt/nltk_data \
    TORCH_HOME=/opt/torch-cache \
    TENSORBOARD_LOG_DIR=/workspace/25_transformer/logs \
    APP_DIR=/opt/workspace-internal/25_transformer


# --------------------------------------------------------------------------
# System packages
# --------------------------------------------------------------------------
# build-essential: pyworld has no cp312 linux wheel and is built from sdist.
# ffmpeg + libsndfile1 + sox: librosa / soundfile / the `sox` bindings used by
# qwen-asr.  The rest is the shell tooling one wants on a rented box.
RUN apt-get update && apt-get install -y --no-install-recommends \
        aria2 \
        build-essential \
        ca-certificates \
        curl \
        ffmpeg \
        git \
        git-lfs \
        htop \
        less \
        libsndfile1 \
        nano \
        rsync \
        sox \
        libsox-fmt-all \
        tmux \
        tzdata \
        unzip \
        vim \
        wget \
    && rm -rf /var/lib/apt/lists/* \
    && mkdir -p "${HF_HOME}" "${NLTK_DATA}" "${TORCH_HOME}"

# --------------------------------------------------------------------------
# Python dependencies
# --------------------------------------------------------------------------
# Installed into the base image's primary venv (/venv/main); torch itself is
# already there — see requirements.txt for why it is not pinned there.
WORKDIR /opt/build
COPY requirements.txt ./
# torchaudio is not part of the pytorch derivative; the vendored code imports
# it (modeling/pitch.py, modeling/vits_decoder/med.py).  Installed from the
# same CUDA wheel index as the base torch build, with --no-deps so it cannot
# swap torch out (torchaudio declares no torch requirement of its own).
ARG TORCH_BACKEND=cu130
RUN --mount=type=cache,target=/root/.cache/pip \
    . /venv/main/bin/activate \
    && pip install --no-deps \
         --index-url "https://download.pytorch.org/whl/${TORCH_BACKEND}" \
         torchaudio==2.11.0
RUN --mount=type=cache,target=/root/.cache/pip \
    . /venv/main/bin/activate \
    && pip install --upgrade pip setuptools wheel \
    && pip install -r requirements.txt \
    && pip check

# The base image supplies torch; make sure it survived dependency resolution
# (a PyPI torch would pull its own nvidia-* wheels and shadow the CUDA build
# the base image was matched to) and that it is new enough for
# `torch.optim.Muon` used by 2_training.py.
RUN . /venv/main/bin/activate \
    && python -c "import torch, torchaudio; \
assert torch.backends.cuda.is_built(), 'base torch has no CUDA support'; \
assert tuple(int(p) for p in torch.__version__.split('+')[0].split('.')[:2]) >= (2, 9), \
    'torch ' + torch.__version__ + ' is too old: torch.optim.Muon needs >= 2.9'; \
print('torch', torch.__version__, 'cuda', torch.version.cuda, \
      '| torchaudio', torchaudio.__version__, '| Muon', bool(torch.optim.Muon))"

# --------------------------------------------------------------------------
# Model / tokenizer / NLTK assets
# --------------------------------------------------------------------------
# Small assets are always baked (RMVPE ~100 MB, g2p tokenizer, NLTK tagger):
# every preprocessing and test path wants them, and an offline instance would
# otherwise fail at the first `import horsephonemizer`.
# The Qwen3-ASR + forced-aligner weights (~2.5 GB) are opt-in via PREFETCH_QWEN.
# Copied separately from the source tree so editing code does not invalidate
# this (expensive) layer.
COPY docker/prefetch_models.py /opt/build/prefetch_models.py
ARG PREFETCH_QWEN=0
RUN . /venv/main/bin/activate \
    && mkdir -p "${HF_HOME}" "${NLTK_DATA}" "${TORCH_HOME}" \
    && chmod 0777 "${HF_HOME}" "${NLTK_DATA}" "${TORCH_HOME}" \
    && if [ "${PREFETCH_QWEN}" = "1" ]; then \
           python /opt/build/prefetch_models.py --qwen; \
       else \
           python /opt/build/prefetch_models.py; \
       fi

# --------------------------------------------------------------------------
# Project source
# --------------------------------------------------------------------------
# Goes into /opt/workspace-internal, which the vast.ai base image copies into
# $WORKSPACE (default /workspace) on first boot — that is what makes the code
# survive a workspace volume mount.  .dockerignore keeps datasets, checkpoints,
# logs and audio out of the image.
COPY . "${APP_DIR}"
# Modes are NOT fixed here: `chmod -R` over the (large) model cache would copy
# every file up into a new layer (~250 MB wasted).  Writability for the
# non-root `user` account is applied at boot instead
# (docker/boot/90-25-transformer-perms.sh).
RUN find "${APP_DIR}" -name '__pycache__' -type d -prune -exec rm -rf {} + \
    && find "${APP_DIR}" -name '*.pyc' -delete

# The first-boot workspace copy runs as root and does not preserve modes, so a
# late boot stage re-applies them (see the script for why this is needed).
COPY docker/boot/90-25-transformer-perms.sh /etc/vast_boot.d/
RUN chmod +x /etc/vast_boot.d/90-25-transformer-perms.sh

# Runtime cwd.
# NOTE: `HorsePhonemizer` opens `new_horsewords.clean` by relative path and
# every script resolves `configs/…`, `data/…`, `checkpoints/…` against the cwd,
# so the project must be run from its own directory (`/workspace/25_transformer`).
# /workspace itself stays empty on purpose: the vast.ai boot stage only copies
# /opt/workspace-internal/<item> into $WORKSPACE when that target does not exist
# yet (/etc/vast_boot.d/36-sync-workspace.sh), so pre-creating
# /workspace/25_transformer in the image would suppress the copy.
WORKDIR /workspace

# ssh (base image), Instance Portal, Jupyter, TensorBoard
EXPOSE 22 1111 8080 6006

# No ENTRYPOINT/CMD: the base image's own entrypoint/supervisor (sshd, portal,
# Jupyter, TensorBoard) keeps working, and vast.ai supplies the launch command.

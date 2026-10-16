# GIB-15K reproducibility image.
#
# CUDA base with every inference dependency pinned, so a reviewer can reproduce
# the open-model tier exactly. The image deliberately does NOT bake in the
# dataset: COCO and Visual Genome carry their own licences, so the container
# ships the transformation scripts and regenerates the corpus from sources the
# user obtains under those terms.

FROM nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    # Deterministic cuBLAS reductions; without this, attention captures and
    # teacher-forced logprobs are not bitwise reproducible across runs.
    CUBLAS_WORKSPACE_CONFIG=:4096:8 \
    HF_HOME=/workspace/.hf

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.11 python3.11-venv python3-pip \
        git curl unzip ca-certificates \
        libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.11 1 \
    && python -m pip install --upgrade pip setuptools wheel

WORKDIR /workspace

COPY requirements.txt requirements-heavy.txt ./
RUN pip install -r requirements.txt -r requirements-heavy.txt

COPY . .
RUN pip install -e .

# Fail fast if the image is broken, rather than three hours into a campaign.
RUN python -m pytest tests/ -q

CMD ["make", "help"]

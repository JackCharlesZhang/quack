# All CUDA runtime libs come from torch's pip wheels (under
# /usr/local/lib/python3.12/dist-packages/nvidia/), and ptxas comes from triton.
# A plain ubuntu base is ~3-7 GB smaller than the nvidia/cuda:*-runtime/-devel bases.
FROM ubuntu:24.04

# Defaults match build.sh's cu12.9 invocation so a bare `docker build` (no
# --build-arg) produces a working image on our cu12.9 runners. Pass
# --build-arg TORCH_CUDA=cu130 / QUACK_EXTRAS=cu13,dev for the cu13.2 variant.
ARG TORCH_CUDA=cu129
ARG QUACK_EXTRAS=dev

ENV PATH=/root/.local/bin:$PATH
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update -y && apt-get install -y \
        python3 python3.12 python3.12-dev python3.12-venv \
        curl git ca-certificates gcc && \
    rm -rf /var/lib/apt/lists/*

RUN curl -LsSf https://astral.sh/uv/install.sh | sh

# Install latest stable torch from the matching cu wheel index. Torch's wheel
# brings cublas/cudnn/cudart/etc. and triton brings ptxas — together these are
# everything cute-dsl needs to JIT. cusolver isn't linked from libtorch_cuda
# (we don't use torch.linalg here), so we drop it in the same RUN to shrink
# the layer (~470 MB on cu12.9, ~235 MB on cu13.2).
#
# Note: build.sh pins TORCH_CUDA=cu129 for both image variants. Our CI host
# runs driver 575.x (CUDA 12.9 max), which can't initialize cu130 torch
# ("driver too old, found 12090"). cute-dsl[cu13] still JIT-compiles fine
# under a cu12.9 driver, so the cu13.2 variant just pairs cu13 cute-dsl with
# cu129 torch. Move back to cu130 once the runner driver is upgraded to >= 580.
RUN uv pip install --system --break-system-packages --no-cache \
        "torch==2.11.*" \
        --index-url https://download.pytorch.org/whl/${TORCH_CUDA} && \
    rm -rf /usr/local/lib/python3.12/dist-packages/nvidia/cusolver \
           /usr/local/lib/python3.12/dist-packages/nvidia/cu13/lib/libcusolver*.so.*

# Install quack-kernels' transitive dependencies. The package itself is
# overwritten at run time by an editable install from the mounted repo.
COPY pyproject.toml /tmp/quack/pyproject.toml
COPY quack/ /tmp/quack/quack/
RUN uv pip install --system --break-system-packages --no-cache "/tmp/quack[${QUACK_EXTRAS}]"

CMD ["/bin/bash"]

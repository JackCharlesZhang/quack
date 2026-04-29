#!/bin/bash
# Build quack CI images for CUDA 12.9 and CUDA 13.2.
# Usage: build.sh [cu129|cu132]   (default: build both)
set -e

DATE=$(date +%y.%m.%d)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(git -C "$SCRIPT_DIR" rev-parse --show-toplevel 2>/dev/null || cd "$SCRIPT_DIR/../../.." && pwd)"

build_image() {
    local tag=$1 torch_cuda=$2 quack_extras=$3
    local image="quack-kernels:${tag}-${DATE}"
    echo "=== Building $image (torch $torch_cuda, extras [$quack_extras]) ==="
    docker build \
        --build-arg "TORCH_CUDA=$torch_cuda" \
        --build-arg "QUACK_EXTRAS=$quack_extras" \
        -t "$image" \
        -f "$SCRIPT_DIR/Dockerfile" \
        "$REPO_ROOT"
    echo "Done: $image"
}

# Both variants pin torch to cu129 wheels: our CI host runs driver 575.x (CUDA
# 12.9 max), which can't initialize cu130 torch ("driver too old, found 12090").
# cute-dsl[cu13] still JIT-compiles fine to the local SM under a cu12.9 driver,
# so the cu13.2 variant exercises the cu13 cute-dsl bundle while staying
# runnable. Switch back to cu130 for either variant once the runner driver is
# upgraded to >= 580.
case "${1:-all}" in
    cu129)  build_image cu12.9 cu129 dev ;;
    cu132)  build_image cu13.2 cu129 cu13,dev ;;
    all)    build_image cu12.9 cu129 dev
            build_image cu13.2 cu129 cu13,dev ;;
    *)      echo "Unknown variant: $1 (expected cu129, cu132, or all)" >&2; exit 1 ;;
esac

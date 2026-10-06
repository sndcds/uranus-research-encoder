#!/bin/sh
# Explicit isolated experiment only. No existing service/container is managed.
set -eu
base=/home/awendelk/v5-merged-onnx-20261006
mode=$1
threads=${2:-8}
runtime=/home/awendelk/uranus-research-encoder/.venv
if [ "$mode" = smoke ]; then
    runtime=$base/runtime-onnx
    set -- /encoder/scripts/smoke_v5_onnx.py --graph-dir /graph \
        --plan-dir /encoder/validation/jina-v5-merged-v1 \
        --reference-vectors /output/native.npz --output /output/http-contract-onnx-only.json
else
    set -- /encoder/scripts/benchmark_v5_onnx.py "$mode" --threads "$threads" \
        --model-root /v5cache --graph-dir /graph \
        --plan-dir /encoder/validation/jina-v5-merged-v1 \
        --parity-report /output/parity.json --contract-report /output/http-contract-onnx-only.json \
        --output "/output/performance-$mode-$threads.json"
    if [ "$threads" != 8 ]; then set -- "$@" --workload query; fi
fi
sudo -n docker run --rm --name "encoder-v5-merged-$mode-$threads-final-20261006" \
    --network none --read-only --cpus 8 --memory 6g --memory-swap 8g \
    --pids-limit 256 --user "$(id -u):$(id -g)" --tmpfs /tmp:rw,size=512m \
    -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 -e PYTHONDONTWRITEBYTECODE=1 \
    -e TORCHINDUCTOR_CACHE_DIR=/tmp/torchinductor -e USER=awendelk \
    -e PYTHONPATH=/encoder/src -e TOKENIZERS_PARALLELISM=false \
    -v "$runtime":/bench-venv:ro \
    -v /home/awendelk/.local/share/uv/python:/home/awendelk/.local/share/uv/python:ro \
    -v "$base/source-7704b0c":/encoder:ro -v "$base/output":/output:rw \
    -v "$base/export-v2/graph":/graph:ro \
    -v /srv/uranus-research-encoder/v5-stage/hf/hub:/v5cache:ro \
    --workdir /encoder --entrypoint /bench-venv/bin/python \
    sha256:6834c50c52bd7e60f6a6d25772102198dd9638b3dc33859e6bf05df9a67b2c1b "$@"

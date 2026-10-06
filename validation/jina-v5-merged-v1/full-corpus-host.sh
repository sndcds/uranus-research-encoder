#!/bin/sh
# New experiment containers only. No ports, network, production or Qdrant access.
set -eu
base=/home/awendelk/v5-merged-onnx-20261006
run=$base/full-corpus-f88cf45
mkdir "$run/source" "$run/output"
tar -xzf "$run/source.tar.gz" -C "$run/source"
for backend in native onnx; do
    sudo -n docker run --rm --name "encoder-v5-full-corpus-$backend-f88cf45" \
        --network none --read-only --cpus 8 --memory 6g --memory-swap 8g \
        --pids-limit 256 --user "$(id -u):$(id -g)" --tmpfs /tmp:rw,size=512m \
        -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 -e PYTHONDONTWRITEBYTECODE=1 \
        -e HF_HUB_DISABLE_TELEMETRY=1 -e PYTHONPATH=/encoder/src \
        -e TOKENIZERS_PARALLELISM=false -e USER=awendelk \
        -v /home/awendelk/uranus-research-encoder/.venv:/bench-venv:ro \
        -v /home/awendelk/.local/share/uv/python:/home/awendelk/.local/share/uv/python:ro \
        -v "$run/source":/encoder:ro -v "$run/output":/output:rw \
        -v "$base/export-v2/graph":/graph:ro \
        -v /srv/uranus-research-encoder/v5-stage/hf/hub:/v5cache:ro \
        --workdir /encoder --entrypoint /bench-venv/bin/python \
        sha256:6834c50c52bd7e60f6a6d25772102198dd9638b3dc33859e6bf05df9a67b2c1b \
        /encoder/scripts/embed_v5_full_corpus.py "$backend" \
        --plan-dir /encoder/validation/jina-v5-merged-v1 --model-root /v5cache \
        --graph-dir /graph --output-dir /output
done
printf 'FULL_CORPUS_EMBEDDING_COMPLETE\n'

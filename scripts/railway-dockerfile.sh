#!/bin/sh
# Regenerate Dockerfile.railway from upstream's Dockerfile after every upstream sync.
# Railway rejects BuildKit cache mounts without an id; dropping them only costs build time.
set -eu
cd "$(dirname "$0")/.."
sed -E 's#--mount=type=cache,target=/root/.cache/uv ##' Dockerfile > Dockerfile.railway
echo "Dockerfile.railway regenerated from Dockerfile"

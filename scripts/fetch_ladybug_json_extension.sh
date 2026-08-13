#!/usr/bin/env bash
# Fetch official Ladybug JSON extension binaries from the GHCR extension-repo
# image and place them in cognee_db_workers/ladybug_extensions/, where
# _kuzu_helpers.load_json_extension() picks them up (see the README there).
#
# ghcr.io/ladybugdb/extension-repo is the origin content behind
# extension.ladybugdb.com (an nginx image whose html root is the extension
# file tree), so these are the same official binaries INSTALL would download —
# available even when that server is unreachable.
#
# Usage: scripts/fetch_ladybug_json_extension.sh <ext-version> [platform ...]
#   e.g. scripts/fetch_ladybug_json_extension.sh v0.18.1 linux_amd64 linux_arm64
#
# The version directory must match what the installed ladybug requests
# (see _EXTENSION_REPO_VERSIONS in cognee_db_workers/_kuzu_helpers.py).
set -euo pipefail

VERSION="${1:?usage: $0 <ext-version e.g. v0.18.1> [platform ...]}"
shift
PLATFORMS=("${@:-linux_amd64}")
if [ $# -eq 0 ]; then
  PLATFORMS=(linux_amd64 linux_arm64)
fi

IMAGE="ghcr.io/ladybugdb/extension-repo:latest"
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
OUT_BASE="$REPO_ROOT/cognee_db_workers/ladybug_extensions"

docker pull -q "$IMAGE"
CONTAINER="$(docker create "$IMAGE")"
trap 'docker rm "$CONTAINER" >/dev/null' EXIT

for platform in "${PLATFORMS[@]}"; do
  out_dir="$OUT_BASE/$VERSION/$platform"
  mkdir -p "$out_dir"
  docker cp -q \
    "$CONTAINER:/usr/share/nginx/html/$VERSION/$platform/json/libjson.lbug_extension" \
    "$out_dir/libjson.lbug_extension"
  ls -l "$out_dir/libjson.lbug_extension"
done

echo "Done. Verify with an offline LOAD before shipping (see ladybug_extensions/README.md)."

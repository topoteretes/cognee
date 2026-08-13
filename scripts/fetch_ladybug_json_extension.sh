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
# Usage:
#   scripts/fetch_ladybug_json_extension.sh
#       Fetch every extension version the ladybug constraint in pyproject.toml
#       supports (derived via scripts/ladybug_extension_versions.py — the
#       constraint is the source of truth), for all five platforms.
#   scripts/fetch_ladybug_json_extension.sh <ext-version> [platform ...]
#       Fetch one explicit version, e.g. for a Docker image build:
#       scripts/fetch_ladybug_json_extension.sh v0.18.1 linux_amd64 linux_arm64
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
OUT_BASE="$REPO_ROOT/cognee_db_workers/ladybug_extensions"
ALL_PLATFORMS=(linux_amd64 linux_arm64 osx_amd64 osx_arm64 win_amd64)

if [ $# -eq 0 ]; then
  # shellcheck disable=SC2207  # dir names never contain whitespace
  VERSIONS=($(python3 "$REPO_ROOT/scripts/ladybug_extension_versions.py"))
  PLATFORMS=("${ALL_PLATFORMS[@]}")
else
  VERSIONS=("$1")
  shift
  if [ $# -eq 0 ]; then
    PLATFORMS=("${ALL_PLATFORMS[@]}")
  else
    PLATFORMS=("$@")
  fi
fi

IMAGE="ghcr.io/ladybugdb/extension-repo:latest"
docker pull -q "$IMAGE"
CONTAINER="$(docker create "$IMAGE")"
trap 'docker rm "$CONTAINER" >/dev/null' EXIT

for version in "${VERSIONS[@]}"; do
  for platform in "${PLATFORMS[@]}"; do
    out_dir="$OUT_BASE/$version/$platform"
    mkdir -p "$out_dir"
    docker cp -q \
      "$CONTAINER:/usr/share/nginx/html/$version/$platform/json/libjson.lbug_extension" \
      "$out_dir/libjson.lbug_extension"
    ls -l "$out_dir/libjson.lbug_extension"
  done
done

echo "Done. Verify with an offline LOAD before shipping (see ladybug_extensions/README.md)."

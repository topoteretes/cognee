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
#       supports — the image's published version dirs are filtered through
#       scripts/ladybug_extension_versions.py, so the constraint is the source
#       of truth and no version list is maintained anywhere. All five
#       platforms.
#   scripts/fetch_ladybug_json_extension.sh <ext-version> [platform ...]
#       Fetch one explicit version, e.g. for a Docker image build:
#       scripts/fetch_ladybug_json_extension.sh v0.18.1 linux_amd64 linux_arm64
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
OUT_BASE="$REPO_ROOT/cognee_db_workers/ladybug_extensions"
ALL_PLATFORMS=(linux_amd64 linux_arm64 osx_amd64 osx_arm64 win_amd64)

# Pinned by digest, not tag: these binaries ship inside cognee's wheels and
# Docker image, so a compromised :latest tag must not be able to inject
# content into a release. The digest is the exact bytes the offline e2e
# (cognee/tests/e2e/bundled_extension/) validated. When a ladybug bump needs
# a version dir this digest predates, the resolver fails loudly ("no
# candidate extension dir satisfies") — refresh with:
#   docker pull ghcr.io/ladybugdb/extension-repo:latest \
#     && docker inspect --format '{{index .RepoDigests 0}}' ghcr.io/ladybugdb/extension-repo:latest
# and keep Dockerfile's ladybug-extensions stage on the same digest.
IMAGE="ghcr.io/ladybugdb/extension-repo@sha256:180c83fb190e9d6ef8d324850b192db26794ab7cb866a38813a45365f14bd46d"
docker pull -q "$IMAGE"

if [ $# -eq 0 ]; then
  # shellcheck disable=SC2207  # dir names never contain whitespace
  VERSIONS=($(docker run --rm --entrypoint ls "$IMAGE" /usr/share/nginx/html \
    | python3 "$REPO_ROOT/scripts/ladybug_extension_versions.py"))
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

#!/usr/bin/env bash
# Offline e2e for the bundled Ladybug JSON extension.
#
# Proves the whole chain on a real engine with the network disabled:
# pyproject constraint -> resolver -> fetched GHCR binaries -> probe ->
# LOAD by absolute path -> working JSON function. One container run per
# ladybug version pinned in uv.lock, so ladybug bumps are covered
# automatically — including the probe-format regression the mocked unit
# tests cannot see (see offline_extension_check.py).
#
# Needs: docker, python3. Network is used only to fetch binaries and build
# the image; the check itself runs under --network=none.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../../../.." && pwd)"
HERE="$(cd "$(dirname "$0")" && pwd)"

# Every ladybug version the lock pins (one per platform-marker constraint
# line; see cognee/tests/unit/test_ladybug_requirement.py).
VERSIONS=$(grep -A1 '^name = "ladybug"$' "$REPO_ROOT/uv.lock" \
  | sed -n 's/^version = "\(.*\)"$/\1/p' | sort -u)
[ -n "$VERSIONS" ] || { echo "no ladybug version found in uv.lock"; exit 1; }
echo "locked ladybug versions: $(echo "$VERSIONS" | tr '\n' ' ')"

# Bundle exactly what a release would: constraint-driven, all platforms.
"$REPO_ROOT/scripts/fetch_ladybug_json_extension.sh"

BUILD_DIR="$(mktemp -d)"
trap 'rm -rf "$BUILD_DIR"' EXIT
cp -R "$REPO_ROOT/cognee_db_workers" "$BUILD_DIR/cognee_db_workers"
rm -rf "$BUILD_DIR"/cognee_db_workers/__pycache__
cp "$HERE/offline_extension_check.py" "$BUILD_DIR/offline_extension_check.py"

for version in $VERSIONS; do
  echo "=== ladybug $version ==="
  cat > "$BUILD_DIR/Dockerfile" <<EOF
FROM python:3.11-slim
RUN pip install --no-cache-dir ladybug==$version
WORKDIR /app
COPY cognee_db_workers /app/cognee_db_workers
COPY offline_extension_check.py /app/offline_extension_check.py
CMD ["python", "/app/offline_extension_check.py"]
EOF
  image="cognee-bundled-ext-e2e:$version"
  docker build -q -t "$image" "$BUILD_DIR"
  docker run --rm --network=none "$image"
  docker rmi "$image" >/dev/null
done

echo "All locked ladybug versions passed the offline bundled-extension check."

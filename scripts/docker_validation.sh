#!/usr/bin/env bash
# Validate a published Cognee Docker image: metadata, boot, and /health.
# Trivy scanning is run separately in CI (see docker_validation_nightly.yml).
set -euo pipefail

IMAGE="${1:?image name required (e.g. cognee/cognee)}"
TAG="${2:?tag required (e.g. main)}"
MAX_BYTES="${3:?max image size in bytes required}"

FULL_IMAGE="${IMAGE}:${TAG}"
CONTAINER_NAME="cognee-validation-${RANDOM}"
HEALTH_URL="http://127.0.0.1:8000/health"
HEALTH_TIMEOUT_SECONDS="${HEALTH_TIMEOUT_SECONDS:-180}"

cleanup() {
  docker rm -f "$CONTAINER_NAME" >/dev/null 2>&1 || true
}
trap cleanup EXIT

echo "=== Pull ${FULL_IMAGE} ==="
docker pull "$FULL_IMAGE"

echo "=== Assert HEALTHCHECK in image metadata ==="
if [ "$(docker inspect --format='{{if .Config.Healthcheck}}yes{{end}}' "$FULL_IMAGE")" != "yes" ]; then
  echo "FAIL: ${FULL_IMAGE} has no HEALTHCHECK instruction"
  exit 1
fi

echo "=== Assert image size budget ==="
SIZE="$(docker image inspect --format='{{.Size}}' "$FULL_IMAGE")"
if (( SIZE > MAX_BYTES )); then
  echo "FAIL: ${FULL_IMAGE} size ${SIZE} bytes exceeds budget ${MAX_BYTES} bytes"
  exit 1
fi
echo "OK: ${FULL_IMAGE} size ${SIZE} bytes (budget ${MAX_BYTES} bytes)"

echo "=== Boot container ==="
RUN_ARGS=(
  -d
  --name "$CONTAINER_NAME"
  -p 8000:8000
  -e COGNEE_SKIP_CONNECTION_TEST=true
  -e ENV=dev
)
if [[ "$IMAGE" == *"cognee-mcp"* ]]; then
  RUN_ARGS+=(-e TRANSPORT_MODE=http)
fi
docker run "${RUN_ARGS[@]}" "$FULL_IMAGE"

if [[ "$IMAGE" == *"cognee-mcp"* ]]; then
  echo "=== Assert non-root uid 1000 (cognee-mcp only) ==="
  RUNNING_UID="$(docker exec "$CONTAINER_NAME" id -u)"
  if [[ "$RUNNING_UID" != "1000" ]]; then
    echo "FAIL: expected uid 1000, got ${RUNNING_UID}"
    docker logs "$CONTAINER_NAME" || true
    exit 1
  fi
  echo "OK: running as uid ${RUNNING_UID}"
fi

echo "=== Wait for ${HEALTH_URL} ==="
deadline=$((SECONDS + HEALTH_TIMEOUT_SECONDS))
until curl -sf "$HEALTH_URL" >/dev/null; do
  if (( SECONDS >= deadline )); then
    echo "FAIL: health check timed out after ${HEALTH_TIMEOUT_SECONDS}s"
    docker logs "$CONTAINER_NAME" || true
    exit 1
  fi
  sleep 2
done
echo "OK: ${HEALTH_URL} returned 200"

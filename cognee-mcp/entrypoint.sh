#!/bin/bash

set -e  # Exit on error
echo "Environment: $ENVIRONMENT"

# Install optional dependencies if EXTRAS is set
if [ -n "$EXTRAS" ]; then
    echo "Installing optional dependencies: $EXTRAS"

    # Get the cognee version that's currently installed
    COGNEE_VERSION=$(uv pip show cognee | grep "Version:" | awk '{print $2}')
    echo "Current cognee version: $COGNEE_VERSION"

    # Build the extras list for cognee
    IFS=',' read -ra EXTRA_ARRAY <<< "$EXTRAS"
    # Combine base extras from pyproject.toml with requested extras
    ALL_EXTRAS=""
    for extra in "${EXTRA_ARRAY[@]}"; do
        # Trim whitespace
        extra=$(echo "$extra" | xargs)
        # Add to extras list if not already present
        if [[ ! "$ALL_EXTRAS" =~ (^|,)"$extra"(,|$) ]]; then
            if [ -z "$ALL_EXTRAS" ]; then
                ALL_EXTRAS="$extra"
            else
                ALL_EXTRAS="$ALL_EXTRAS,$extra"
            fi
        fi
    done

    echo "Installing cognee with extras: $ALL_EXTRAS"
    echo "Running: uv pip install 'cognee[$ALL_EXTRAS]==$COGNEE_VERSION'"
    uv pip install "cognee[$ALL_EXTRAS]==$COGNEE_VERSION"

    # Verify installation
    echo ""
    echo "✓ Optional dependencies installation completed"
else
    echo "No optional dependencies specified"
fi

ARGS=("$@") #forward any args passed to the container at runtime

# Set default transport mode if not specified
TRANSPORT_MODE=${TRANSPORT_MODE:-"stdio"}
echo "Transport mode: $TRANSPORT_MODE"

# Set default ports if not specified

if [ "$TRANSPORT_MODE" != "stdio" ]; then
    HTTP_PORT=${HTTP_PORT:-8000}
    echo "HTTP port: $HTTP_PORT"
    ARGS+=("--host" "0.0.0.0" "--port" "$HTTP_PORT")
fi

echo "Starting Cognee MCP Server with transport mode: $TRANSPORT_MODE"
ARGS+=("--transport" "$TRANSPORT_MODE")

# Add startup delay to ensure DB is ready
sleep 2

# Build API arguments if API_URL is set
if [ -n "$API_URL" ]; then
    echo "API mode enabled: $API_URL"

    # Handle localhost in API_URL - convert to host-accessible address
    if [[ "$API_URL" =~ "localhost" || "$API_URL" =~ "127.0.0.1" ]]; then
        echo "⚠️  Warning: API_URL contains localhost/127.0.0.1"
        echo "   Original: $API_URL"

        # Resolve the best hostname to reach the host from inside the container.
        # Supports Docker Desktop, Colima / Lima, and plain Linux Docker.
        HOST_ADDR=""

        # 1. Docker Desktop (macOS, Windows, recent Linux Desktop)
        if getent hosts host.docker.internal >/dev/null 2>&1; then
            HOST_ADDR="host.docker.internal"
            echo "   ✓ Resolved via host.docker.internal (Docker Desktop)"

        # 2. Colima / Lima on macOS / Linux
        elif getent hosts host.lima.internal >/dev/null 2>&1; then
            HOST_ADDR="host.lima.internal"
            echo "   ✓ Resolved via host.lima.internal (Colima / Lima)"

        # 3. Fallback: default gateway IP (works on plain Linux Docker).
        # Read it from /proc/net/route, which exists in every Linux container with
        # no extra package. (The runtime image is debian-slim and does NOT ship the
        # `ip` binary / iproute2, so `ip route` would fail and leave this empty.)
        # awk only extracts the little-endian hex gateway of the default route
        # (destination 00000000) — no gawk-only strtonum, so it works under mawk too.
        else
            GATEWAY_HEX=$(awk '$2 == "00000000" { print $3; exit }' /proc/net/route 2>/dev/null || true)
            GATEWAY_IP=""
            if [ -n "$GATEWAY_HEX" ]; then
                # /proc/net/route stores the gateway little-endian, so reverse the
                # byte pairs to get dotted-decimal (e.g. 010011AC -> 172.17.0.1).
                GATEWAY_IP=$(printf "%d.%d.%d.%d" \
                    "0x${GATEWAY_HEX:6:2}" "0x${GATEWAY_HEX:4:2}" \
                    "0x${GATEWAY_HEX:2:2}" "0x${GATEWAY_HEX:0:2}" 2>/dev/null || true)
            fi
            if [ -n "$GATEWAY_IP" ]; then
                HOST_ADDR="$GATEWAY_IP"
                echo "   ✓ Resolved via default gateway IP: $GATEWAY_IP"
            else
                # Last resort: keep host.docker.internal and let the user know
                HOST_ADDR="host.docker.internal"
                echo "   ⚠️  Could not auto-detect host address; defaulting to host.docker.internal"
                echo "   If this fails, try one of:"
                echo "     - Colima: colima start --network-address"
                echo "     - Linux:  use --network host, or set API_URL=http://172.17.0.1:<port>"
            fi
        fi

        FIXED_API_URL=$(echo "$API_URL" | sed "s/localhost/$HOST_ADDR/g" | sed "s/127\.0\.0\.1/$HOST_ADDR/g")
        echo "   Converted to: $FIXED_API_URL"

        API_URL="$FIXED_API_URL"
    fi

    ARGS+=("--api-url" "$API_URL")
    if [ -n "$API_TOKEN" ]; then
        ARGS+=("--api-token" "$API_TOKEN")
    fi
else
    echo "Direct mode: Using local cognee instance"
fi

echo "calling cognee-mcp" "${ARGS[@]}"

if [ "$DEBUG" = "true" ] && { [ "$ENVIRONMENT" = "dev" ] || [ "$ENVIRONMENT" = "local" ]; }; then
    DEBUG_PORT=${DEBUG_PORT:-5678}
    echo "Running in debug mode"
    echo "Debug port: $DEBUG_PORT"
    echo "Waiting for the debugger to attach..."
    exec python -m debugpy --wait-for-client --listen 0.0.0.0:"$DEBUG_PORT" -m cognee-mcp "${ARGS[@]}"
else
    exec cognee-mcp "${ARGS[@]}"
fi

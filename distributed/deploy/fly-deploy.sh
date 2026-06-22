#!/usr/bin/env bash
# Cognee 1-click deployment to Fly.io
#
# Prerequisites: fly CLI installed (https://fly.io/docs/flyctl/install/)
#
# Usage:
#   export LLM_API_KEY=sk-xxx
#   bash distributed/deploy/fly-deploy.sh

set -euo pipefail

APP_NAME="${FLY_APP_NAME:-cognee}"
REGION="${FLY_REGION:-iad}"
VOLUME_SIZE="${FLY_VOLUME_SIZE:-10}"

echo "Deploying Cognee to Fly.io..."
echo "  App:    $APP_NAME"
echo "  Region: $REGION"

# Create app if it doesn't exist
if ! fly apps list --json | grep -q "\"$APP_NAME\""; then
    echo "Creating app..."
    fly apps create "$APP_NAME"
fi

# Set secrets
if [ -z "${LLM_API_KEY:-}" ]; then
    echo "ERROR: LLM_API_KEY environment variable is required"
    exit 1
fi

echo "Setting secrets..."
fly secrets set \
    LLM_API_KEY="$LLM_API_KEY" \
    LLM_MODEL="${LLM_MODEL:-openai/gpt-4o-mini}" \
    LLM_PROVIDER="${LLM_PROVIDER:-openai}" \
    -a "$APP_NAME"

# Create volume if it doesn't exist
if ! fly volumes list -a "$APP_NAME" --json | grep -q "cognee_data"; then
    echo "Creating persistent volume..."
    fly volumes create cognee_data \
        --size "$VOLUME_SIZE" \
        --region "$REGION" \
        -a "$APP_NAME" \
        --yes
fi

# Deploy using the fly.toml in this directory
echo "Deploying..."
fly deploy \
    --config distributed/deploy/fly.toml \
    -a "$APP_NAME"

echo ""
echo "Cognee deployed successfully!"
echo "API: https://$APP_NAME.fly.dev"
echo "Health: https://$APP_NAME.fly.dev/health"

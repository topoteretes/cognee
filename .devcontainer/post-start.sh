#!/bin/bash
set -e

echo "=================================================="
echo "🔄 Starting Cognee Codespace Post-Start Tasks"
echo "=================================================="

# This script runs every time the codespace starts
# Use it for lightweight tasks that should run on each startup

echo "🔍 Checking environment..."

# Check if .env file exists
if [ -f /app/.env ]; then
    echo "✅ .env file found"
else
    echo "⚠️  .env file not found - copying from template"
    cp /app/.env.template /app/.env
fi

# Display helpful information
echo ""
echo "📊 Codespace Status:"
echo "  - Python version: $(python --version)"
echo "  - Working directory: $(pwd)"
echo "  - Git branch: $(git branch --show-current 2>/dev/null || echo 'not in git repo')"
echo ""

# Optional: Start services in the background (delayed)
# Uncomment the following lines to auto-start services
# echo "🐳 Starting background services..."
# nohup docker-compose up postgres neo4j chromadb > /tmp/services.log 2>&1 &
# echo "✅ Services starting in background (check /tmp/services.log for logs)"

echo ""
echo "=================================================="
echo "✨ Codespace Ready!"
echo "=================================================="
echo ""
echo "🎯 Quick commands:"
echo "  - cognee-cli --help         : View CLI help"
echo "  - pytest                    : Run tests"
echo "  - docker-compose up -d      : Start all services"
echo "  - docker-compose ps         : Check service status"
echo ""

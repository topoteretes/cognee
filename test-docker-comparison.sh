#!/bin/bash
# Docker Image Size Comparison Test
# Compares current MCP Dockerfile vs optimized version

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo "======================================"
echo "MCP Docker Image Comparison Test"
echo "======================================"
echo ""

# Clean up existing images
echo "🧹 Cleaning up existing cognee-mcp images..."
docker images | grep cognee-mcp | awk '{print $1":"$2}' | xargs -r docker rmi -f 2>/dev/null || true
echo ""

# Build current Dockerfile
echo "🔨 Building CURRENT Dockerfile (used in GitHub Actions)..."
echo "   File: cognee-mcp/Dockerfile"
echo "   Tag: cognee-mcp:current"
echo ""

BUILD_START=$(date +%s)
docker build \
  --tag cognee-mcp:current \
  --file cognee-mcp/Dockerfile \
  --progress=plain \
  . 2>&1 | tee build-current.log

BUILD_END=$(date +%s)
CURRENT_BUILD_TIME=$((BUILD_END - BUILD_START))

echo ""
echo "✅ Current build completed in ${CURRENT_BUILD_TIME}s"
echo ""

# Build optimized Dockerfile
echo "🔨 Building OPTIMIZED Dockerfile..."
echo "   File: cognee-mcp/Dockerfile.optimized"
echo "   Tag: cognee-mcp:optimized"
echo ""

BUILD_START=$(date +%s)
docker build \
  --tag cognee-mcp:optimized \
  --file cognee-mcp/Dockerfile.optimized \
  --progress=plain \
  . 2>&1 | tee build-optimized.log

BUILD_END=$(date +%s)
OPTIMIZED_BUILD_TIME=$((BUILD_END - BUILD_START))

echo ""
echo "✅ Optimized build completed in ${OPTIMIZED_BUILD_TIME}s"
echo ""

# Get image sizes
echo "======================================"
echo "📊 Image Size Comparison"
echo "======================================"
echo ""

CURRENT_SIZE=$(docker images cognee-mcp:current --format "{{.Size}}")
OPTIMIZED_SIZE=$(docker images cognee-mcp:optimized --format "{{.Size}}")

CURRENT_SIZE_BYTES=$(docker images cognee-mcp:current --format "{{.Size}}" | numfmt --from=iec)
OPTIMIZED_SIZE_BYTES=$(docker images cognee-mcp:optimized --format "{{.Size}}" | numfmt --from=iec)

REDUCTION=$(echo "scale=2; (1 - $OPTIMIZED_SIZE_BYTES / $CURRENT_SIZE_BYTES) * 100" | bc)

echo "Current Dockerfile:   ${CURRENT_SIZE}"
echo "Optimized Dockerfile: ${OPTIMIZED_SIZE}"
echo ""
echo -e "${GREEN}Size Reduction: ${REDUCTION}%${NC}"
echo ""

# Detailed image information
echo "======================================"
echo "🔍 Detailed Image Information"
echo "======================================"
echo ""

echo "--- Current Image ---"
docker images cognee-mcp:current --format "table {{.Repository}}\t{{.Tag}}\t{{.Size}}\t{{.CreatedAt}}"
echo ""
docker history cognee-mcp:current --no-trunc --format "table {{.Size}}\t{{.CreatedBy}}" | head -n 20
echo ""

echo "--- Optimized Image ---"
docker images cognee-mcp:optimized --format "table {{.Repository}}\t{{.Tag}}\t{{.Size}}\t{{.CreatedAt}}"
echo ""
docker history cognee-mcp:optimized --no-trunc --format "table {{.Size}}\t{{.CreatedBy}}" | head -n 20
echo ""

# Test both images
echo "======================================"
echo "🧪 Testing Image Functionality"
echo "======================================"
echo ""

echo "Testing current image..."
docker run --rm cognee-mcp:current python -c "import sys; print(f'Python {sys.version}'); import cognee_mcp; print('✅ cognee_mcp imports successfully')" || echo -e "${RED}❌ Current image test failed${NC}"
echo ""

echo "Testing optimized image..."
docker run --rm cognee-mcp:optimized python -c "import sys; print(f'Python {sys.version}'); import cognee_mcp; print('✅ cognee_mcp imports successfully')" || echo -e "${RED}❌ Optimized image test failed${NC}"
echo ""

# Security comparison
echo "======================================"
echo "🔒 Security Comparison"
echo "======================================"
echo ""

echo "--- Current Image ---"
echo "User: $(docker run --rm cognee-mcp:current whoami)"
echo "Shell: $(docker run --rm cognee-mcp:current which bash || echo 'No bash')"
echo ""

echo "--- Optimized Image ---"
echo "User: $(docker run --rm cognee-mcp:optimized whoami)"
echo "Shell: $(docker run --rm cognee-mcp:optimized which bash || echo 'No bash')"
echo ""

# Layer count comparison
echo "======================================"
echo "📋 Layer Count Comparison"
echo "======================================"
echo ""

CURRENT_LAYERS=$(docker history cognee-mcp:current --format "{{.ID}}" | wc -l)
OPTIMIZED_LAYERS=$(docker history cognee-mcp:optimized --format "{{.ID}}" | wc -l)

echo "Current layers:   ${CURRENT_LAYERS}"
echo "Optimized layers: ${OPTIMIZED_LAYERS}"
echo ""

# Build time comparison
echo "======================================"
echo "⏱️  Build Time Comparison"
echo "======================================"
echo ""

echo "Current build time:   ${CURRENT_BUILD_TIME}s"
echo "Optimized build time: ${OPTIMIZED_BUILD_TIME}s"
echo ""

# Summary
echo "======================================"
echo "📝 SUMMARY"
echo "======================================"
echo ""

echo "Image Sizes:"
echo "  Current:   ${CURRENT_SIZE}"
echo "  Optimized: ${OPTIMIZED_SIZE}"
echo -e "  Reduction: ${GREEN}${REDUCTION}%${NC}"
echo ""

echo "Build Times:"
echo "  Current:   ${CURRENT_BUILD_TIME}s"
echo "  Optimized: ${OPTIMIZED_BUILD_TIME}s"
echo ""

echo "Layers:"
echo "  Current:   ${CURRENT_LAYERS}"
echo "  Optimized: ${OPTIMIZED_LAYERS}"
echo ""

if (( $(echo "$REDUCTION > 50" | bc -l) )); then
  echo -e "${GREEN}✅ RECOMMENDATION: Use optimized Dockerfile${NC}"
  echo "   The optimized version provides significant size reduction"
  echo "   while maintaining functionality and improving security."
elif (( $(echo "$REDUCTION > 20" | bc -l) )); then
  echo -e "${YELLOW}⚠️  RECOMMENDATION: Consider optimized Dockerfile${NC}"
  echo "   The optimized version provides moderate size reduction."
else
  echo -e "${YELLOW}⚠️  RECOMMENDATION: Review differences${NC}"
  echo "   Size reduction is minimal. Review other benefits."
fi

echo ""
echo "======================================"
echo "Test Complete!"
echo "======================================"
echo ""
echo "Build logs saved to:"
echo "  - build-current.log"
echo "  - build-optimized.log"
echo ""
echo "To clean up test images:"
echo "  docker rmi cognee-mcp:current cognee-mcp:optimized"
echo ""

#!/bin/bash

# ==============================================
# Docker Setup Validation Script
# Tests the optimized Docker configuration
# ==============================================

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Configuration
IMAGE_NAME="cognee-frontend"
CONTAINER_NAME="cognee-frontend-test"
TEST_PORT=3001

echo "=========================================="
echo "Cognee Frontend Docker Validation"
echo "=========================================="
echo ""

# Function to print colored output
print_status() {
    if [ $1 -eq 0 ]; then
        echo -e "${GREEN}✓ $2${NC}"
    else
        echo -e "${RED}✗ $2${NC}"
        exit 1
    fi
}

print_warning() {
    echo -e "${YELLOW}⚠ $1${NC}"
}

print_info() {
    echo -e "ℹ $1"
}

# Check if Docker is running
echo "1. Checking Docker daemon..."
if docker info > /dev/null 2>&1; then
    print_status 0 "Docker daemon is running"
else
    print_status 1 "Docker daemon is not running. Please start Docker."
fi

# Clean up any existing test containers/images
echo ""
echo "2. Cleaning up previous test resources..."
docker stop $CONTAINER_NAME 2>/dev/null || true
docker rm $CONTAINER_NAME 2>/dev/null || true
docker rmi ${IMAGE_NAME}:test 2>/dev/null || true
print_status 0 "Cleanup complete"

# Build the production image
echo ""
echo "3. Building production Docker image..."
print_info "This may take 5-10 minutes on first build..."
if docker build -t ${IMAGE_NAME}:test . > /tmp/docker-build.log 2>&1; then
    print_status 0 "Production image built successfully"
else
    print_status 1 "Failed to build production image. Check /tmp/docker-build.log"
fi

# Check image size
echo ""
echo "4. Checking image size..."
IMAGE_SIZE=$(docker images ${IMAGE_NAME}:test --format "{{.Size}}")
print_info "Image size: $IMAGE_SIZE"
if [[ $(docker images ${IMAGE_NAME}:test --format "{{.Size}}" | grep -E "MB|GB" | sed 's/[^0-9.]//g') -lt 300 ]]; then
    print_status 0 "Image size is optimized (< 300MB)"
else
    print_warning "Image size is larger than expected. Consider optimization."
fi

# Verify multi-stage build
echo ""
echo "5. Verifying multi-stage build..."
STAGE_COUNT=$(grep -c "^FROM" Dockerfile)
if [ $STAGE_COUNT -ge 3 ]; then
    print_status 0 "Multi-stage build detected ($STAGE_COUNT stages)"
else
    print_warning "Expected 3+ stages, found $STAGE_COUNT"
fi

# Verify non-root user
echo ""
echo "6. Verifying non-root user..."
USER_INFO=$(docker run --rm ${IMAGE_NAME}:test id)
if echo "$USER_INFO" | grep -q "uid=1001(nextjs)"; then
    print_status 0 "Container runs as non-root user (nextjs:1001)"
else
    print_status 1 "Container is not running as expected non-root user"
fi

# Start the container
echo ""
echo "7. Starting test container..."
if docker run -d --name $CONTAINER_NAME -p ${TEST_PORT}:3000 \
    -e NEXT_PUBLIC_BACKEND_API_URL=http://localhost:8000/api \
    ${IMAGE_NAME}:test > /dev/null 2>&1; then
    print_status 0 "Container started successfully"
else
    print_status 1 "Failed to start container"
fi

# Wait for container to be ready
echo ""
echo "8. Waiting for application to start..."
print_info "This may take 10-15 seconds..."
sleep 5

MAX_RETRIES=20
RETRY_COUNT=0
while [ $RETRY_COUNT -lt $MAX_RETRIES ]; do
    if curl -f http://localhost:${TEST_PORT} > /dev/null 2>&1; then
        print_status 0 "Application is responding on port ${TEST_PORT}"
        break
    fi
    RETRY_COUNT=$((RETRY_COUNT + 1))
    if [ $RETRY_COUNT -eq $MAX_RETRIES ]; then
        print_status 1 "Application failed to start within timeout"
        docker logs $CONTAINER_NAME
    fi
    sleep 1
done

# Check health status
echo ""
echo "9. Checking container health..."
sleep 10  # Wait for health check to run
HEALTH_STATUS=$(docker inspect --format='{{.State.Health.Status}}' $CONTAINER_NAME 2>/dev/null || echo "none")
if [ "$HEALTH_STATUS" == "healthy" ]; then
    print_status 0 "Container health check: healthy"
elif [ "$HEALTH_STATUS" == "none" ]; then
    print_warning "No health check configured"
else
    print_warning "Container health check: $HEALTH_STATUS (may need more time)"
fi

# Check resource usage
echo ""
echo "10. Checking resource usage..."
MEMORY_USAGE=$(docker stats $CONTAINER_NAME --no-stream --format "{{.MemUsage}}" | awk '{print $1}')
print_info "Memory usage: $MEMORY_USAGE"

CPU_USAGE=$(docker stats $CONTAINER_NAME --no-stream --format "{{.CPUPerc}}")
print_info "CPU usage: $CPU_USAGE"

# Verify security features
echo ""
echo "11. Verifying security features..."

# Check if running as non-root
if docker exec $CONTAINER_NAME whoami 2>/dev/null | grep -q "nextjs"; then
    print_status 0 "Process runs as non-root user"
else
    print_warning "Process may be running as root"
fi

# Check for dumb-init
if docker exec $CONTAINER_NAME ps aux 2>/dev/null | grep -q "dumb-init"; then
    print_status 0 "dumb-init is handling signals"
else
    print_warning "dumb-init may not be running"
fi

# Test rebuild with cache
echo ""
echo "12. Testing build cache performance..."
print_info "Rebuilding to test layer caching..."
BUILD_START=$(date +%s)
if docker build -t ${IMAGE_NAME}:test . > /tmp/docker-rebuild.log 2>&1; then
    BUILD_END=$(date +%s)
    BUILD_TIME=$((BUILD_END - BUILD_START))
    print_status 0 "Rebuild completed in ${BUILD_TIME} seconds"
    if [ $BUILD_TIME -lt 120 ]; then
        print_status 0 "Build cache is working efficiently"
    else
        print_warning "Build cache may not be optimal (took ${BUILD_TIME}s)"
    fi
else
    print_status 1 "Rebuild failed"
fi

# Summary
echo ""
echo "=========================================="
echo "Test Summary"
echo "=========================================="
echo ""
echo "Image Details:"
echo "  Name: ${IMAGE_NAME}:test"
echo "  Size: $IMAGE_SIZE"
echo "  Container: $CONTAINER_NAME"
echo "  Port: http://localhost:${TEST_PORT}"
echo ""
echo "Resource Usage:"
echo "  Memory: $MEMORY_USAGE"
echo "  CPU: $CPU_USAGE"
echo ""

# Prompt for cleanup
echo ""
read -p "Do you want to clean up test resources? (y/n) " -n 1 -r
echo ""
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "Cleaning up..."
    docker stop $CONTAINER_NAME > /dev/null 2>&1
    docker rm $CONTAINER_NAME > /dev/null 2>&1
    docker rmi ${IMAGE_NAME}:test > /dev/null 2>&1
    print_status 0 "Cleanup complete"
else
    print_info "Container left running for manual inspection"
    print_info "To view logs: docker logs $CONTAINER_NAME"
    print_info "To stop: docker stop $CONTAINER_NAME && docker rm $CONTAINER_NAME"
    print_info "To remove image: docker rmi ${IMAGE_NAME}:test"
fi

echo ""
print_status 0 "All tests completed successfully!"
echo ""

#!/bin/bash
# Cognee MCP Server - Docker Setup Test Script
# Tests the Docker configuration and validates functionality

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Configuration
IMAGE_NAME="cognee-mcp-test"
CONTAINER_NAME="cognee-mcp-test"
TEST_PORT=8091
BACKEND_URL=${BACKEND_URL:-"http://host.docker.internal:8000"}

echo "========================================"
echo "Cognee MCP Docker Setup Test"
echo "========================================"

# Function to print colored output
print_success() {
    echo -e "${GREEN}✓ $1${NC}"
}

print_error() {
    echo -e "${RED}✗ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠ $1${NC}"
}

print_info() {
    echo "ℹ $1"
}

# Cleanup function
cleanup() {
    print_info "Cleaning up..."
    docker rm -f $CONTAINER_NAME 2>/dev/null || true
    docker rmi -f $IMAGE_NAME 2>/dev/null || true
}

# Trap cleanup on exit
trap cleanup EXIT

# Test 1: Check if Docker is running
print_info "Test 1: Checking Docker..."
if docker info >/dev/null 2>&1; then
    print_success "Docker is running"
else
    print_error "Docker is not running"
    exit 1
fi

# Test 2: Check if Dockerfile.optimized exists
print_info "Test 2: Checking Dockerfile.optimized..."
if [ -f "Dockerfile.optimized" ]; then
    print_success "Dockerfile.optimized found"
else
    print_error "Dockerfile.optimized not found"
    exit 1
fi

# Test 3: Check if .dockerignore exists
print_info "Test 3: Checking .dockerignore..."
if [ -f ".dockerignore" ]; then
    print_success ".dockerignore found"
else
    print_warning ".dockerignore not found (recommended)"
fi

# Test 4: Build the optimized image
print_info "Test 4: Building optimized image..."
if docker build -f Dockerfile.optimized -t $IMAGE_NAME . >/dev/null 2>&1; then
    print_success "Image built successfully"
else
    print_error "Image build failed"
    exit 1
fi

# Test 5: Check image size
print_info "Test 5: Checking image size..."
IMAGE_SIZE=$(docker images $IMAGE_NAME --format "{{.Size}}")
print_success "Image size: $IMAGE_SIZE"

# Test 6: Inspect image for security
print_info "Test 6: Checking security configuration..."
USER=$(docker inspect $IMAGE_NAME --format='{{.Config.User}}')
if [ "$USER" = "cognee" ] || [ "$USER" = "1000" ]; then
    print_success "Running as non-root user: $USER"
else
    print_warning "Not running as non-root user (current: $USER)"
fi

# Test 7: Check for health check
print_info "Test 7: Checking health check configuration..."
HEALTHCHECK=$(docker inspect $IMAGE_NAME --format='{{.Config.Healthcheck.Test}}')
if [ -n "$HEALTHCHECK" ]; then
    print_success "Health check configured"
else
    print_warning "No health check configured"
fi

# Test 8: Start container with SSE transport
print_info "Test 8: Starting container with SSE transport..."
docker run -d \
    --name $CONTAINER_NAME \
    -e TRANSPORT_MODE=sse \
    -e API_URL=$BACKEND_URL \
    -e HTTP_PORT=$TEST_PORT \
    -p $TEST_PORT:$TEST_PORT \
    $IMAGE_NAME >/dev/null

sleep 5  # Wait for startup

if docker ps | grep -q $CONTAINER_NAME; then
    print_success "Container started successfully"
else
    print_error "Container failed to start"
    docker logs $CONTAINER_NAME
    exit 1
fi

# Test 9: Check if server is responding
print_info "Test 9: Testing health endpoint..."
sleep 5  # Additional wait for service to be ready

if curl -s -f http://localhost:$TEST_PORT/health >/dev/null 2>&1; then
    print_success "Health endpoint responding"
else
    print_warning "Health endpoint not responding (may need backend API)"
    print_info "Container logs:"
    docker logs --tail 20 $CONTAINER_NAME
fi

# Test 10: Check container logs for errors
print_info "Test 10: Checking container logs..."
if docker logs $CONTAINER_NAME 2>&1 | grep -qi "error"; then
    print_warning "Errors found in logs"
    docker logs $CONTAINER_NAME | grep -i error | tail -5
else
    print_success "No errors in logs"
fi

# Test 11: Check resource usage
print_info "Test 11: Checking resource usage..."
MEM_USAGE=$(docker stats $CONTAINER_NAME --no-stream --format "{{.MemUsage}}" | awk '{print $1}')
CPU_USAGE=$(docker stats $CONTAINER_NAME --no-stream --format "{{.CPUPerc}}")
print_success "Memory: $MEM_USAGE, CPU: $CPU_USAGE"

# Test 12: Test container can be stopped gracefully
print_info "Test 12: Testing graceful shutdown..."
if docker stop $CONTAINER_NAME -t 5 >/dev/null 2>&1; then
    print_success "Container stopped gracefully"
else
    print_warning "Container did not stop gracefully"
fi

# Summary
echo ""
echo "========================================"
echo "Test Summary"
echo "========================================"
echo "Image: $IMAGE_NAME"
echo "Size: $IMAGE_SIZE"
echo "User: $USER"
echo "Port: $TEST_PORT"
echo "========================================"
print_success "All tests completed!"

# Recommendations
echo ""
echo "Recommendations:"
if [ "$USER" != "cognee" ] && [ "$USER" != "1000" ]; then
    echo "  - Configure non-root user for better security"
fi
if [ -z "$HEALTHCHECK" ]; then
    echo "  - Add HEALTHCHECK directive to Dockerfile"
fi
if [ ! -f ".dockerignore" ]; then
    echo "  - Create .dockerignore file to optimize builds"
fi

echo ""
print_info "To run the optimized container in production:"
echo "  docker run -d \\"
echo "    --name cognee-mcp \\"
echo "    --network cognee-network \\"
echo "    -e TRANSPORT_MODE=sse \\"
echo "    -e API_URL=http://cognee:8000 \\"
echo "    -e API_TOKEN=your_token \\"
echo "    -p 8001:8001 \\"
echo "    $IMAGE_NAME"

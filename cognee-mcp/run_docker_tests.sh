#!/bin/bash
# Script to run Docker tests for the Cognee MCP server

set -e  # Exit on error

# Print header
echo "====================================="
echo "  Cognee MCP Docker Tests"
echo "====================================="
echo

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    echo "Error: Docker is not installed or not in PATH"
    exit 1
fi

# Check if Docker daemon is running
if ! docker info &> /dev/null; then
    echo "Error: Docker daemon is not running"
    exit 1
fi

# Set working directory to the project root
cd "$(dirname "$0")"

# Clean up any existing test containers or images
echo "Cleaning up any existing test resources..."
docker rm -f cognee-mcp-test 2>/dev/null || true
docker rmi -f cognee-mcp:test 2>/dev/null || true

# Build the Docker image first to ensure it's fresh
echo "Building Docker image for testing..."
docker build -t cognee-mcp:test .

# Run the Docker tests with verbose output
echo "Running Docker tests..."
python3 -v tests/test_docker_image.py

# Check the exit code
if [ $? -eq 0 ]; then
    echo
    echo "====================================="
    echo "  All Docker tests passed!"
    echo "====================================="
    exit 0
else
    echo
    echo "====================================="
    echo "  Docker tests failed!"
    echo "====================================="
    exit 1
fi 
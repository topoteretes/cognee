#!/bin/bash
# Script to run MCP tool tests

set -e

# Set colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${BLUE}=====================================${NC}"
echo -e "${BLUE}  Cognee MCP Tools Tests${NC}"
echo -e "${BLUE}=====================================${NC}"
echo ""

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

# Get the directory of this script
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

# Install test dependencies
echo "Installing test dependencies..."
if [ -f "tests/requirements-test.txt" ]; then
    pip install -r tests/requirements-test.txt
else
    # Install minimal dependencies if requirements file doesn't exist
    pip install python-dotenv
fi

# Check if .env file exists
if [ ! -f ".env" ]; then
    echo "Warning: .env file not found. Creating from example if available..."
    if [ -f ".env.example" ]; then
        cp .env.example .env
        echo "Created .env from .env.example. Please update with your actual values."
    else
        echo "Warning: No .env or .env.example file found. Tests may fail without proper environment variables."
        # Create minimal .env file
        echo "# Minimal .env file created by test script" > .env
        echo "DEBUG=true" >> .env
    fi
fi

# Clean up any existing test containers or images
echo -e "${BLUE}Cleaning up any existing test resources...${NC}"
docker stop cognee-mcp-test 2>/dev/null || true
docker rmi cognee-mcp:test 2>/dev/null || true

echo -e "${BLUE}Running MCP tool tests...${NC}"
echo ""

# Run the tests
python -u tests/test_mcp_docker_tools.py

# Check the exit code
EXIT_CODE=$?
if [ $EXIT_CODE -eq 0 ]; then
    echo ""
    echo -e "${GREEN}=====================================${NC}"
    echo -e "${GREEN}  All MCP tool tests passed!${NC}"
    echo -e "${GREEN}=====================================${NC}"
    echo ""
    echo -e "The tests verified the following functionality:"
    echo -e "1. Docker image build"
    echo -e "2. Cognify tool"
    echo -e "3. Search tool (after cognify)"
    echo -e "4. Codify tool"
    exit 0
else
    echo ""
    echo -e "${RED}=====================================${NC}"
    echo -e "${RED}  MCP tool tests failed!${NC}"
    echo -e "${RED}=====================================${NC}"
    echo ""
    echo -e "There was an issue with the MCP tool tests."
    echo -e "Please check the error messages above for more information."
    exit 1
fi 
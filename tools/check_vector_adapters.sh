#!/bin/bash

# Vector Database Adapters MyPy Check Script

set -e  # Exit on any error

echo "🔍 Discovering Vector Database Adapters..."

# Ensure we're in the project root directory
cd "$(dirname "$0")/.."

# Activate virtual environment
source .venv/bin/activate

# Find all *Adapter.py files in vector database directories
vector_adapters=$(find cognee/infrastructure/databases/vector -name "*Adapter.py" -type f | sort)

if [ -z "$vector_adapters" ]; then
  echo "No vector database adapters found"
  exit 0
else
  echo "Found vector database adapters:"
  echo "$vector_adapters" | sed 's/^/  • /'
  echo ""
  
  echo "Running MyPy on vector database adapters..."
  
  # Use while read to properly handle each file
  echo "$vector_adapters" | while read -r adapter; do
    if [ -n "$adapter" ]; then
      echo "Checking: $adapter"
      uv run mypy "$adapter" \
        --config-file mypy.ini \
        --show-error-codes \
        --no-error-summary
      echo ""
    fi
  done
fi

echo "✅ Vector Database Adapters MyPy Check Complete!"

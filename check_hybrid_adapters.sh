#!/bin/bash

# Hybrid Database Adapters MyPy Check Script

set -e  # Exit on any error

echo "🔍 Discovering Hybrid Database Adapters..."

# Ensure we're in the right directory
cd "$(dirname "$0")"

# Activate virtual environment
source .venv/bin/activate

# Find all *Adapter.py files in hybrid database directories
hybrid_adapters=$(find cognee/infrastructure/databases/hybrid -name "*Adapter.py" -type f | sort)

if [ -z "$hybrid_adapters" ]; then
  echo "No hybrid database adapters found"
  exit 0
else
  echo "Found hybrid database adapters:"
  echo "$hybrid_adapters" | sed 's/^/  • /'
  echo ""
  
  echo "Running MyPy on hybrid database adapters..."
  
  # Use while read to properly handle each file
  echo "$hybrid_adapters" | while read -r adapter; do
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

echo "✅ Hybrid Database Adapters MyPy Check Complete!"

#!/bin/bash

# Graph Database Adapters MyPy Check Script

set -e  # Exit on any error

echo "🔍 Discovering Graph Database Adapters..."

# Ensure we're in the right directory
cd "$(dirname "$0")"

# Activate virtual environment
source .venv/bin/activate

# Find all adapter.py and *adapter.py files in graph database directories, excluding utility files
graph_adapters=$(find cognee/infrastructure/databases/graph -name "*adapter.py" -o -name "adapter.py" | grep -v "use_graph_adapter.py" | sort)

if [ -z "$graph_adapters" ]; then
  echo "No graph database adapters found"
  exit 0
else
  echo "Found graph database adapters:"
  echo "$graph_adapters" | sed 's/^/  • /'
  echo ""
  
  echo "Running MyPy on graph database adapters..."
  
  # Use while read to properly handle each file
  echo "$graph_adapters" | while read -r adapter; do
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

echo "✅ Graph Database Adapters MyPy Check Complete!"

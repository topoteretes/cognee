#!/bin/bash

# All Database Adapters MyPy Check Script

set -e  # Exit on any error

echo "🚀 Running MyPy checks on all database adapters..."
echo ""

# Ensure we're in the project root directory
cd "$(dirname "$0")/.."

# Run all three adapter checks
echo "========================================="
echo "1️⃣  VECTOR DATABASE ADAPTERS"
echo "========================================="
./tools/check_vector_adapters.sh

echo ""
echo "========================================="
echo "2️⃣  GRAPH DATABASE ADAPTERS"
echo "========================================="
./tools/check_graph_adapters.sh

echo ""
echo "========================================="
echo "3️⃣  HYBRID DATABASE ADAPTERS"
echo "========================================="
./tools/check_hybrid_adapters.sh

echo ""
echo "🎉 All Database Adapters MyPy Checks Complete!"
echo ""
echo "🔍 Auto-Discovery Approach:"
echo "  • Vector Adapters: cognee/infrastructure/databases/vector/**/*Adapter.py"
echo "  • Graph Adapters: cognee/infrastructure/databases/graph/**/*adapter.py"
echo "  • Hybrid Adapters: cognee/infrastructure/databases/hybrid/**/*Adapter.py"
echo ""
echo "🎯 Purpose: Enforce that database adapters are properly typed"
echo "🔧 MyPy Configuration: mypy.ini (strict mode enabled)"
echo "🚀 Maintenance-Free: Automatically discovers new adapters"

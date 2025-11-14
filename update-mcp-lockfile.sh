#!/bin/bash
# Script to regenerate MCP lockfile after removing heavy dependencies

set -e

echo "======================================"
echo "MCP Lockfile Update Script"
echo "======================================"
echo ""

# Check we're in the right directory
if [ ! -f "cognee-mcp/pyproject.toml" ]; then
    echo "❌ Error: Must run from repo root"
    echo "   Current directory: $(pwd)"
    echo "   Expected: cognee-mcp/pyproject.toml to exist"
    exit 1
fi

# Check uv is installed
if ! command -v uv &> /dev/null; then
    echo "❌ Error: uv is not installed"
    echo ""
    echo "Install uv:"
    echo "  curl -LsSf https://astral.sh/uv/install.sh | sh"
    echo ""
    exit 1
fi

echo "✅ Found uv: $(uv --version)"
echo ""

# Show current dependencies
echo "Current dependencies in pyproject.toml:"
echo "----------------------------------------"
grep "cognee" cognee-mcp/pyproject.toml | grep -v "^#" || true
echo ""

# Backup old lockfile
if [ -f "cognee-mcp/uv.lock" ]; then
    echo "📦 Backing up old lockfile..."
    cp cognee-mcp/uv.lock cognee-mcp/uv.lock.backup
    echo "   Saved to: cognee-mcp/uv.lock.backup"
    echo ""

    # Show old lockfile size
    OLD_SIZE=$(wc -l < cognee-mcp/uv.lock)
    echo "Old lockfile: $OLD_SIZE lines"
    echo ""
fi

# Regenerate lockfile
echo "🔄 Regenerating lockfile without heavy dependencies..."
echo "   This may take 2-3 minutes..."
echo ""

cd cognee-mcp

# Remove old .venv if it exists
if [ -d ".venv" ]; then
    echo "   Removing old .venv..."
    rm -rf .venv
fi

# Sync with new dependencies
uv sync --reinstall

cd ..

echo ""
echo "✅ Lockfile regenerated!"
echo ""

# Show new lockfile size
if [ -f "cognee-mcp/uv.lock" ]; then
    NEW_SIZE=$(wc -l < cognee-mcp/uv.lock)
    echo "New lockfile: $NEW_SIZE lines"

    if [ -f "cognee-mcp/uv.lock.backup" ]; then
        REDUCTION=$((OLD_SIZE - NEW_SIZE))
        PERCENT=$((REDUCTION * 100 / OLD_SIZE))
        echo "Reduction: $REDUCTION lines (${PERCENT}%)"
    fi
    echo ""
fi

# Verify no heavy dependencies
echo "🔍 Verifying heavy dependencies are removed..."
cd cognee-mcp

HEAVY_DEPS=$(uv tree | grep -E "(unstructured|psycopg2-binary|neo4j|tesseract|poppler)" || true)

if [ -z "$HEAVY_DEPS" ]; then
    echo "✅ No heavy dependencies found!"
else
    echo "⚠️  Warning: Found heavy dependencies:"
    echo "$HEAVY_DEPS"
fi

cd ..
echo ""

# Show dependency count
DEP_COUNT=$(cd cognee-mcp && uv tree | wc -l)
echo "Total dependencies: $DEP_COUNT packages"
echo ""

# Test import
echo "🧪 Testing imports..."
cd cognee-mcp

if uv run python -c "from src.cognee_client import CogneeClient; print('✅ CogneeClient imports OK')" 2>&1; then
    echo ""
else
    echo "❌ Import test failed!"
    echo ""
    echo "Restoring backup..."
    if [ -f "uv.lock.backup" ]; then
        cp uv.lock.backup uv.lock
        echo "✅ Restored old lockfile"
    fi
    exit 1
fi

cd ..

# Summary
echo "======================================"
echo "✅ Lockfile Update Complete!"
echo "======================================"
echo ""
echo "Next steps:"
echo ""
echo "1. Review changes:"
echo "   git diff cognee-mcp/uv.lock"
echo ""
echo "2. Build Docker image to verify size:"
echo "   docker build -f cognee-mcp/Dockerfile -t cognee-mcp:test ."
echo "   docker images cognee-mcp:test"
echo "   # Should show ~600MB, not 4GB"
echo ""
echo "3. Commit changes:"
echo "   git add cognee-mcp/pyproject.toml cognee-mcp/uv.lock"
echo "   git commit -m \"fix: remove heavy deps from MCP server (4GB->600MB)\""
echo ""
echo "4. Push to main:"
echo "   git push origin main"
echo ""

# Clean up backup
if [ -f "cognee-mcp/uv.lock.backup" ]; then
    echo "Backup saved at: cognee-mcp/uv.lock.backup"
    echo "(You can delete this after verifying everything works)"
    echo ""
fi

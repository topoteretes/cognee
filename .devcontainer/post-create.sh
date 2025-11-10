#!/bin/bash
set -e

echo "=================================================="
echo "🚀 Starting Cognee Codespace Post-Create Setup"
echo "=================================================="

# This script runs once after the codespace is created
# It performs delayed initialization to speed up codespace creation

echo "📦 Installing Python dependencies..."
# Install the package in development mode with all extras
pip install --no-cache-dir -e .[dev,neo4j,postgres,chromadb,redis] || {
    echo "⚠️  Full installation failed, trying minimal installation..."
    pip install --no-cache-dir -e .
}

echo "🔧 Setting up pre-commit hooks..."
pre-commit install || echo "⚠️  Pre-commit hooks installation skipped"

echo "📝 Creating .env file from template..."
if [ ! -f /app/.env ]; then
    cp /app/.env.template /app/.env
    echo "✅ .env file created from template"
else
    echo "ℹ️  .env file already exists"
fi

echo "🧪 Verifying installation..."
python -c "import cognee; print(f'✅ Cognee version: {cognee.__version__ if hasattr(cognee, \"__version__\") else \"installed\"}')" || echo "⚠️  Cognee import failed"

echo ""
echo "=================================================="
echo "✨ Post-Create Setup Complete!"
echo "=================================================="
echo ""
echo "📚 Next steps:"
echo "  1. Configure your .env file with API keys"
echo "  2. Run 'docker-compose up -d postgres neo4j' to start services"
echo "  3. Run 'cognee-cli' to get started"
echo ""

#!/bin/bash
# ═══════════════════════════════════════════════════════════════
# Research Paper Intelligence System — Docker Startup Script
# ═══════════════════════════════════════════════════════════════

set -e

# Default to backend if not specified
SERVICE=${SERVICE:-backend}

echo "🚀 Starting Research Paper Intelligence System..."
echo "📦 Service: $SERVICE"
echo ""

# Check if .env file exists, if not create a template
if [ ! -f /app/.env ]; then
    echo "⚠️  Warning: .env file not found. Creating from environment variables..."
    
    # Create .env from environment variables
    cat > /app/.env << EOF
# Auto-generated from environment variables
GROQ_API_KEY=${GROQ_API_KEY:-}
QDRANT_HOST=${QDRANT_HOST:-qdrant}
QDRANT_PORT=${QDRANT_PORT:-6333}
MONGODB_URI=${MONGODB_URI:-mongodb://mongodb:27017}
LANGFUSE_PUBLIC_KEY=${LANGFUSE_PUBLIC_KEY:-}
LANGFUSE_SECRET_KEY=${LANGFUSE_SECRET_KEY:-}
LANGFUSE_HOST=${LANGFUSE_HOST:-http://localhost:3000}
ENABLE_LANGFUSE=${ENABLE_LANGFUSE:-false}
EOF
fi

case "$SERVICE" in
    backend)
        echo "🔧 Starting FastAPI Backend..."
        echo "📡 API will be available at: http://localhost:8000"
        echo "📚 API Docs at: http://localhost:8000/docs"
        echo ""
        cd /app/backend && uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
        ;;
        
    frontend)
        echo "🎨 Starting Streamlit Frontend..."
        echo "🌐 UI will be available at: http://localhost:8501"
        echo ""
        cd /app && streamlit run frontend/app.py --server.port=8501 --server.address=0.0.0.0
        ;;
        
    corpus)
        echo "📚 Building corpus from PDFs..."
        if [ -z "$(ls -A /app/corpus/*.pdf 2>/dev/null)" ]; then
            echo "⚠️  Warning: No PDF files found in /app/corpus/"
            echo "Please mount your PDF files to /app/corpus/"
            exit 1
        fi
        cd /app && python build_corpus.py
        ;;
        
    interactive)
        echo "💬 Starting Interactive Query CLI..."
        cd /app && python interactive_query.py
        ;;
        
    test)
        echo "🧪 Running tests..."
        cd /app/backend && pytest -v
        ;;
        
    *)
        echo "❌ Unknown service: $SERVICE"
        echo ""
        echo "Available services:"
        echo "  backend      - FastAPI REST API (port 8000)"
        echo "  frontend     - Streamlit Web UI (port 8501)"
        echo "  corpus       - Build corpus from PDFs"
        echo "  interactive  - Interactive CLI for queries"
        echo "  test         - Run test suite"
        echo ""
        echo "Usage: docker run -e SERVICE=backend <image>"
        exit 1
        ;;
esac

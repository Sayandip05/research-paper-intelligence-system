# Research Paper Analyzer - Week 1: Core Document Processing

## 🎯 Week 1 Goal

Get PDFs in, parsed, and searchable with section-aware chunking and hybrid retrieval.

## 📋 What You've Built

- ✅ FastAPI application structure
- ✅ PDF parsing with PyMuPDF (sections, tables, figures, metadata)
- ✅ Section-aware chunking strategy
- ✅ Embedding generation (OpenAI + Local options)
- ✅ Qdrant vector database integration
- ✅ Upload and search API endpoints
- ✅ Docker containerization

## 🚀 Quick Start

### Option 1: Docker (Recommended)

```bash
# 1. Clone/create project
git clone <your-repo>
cd research-paper-analyzer

# 2. Create .env file
cp .env.example .env
# Edit .env and add your OPENAI_API_KEY

# 3. Start all services
docker-compose up -d

# 4. Check if services are running
docker-compose ps

# 5. Test the API
curl http://localhost:8000/health
```

### Option 2: Local Development

```bash
# 1. Create virtual environment
cd backend
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Create .env file
cp .env.example .env
# Add your OPENAI_API_KEY

# 4. Start Qdrant and MongoDB (Docker)
docker-compose up qdrant mongodb -d

# 5. Run FastAPI
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

## 📝 Testing Your Setup

### 1. Check Health

```bash
curl http://localhost:8000/health
```

Expected response:
```json
{
  "status": "healthy",
  "qdrant": "localhost:6333",
  "mongodb": "mongodb://localhost:27017"
}
```

### 2. Upload a PDF

```bash
curl -X POST "http://localhost:8000/api/upload/pdf" \
  -H "accept: application/json" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@your_paper.pdf"
```

Expected response:
```json
{
  "paper_id": "abc-123-def-456",
  "filename": "your_paper.pdf",
  "status": "success",
  "message": "Successfully processed 45 chunks",
  "num_chunks": 45,
  "metadata": {
    "title": "Attention Is All You Need",
    "authors": [...],
    "year": 2017
  }
}
```

### 3. Search Papers

```bash
curl -X POST "http://localhost:8000/api/search/" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What is the transformer architecture?",
    "top_k": 5
  }'
```

Expected response:
```json
{
  "query": "What is the transformer architecture?",
  "results": [
    {
      "chunk_id": "xyz-789",
      "text": "The Transformer is a model architecture...",
      "score": 0.89,
      "metadata": {
        "paper_title": "Attention Is All You Need",
        "section_title": "Model Architecture",
        "page_start": 3,
        "page_end": 4
      }
    }
  ],
  "total_found": 5,
  "search_time_ms": 123.45
}
```

## 🏗️ Project Structure

```
research-paper-analyzer/
├── backend/
│   ├── app/
│   │   ├── main.py              # FastAPI app
│   │   ├── config.py            # Configuration
│   │   ├── api/
│   │   │   └── routes/
│   │   │       ├── upload.py    # PDF upload endpoint
│   │   │       └── search.py    # Search endpoint
│   │   ├── services/
│   │   │   ├── pdf_parser.py    # PDF parsing
│   │   │   ├── chunking.py      # Section-aware chunking
│   │   │   └── embeddings.py    # Embedding generation
│   │   ├── db/
│   │   │   └── qdrant_client.py # Qdrant operations
│   │   └── models/
│   │       ├── paper.py         # Paper data models
│   │       └── chunk.py         # Chunk data models
│   ├── uploads/                 # Uploaded PDFs
│   ├── requirements.txt
│   └── Dockerfile
├── docker-compose.yml
└── README.md
```

## 🔧 Configuration (.env)

```bash
# API Keys
OPENAI_API_KEY=sk-...

# Qdrant
QDRANT_HOST=localhost
QDRANT_PORT=6333

# MongoDB
MONGODB_URL=mongodb://localhost:27017

# Embeddings
EMBEDDING_MODEL=text-embedding-3-small
EMBEDDING_DIM=1536

# Chunking
CHUNK_SIZE=1000
CHUNK_OVERLAP=200
```

## 📊 API Endpoints

### Upload PDF
```
POST /api/upload/pdf
- Upload and process a research paper
- Returns: paper_id, metadata, num_chunks
```

### Search Papers
```
POST /api/search/
- Search across all papers
- Body: { "query": "...", "top_k": 10 }
- Returns: ranked chunks with citations
```

### Health Check
```
GET /health
- Check if services are running
```

### API Documentation
```
GET /docs
- Interactive Swagger UI
```

## 🧪 Running Tests

```bash
# Run all tests
make test

# Or manually
cd backend
pytest tests/ -v
```

## 🐛 Common Issues

### Issue 1: Qdrant connection error
```
Error: Connection refused to localhost:6333
```
**Solution:**
```bash
# Check if Qdrant is running
docker-compose ps

# Restart Qdrant
docker-compose restart qdrant
```

### Issue 2: OpenAI API key error
```
Error: Invalid API key
```
**Solution:**
- Check `.env` file has correct `OPENAI_API_KEY`
- Restart the app after changing `.env`

### Issue 3: PDF parsing fails
```
Error: Could not parse PDF
```
**Solution:**
- Ensure PDF is not corrupted
- Check PDF is text-based (not scanned image)
- Try with a different PDF

## 📈 What's Next (Week 2)

- [ ] Add LlamaIndex agent system
- [ ] Implement query planning
- [ ] Build specialized agents (methodology, results, gaps)
- [ ] Add MongoDB for paper metadata
- [ ] Create conversation history

## 🎓 Learning Resources

- [PyMuPDF Documentation](https://pymupdf.readthedocs.io/)
- [Qdrant Documentation](https://qdrant.tech/documentation/)
- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [LlamaIndex Documentation](https://docs.llamaindex.ai/)

## 📧 Support

If you encounter any issues:
1. Check the logs: `docker-compose logs app`
2. Verify all services are running: `docker-compose ps`
3. Check the API docs: http://localhost:8000/docs

---

**Status: Week 1 Complete ✅**

You now have:
- PDF upload and parsing ✅
- Section-aware chunking ✅
- Embedding generation ✅
- Vector search with Qdrant ✅
- RESTful API ✅
- Docker containerization ✅
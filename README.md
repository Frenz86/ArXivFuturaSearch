# ArXiv RAG Copilot v0.3.0

**Advanced Retrieval-Augmented Generation (RAG) system** powered by LangChain for querying ArXiv research papers with enterprise-grade features.

## 🚀 What's New in v0.3.0 (LangChain Refactoring)

- **🔗 Full LangChain Integration** - Complete refactoring to use LangChain framework
- **📦 LangChain Components** - Vector stores, retrievers, chains, and LLMs via LangChain
- **🔄 Backward Compatible** - Existing API preserved while using LangChain under the hood
- **📚 ArXiv Document Loader** - Native LangChain document loader for ArXiv papers
- **⛓️ RAG Chains** - Composable chains for retrieval and generation
- **🎯 Ensemble Retrievers** - Hybrid search (semantic + BM25) via LangChain retrievers
- **🤖 Multiple LLM Support** - OpenRouter, Ollama, or Mock via unified LangChain interface

## ✨ Key Features

### Retrieval & Search
- **Hybrid Search**: Semantic (ChromaDB) + Lexical (BM25) with Reciprocal Rank Fusion
- **ChromaDB Vector Store**: Scalable HNSW indexing with persistent storage
- **Semantic Chunking**: Content-aware splitting based on sentence similarity
- **Advanced Reranking**:
  - Cross-encoder reranking for precision
  - MMR (Maximal Marginal Relevance) for diversity

### LLM Integration
- **Streaming Responses**: Real-time answer generation
- **Multiple Providers**: OpenRouter, Ollama, or Mock mode
- **Chain-of-Thought**: Structured reasoning with few-shot examples
- **Smart Caching**: Redis-backed query cache

### Monitoring & Performance
- **Prometheus Metrics**: Request latency, cache hit rates, LLM token usage
- **Health Checks**: Comprehensive system status monitoring
- **Structured Logging**: JSON/console formats with context

### Evaluation
- **RAGAS Integration**: Automated quality assessment
- **Retrieval Metrics**: Track precision, recall, and relevance

## 📋 Quick Start

### 1. Install Dependencies

```bash
# Using uv (recommended)
uv sync

# Or using pip
pip install -e .
```

### 2. Configure Environment

Copy `.env.example` to `.env`:

```bash
cp .env.example .env
```

**Required Configuration:**

```env
# ChromaDB directory
CHROMA_DIR=data/chroma_db

# Embeddings - E5 Multilingual
EMBED_MODEL=intfloat/multilingual-e5-large

# LLM Provider
LLM_MODE=openrouter
OPENROUTER_API_KEY=sk-or-v1-your-key-here
OPENROUTER_MODEL=anthropic/claude-3.5-sonnet

# Optional: Redis for caching (recommended for production)
CACHE_ENABLED=true
REDIS_HOST=localhost
REDIS_PORT=6379
```

### 3. Optional: Start Redis (for caching)

```bash
# Using Docker
docker run -d -p 6379:6379 redis:alpine

# Or install locally
# macOS: brew install redis && redis-server
# Ubuntu: sudo apt install redis-server && sudo systemctl start redis
```

### 4. Build the Index

```bash
uv run uvicorn app.main:app --port 8000 --reload

```

Access the web interface at `http://localhost:8000`

## 🏗️ Architecture

```
arxiv-rag-copilot/
├── app/
│   ├── main.py                   # FastAPI server & routes
│   ├── config.py                 # Configuration management
│   ├── arxiv_loader.py           # ArXiv API client
│   ├── arxiv_document_loader.py  # LangChain document loader
│   ├── chunking.py               # Text chunking (semantic & sentence-aware)
│   ├── embeddings.py             # Embedding & reranking (E5, MMR)
│   ├── vectorstore_chroma.py     # ChromaDB hybrid store
│   ├── rag.py                    # RAG logic & LLM generation
│   ├── cache.py                  # Redis caching layer
│   ├── metrics.py                # Prometheus metrics
│   └── evals.py                  # RAGAS evaluation
├── templates/                    # Web UI templates
├── data/
│   ├── raw/                      # Raw ArXiv data
│   ├── processed/                # Evaluations
│   └── chroma_db/                # ChromaDB storage
└── pyproject.toml                # Dependencies
```

## 🔧 How It Works

### Index Building

1. **Fetch Papers**: ArXiv API retrieves papers based on search query
2. **Semantic Chunking**:
   - Sentence tokenization (NLTK)
   - Embedding-based similarity analysis
   - Intelligent boundary detection
3. **Generate Embeddings**: E5 Multilingual creates 1024-dim vectors
4. **Build Index**: ChromaDB HNSW index with persistent storage

### Query Processing

1. **Embed Query**: E5 Multilingual embedding
2. **Hybrid Search**:
   - **Semantic**: ChromaDB cosine similarity (70% weight)
   - **Lexical**: BM25 keyword matching (30% weight)
   - **Fusion**: Reciprocal Rank Fusion combines results
3. **Rerank** (configurable):
   - **Cross-Encoder**: Precision-focused reranking
   - **MMR**: Diversity-aware selection (balances relevance & novelty)
4. **Generate Answer**:
   - Chain-of-thought prompting
   - Few-shot examples
   - Source citations

### Response Format

```
Answer:
Let's think step by step:
1. Relevant sources: [1], [3]
2. Key information: ...
3. Synthesis: ...

According to recent research [1], RAG systems...

Sources:
#1: Title of Paper (Score: 0.854)
By: Author Name | Published: 2024-01-15

Retrieval Info:
Retrieved 20 candidates, reranked with MMR, returned 5 results (12ms)
```

## 🔗 Using LangChain Components

The system is now built on LangChain, allowing direct access to powerful composable components:

### Document Loader

```python
from app.arxiv_document_loader import ArXivLoader

# Load and chunk ArXiv papers
loader = ArXivLoader(
    search_query="cat:cs.AI",
    max_results=10,
    chunk_documents=True,
    use_semantic_chunking=True
)

documents = loader.load()  # Returns list of LangChain Document objects
```

### Retrievers

```python
from app.vectorstore_chroma import ChromaHybridStore

# Initialize vector store
store = ChromaHybridStore()

# Get LangChain retriever with MMR
retriever = store.get_langchain_retriever(
    search_type="mmr",
    search_kwargs={"k": 5, "fetch_k": 20, "lambda_mult": 0.5}
)

# Or get ensemble retriever (semantic + BM25)
ensemble = store.get_ensemble_retriever(
    top_k=5,
    semantic_weight=0.7,
    bm25_weight=0.3
)

# Use in chains
docs = retriever.invoke("What is attention mechanism?")
```

### RAG Chains

```python
from app.rag import create_rag_chain

# Create complete RAG chain
chain = create_rag_chain(
    retriever=retriever,
    use_cot=True,  # Chain-of-thought prompting
    streaming=False
)

# Invoke chain
answer = chain.invoke("Explain transformer architecture")

# Or with streaming
for chunk in chain.stream("Explain transformer architecture"):
    print(chunk, end="", flush=True)
```

### Direct LLM Access

```python
from app.rag import get_llm

# Get LangChain LLM
llm = get_llm(streaming=False)

# Use directly
response = await llm.ainvoke("Your prompt here")

# Or with streaming
async for chunk in llm.astream("Your prompt here"):
    print(chunk.content, end="", flush=True)
```

### Custom Chains

```python
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from app.rag import get_llm

# Build custom chain
prompt = PromptTemplate.from_template("Summarize: {text}")
llm = get_llm()
parser = StrOutputParser()

chain = prompt | llm | parser
result = chain.invoke({"text": "Your text here"})
```

## ⚙️ Configuration

### ChromaDB Vector Store

```env
# ChromaDB directory (persistent storage)
CHROMA_DIR=data/chroma_db
```

### Embeddings

```env
# Best quality (1024 dim, multilingual)
EMBED_MODEL=intfloat/multilingual-e5-large

# Alternatives:
# intfloat/multilingual-e5-base        # 768 dim, faster
# intfloat/multilingual-e5-small       # 384 dim, fastest
# sentence-transformers/all-MiniLM-L6-v2  # 384 dim, English only
```

### LLM Providers

**OpenRouter** (default):
```env
LLM_MODE=openrouter
OPENROUTER_API_KEY=sk-or-v1-...
OPENROUTER_MODEL=anthropic/claude-3.5-sonnet
```

**Ollama** (local):
```env
LLM_MODE=ollama
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=llama3.1
```

**Mock** (testing):
```env
LLM_MODE=mock
```

### Retrieval Settings

```env
# Results configuration
TOP_K=5                   # Final results to return
RETRIEVAL_K=20            # Candidates before reranking

# Chunking
CHUNK_SIZE=900
CHUNK_OVERLAP=150
USE_SEMANTIC_CHUNKING=true
SEMANTIC_SIMILARITY_THRESHOLD=0.7

# Hybrid search weights
SEMANTIC_WEIGHT=0.7       # Semantic search weight
BM25_WEIGHT=0.3           # Lexical search weight
```

### Reranking

```env
# Cross-encoder reranking
RERANK_ENABLED=true
RERANK_MODEL=cross-encoder/ms-marco-MiniLM-L-6-v2

# MMR (Maximal Marginal Relevance) for diversity
RERANK_USE_MMR=true
MMR_LAMBDA=0.5            # 0=max diversity, 1=max relevance
```

### Caching

```env
CACHE_ENABLED=true
REDIS_HOST=localhost
REDIS_PORT=6379
REDIS_DB=0
CACHE_TTL=3600            # 1 hour
```

### Monitoring

```env
METRICS_ENABLED=true
PROMETHEUS_PORT=8001
```

## 🌐 API Endpoints

### Web Interface
- `GET /` - Main search page
- `GET /web/build` - Index building page
- `GET /docs` - Interactive API documentation (Swagger)

### Core API
- `POST /build` - Build new search index from ArXiv papers
- `POST /ask` - Ask questions with streaming support
- `GET /health` - System health check
- `GET /config` - Current configuration

### Search
- `GET /search` - Search without LLM generation

### Monitoring
- `GET /metrics` - Prometheus metrics (text format)
- `GET /metrics/summary` - Human-readable metrics summary
- `GET /cache/stats` - Cache statistics
- `POST /cache/clear` - Clear cache entries

### Streaming

Server-Sent Events for real-time responses:

```bash
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "What is RAG?", "stream": true}'
```

## 📊 Monitoring

### Prometheus Metrics

Access at `http://localhost:8001/metrics`:

- **Request metrics**: `arxiv_rag_requests_total`, `arxiv_rag_request_latency_seconds`
- **Retrieval metrics**: `arxiv_rag_retrieval_latency_seconds`, `arxiv_rag_documents_retrieved`
- **LLM metrics**: `arxiv_rag_llm_latency_seconds`, `arxiv_rag_llm_tokens_total`
- **Cache metrics**: `arxiv_rag_cache_hits_total`, `arxiv_rag_cache_misses_total`

### Health Check

```bash
curl http://localhost:8000/health
```

Returns:
```json
{
  "status": "ok",
  "llm_mode": "openrouter",
  "llm_health": {"healthy": true, "details": "..."},
  "index_loaded": true,
  "index_documents": 150,
  "embedder_loaded": true,
  "reranker_enabled": true
}
```

## 🧪 Development

### Running Tests

```bash
pytest
```

### Evaluation with RAGAS

The system includes comprehensive evaluation capabilities using RAGAS metrics:

**Command Line:**

```bash
# Run single evaluation
python -m app.evals

# Run comparative evaluation (MMR vs Cross-Encoder vs None)
python -m app.evals --compare
```

**Via API:**

```bash
# Run evaluation
curl -X POST http://localhost:8000/evaluate \
  -H "Content-Type: application/json" \
  -d '{
    "rerank_method": "mmr",
    "use_cot_prompting": true
  }'

# Run comparative evaluation
curl -X POST http://localhost:8000/evaluate/compare \
  -H "Content-Type: application/json" \
  -d '{"compare_rerank_methods": true}'

# Get latest results
curl http://localhost:8000/evaluate/results

# Get evaluation history
curl http://localhost:8000/evaluate/history?limit=5
```

**Custom Test Dataset:**

Create a JSON file with your test questions:

```json
[
  {
    "question": "Your question here",
    "ground_truth": "Expected answer here"
  }
]
```

Then run with:

```bash
curl -X POST http://localhost:8000/evaluate \
  -H "Content-Type: application/json" \
  -d '{"test_dataset_path": "path/to/your/dataset.json"}'
```

**RAGAS Metrics:**

- **Faithfulness**: Answer consistency with retrieved sources
- **Answer Relevancy**: How well the answer addresses the question
- **Context Precision**: Quality of retrieved documents
- **Context Recall**: Coverage of relevant information
- **Answer Similarity**: Semantic similarity to ground truth
- **Answer Correctness**: Factual accuracy

**Output:**

Results are saved to `data/processed/`:
- `eval_summary_TIMESTAMP.json` - Complete results with metrics
- `eval_results_TIMESTAMP.csv` - Detailed per-question results
- `ragas_scores_latest.csv` - Latest RAGAS scores
- `eval_comparison_TIMESTAMP.json` - Comparative analysis (if using --compare)

### Logging

```env
LOG_LEVEL=INFO          # DEBUG, INFO, WARNING, ERROR
LOG_FORMAT=console      # "console" or "json" (for production)
```

## 🎯 Performance Tips

1. **Enable Redis Caching**: 10-100x speedup for repeated queries
2. **Tune MMR Lambda**:
   - λ=0.8-1.0 for high precision
   - λ=0.3-0.5 for diverse results
3. **Adjust RETRIEVAL_K**: Higher values improve recall but increase latency
4. **Semantic Chunking**: Better quality but slower indexing

## 🐛 Troubleshooting

### No Answer Displayed

1. Check logs for OpenRouter errors
2. Verify `OPENROUTER_API_KEY` in `.env`
3. Confirm model name: `anthropic/claude-3.5-sonnet` (not `claude-4.5-sonnet`)
4. Restart server after `.env` changes

### Index Not Loading

1. Run `/web/build` to create new index
2. Check `data/chroma_db/` directory exists
3. Verify ArXiv API accessibility

### Cache Not Working

1. Ensure Redis is running: `redis-cli ping` (should return PONG)
2. Check `CACHE_ENABLED=true` in `.env`
3. View cache stats: `curl http://localhost:8000/cache/stats`

### Slow Queries

1. Enable Redis caching
2. Reduce `RETRIEVAL_K` (try 10-15)
3. Disable reranking if not needed
4. Use smaller embedding model (e5-base or e5-small)

## 📈 Benchmarks

**System**: E5-large embeddings, ChromaDB, Claude 3.5 Sonnet

| Operation | Latency | Notes |
|-----------|---------|-------|
| Embedding (query) | ~50ms | E5-large, CPU |
| Retrieval (hybrid) | ~30ms | ChromaDB, 10K docs |
| Reranking (MMR) | ~100ms | 20 candidates → 5 results |
| LLM Generation | ~2-5s | Claude 3.5, streaming |
| **Total (cached)** | **~50ms** | Cache hit |
| **Total (uncached)** | **~3-6s** | Full pipeline |

## 🤝 Contributing

Contributions welcome! Please:

1. Fork the repository
2. Create a feature branch
3. Submit a pull request with tests

## 📄 License

MIT License - see LICENSE file for details

## 🙏 Acknowledgments

- **ChromaDB** - Scalable vector database
- **Sentence Transformers** - E5 multilingual embeddings
- **RAGAS** - RAG evaluation framework
- **FastAPI** - Modern web framework
- **ArXiv** - Open access research papers

---

**Built with ❤️ for AI researchers and developers**

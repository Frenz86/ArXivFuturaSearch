# 🔍 ArXiv Futura Search - AI-Powered Research Paper Assistant

**Search, explore, and understand ML research papers with the power of AI. Built by [Futura AI](https://futura.ai)**

![ArXiv Futura Search](https://img.shields.io/badge/version-1.0-blue)
![Python](https://img.shields.io/badge/python-3.10+-green)
![FastAPI](https://img.shields.io/badge/FastAPI-0.104+-red)
![License](https://img.shields.io/badge/license-MIT-purple)

---

## ✨ Features

### 🧠 Advanced AI Search
- **E5 Multilingual Embeddings** - State-of-the-art multilingual model for 100+ languages
- **Hybrid Search** - Combines semantic (vector) and lexical (BM25) search for optimal results
- **Query Expansion** - Automatically expands your queries with related terms and acronyms
- **Smart Reranking** - Cross-encoder reranking with MMR for diverse, relevant results

### 📚 Research-Ready Tools
- **One-Click BibTeX Export** - Copy citations instantly for your papers
- **Semantic Chunking** - Intelligently splits papers into coherent chunks
- **Source Attribution** - Every answer comes with proper citations and scores
- **Multi-Paper Indexing** - Index hundreds of papers with preset topics

### 🎨 Beautiful Interface
- Clean, modern UI with real-time streaming responses
- Markdown-rendered answers with syntax highlighting
- Visual quality indicators (color-coded relevance scores)
- Mobile-responsive design

### ⚡ Fast & Efficient
- Local embedding model (no API costs!)
- Streaming responses for instant feedback
- Built-in caching for faster repeated queries
- Configurable search parameters

---

## 🚀 Quick Start

```bash
# Clone the repository
git clone https://github.com/yourusername/arxiv-futura-search.git
cd arxiv-futura-search

# Install dependencies
pip install -r requirements.txt

# Set up your environment
cp .env.example .env
# Edit .env with your OpenRouter API key

# Run the server
uv run uvicorn app.main:app --reload
```

Visit **http://localhost:8000** and start searching!

---

## 🎯 Use Cases

- **Literature Review** - Quickly find relevant papers for your research
- **Concept Exploration** - Ask questions like "What is chain-of-thought prompting?"
- **Citation Management** - Export BibTeX with one click
- **Learning** - Understand complex ML concepts with AI-generated explanations
- **Writing** - Get AI assistance while writing your papers

---

## 🏗️ Architecture

```
┌─────────────┐    ┌──────────────┐    ┌─────────────┐
│   ArXiv API │ ──▶│   FastAPI    │ ──▶│   Web UI     │
└─────────────┘    └──────────────┘    └─────────────┘
                          │
                          ▼
              ┌───────────────────────┐
              │  ChromaDB Vector Store  │
              │  + BM25 Retrieval      │
              │  + Cross-Encoder Rerank│
              └───────────────────────┘
                          │
                          ▼
              ┌───────────────────────┐
              │   E5 Multilingual      │
              │   Embedding Model     │
              └───────────────────────┘
```

### Tech Stack
- **Backend**: FastAPI + Python 3.10+
- **Vector DB**: ChromaDB with LangChain
- **Embeddings**: `intfloat/multilingual-e5-large`
- **Reranking**: `cross-encoder/ms-marco-MiniLM-L-6-v2`
- **LLM**: OpenRouter (Claude, GPT-4, Gemini, Llama)

---

## 📖 Example Usage

### Search Interface
Simply ask a question in natural language:

> "What are the main challenges in retrieval-augmented generation?"

The system will:
1. Search across all indexed papers
2. Retrieve the most relevant chunks
3. Generate a comprehensive answer with citations
4. Show you exactly which sources were used

### BibTeX Export
Click the **📋 BibTeX** button on any source to instantly copy:

```bibtex
@misc{kahana2024disc,
  title={Discovering Hidden Gems in Model Repositories},
  author={Kahana, J and Horwitz, E and Hoshen, Y},
  year={2024},
  eprint={2401.12345},
  archivePrefix={arXiv},
  primaryClass={cs.CL}
}
```

---

## 🎨 Preset Topics

One-click indexing for popular research areas:

| Topic | Papers | Description |
|-------|--------|-------------|
| 🤖 AI & ML | 30/100/200 | Artificial Intelligence & Machine Learning |
| 🧠 Transformers | 30/100/200 | Attention mechanisms & transformer architectures |
| 🔍 RAG Systems | 30/100/200 | Retrieval-Augmented Generation |
| 💬 NLP | 30/100/200 | Natural Language Processing |
| 🖼️ Computer Vision | 30/100/200 | Vision models & image processing |
| 🎯 RL & Optimization | 30/100/200 | Reinforcement Learning |
| 🌐 Multilingual | 30/100/200 | Multilingual & cross-lingual models |
| ⚡ Efficient ML | 30/100/200 | Model compression & efficiency |

---

## 🌍 Why ArXiv Futura Search?

Traditional search engines struggle with research questions because:
- ❌ They match keywords, not meaning
- ❌ They can't synthesize information from multiple papers
- ❌ They require exact terminology
- ❌ They don't provide citations

**ArXiv Futura Search** solves all of these problems with:
- ✅ **Semantic understanding** - Finds relevant papers even with different terminology
- ✅ **Answer synthesis** - Combines information from multiple sources
- ✅ **Natural language** - Ask questions however you prefer
- ✅ **Proper citations** - Every claim is backed by sources

---

## 🤝 Contributing

We welcome contributions! Areas for improvement:
- [ ] Additional embedding model options
- [ ] Search history & saved queries
- [ ] Dark mode
- [ ] Advanced filters (date, author, venue)
- [ ] Export results as PDF/Markdown
- [ ] Multi-language UI support

---

## 📄 License

MIT License - feel free to use this project for your research!

---

## 🙏 Acknowledgments

- **ArXiv** - Open access to scientific literature
- **Hugging Face** - E5 embeddings and cross-encoder models
- **LangChain** - Framework for LLM applications
- **ChromaDB** - Vector database for semantic search
- **OpenRouter** - Access to frontier LLMs

---

## 📬 Contact

Built with ❤️ by **[Futura AI](https://futuraaigroup.com)**

- **Issues**: Report bugs and feature requests on GitHub Issues
- **Discussions**: Join the community discussions

---

## 🌟 Star the repo if you find it useful! ⭐

Made with research in mind. Happy searching! 🎓

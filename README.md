# 🚀 ROCm-Pilot: AI-Powered AMD Setup Assistant

> An intelligent RAG agent that helps developers set up AI/ML workloads on AMD hardware — grounded in official ROCm documentation.

**Built for:** [AMD Developer Hackathon ACT II](https://lablab.ai/ai-hackathons/amd-developer-hackathon-act-ii) — Track 3 (Unicorn)

---

## 🎯 The Problem

Setting up AI and machine learning environments on AMD hardware is hard. Developers face:

- Fragmented documentation across dozens of repos and websites
- Complex version compatibility matrices (ROCm ↔ PyTorch ↔ GPU ↔ OS ↔ Kernel)
- Different setup paths for Instinct (data center) vs. Radeon (consumer) GPUs
- Confusing CUDA-to-ROCm migration guidance

## 💡 The Solution

**ROCm-Pilot** is an AI assistant that:

1. **Auto-detects your AMD hardware** (GPU model, ROCm version, installed frameworks)
2. **Retrieves relevant official documentation** from AMD's GitHub repos
3. **Generates tailored, copy-pasteable setup instructions** via Fireworks AI
4. **Cites its sources** so you can verify every recommendation

## 🏗️ Architecture

```
User Question → Environment Detection → Document Retrieval (ChromaDB)
                                              ↓
                              Context Assembly → Fireworks AI LLM → Grounded Answer
```

| Component | Technology |
|---|---|
| **LLM** | Fireworks AI (Llama 3.1 70B) |
| **Embeddings** | sentence-transformers on AMD GPU (ROCm) |
| **Vector Store** | ChromaDB (persistent, local) |
| **Knowledge Base** | Official ROCm docs from GitHub |
| **Environment Detection** | rocm-smi, rocminfo, PyTorch introspection |

## 📦 Knowledge Base Sources

| Source | Repository |
|---|---|
| ROCm Platform Docs | [ROCm/ROCm](https://github.com/ROCm/ROCm) |
| Linux Install Guides | [ROCm/rocm-install-on-linux](https://github.com/ROCm/rocm-install-on-linux) |
| Technical Blogs | [ROCm/rocm-blogs](https://github.com/ROCm/rocm-blogs) |
| AI Developer Hub | [ROCm/gpuaidev](https://github.com/ROCm/gpuaidev) |

## 🚀 Quick Start

### 1. Clone the repo

```bash
git clone https://github.com/YOUR_USERNAME/ROCm-Pilot.git
cd ROCm-Pilot
```

### 2. Set your Fireworks AI API key

```bash
export FIREWORKS_API_KEY='your-fireworks-api-key'
```

### 3. Run the setup script

```bash
chmod +x setup.sh
bash setup.sh
```

This will:
- Install PyTorch with ROCm support
- Install all Python dependencies
- Clone official AMD documentation repos
- Build the vector knowledge base (GPU-accelerated embeddings)

### 4. Start ROCm-Pilot

```bash
python3 -m src.agent
```

## 💬 Example Queries

```
🧑 You: How do I install PyTorch on MI300X?
🧑 You: What ROCm version supports Ubuntu 24.04?
🧑 You: How do I run Llama 3 with vLLM on AMD?
🧑 You: What's the difference between MI300X and MI250?
🧑 You: How do I check if my GPU is detected by ROCm?
```

## 📁 Project Structure

```
ROCm-Pilot/
├── setup.sh                 # One-shot setup (deps + knowledge base)
├── requirements.txt         # Python dependencies
├── README.md
├── src/
│   ├── __init__.py
│   ├── scraper.py           # Collects docs from cloned GitHub repos
│   ├── chunker.py           # Splits docs into embeddable chunks
│   ├── embedder.py          # GPU-accelerated embedding + ChromaDB storage
│   ├── retriever.py         # Semantic search over the knowledge base
│   ├── agent.py             # Main RAG agent (orchestrator)
│   ├── env_detector.py      # AMD hardware auto-detection
│   └── fireworks_client.py  # Fireworks AI API wrapper
├── data/
│   ├── raw_docs/            # Cloned documentation repos
│   └── chroma_db/           # Persistent vector store
└── notebooks/               # Jupyter demo notebooks
```

## 🔧 Tech Stack

- **AMD Instinct MI300X** — GPU compute for embeddings
- **ROCm 6.x** — AMD's open-source GPU compute platform
- **PyTorch (ROCm)** — Deep learning framework
- **Fireworks AI** — LLM inference API (Llama 3.1 70B)
- **ChromaDB** — Lightweight vector database
- **sentence-transformers** — Embedding generation

## 🏆 Why This Matters

ROCm-Pilot directly addresses the biggest barrier to AMD GPU adoption in AI: **setup complexity**. By making it trivially easy to get started with AMD hardware, this tool can:

- **Lower the barrier to entry** for developers new to AMD
- **Reduce support burden** for AMD's developer relations team
- **Accelerate AMD ecosystem adoption** in the AI community

## 👤 Team

- **Solo Developer** — AMD Developer Hackathon ACT II

## 📄 License

MIT

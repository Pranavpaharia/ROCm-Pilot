# 🚀 ROCm-Pilot: AI-Powered AMD Setup Assistant

> An intelligent RAG agent that provides technical assistance to run AI/ML tools and workflows on AMD-based hardware — grounded in official ROCm documentation.

**Built for:** [AMD Developer Hackathon ACT II](https://lablab.ai/ai-hackathons/amd-developer-hackathon-act-ii) — Track 3 (Unicorn)

---

## 🎯 The Problem

Setting up AI and machine learning environments on AMD hardware is hard. Developers face:

- Fragmented documentation across dozens of repos and websites
- Complex version compatibility matrices (ROCm ↔ PyTorch ↔ GPU ↔ OS ↔ Kernel)
- Different setup paths for **AMD Instinct™** (Data Center) vs. **AMD Radeon™ RX** (Consumer/Workstation) GPUs
- Confusing CUDA-to-ROCm migration guidance

## 💡 The Solution

**ROCm-Pilot** is an AI assistant that:

1. **Auto-detects your AMD hardware** (GPU model, ROCm version, installed frameworks)
2. **Retrieves relevant official documentation** from AMD's GitHub repos using **Hybrid Search (BM25 + ChromaDB Vector)**
3. **Generates tailored, copy-pasteable setup instructions** using **Native On-Device LLMs** or Cloud APIs.
4. **Cites its sources** so you can verify every recommendation

---

## ✨ Key Features & Tech Stack

We've massively expanded ROCm-Pilot to be a production-ready, resilient RAG pipeline. Here is the tech stack and our core features:

- 🧠 **Native Local AI Integration:** Run completely private, open-weight models natively on AMD GPUs. **Currently defaults to the bleeding-edge Google Gemma-4-12B-it model**, powered by the latest `transformers` library compiled from source. It runs natively in the VRAM of your AMD hardware without requiring API keys!
- ☁️ **AMD GPU Cloud (DigitalOcean):** Fully tested and deployed on DigitalOcean's **AMD Instinct MI300X (192GB VRAM)** instances, ensuring massive scalability and lightning-fast inference.
- 🔍 **Hybrid Search (Dense + Sparse):** We overhauled the retrieval engine to use a Hybrid approach. It combines **ChromaDB dense vectors** (`BAAI/bge-large-en-v1.5`) for semantic matching with the **BM25 Sparse Algorithm** (`rank_bm25`) for exact keyword matching (like specific ROCm version numbers or GPU IDs).
- 📚 **Dual-Tiered Knowledge Base:** Our RAG documents are structured with both **core** and **full versions**, ensuring the agent can retrieve either quick reference snippets or deep, comprehensive technical guides depending on the complexity of the query.
- 🛠️ **Agentic Tool Calling (Read-Only):** The AI operates autonomously with **read-only tool calling enabled**, allowing it to safely introspect system environments (like `rocm-smi`) and verify hardware states without executing destructive commands.
- 🛠️ **Ultra-Fast `uv` Dependency Management:** The monolithic setup script has been replaced by an environment-aware, modular system powered by Astral's `uv`. This guarantees lightning-fast, reproducible, and isolated `.venv` builds across any AMD machine.
- 🖥️ **Sleek Web Interface:** A beautiful, dark-mode Gradio web app with source citations, diagnostic readouts, and model toggles.
- 🛡️ **Resilient AI Pipeline:** Automatic model fallback capabilities and error handling to ensure continuous operation even during configuration mismatches.
- 🍋 **Lemonade SDK Integration:** Automated deployment and robust model serving via the **Lemonade SDK**, seamlessly integrating partner tech with AMD hardware.

---

## 🏗️ Architecture

```text
User Question → Environment Detection 
                      ↓
Document Retrieval (Hybrid: BM25 + ChromaDB) → Context Assembly
                      ↓
LLM (Native Gemma-4 on AMD / Cloud) → Grounded Answer
```

| Component | Technology |
|---|---|
| **LLM Inference** | Native `google/gemma-4-12B-it` on AMD GPU or Fireworks Cloud API |
| **Embeddings** | `BAAI/bge-large-en-v1.5` on AMD GPU |
| **Sparse Retrieval** | `rank_bm25` for precise keyword matching |
| **Vector Store** | ChromaDB (persistent, local) |
| **Environment Detection** | rocm-smi, rocminfo, PyTorch introspection |
| **Cloud Infrastructure** | DigitalOcean AMD Instinct MI300X Droplets |

## 📦 Knowledge Base Sources

| Source | Repository |
|---|---|
| ROCm Platform Docs | [ROCm/ROCm](https://github.com/ROCm/ROCm) |
| Linux Install Guides | [ROCm/rocm-install-on-linux](https://github.com/ROCm/rocm-install-on-linux) |
| Technical Blogs | [ROCm/rocm-blogs](https://github.com/ROCm/rocm-blogs) |
| AI Developer Hub | [ROCm/gpuaidev](https://github.com/ROCm/gpuaidev) |
| GPU Database | Internal JSON device registry |

## 🚀 Quick Start

### 1. Clone the repo

```bash
git clone https://github.com/YOUR_USERNAME/ROCm-Pilot.git
cd ROCm-Pilot
```

### 2. Run the environment setup

We use `uv` for blazing-fast, isolated Python dependency management.

```bash
chmod +x setup-deps.sh
./setup-deps.sh
```

This will automatically create a `.venv`, install PyTorch for ROCm, and pull the latest HuggingFace libraries.

### 3. Start the Agent

You can start the RAG agent in two ways:

**To run natively on your AMD GPU (Downloads Gemma-4-12B into VRAM):**
```bash
./run.sh --provider local_gpu
```

**To run using the cloud API:**
*(Make sure to set `FIREWORKS_API_KEY` in your `.env` file!)*
```bash
./run.sh --provider cloud
```

## 💬 Example Queries

```
🧑 You: How do I install PyTorch on MI300X?
🧑 You: What ROCm version supports Ubuntu 24.04?
🧑 You: How do I run Llama 3 with vLLM on AMD?
🧑 You: What's the difference between MI300X and MI250?
🧑 You: What version of PyTorch do I need for gfx942?
```

## 📁 Project Structure

```text
ROCm-Pilot/
├── setup-deps.sh            # Ultra-fast dependency setup using uv
├── run.sh                   # Entrypoint for the agent
├── src/
│   ├── app_web.py           # Gradio Web Interface
│   ├── scraper.py           # Collects docs from cloned GitHub repos
│   ├── chunker.py           # Splits docs into embeddable chunks
│   ├── embedder.py          # GPU-accelerated embedding + ChromaDB storage
│   ├── retriever.py         # Hybrid search (Chroma + BM25)
│   ├── agent.py             # Main RAG agent (orchestrator)
│   ├── env_detector.py      # AMD hardware auto-detection & GPU Monitor
│   └── llm_provider.py      # LLM wrapper (Local Gemma 4 & Cloud Fallback)
├── data/                    # Vector store and raw docs
└── notebooks/               # Jupyter demo notebooks
```

## 🏆 Why This Matters

ROCm-Pilot directly addresses the biggest barrier to AMD GPU adoption in AI: **setup complexity**. By making it trivially easy to get started with AMD hardware, this tool can:

- **Lower the barrier to entry** for developers new to AMD
- **Reduce support burden** for AMD's developer relations team
- **Accelerate AMD ecosystem adoption** in the AI community

## 👤 Team

- **Solo Developer** — AMD Developer Hackathon ACT II

## 📄 License

MIT

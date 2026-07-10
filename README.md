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
3. **Generates tailored, copy-pasteable setup instructions** via Fireworks AI or Local Models
4. **Cites its sources** so you can verify every recommendation

---

## ✨ Key Features (New Updates!)

We've massively expanded ROCm-Pilot during the hackathon. Here is what's new:

- 🖥️ **Sleek Web Interface:** A beautiful, dark-mode Gradio web app with source citations, diagnostic readouts, and model toggles.
- 🚀 **Universal AMD Support (Tested on MI300X):** While we extensively tested and optimized for the massive 192GB Instinct MI300X, ROCm-Pilot's hardware detection and setup guidance **works seamlessly across all ROCm-compatible AMD hardware** (including Radeon consumer GPUs like RX 7900 XTX and older Instinct accelerators).
- 🧠 **Local Open-Weights Inference:** Run completely private, local open-weight models (like `gemma-4-12b-it` up to `31b-it`) directly on your AMD GPU VRAM using HuggingFace Accelerate and PyTorch native integration.
- ☁️ **Cloud API Integration:** Blazing-fast inference via Fireworks AI API, fully supporting next-gen models like `deepseek-v4-pro` and `glm-5p2`.
- 📊 **Live AMD GPU Monitor:** Real-time VRAM, Temperature, and GPU Utilization progress bars with HTML color coding right inside the Web UI.
- 🛡️ **Resilient AI Pipeline (OOM Failover):** If a local LLM or ghost process fills up all 192GB of your MI300X VRAM, the semantic embedding and cross-encoder models will gracefully catch the `OutOfMemoryError` and **dynamically fail over to the CPU** so the app never crashes!
- 🤔 **Dynamic Reasoning UI:** Intercepts and beautifully formats hidden "Chain-of-Thought" reasoning streams (from models like DeepSeek Pro) into sleek HTML accordions.
- 📡 **Remote Machine Diagnostics:** Running the UI on your laptop but deploying to a remote server? Run our one-line `diagnose_system.py` script on any remote AMD machine and paste the JSON output directly into the Web UI to inject the remote machine's context into the LLM!
- 🎯 **Cross-Encoder Reranking:** We added an MS-MARCO Cross-Encoder pipeline to re-score and re-rank vector search results, drastically improving documentation retrieval precision.

---

## 🏗️ Architecture

```
User Question → Environment Detection → Document Retrieval (ChromaDB + Cross-Encoder)
                                              ↓
                               Context Assembly → LLM (Local or Cloud) → Grounded Answer
```

| Component | Technology |
|---|---|
| **LLM Inference** | Fireworks AI API or Local AMD VRAM (`device_map="auto"`) |
| **Embeddings** | `BAAI/bge-large-en-v1.5` via sentence-transformers on AMD GPU |
| **Reranking** | `cross-encoder/ms-marco-MiniLM-L-6-v2` |
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

### 2. Set your API Key (Optional)

If using Cloud Models:
```bash
export FIREWORKS_API_KEY='your-fireworks-api-key'
```

### 3. Run the setup script

```bash
chmod +x setup.sh
bash setup.sh
```

This will:
- Install PyTorch with ROCm support and dependencies (including `accelerate` for local multi-GPU).
- Clone official AMD documentation repos.
- Build the vector knowledge base using GPU-accelerated embeddings.

### 4. Start the Web UI

```bash
python3 src/app_web.py --share
```
This will spin up the web interface and print a public `gradio.live` link so you can access the UI from anywhere.

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
├── src/
│   ├── app_web.py           # Gradio Web Interface
│   ├── scraper.py           # Collects docs from cloned GitHub repos
│   ├── chunker.py           # Splits docs into embeddable chunks
│   ├── embedder.py          # GPU-accelerated embedding + ChromaDB storage
│   ├── retriever.py         # Semantic search & Cross-Encoder reranking
│   ├── agent.py             # Main RAG agent (orchestrator)
│   ├── env_detector.py      # AMD hardware auto-detection & GPU Monitor
│   ├── llm_provider.py      # LLM wrapper (Cloud & Local)
│   ├── fireworks_client.py  # Fireworks API integration
│   └── diagnose_system.py   # Remote environment diagnosis script
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

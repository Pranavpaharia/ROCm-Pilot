# ROCm-Pilot — Architecture Document

> **Purpose:** This document gives an AI agent (or a human developer) a complete, machine-readable understanding of every file in the ROCm-Pilot repository, its role, dependencies, and how all pieces fit together.

---

## 1. Project Overview

**ROCm-Pilot** is an AI-powered AMD setup assistant built for the [AMD Developer Hackathon ACT II](https://lablab.ai/ai-hackathons/amd-developer-hackathon-act-ii). It is a **RAG (Retrieval-Augmented Generation)** system that:

1. Auto-detects the user's AMD hardware (GPU model, ROCm version, frameworks)
2. Retrieves relevant official ROCm documentation from cloned GitHub repos
3. Generates tailored, copy-pasteable setup instructions via an LLM
4. Cites its sources so every recommendation is verifiable

**Target Hardware:** AMD Instinct MI300X (GPU compute via ROCm 6.x / 7.2)
**LLM Backend:** Fireworks AI (cloud) or local Gemma 4 12B on AMD GPU
**Vector Store:** ChromaDB (persistent, local)
**Embeddings:** sentence-transformers on AMD GPU via ROCm

---

## 2. Directory Layout

```
ROCm-Pilot/
├── README.md                  # Project overview, quick start, tech stack
├── setup.sh                   # One-shot bootstrap: deps + docs clone + KB build
├── requirements.txt           # Python dependencies (pip)
├── .gitignore                 # Excludes data/, __pycache__, .env, venv/
├── src/
│   ├── __init__.py            # Package marker, version string ("1.0.0")
│   ├── scraper.py             # Collects .md/.rst docs from cloned repos
│   ├── chunker.py             # Splits documents into overlapping chunks (preserves tables)
│   ├── embedder.py            # GPU-accelerated embedding + ChromaDB vector store builder
│   ├── retriever.py           # Semantic search over ChromaDB + Cross-Encoder reranking
│   ├── env_detector.py        # AMD hardware/software auto-detection (multi-GPU aware)
│   ├── llm_provider.py        # Abstraction: Cloud (Fireworks) or Local (HuggingFace)
│   ├── fireworks_client.py    # Fireworks AI API wrapper (OpenAI-compatible endpoint)
│   ├── agent.py               # Main RAG orchestrator + interactive CLI
│   ├── app_web.py             # Gradio-based web UI (chat, GPU monitor, source citations)
│   └── diagnose_system.py     # Standalone diagnostic script (zero deps, for remote machines)
├── tests/
│   └── test_basic.py          # Sanity checks: env detector, provider compilation
├── data/                      # [Generated at runtime — gitignored]
│   ├── raw_docs/              # Cloned ROCm documentation repos
│   └── chroma_db/             # Persistent ChromaDB vector store
└── notebooks/                 # [Referenced in README but not yet created]
```

---

## 3. Module-by-Module Breakdown

### 3.1 `src/__init__.py`

| Attribute | Value |
|---|---|
| Purpose | Package marker |
| Exports | `__version__ = "1.0.0"` |

---

### 3.2 `src/scraper.py`

**Role:** Collects documentation from cloned GitHub repos into a unified list of document dicts.

**Key Functions:**
- `classify_doc_type(filepath)` -> Classifies a doc by path keywords into: `installation`, `blog`, `tutorial`, `reference`, `conceptual`, `example`, or `general`.
- `_extract_markdown_tables(content)` -> Extracts GFM-style markdown tables (header + separator + data rows) into structured dicts with `headers`, `rows`, and `raw` text.
- `_parse_table_row(row)` -> Splits a pipe-delimited row into cell list.
- `_format_table_for_embedding(table)` -> Converts a table dict to plain text for embedding.
- `collect_documents(raw_docs_dir)` -> Walks the directory tree, reads `.md`/`.rst`/`.txt` files (skipping build dirs), extracts tables, and returns a list of dicts:
  ```python
  {
      "content": str,           # Raw file text (truncated at 50k chars)
      "source_repo": str,       # e.g. "ROCm"
      "source_file": str,       # Relative path within repo
      "source_url": str,        # Estimated URL on rocm.docs.amd.com
      "doc_type": str,          # Classification from classify_doc_type()
      "tables": List[Dict],     # Extracted markdown tables
  }
  ```

**Dependencies:** `os`, `re`, `pathlib.Path`, `tqdm`

---

### 3.3 `src/chunker.py`

**Role:** Splits scraped documents into overlapping chunks suitable for embedding, while preserving markdown tables as intact blocks.

**Key Functions:**
- `_split_by_headers(content, file_ext)` -> Splits by Markdown (`#`) or RST headers into sections.
- `_detect_and_replace_tables(text)` -> Replaces markdown tables with placeholder tokens (`<<TABLE_BLOCK_{idx}>>`) so they are not split across chunks. Returns modified text + list of table dicts.
- `_chunk_text_preserving_tables(modified_text, tables, chunk_size, overlap)` -> Chunks text by words with configurable overlap, restoring table placeholders.
- `_restore_tables_in_chunk(chunk_text, tables)` -> Replaces placeholder tokens back with formatted table text.
- `chunk_documents(documents, chunk_size=500, overlap=50)` -> Main entry point. Returns a list of chunk dicts:
  ```python
  {
      "text": str,
      "source_repo": str,
      "source_file": str,
      "source_url": str,
      "doc_type": str,
      "section_title": str,
      "chunk_index": int,
      "has_table": bool,       # Whether this chunk contains table data
  }
  ```

**Dependencies:** `re`, `typing`, `tqdm`

---

### 3.4 `src/embedder.py`

**Role:** Generates GPU-accelerated embeddings for all chunks and stores them in a persistent ChromaDB collection.

**Key Functions:**
- `get_device()` -> Returns `'cuda'` if AMD GPU detected via PyTorch, else `'cpu'`.
- `build_vector_store(chunks, db_path='data/chroma_db', model_name=DEFAULT_MODEL, collection_name='rocm_docs')` ->
  1. Loads `SentenceTransformer(model_name)` on the detected device.
  2. Creates (or drops + recreates) a ChromaDB `PersistentClient` collection with cosine similarity.
  3. Embeds all chunks in batches of 64 (GPU-accelerated).
  4. Inserts into ChromaDB in batches of 5000 (ChromaDB API limit).

**Constants:**
- `DEFAULT_MODEL = 'sentence-transformers/all-MiniLM-L6-v2'`
- `BATCH_SIZE = 64`

**Dependencies:** `os`, `torch`, `tqdm`, `chromadb`, `sentence_transformers`

---

### 3.5 `src/retriever.py`

**Role:** Queries the ChromaDB vector store for relevant documentation chunks, with optional Cross-Encoder reranking.

**Key Classes:**
- `CrossEncoderReranker` -> Lazy-loaded GPU-accelerated reranker using `cross-encoder/ms-marco-MiniLM-L-6-v2`.
  - `_load_model()` -> Lazily loads the CrossEncoder on `cuda` or `cpu`.
  - `rerank(query, results, top_k=8)` -> Scores each result against the query and returns the top-k highest-scoring results with a `rerank_score` field.

**Key Functions:**
- `get_reranker()` -> Module-level singleton accessor for `CrossEncoderReranker`.
- `get_retriever(db_path='data/chroma_db', collection_name='rocm_docs')` -> Returns a ChromaDB `Collection` handle (lazy load).
- `retrieve(query, collection, embedding_model=None, top_k=5, doc_type_filter=None, use_reranker=True)` ->
  1. Fetches `top_k * 3` candidates from ChromaDB (or just `top_k` if no reranker).
  2. Optionally filters by `doc_type`.
  3. Flattens ChromaDB's nested list structure into a list of dicts.
  4. If `use_reranker=True`, re-scores with Cross-Encoder and returns top-k.
  Returns: `List[Dict]` with keys `text`, `metadata`, `distance`, `id`.
- `format_context(results, max_words=3000)` -> Formats retrieved chunks into a numbered context string with source citations for injection into the LLM prompt.

**Dependencies:** `logging`, `time`, `typing`

---

### 3.6 `src/env_detector.py`

**Role:** Auto-detects the AMD hardware and software environment on the local machine.

**Key Functions:**
- `_run_cmd(cmd, timeout=10)` -> Runs a shell command; returns stdout or `None`.
- `_detect_container()` -> Detects Docker/Podman/LXC containers via `.dockerenv`, cgroup, or `container` env var.
- `detect_gpus()` -> Detects all AMD GPUs via `rocm-smi` (multi-GPU aware). Returns list of GPU dicts with: `detected`, `card_id`, `model`, `vram`, `temperature`, `rocm_version`, `arch`, `device_index`, `pytorch_version`, `hip_version`, `detection_errors`.
- `detect_gpu_utilization()` -> Reads GPU utilization, memory usage, and temperature from `rocm-smi`.
- `detect_gpu_processes()` -> Lists GPU processes via `rocm-smi --showprocesses`.
- `detect_software()` -> Detects ROCm version, Python version, OS, and installed AI/ML frameworks (PyTorch, TensorFlow, JAX, vLLM, Transformers).
- `detect_environment()` -> Orchestrates all detection routines. Returns a comprehensive dict:
  ```python
  {
      "gpus": List[Dict],
      "software": Dict,
      "container": Dict,
      "gpu_utilization": Dict,
      "gpu_processes": List[Dict],
  }
  ```
- `format_env_context(env)` -> Formats the detected environment as a human-readable text block for injection into the LLM system prompt.
- `format_gpu_monitor(env)` -> Formats GPU utilization + processes as a markdown table for the web UI.

**Dependencies:** `json`, `logging`, `os`, `subprocess`, `sys`, `typing`

---

### 3.7 `src/llm_provider.py`

**Role:** Abstraction layer for LLM inference, supporting both cloud (Fireworks AI) and local (AMD GPU) modes.

**Key Classes:**
- `BaseLLMProvider` -> Abstract base class with `chat(messages, stream)` interface.
- `FireworksProvider(BaseLLMProvider)` -> Wraps Fireworks AI API (OpenAI-compatible endpoint).
  - `__init__(model)` -> Stores model identifier.
  - `chat(messages, stream)` -> Delegates to `src.fireworks_client.chat()`.
- `LocalGPUProvider(BaseLLMProvider)` -> Runs Gemma 4 locally on AMD GPU via HuggingFace `transformers`.
  - `DEFAULT_MODEL_ID = os.environ.get("LOCAL_MODEL_ID", "google/gemma-4-12b-it")`
  - `__init__(model_id=None)` -> Loads tokenizer, model (fp16), and pipeline on `device_map="auto"`.
  - Logs GPU diagnostics (device name, VRAM allocated/reserved, HIP version) and model size.
  - `chat(messages, stream=False, max_new_tokens=512)` -> Supports both blocking and streaming generation via `TextIteratorStreamer`.
  - `_generate_full(prompt, max_new_tokens)` -> Blocking generation returning complete response.
  - `_generate_stream(prompt, max_new_tokens)` -> Streaming via `TextIteratorStreamer` in a background thread.

**Factory Function:**
- `get_provider(provider_type, model=None)` -> Returns `LocalGPUProvider` if `provider_type == "local"`, else `FireworksProvider`.

**Dependencies:** `os`, `logging`, `threading`, `typing` (lazy imports for `torch`, `transformers`)

---

### 3.8 `src/fireworks_client.py`

**Role:** Thin wrapper around the OpenAI SDK pointing at Fireworks AI's inference endpoint.

**Key Functions:**
- `get_client()` -> Returns an `OpenAI` client with `base_url="https://api.fireworks.ai/inference/v1"` and API key from `FIREWORKS_API_KEY` env var.
- `chat(messages, model=DEFAULT_MODEL, temperature=0.3, max_tokens=2048, stream=False)` -> Sends a chat completion request. Returns full string or generator if `stream=True`.
- `_stream_response(response)` -> Yields content chunks from a streaming response.
- `test_connection(model=DEFAULT_MODEL)` -> Quick connectivity test (returns bool).

**Constants:**
- `DEFAULT_MODEL = "accounts/fireworks/models/deepseek-v4-pro"`
- `FIREWORKS_BASE_URL = "https://api.fireworks.ai/inference/v1"`

**Dependencies:** `os`, `typing` (lazy import of `openai.OpenAI`)

---

### 3.9 `src/agent.py`

**Role:** The main RAG orchestrator and interactive CLI. This is the heart of ROCm-Pilot.

**Key Components:**
- `_check_api_key()` -> Validates `FIREWORKS_API_KEY` from env or `.env` file. Prints clear error if missing.
- `SYSTEM_PROMPT_TEMPLATE` -> A large system prompt string (~150 lines) that:
  - Identifies ROCm-Pilot as an AMD setup assistant
  - Injects the detected environment context
  - Provides tool execution instructions (LLM can request `[TOOL: tool_name]`)
  - Specifies answer format with source citations
- `ToolExecutor` -> Registry of safe, read-only diagnostic commands:
  - `rocm-smi`, `rocm-smi-gpus`, `rocminfo`, `rocm-version`, `hipconfig`, `pytorch-rocm-check`, `vram-info`
  - Each tool has a description, shell command, and timeout.
  - `execute_tool(tool_name)` -> Runs the tool via `subprocess.run()`, returns stdout/stderr.
  - Supports manual approval or auto-approve mode.
- `RocmPilotAgent` -> The main agent class:
  - `__init__(db_path, auto_approve_tools)` -> Initializes vector store, embedding model, LLM provider.
  - `ask(question, stream=False)` -> Core RAG pipeline:
    1. Detects environment and formats context
    2. Retrieves relevant docs from ChromaDB
    3. Assembles system prompt with environment + retrieved context
    4. Sends to LLM (Fireworks or local)
    5. Parses response for `[TOOL: tool_name]` directives
    6. Executes tools if requested (with approval)
    7. Returns final answer with source citations
  - `clear_history()` -> Resets conversation history.
  - `get_sources(query)` -> Returns raw source documents for a query (for debugging).
  - `_log_gpu_status()` -> Logs GPU status to the conversation history.
- `interactive_session(db_path, auto_approve_tools)` -> CLI entry point:
  - Prompts for questions
  - Supports commands: `quit`, `exit`, `clear`, `sources <query>`
  - Streams LLM responses in real-time

**Dependencies:** `logging`, `os`, `re`, `subprocess`, `typing`, `pathlib` (imports from `env_detector`, `retriever`, `llm_provider`)

---

### 3.10 `src/app_web.py`

**Role:** Gradio-based web UI providing a chat interface with AMD GPU monitoring.

**Key Components:**
- `WebAgent` -> Stateful agent for Gradio sessions:
  - `__init__(db_path)` -> Initializes with cloud provider by default.
  - `initialize()` -> Lazy initialization: validates API key, loads vector store, detects environment.
  - `respond(message, history)` -> Core RAG pipeline (same as agent but returns plain text).
  - `switch_provider(choice)` -> Switches between cloud and local providers.
  - `set_remote_env(json_text)` -> Processes pasted remote diagnostics JSON to override local detection.
  - `get_status_md()` / `get_gpu_monitor_md()` -> Returns formatted markdown for UI panels.
  - `clear_history()` -> Resets conversation history.
- `build_ui(db_path)` -> Constructs the Gradio interface:
  - **Chat panel** with message history
  - **Hardware status panel** (GPU model, ROCm version, frameworks)
  - **Source citations panel**
  - **GPU monitor panel** (real-time VRAM, utilization, processes)
  - **LLM provider selector** (Cloud / Local radio buttons)
  - **Remote diagnostics panel** (paste JSON from `diagnose_system.py`)
- Event bindings for: message submit, clear history, refresh GPU, provider switch, remote JSON submission.
- Entry point: `python3 -m src.app_web --db-path data/chroma_db --port 7860 [--share]`

**Dependencies:** `os`, `sys`, `json`, `logging`, `gradio`, `pathlib` (imports from `env_detector`, `retriever`, `llm_provider`, `agent`)

---

### 3.11 `src/diagnose_system.py`

**Role:** Standalone diagnostic script designed to be copied and run on a **remote AMD GPU machine**. Has **zero external dependencies** (only Python 3.6+ stdlib).

**Key Functions:**
- `_run(cmd, timeout=15)` -> Runs a shell command; returns stdout or `None`.
- `_detect_os()` -> OS name, version, pretty_name from `/etc/os-release`.
- `_detect_kernel()` -> Kernel version via `uname -r`.
- `_detect_python()` -> Python version, executable path, implementation.
- `_detect_rocm_version()` -> ROCm version from `/opt/rocm/.info/version` or apt metadata.
- `_detect_gpus()` -> GPU model, VRAM, temperature from `rocm-smi --json`.
- `_detect_gpu_arch()` -> GPU architecture (gfx ID) from `rocminfo`.
- `_detect_pip_packages()` -> Checks for torch, tensorflow, jax, vllm, transformers.
- `_detect_pytorch_rocm()` -> Checks if PyTorch has ROCm/HIP support.
- `_detect_container()` -> Detects Docker/Podman container environment.
- `diagnose()` -> Orchestrates all detection routines. Returns a single JSON dict:
  ```python
  {
      "os": Dict,
      "kernel": str,
      "python": Dict,
      "rocm_version": str,
      "gpus": List[Dict],
      "gpu_arch": List[Dict],
      "pip_packages": Dict,
      "pytorch_rocm": Dict,
      "container": Dict,
  }
  ```

**Usage:** `python3 diagnose_system.py` (pretty JSON) or `--compact` (single-line).

---

## 4. Data Flow Diagram

```
+-----------------------------------------------------------------------+
|                         SETUP PHASE (setup.sh)                        |
|                                                                       |
|  1. Install PyTorch (ROCm) + Python deps                              |
|  2. Clone ROCm doc repos -> data/raw_docs/                            |
|  3. scraper.collect_documents() -> document dicts                     |
|  4. chunker.chunk_documents() -> overlapping chunks                   |
|  5. embedder.build_vector_store() -> ChromaDB (data/chroma_db/)       |
+-----------------------------------------------------------------------+

+-----------------------------------------------------------------------+
|                         RUNTIME PHASE (agent / app_web)               |
|                                                                       |
|  User Question                                                        |
|       +                                                               |
|  +------------------+                                               |
|  | env_detector.py  |  -> Detect GPU, ROCm version, frameworks       |
|  +--------+---------+                                               |
|           + (formatted context)                                       |
|  +------------------+                                               |
|  | retriever.py     |  -> ChromaDB semantic search + reranking       |
|  +--------+---------+                                               |
|           + (retrieved chunks)                                        |
|  +------------------+                                               |
|  | agent.py         |  -> Assemble system prompt + context           |
|  +--------+---------+                                               |
|           + (prompt)                                                  |
|  +------------------+                                               |
|  | llm_provider.py  |  -> Fireworks AI or Local GPU inference        |
|  +--------+---------+                                               |
|           + (answer + source citations)                               |
|  User receives grounded response                                      |
+-----------------------------------------------------------------------+
```

---

## 5. Dependency Graph (Code-Level)

```
__init__.py
    |
    +-- scraper.py          (no internal deps -- stdlib only)
    |       |
    |       +-- chunker.py  (no internal deps -- stdlib only)
    |               |
    |               +-- embedder.py  (no internal deps -- stdlib only)
    |                       |
    |               +-------+--------+
    |               v                v
    |          retriever.py   data/chroma_db/ (generated)
    |               |
    +-- env_detector.py  (no internal deps -- stdlib only)
    |               |
    +-- fireworks_client.py  (no internal deps -- stdlib only)
    |               |
    +-- llm_provider.py  ----+
    |         |
    |         v
    +-- agent.py     <-----+
            |
            +-- app_web.py  (imports from agent, env_detector, retriever, llm_provider)
            +-- diagnose_system.py  (standalone -- no internal deps)
```

---

## 6. External Dependencies (pip packages)

| Package | Purpose | Used By |
|---|---|---|
| `chromadb>=0.4.0` | Persistent vector database | `embedder.py`, `retriever.py` |
| `sentence-transformers>=2.2.0` | Embedding generation + Cross-Encoder reranking | `embedder.py`, `retriever.py` |
| `openai>=1.0.0` | Fireworks AI API client (OpenAI-compatible) | `fireworks_client.py`, `llm_provider.py` |
| `requests>=2.31.0` | HTTP requests (unused currently) | -- |
| `tqdm>=4.65.0` | Progress bars | `scraper.py`, `chunker.py`, `embedder.py` |
| `gradio>=4.0.0` | Web UI framework | `app_web.py` |
| `transformers==4.45.2` | Local LLM inference (Gemma 4) | `llm_provider.py` |
| `accelerate>=0.25.0` | Model loading optimization | (not yet used) |
| `bitsandbytes>=0.41.0` | 4-bit model quantization | (not yet used) |
| `ipywidgets>=8.0.0` | Jupyter widgets | (not yet used) |
| `IPython>=8.0.0` | Jupyter kernel | (not yet used) |

---

## 7. Runtime Data Artifacts

| Artifact | Location | Description |
|---|---|---|
| `data/raw_docs/` | Cloned repos | 4 ROCm documentation repositories (cloned by `setup.sh`) |
| `data/chroma_db/` | Vector store | Persistent ChromaDB collection named `rocm_docs` |
| `.env` | Project root | Contains `FIREWORKS_API_KEY` (gitignored) |
| `/var/log/rocm-pilot-init.log` | Remote server | Initialization log (`startup_script.sh`) |

---

## 8. Deployment Modes

### Mode A: Interactive CLI (Local)
```bash
python3 -m src.agent --db-path data/chroma_db [--auto-approve]
```
- Reads from local ChromaDB
- Uses Fireworks AI (cloud) or local Gemma 4
- Interactive terminal session with tool execution support

### Mode B: Web UI (Local/Remote)
```bash
python3 -m src.app_web --db-path data/chroma_db --port 7860 [--share]
```
- Gradio-based chat interface
- Real-time AMD GPU monitoring panel
- Source citation display
- LLM provider selector (Cloud / Local)
- Remote diagnostics upload support

### Mode C: Cloud-Hosted (DigitalOcean GPU Droplet)
```bash
bash startup_script.sh
```
- Full system preparation (ROCm, PyTorch, permissions)
- Runs `setup.sh` to build knowledge base
- Pre-caches all ML models (embeddings, reranker, Gemma 4)
- Starts Gradio Web UI as a `systemd` service on port 7860
- Accessible at `http://<droplet-ip>:7860`

### Mode D: Remote Diagnostics (Zero-Dep Script)
```bash
python3 diagnose_system.py [--compact]
```
- Standalone script with **zero external dependencies**
- Designed to be copied and run on any remote AMD GPU machine
- Outputs JSON describing hardware, OS, ROCm stack, and AI packages
- Results can be pasted into the Web UI's remote diagnostics panel

---

## 9. Key Design Decisions

1. **RAG over fine-tuning:** The system retrieves from official ROCm docs rather than training a model, ensuring answers are always grounded in current documentation.

2. **Two-tier retrieval:** ChromaDB (fast, approximate) + Cross-Encoder reranker (precise, GPU-accelerated) for high-quality results.

3. **Table preservation:** Markdown tables (compatibility matrices, version info) are detected and preserved as intact blocks during chunking -- critical for ROCm documentation which is heavy on tables.

4. **Dual LLM support:** Cloud (Fireworks AI) for convenience, local (Gemma 4 on AMD GPU) for offline/air-gapped scenarios.

5. **Tool execution:** The LLM can request diagnostic commands (`rocm-smi`, `rocminfo`, etc.) to gather real-time system info, with optional auto-approve mode.

6. **Remote diagnostics:** `diagnose_system.py` enables the system to work with remote machines by outputting a JSON snapshot that can be pasted into the Web UI.

7. **ChromaDB over cloud DB:** Persistent local storage avoids external database dependencies and works offline.

---

## 10. Quick Reference: Entry Points

| Entry Point | Command | Description |
|---|---|---|
| Setup | `bash setup.sh` | Bootstrap: deps, docs clone, KB build |
| CLI Agent | `python3 -m src.agent` | Interactive terminal session |
| Web UI | `python3 -m src.app_web --port 7860` | Gradio web interface |
| Remote Diag | `python3 src/diagnose_system.py` | Zero-dep remote machine diagnostics |
| Tests | `python3 tests/test_basic.py` | Sanity checks |
| Knowledge Base Build | (inside `setup.sh`) | scraper -> chunker -> embedder pipeline |

---

*Document generated for AMD Developer Hackathon ACT II -- ROCm-Pilot v1.0.0*

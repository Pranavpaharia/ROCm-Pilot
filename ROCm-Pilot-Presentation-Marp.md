---
marp: true
theme: default
class: lead
---

# 🚀 ROCm-Pilot
## The AI-Powered Setup Assistant for AMD Workflows
**Built for: AMD Developer Hackathon ACT II**

*Providing seamless technical assistance to run AI/ML tools and workflows across all AMD-based hardware.*

---

# The Problem: Setup Complexity
**Setting up AI/ML environments shouldn't be a bottleneck.**

* **Fragmented Documentation:** Critical setup information is scattered across dozens of repos, blogs, and websites.
* **Complex Compatibility Matrices:** Navigating the exact combination of ROCm ↔ PyTorch ↔ GPU ↔ OS is daunting.
* **Hardware-Specific Nuances:** Setup paths vary significantly between **AMD Instinct™** (Data Center) and **AMD Radeon™ RX** (Consumer/Workstation) GPUs.
* **Migration Friction:** Developers coming from CUDA lack clear, unified guidance to port their workflows.

---

# The Solution: ROCm-Pilot
**Your Intelligent, Grounded AI Co-Pilot**
An intelligent RAG agent that provides technical assistance and generates setup instructions grounded strictly in official AMD documentation.

1. **Auto-Detects AMD Hardware:** Understands your specific environment (GPU model, ROCm version, frameworks) — fully supporting both **Instinct** and **Radeon RX** lines.
2. **Retrieves Official Docs:** Searches our auto-updating vector database built directly from AMD's GitHub repos.
3. **Generates Tailored Instructions:** Outputs perfectly tailored, copy-pasteable bash commands and setup scripts.
4. **Verifiable Citations:** Always cites its sources so developers can trust the recommendations.

---

# Architecture & Tech Stack (Part 1)
**Built with the Best of the AMD Ecosystem**

* **AMD Hardware & Cloud:** Tested on **Instinct™** and **Radeon™ RX** GPUs. Deployed on **DigitalOcean MI300X (192GB)** instances.
* **AMD Software:** Leverages **ROCm™**, `rocm-smi`, and `rocminfo` for deep environment introspection.
* **Fast Setup:** Powered by Astral's `uv` for sub-second, isolated dependency management.

---

# Architecture & Tech Stack (Part 2)
**AI Models & Retrieval Engine**

* **Supported AI Tech:** Dual LLM support featuring **Google Gemma 4 (12B)** natively on AMD GPUs (via source-compiled `transformers`), plus Fireworks AI & Lemonade SDK.
* **Hybrid RAG Pipeline:** GPU-accelerated embeddings via ROCm + dense semantic (ChromaDB) and sparse keyword (BM25) retrieval.

---

# Key Design Decisions (Part 1)
**Engineered for Accuracy and Reliability**

* **RAG over Fine-tuning:** Ensures answers are grounded in the *most current* official documentation without hallucinating APIs.
* **Hybrid Search:** Combines semantic vectors with exact keyword matching via the **BM25 Algorithm** to never miss a specific release note.
* **Dual-Tiered Documents:** Documents are chunked into **core** and **full versions** for flexible snippet or deep-guide retrieval.

---

# Key Design Decisions (Part 2)
**Privacy and Diagnostics**

* **Read-Only Tool Calling:** Safely introspects user environments without risk of executing destructive commands.
* **Native Local-First AI:** **Gemma-4-12B-it** runs entirely offline in VRAM for strict data privacy.
* **Zero-Dep Remote Diagnostics:** Standalone script ingests environment data from headless compute instances.

---

# Knowledge Base Sources
**The Source of Truth**

Auto-scraped and indexed directly from official AMD GitHub repositories:
* **ROCm Platform Docs** (`ROCm/ROCm`)
* **Linux Install Guides** (`ROCm/rocm-install-on-linux`)
* **Technical Blogs** (`ROCm/rocm-blogs`)
* **AI Developer Hub** (`ROCm/gpuaidev`)

---

# Why This Matters
**Accelerating the AMD AI Ecosystem**

* 🚀 **Lowers the Barrier to Entry:** Makes AMD hardware instantly accessible to developers new to the ecosystem.
* 📉 **Reduces Support Burden:** Automates the most common setup and environment troubleshooting questions for AMD's Developer Relations teams.
* 🤝 **Showcases Partner Tech:** Demonstrates how seamlessly partner technologies like **Google Gemma 4** and **Lemonade** run on AMD hardware.
* 📈 **Drives Ecosystem Adoption:** Faster setup times mean more developers building on ROCm, accelerating community growth.

---

# Thank You!

**Project Name:** ROCm-Pilot
**Team:** SoulsLikeAMD
**Track:** Unicorn (Track 3)

*Scan to view the GitHub Repo / Try the Demo!*

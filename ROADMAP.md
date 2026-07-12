# ROCm-Pilot — Brief Roadmap (1–2 Weeks)

**Goal:** Turn a strong hackathon demo into a reliable AMD setup agent.  
**Owner:** Solo (SoulsLikeAMD) — swap names if the team grows.  
**Success criteria:** Measurable answer quality, install-and-verify loop, docs that stay version-aware.

---

## Week 1 — Correctness & foundations

| Day | Focus | Deliverable | Owner |
|-----|--------|-------------|--------|
| **1–2** | Eval harness | 50 golden Q&A (install, matrices, vLLM, Radeon vs Instinct, CUDA→ROCm). Script scores retrieval hit-rate + fact presence. | Solo |
| **2–3** | Index consistency | One embed model at build *and* query time; rebuild `chroma_db`; write index manifest (model, docs SHAs, date). | Solo |
| **3–4** | Config cleanup | Single `.env.example` + config for model IDs, Lemonade URL, embed model, top_k. Local mode works without Fireworks key. | Solo |
| **4–5** | Tests | Hardware-agnostic pytest (mock env); drop hard-coded MI300X/`/root` paths; CI-friendly unit suite for chunker, gpu_compat, classify. | Solo |

**Week 1 exit:** “We know when answers are wrong” + clean local/cloud bootstrap.

---

## Week 2 — Agent outcomes & freshness

| Day | Focus | Deliverable | Owner |
|-----|--------|-------------|--------|
| **6–7** | Install-plan mode | Multi-step plans with copy-paste commands + per-step verify checks. | Solo |
| **7–8** | Verify loop | Structured tools (not free-text `[TOOL:]` only): re-run `rocm-smi` / PyTorch HIP checks; report pass/fail. | Solo |
| **8–9** | Version-aware retrieval | Prefer chunks matching detected ROCm major version; hybrid BM25+dense if time. | Solo |
| **9–10** | Docs refresh | One-command re-index (+ optional scheduled job); `live_scraper` for high-value pages. | Solo |
| **10** | Demo polish | Error-log doctor (paste stack trace → diagnose); consumer-GPU sample path; “download checklist/script.” | Solo |

**Week 2 exit:** Chat that *gets you set up* and can re-check the machine after advice.

---

## Stretch (if time)

- Auto OOM failover: local LLM → Lemonade → Fireworks  
- Quantized local models (`bitsandbytes`) for non-MI300X cards  
- Curated recipe library (last-verified ROCm + date)  
- Public diagnose one-liner + nicer remote JSON UX  

---

## Explicit non-goals (this window)

- Full UI redesign off Gradio  
- Fine-tuning a custom LLM  
- Indexing all of AMD.com  

---

## Tracking

| Priority | Theme | Status |
|----------|--------|--------|
| P0 | Eval + embedding/index consistency | Not started |
| P0 | Config / local-without-API-key | Not started |
| P1 | Install-plan + verify tools | Not started |
| P1 | Version-aware retrieval + re-index | Not started |
| P2 | Error-log doctor + demo polish | Not started |

Update the Status column as you ship. Revisit after Day 5: if eval scores are weak, stay on retrieval before building more UI.

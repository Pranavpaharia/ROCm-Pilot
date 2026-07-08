"""Comprehensive test suite for ROCm-Pilot."""

import sys
from pathlib import Path
import os

sys.path.insert(0, str(Path(__file__).parent.parent))

passed = 0
failed = 0


def test(name, fn):
    global passed, failed
    try:
        print("\n" + "=" * 50)
        print(f"  TEST: {name}")
        print("=" * 50)
        fn()
        print(f"✅ PASSED: {name}")
        passed += 1
    except Exception as e:
        print(f"❌ FAILED: {name} — {e}")
        failed += 1


# ── Test 1: Environment Detector ───────────────────────────────
def test_env_detector():
    from src.env_detector import detect_environment, format_env_context
    
    env = detect_environment()
    
    # Check GPU detection
    detected_gpus = [g for g in env.get("gpus", []) if g.get("detected")]
    assert len(detected_gpus) >= 1, f"Expected >=1 GPU, got {len(detected_gpus)}"
    gpu = detected_gpus[0]
    print(f"  GPU: {gpu.get('model')}")
    assert "MI300X" in gpu.get("model", ""), f"Expected MI300X GPU, got {gpu.get('model')}"
    
    # Check software detection
    sw = env.get("software", {})
    assert "python_version" in sw, "Python version not detected"
    assert "frameworks" in sw, "Frameworks not detected"
    
    # Check container detection
    assert "in_container" in env.get("container", {}), "Container detection missing"
    
    # Check format_env_context
    ctx = format_env_context(env)
    assert "AMD Instinct MI300X" in ctx or "MI300X" in ctx, "GPU model not in context"
    assert "ROCm Version: 7.2.4" in ctx or "ROCm" in ctx, "ROCm version not in context"
    words = len(ctx.split())
    assert words > 50, f"Context too short: {words} words"


# ── Test 2: System Diagnostics Script ──────────────────────────
def test_diagnose_script():
    import subprocess
    
    result = subprocess.run(
        ["python3", "/root/ROCm-Pilot/src/diagnose_system.py"],
        capture_output=True, text=True, timeout=30,
    )
    assert result.returncode == 0, f"diagnose_system.py failed: {result.stderr}"
    
    import json
    data = json.loads(result.stdout)
    
    assert "os" in data, "OS info missing"
    assert "gpus" in data, "GPU info missing"
    assert "python" in data, "Python info missing"
    assert "rocm_version" in data, "ROCm version missing"
    
    print(f"  OS: {data['os'].get('pretty_name', 'Unknown')}")
    print(f"  ROCm: {data['rocm_version']}")
    print(f"  GPUs: {len(data['gpus'])} detected")


# ── Test 3: Scraper Module ─────────────────────────────────────
def test_scraper():
    from src.scraper import collect_documents, classify_doc_type
    
    # Test doc classification
    assert classify_doc_type("docs/installation/guide.rst") == "installation"
    assert classify_doc_type("blogs/post.md") == "blog"
    assert classify_doc_type("docs/tutorial/how-to.rst") == "tutorial"
    assert classify_doc_type("docs/api/reference.rst") == "reference"
    print("  Doc classification works correctly")
    
    # Test document collection (from existing raw_docs)
    docs = collect_documents("data/raw_docs")
    assert len(docs) > 0, "No documents collected"
    
    types = {}
    for d in docs:
        dt = d.get("doc_type")
        types[dt] = types.get(dt, 0) + 1
    
    print(f"  Collected {len(docs)} documents")
    for dt, count in sorted(types.items()):
        print(f"    {dt}: {count}")


# ── Test 4: Chunker Module ─────────────────────────────────────
def test_chunker():
    from src.chunker import chunk_documents, _detect_and_replace_tables
    
    # Test table detection
    text = """Some intro text.

| Column A | Column B |
|----------|----------|
| Value 1  | Value 2  |
| Value 3  | Value 4  |

More text after."""

    modified, tables = _detect_and_replace_tables(text)
    assert len(tables) == 1, f"Expected 1 table, got {len(tables)}"
    assert "TABLE_BLOCK_0" in modified, "Table placeholder not inserted"
    print("  Table detection and placeholder insertion works")


# ── Test 5: Embedder Module (compilation only) ────────────────
def test_embedder_compilation():
    from src.embedder import get_device, DEFAULT_MODEL
    
    device = get_device()
    print(f"  Device: {device}")
    assert device in ("cuda", "cpu"), f"Unexpected device: {device}"
    
    # Verify model name is valid
    assert DEFAULT_MODEL == "sentence-transformers/all-MiniLM-L6-v2"
    print(f"  Default model: {DEFAULT_MODEL}")


# ── Test 6: Retriever Module (compilation only) ───────────────
def test_retriever_compilation():
    from src.retriever import get_reranker, CrossEncoderReranker
    
    # Just verify the class compiles and singleton works
    reranker = get_reranker()
    assert isinstance(reranker, CrossEncoderReranker)
    
    # Test empty results handling
    result = reranker.rerank("test query", [])
    assert result == []
    print("  CrossEncoderReranker compiles and handles empty input")


# ── Test 7: Fireworks Client (compilation only) ───────────────
def test_fireworks_client():
    from src.fireworks_client import get_client, DEFAULT_MODEL
    
    # Verify constants
    assert "fireworks" in DEFAULT_MODEL.lower() or "deepseek" in DEFAULT_MODEL.lower()
    print(f"  Default model: {DEFAULT_MODEL}")


# ── Test 8: Agent Module (compilation only) ───────────────────
def test_agent_compilation():
    from src.agent import RocmPilotAgent, ToolExecutor
    
    # Verify classes compile
    executor = ToolExecutor(auto_approve=False)
    assert len(executor.tools) > 0, "No tools registered"
    print(f"  Registered {len(executor.tools)} diagnostic tools")


# ── Test 9: LLM Provider (compilation only) ───────────────────
def test_llm_provider_compilation():
    from src.llm_provider import get_provider, FireworksProvider
    
    # Cloud provider (no actual API call)
    cloud = get_provider("cloud", model="accounts/fireworks/models/deepseek-v4-pro")
    assert isinstance(cloud, FireworksProvider)
    print("  Cloud provider compiles correctly")


# ── Test 10: Web UI Module (compilation only) ────────────────
def test_web_ui_compilation():
    from src.app_web import build_ui
    
    # Verify the function exists and compiles
    assert callable(build_ui)
    print("  Web UI build_ui function exists")


# ── Run all tests ─────────────────────────────────────────────
if __name__ == "__main__":
    print("\n" + "=" * 50)
    print("  🧪 ROCm-Pilot Comprehensive Test Suite")
    print("=" * 50)
    
    test("Environment Detector", test_env_detector)
    test("System Diagnostics Script", test_diagnose_script)
    test("Scraper Module", test_scraper)
    test("Chunker Module", test_chunker)
    test("Embedder Module (compilation)", test_embedder_compilation)
    test("Retriever Module (compilation)", test_retriever_compilation)
    test("Fireworks Client (compilation)", test_fireworks_client)
    test("Agent Module (compilation)", test_agent_compilation)
    test("LLM Provider (compilation)", test_llm_provider_compilation)
    test("Web UI Module (compilation)", test_web_ui_compilation)
    
    print("\n" + "=" * 50)
    total = passed + failed
    print(f"  Results: {passed}/{total} tests passed")
    if failed > 0:
        print(f"  ❌ {failed} test(s) FAILED")
        sys.exit(1)
    else:
        print("  🎉 ALL TESTS PASSED!")
    print("=" * 50)

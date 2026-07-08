"""
ROCm-Pilot Agent — the main RAG agent that answers AMD/ROCm questions
grounded in official documentation.
"""

import logging
import os
import re
import subprocess
from typing import Optional, List, Dict, Callable
from pathlib import Path

from src.env_detector import detect_environment, format_env_context
from src.retriever import get_retriever, retrieve, format_context
from src.llm_provider import get_provider, FireworksProvider


# ──────────────────────────────────────────────────────────────────
# API Key Validation
# ──────────────────────────────────────────────────────────────────

def _check_api_key() -> str:
    """
    Check for FIREWORKS_API_KEY in environment or .env file.
    Returns the key if found, raises SystemExit with clear message if not.
    """
    # Check environment variable first
    key = os.environ.get('FIREWORKS_API_KEY')
    if key:
        return key

    # Try loading from .env file in project root
    env_paths = [
        Path('.env'),
        Path('ROCm-Pilot/.env'),
        Path.home() / '.env',
    ]
    for env_path in env_paths:
        if env_path.exists():
            with open(env_path, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line.startswith('FIREWORKS_API_KEY='):
                        key = line.split('=', 1)[1].strip().strip('"').strip("'")
                        if key:
                            # Set it in environment for subprocesses
                            os.environ['FIREWORKS_API_KEY'] = key
                            return key

    print("=" * 60)
    print("❌ ERROR: FIREWORKS_API_KEY not found!")
    print("=" * 60)
    print()
    print("ROCm-Pilot requires a Fireworks AI API key to function.")
    print()
    print("To get a key:")
    print("  1. Visit https://fireworks.ai/api")
    print("  2. Sign up and generate an API key")
    print()
    print("To set the key, choose one of:")
    print("  • Export: export FIREWORKS_API_KEY='your-key-here'")
    print("  • .env file: Create a .env file with FIREWORKS_API_KEY=your-key-here")
    print("=" * 60)
    raise SystemExit(1)


# ──────────────────────────────────────────────────────────────────
# Tool Execution System
# ──────────────────────────────────────────────────────────────────

class ToolExecutor:
    """
    Registry and executor of safe, read-only diagnostic commands.
    The LLM can suggest running these tools by outputting [TOOL: tool_name].
    """

    def __init__(self, auto_approve: bool = False):
        self.auto_approve = auto_approve
        self.tools: Dict[str, Dict] = {
            'rocm-smi': {
                'description': 'Show AMD GPU status, utilization, and temperatures',
                'command': 'rocm-smi',
                'timeout': 15,
            },
            'rocm-smi-gpus': {
                'description': 'List all detected AMD GPUs with product names',
                'command': 'rocm-smi --showproductname',
                'timeout': 15,
            },
            'rocminfo': {
                'description': 'Show detailed ROCm system and GPU information',
                'command': 'rocminfo',
                'timeout': 15,
            },
            'rocm-version': {
                'description': 'Show installed ROCm version',
                'command': 'cat /opt/rocm/.info/version 2>/dev/null || rocm-smi --version',
                'timeout': 10,
            },
            'pip-list-ai': {
                'description': 'List installed AI/ML Python packages',
                'command': "pip list 2>/dev/null | grep -iE 'torch|tensorflow|jax|vllm|transformers|rocm|hip'",
                'timeout': 15,
            },
            'pytorch-gpu-check': {
                'description': 'Verify PyTorch can see AMD GPUs',
                'command': "python3 -c \"import torch; print('CUDA available:', torch.cuda.is_available()); print('Device count:', torch.cuda.device_count()); [print(f'  GPU {i}: {torch.cuda.get_device_name(i)}') for i in range(torch.cuda.device_count())]\"",
                'timeout': 20,
            },
            'hip-info': {
                'description': 'Show HIP/ROCm runtime details',
                'command': "python3 -c \"import torch; print('HIP version:', getattr(torch.version, 'hip', 'N/A')); print('ROCm version:', getattr(torch.version, 'rocm', 'N/A'))\"",
                'timeout': 15,
            },
            'disk-space': {
                'description': 'Check available disk space (for large model downloads)',
                'command': "df -h / /opt/rocm 2>/dev/null | head -10",
                'timeout': 10,
            },
            'docker-check': {
                'description': 'Check if running inside a Docker container',
                'command': "if [ -f /.dockerenv ]; then echo 'Running in Docker container'; cat /proc/1/cgroup 2>/dev/null | head -5; else echo 'Not running in Docker'; fi",
                'timeout': 10,
            },
        }

    def get_tool_descriptions(self) -> str:
        """Return a formatted list of available tools for the system prompt."""
        lines = ["Available diagnostic tools (use [TOOL: tool_name] to suggest):"]
        for name, info in self.tools.items():
            lines.append(f"  - {name}: {info['description']}")
        return '\n'.join(lines)

    def parse_tool_calls(self, text: str) -> List[str]:
        """Parse LLM response for [TOOL: tool_name] markers."""
        pattern = r'\[TOOL:\s*([a-zA-Z0-9_-]+)\]'
        return re.findall(pattern, text)

    def execute_tool(self, tool_name: str) -> Dict:
        """
        Execute a tool by name. Returns dict with 'success', 'output', 'error'.
        """
        if tool_name not in self.tools:
            return {
                'success': False,
                'output': '',
                'error': f"Unknown tool: {tool_name}. Available: {list(self.tools.keys())}",
            }

        tool = self.tools[tool_name]
        cmd = tool['command']
        timeout = tool.get('timeout', 15)

        # Ask for user approval unless auto_approve is set
        if not self.auto_approve:
            print(f"\n🔧 Tool suggested: {tool_name}")
            print(f"   Description: {tool['description']}")
            print(f"   Command: {cmd}")
            try:
                response = input("   Run this command? [y/N]: ").strip().lower()
                if response not in ('y', 'yes'):
                    return {
                        'success': False,
                        'output': '',
                        'error': 'User declined to run this command.',
                    }
            except (EOFError, KeyboardInterrupt):
                return {
                    'success': False,
                    'output': '',
                    'error': 'User interrupted.',
                }

        # Execute the command
        try:
            result = subprocess.run(
                cmd,
                shell=True,
                capture_output=True,
                text=True,
                timeout=timeout,
            )
            output = result.stdout
            if result.stderr:
                output += f"\n[stderr]: {result.stderr}"
            return {
                'success': result.returncode == 0,
                'output': output.strip(),
                'error': '' if result.returncode == 0 else f"Exit code: {result.returncode}",
            }
        except subprocess.TimeoutExpired:
            return {
                'success': False,
                'output': '',
                'error': f"Command timed out after {timeout}s",
            }
        except Exception as e:
            return {
                'success': False,
                'output': '',
                'error': str(e),
            }

    def execute_and_format(self, tool_name: str) -> str:
        """Execute a tool and return formatted output for LLM context."""
        result = self.execute_tool(tool_name)
        if result['success']:
            return f"[Tool '{tool_name}' output]:\n{result['output']}"
        else:
            return f"[Tool '{tool_name}' failed]: {result['error']}"


# ──────────────────────────────────────────────────────────────────
# System prompt — this is the most important piece for answer quality
# ──────────────────────────────────────────────────────────────────
SYSTEM_PROMPT_TEMPLATE = """\
You are **ROCm-Pilot**, an expert AI assistant that helps developers set up \
AI and machine learning workloads on AMD hardware using the ROCm (Radeon Open \
Compute) software stack.

You are grounded in **official AMD ROCm documentation**. Follow these rules:

1. **Base your answers on the provided documentation context.** If the context \
   contains relevant information, use it. If it does not cover the topic, say \
   so honestly — never fabricate commands or version numbers.
2. **Provide copy-pasteable commands** in ```bash code blocks whenever you \
   suggest terminal actions.
3. **Tailor advice to the user's detected hardware** when environment info is \
   available. For example, recommend MI300X-specific flags for MI300X users.
4. **Cite your sources** by referencing the document name and section \
   (e.g., "According to the ROCm Linux Install Guide, …").
5. **Structure responses clearly** with headers, numbered steps, and bullet \
   points when appropriate.
6. **Include version-specific details** (ROCm version, PyTorch version, \
   driver version, GPU architecture) wherever relevant.
7. **Warn about common pitfalls** — version mismatches, unsupported GPU \
   models, driver conflicts, kernel incompatibilities, etc.
8. **Suggest follow-up actions** — after answering, recommend a logical next \
   step the user might want to take.
9. **Use diagnostic tools** when you need to verify the user's system state. \
   To suggest running a diagnostic command, output `[TOOL: tool_name]` on its \
   own line. The user will be asked for approval before execution. Only use \
   read-only diagnostic tools — never suggest destructive commands.

Your expertise covers:
- ROCm installation and configuration (Linux)
- PyTorch, TensorFlow, JAX, and ONNX Runtime setup with ROCm
- vLLM, text-generation-inference, and other LLM serving frameworks on AMD
- AMD Instinct (MI100 / MI200 / MI300 series) and Radeon (RX 7000) GPUs
- GPU monitoring and debugging tools (rocm-smi, rocminfo, rocprof)
- Performance tuning (TunableOp, MIOpen, hipBLAS)
- Docker containers for AMD AI workloads

{tool_descriptions}
"""


class RocmPilotAgent:
    """Main RAG agent for AMD/ROCm developer assistance."""

    def __init__(
        self,
        db_path: str = 'data/chroma_db',
        provider_type: str = "cloud", model: str = "accounts/fireworks/models/deepseek-v4-pro",
        auto_detect_env: bool = True,
        auto_approve_tools: bool = False,
    ):
        """
        Initialize ROCm-Pilot.

        Args:
            db_path: Path to the ChromaDB vector store.
            model: Fireworks AI model identifier.
            auto_detect_env: Whether to auto-detect AMD hardware on startup.
            auto_approve_tools: If True, skip user approval for tool execution.
        """
        self.provider = get_provider(provider_type, model)
        self.conversation_history: List[Dict[str, str]] = []
        self.provider_type = provider_type
        self.env_context = ""
        self.embedding_model = None
        self.tool_executor = ToolExecutor(auto_approve=auto_approve_tools)

        # Validate API key at startup
        _check_api_key()

        # Load the vector store
        print("Loading knowledge base...")
        self.collection = get_retriever(db_path)
        doc_count = self.collection.count()
        print(f"✅ Knowledge base loaded: {doc_count:,} chunks")

        # Load the embedding model (for query encoding)
        try:
            from sentence_transformers import SentenceTransformer
            self.embedding_model = SentenceTransformer(
                'sentence-transformers/all-MiniLM-L6-v2',
            )
            print("✅ Embedding model loaded")
        except Exception:
            print("⚠️  Embedding model unavailable — using ChromaDB defaults")

        # Detect hardware environment
        if auto_detect_env:
            print("Detecting AMD hardware environment...")
            env = detect_environment()
            self.env_context = format_env_context(env)
            print(self.env_context)

        # Log GPU status at startup
        self._log_gpu_status()

    # ------------------------------------------------------------------ #
    #  Core Q&A                                                           #
    # ------------------------------------------------------------------ #

    def ask(
        self,
        question: str,
        top_k: int = 8,
        doc_type_filter: Optional[str] = None,
        stream: bool = False,
        max_tool_rounds: int = 3,
    ) -> str:
        """
        Ask a question about AMD/ROCm.

        Args:
            question: The user's natural-language question.
            top_k: Number of documentation chunks to retrieve.
            doc_type_filter: Optional filter (installation | blog | tutorial | …).
            stream: If True, print response tokens as they arrive.
            max_tool_rounds: Maximum number of tool execution rounds per question.

        Returns:
            The agent's full response string.
        """
        # Step 1 — Retrieve relevant documentation
        results = retrieve(
            query=question,
            collection=self.collection,
            embedding_model=self.embedding_model,
            top_k=top_k,
            doc_type_filter=doc_type_filter,
        )
        doc_context = format_context(results)

        # Step 2 — Assemble the system message with tool descriptions
        tool_descriptions = self.tool_executor.get_tool_descriptions()
        system_message = SYSTEM_PROMPT_TEMPLATE.format(
            tool_descriptions=tool_descriptions,
        )
        if self.env_context:
            system_message += f"\n\n{self.env_context}"

        # Step 3 — Build user message with injected context
        user_message = (
            "Based on the following official AMD ROCm documentation, "
            "answer the user's question.\n\n"
            "--- DOCUMENTATION CONTEXT ---\n"
            f"{doc_context}\n"
            "--- END CONTEXT ---\n\n"
            f"**User Question:** {question}\n\n"
            "Provide a clear, actionable answer grounded in the documentation. "
            "Cite the source documents."
        )

        # Step 4 — Build the full message list (system + history + user)
        messages = [{"role": "system", "content": system_message}]
        for msg in self.conversation_history[-6:]:  # keep last 6 turns
            messages.append(msg)
        messages.append({"role": "user", "content": user_message})

        # Step 5 — Call LLM with tool execution loop
        full_response = ""
        for tool_round in range(max_tool_rounds + 1):
            # Call the LLM
            if stream:
                response_text = ""
                for chunk in self.provider.chat(messages=messages, stream=True):
                    print(chunk, end="", flush=True)
                    response_text += chunk
                print()
            else:
                response_text = self.provider.chat(messages=messages, stream=False)

            # Check for tool calls
            tool_calls = self.tool_executor.parse_tool_calls(response_text)
            if not tool_calls:
                full_response = response_text
                break

            # Execute each tool and feed results back to the LLM
            tool_results = []
            for tool_name in tool_calls:
                result_text = self.tool_executor.execute_and_format(tool_name)
                tool_results.append(result_text)

            # Add the assistant's response and tool results to the conversation
            messages.append({"role": "assistant", "content": response_text})
            tool_context = "\n\n".join(tool_results)
            messages.append({
                "role": "user",
                "content": (
                    f"Here are the results of the diagnostic tools you requested:\n\n"
                    f"{tool_context}\n\n"
                    f"Based on these results, please update your answer for the user."
                ),
            })

            # Get a follow-up response from the LLM
            if stream:
                followup = ""
                for chunk in self.provider.chat(messages=messages, stream=True):
                    print(chunk, end="", flush=True)
                    followup += chunk
                print()
            else:
                followup = self.provider.chat(messages=messages, stream=False)
            full_response = response_text + "\n\n" + followup

            # Update messages for potential next round
            messages.append({"role": "assistant", "content": followup})
            # Check if the followup also has tool calls
            if not self.tool_executor.parse_tool_calls(followup):
                break
        else:
            full_response += "\n\n[Note: Maximum diagnostic tool rounds reached.]"

        # Step 6 — Persist in conversation history
        self.conversation_history.append({"role": "user", "content": question})
        self.conversation_history.append(
            {"role": "assistant", "content": full_response}
        )

        return full_response

    # ------------------------------------------------------------------ #
    #  Utilities                                                          #
    # ------------------------------------------------------------------ #

    def get_sources(self, question: str, top_k: int = 5) -> List[Dict]:
        """Retrieve source documents for a query without calling the LLM."""
        return retrieve(
            query=question,
            collection=self.collection,
            embedding_model=self.embedding_model,
            top_k=top_k,
        )

    def clear_history(self):
        """Reset the conversation history."""
        self.conversation_history.clear()
        print("🗑️  Conversation history cleared.")

    def _log_gpu_status(self):
        """Log GPU model, VRAM, and HIP version at startup."""
        logger = logging.getLogger("rocm_pilot")
        try:
            # Get GPU product name
            result = subprocess.run(
                "rocm-smi --showproductname",
                shell=True,
                capture_output=True,
                text=True,
                timeout=10,
            )
            if result.returncode == 0:
                logger.info("GPU info:\n%s", result.stdout.strip())
            else:
                logger.warning("Could not query GPU product name via rocm-smi")

            # Get VRAM info
            result = subprocess.run(
                "rocm-smi --showmeminfo vram",
                shell=True,
                capture_output=True,
                text=True,
                timeout=10,
            )
            if result.returncode == 0:
                logger.info("VRAM info:\n%s", result.stdout.strip())

            # Get HIP version
            result = subprocess.run(
                "hipconfig --version",
                shell=True,
                capture_output=True,
                text=True,
                timeout=10,
            )
            if result.returncode == 0:
                logger.info("HIP version: %s", result.stdout.strip())
            else:
                logger.warning("Could not determine HIP version")
        except Exception as e:
            logger.debug("GPU status logging skipped: %s", e)


# ──────────────────────────────────────────────────────────────────
# Interactive CLI session
# ──────────────────────────────────────────────────────────────────

def interactive_session(
    db_path: str = 'data/chroma_db',
    auto_approve_tools: bool = False,
):
    """Launch an interactive terminal session with ROCm-Pilot."""
    print("=" * 60)
    print("  🚀 ROCm-Pilot: AI-Powered AMD Setup Assistant")
    print("=" * 60)
    print()

    agent = RocmPilotAgent(
        db_path=db_path,
        auto_approve_tools=auto_approve_tools,
    )

    print()
    print("Ask me anything about AMD ROCm, GPU setup, or AI frameworks!")
    print("Commands:")
    print("  quit / exit   — end session")
    print("  clear         — reset conversation history")
    print("  sources <q>   — show raw source documents for a query")
    if auto_approve_tools:
        print("  ⚡ Tool auto-approve is ON")
    print("-" * 60)

    while True:
        try:
            question = input("\n🧑 You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye! 👋")
            break

        if not question:
            continue
        if question.lower() in ('quit', 'exit', 'q'):
            print("Goodbye! 👋")
            break
        if question.lower() == 'clear':
            agent.clear_history()
            continue
        if question.lower().startswith('sources '):
            query = question[8:]
            sources = agent.get_sources(query)
            for i, s in enumerate(sources, 1):
                m = s['metadata']
                print(f"\n📄 Source {i}: {m['source_repo']}/{m['source_file']}")
                print(f"   Type: {m['doc_type']} | Section: {m['section_title']}")
                print(f"   URL: {m['source_url']}")
                print(f"   Preview: {s['text'][:200]}...")
            continue

        print("\n🤖 ROCm-Pilot:")
        agent.ask(question, stream=True)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(
        description='ROCm-Pilot: AI-Powered AMD Setup Assistant',
    )
    parser.add_argument(
        '--db-path',
        default='data/chroma_db',
        help='Path to the ChromaDB vector store (default: data/chroma_db)',
    )
    parser.add_argument(
        '--auto-approve',
        action='store_true',
        help='Auto-approve diagnostic tool execution without user confirmation',
    )
    args = parser.parse_args()

    interactive_session(
        db_path=args.db_path,
        auto_approve_tools=args.auto_approve,
    )

"""
ROCm-Pilot Agent — the main RAG agent that answers AMD/ROCm questions
grounded in official documentation.
"""

import os
from typing import Optional, List, Dict

from src.env_detector import detect_environment, format_env_context
from src.retriever import get_retriever, retrieve, format_context
from src.fireworks_client import chat, DEFAULT_MODEL


# ──────────────────────────────────────────────────────────────────
# System prompt — this is the most important piece for answer quality
# ──────────────────────────────────────────────────────────────────
SYSTEM_PROMPT = """\
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

Your expertise covers:
- ROCm installation and configuration (Linux)
- PyTorch, TensorFlow, JAX, and ONNX Runtime setup with ROCm
- vLLM, text-generation-inference, and other LLM serving frameworks on AMD
- AMD Instinct (MI100 / MI200 / MI300 series) and Radeon (RX 7000) GPUs
- GPU monitoring and debugging tools (rocm-smi, rocminfo, rocprof)
- Performance tuning (TunableOp, MIOpen, hipBLAS)
- Docker containers for AMD AI workloads
"""


class RocmPilotAgent:
    """Main RAG agent for AMD/ROCm developer assistance."""

    def __init__(
        self,
        db_path: str = 'data/chroma_db',
        model: str = DEFAULT_MODEL,
        auto_detect_env: bool = True,
    ):
        """
        Initialize ROCm-Pilot.

        Args:
            db_path: Path to the ChromaDB vector store.
            model: Fireworks AI model identifier.
            auto_detect_env: Whether to auto-detect AMD hardware on startup.
        """
        self.model = model
        self.conversation_history: List[Dict[str, str]] = []
        self.env_context = ""
        self.embedding_model = None

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

    # ------------------------------------------------------------------ #
    #  Core Q&A                                                           #
    # ------------------------------------------------------------------ #

    def ask(
        self,
        question: str,
        top_k: int = 8,
        doc_type_filter: Optional[str] = None,
        stream: bool = False,
    ) -> str:
        """
        Ask a question about AMD/ROCm.

        Args:
            question: The user's natural-language question.
            top_k: Number of documentation chunks to retrieve.
            doc_type_filter: Optional filter (installation | blog | tutorial | …).
            stream: If True, print response tokens as they arrive.

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

        # Step 2 — Assemble the system message
        system_message = SYSTEM_PROMPT
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

        # Step 5 — Call Fireworks AI
        if stream:
            response_text = ""
            for chunk in chat(messages=messages, model=self.model, stream=True):
                print(chunk, end="", flush=True)
                response_text += chunk
            print()  # trailing newline
        else:
            response_text = chat(messages=messages, model=self.model)

        # Step 6 — Persist in conversation history
        self.conversation_history.append({"role": "user", "content": question})
        self.conversation_history.append(
            {"role": "assistant", "content": response_text}
        )

        return response_text

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


# ──────────────────────────────────────────────────────────────────
# Interactive CLI session
# ──────────────────────────────────────────────────────────────────

def interactive_session(db_path: str = 'data/chroma_db'):
    """Launch an interactive terminal session with ROCm-Pilot."""
    print("=" * 60)
    print("  🚀 ROCm-Pilot: AI-Powered AMD Setup Assistant")
    print("=" * 60)
    print()

    agent = RocmPilotAgent(db_path=db_path)

    print()
    print("Ask me anything about AMD ROCm, GPU setup, or AI frameworks!")
    print("Commands:")
    print("  quit / exit   — end session")
    print("  clear         — reset conversation history")
    print("  sources <q>   — show raw source documents for a query")
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
    interactive_session()

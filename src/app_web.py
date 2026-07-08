"""
ROCm-Pilot Web UI — Gradio-based chat interface.
Provides a web-based alternative to the CLI agent with:
  - Chat interface with message history
  - Hardware status panel
  - Source citation display
  - Tool execution approval
"""

import os
import sys
import json
import gradio as gr
from pathlib import Path
from typing import List, Dict, Tuple, Optional

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.env_detector import detect_environment, format_env_context
from src.retriever import get_retriever, retrieve, format_context
from src.fireworks_client import chat, DEFAULT_MODEL
from src.agent import (
    _check_api_key,
    ToolExecutor,
    SYSTEM_PROMPT_TEMPLATE,
)


# ──────────────────────────────────────────────────────────────────
# Agent State (singleton for Gradio session)
# ──────────────────────────────────────────────────────────────────

class WebAgent:
    """Stateful agent for the web UI session."""

    def __init__(self, db_path: str = 'data/chroma_db'):
        self.db_path = db_path
        self.collection = None
        self.embedding_model = None
        self.env_context = ""
        self.env_raw = {}
        self.conversation_history: List[Dict[str, str]] = []
        self.tool_executor = ToolExecutor(auto_approve=False)
        self.model = DEFAULT_MODEL
        self._initialized = False

    def initialize(self):
        """Lazy initialization — called on first message."""
        if self._initialized:
            return

        # Validate API key
        _check_api_key()

        # Load vector store
        print("Loading knowledge base...")
        self.collection = get_retriever(self.db_path)
        doc_count = self.collection.count()
        print(f"✅ Knowledge base loaded: {doc_count:,} chunks")

        # Load embedding model
        try:
            from sentence_transformers import SentenceTransformer
            self.embedding_model = SentenceTransformer(
                'sentence-transformers/all-MiniLM-L6-v2',
            )
            print("✅ Embedding model loaded")
        except Exception:
            print("⚠️  Embedding model unavailable — using ChromaDB defaults")

        # Detect environment
        print("Detecting AMD hardware environment...")
        self.env_raw = detect_environment()
        self.env_context = format_env_context(self.env_raw)
        print(self.env_context)

        self._initialized = True

    def respond(
        self,
        message: str,
        history: List[Dict],
    ) -> Tuple[str, List[Dict], str]:
        """
        Process a user message and return the response.
        
        Returns:
            - response text
            - updated history
            - sources markdown
        """
        self.initialize()

        if not message.strip():
            return "", history, ""

        # Retrieve relevant documentation
        results = retrieve(
            query=message,
            collection=self.collection,
            embedding_model=self.embedding_model,
            top_k=8,
        )
        doc_context = format_context(results)

        # Build sources display
        sources_md = self._format_sources(results)

        # Assemble system message
        tool_descriptions = self.tool_executor.get_tool_descriptions()
        system_message = SYSTEM_PROMPT_TEMPLATE.format(
            tool_descriptions=tool_descriptions,
        )
        if self.env_context:
            system_message += f"\n\n{self.env_context}"

        # Build user message
        user_message = (
            "Based on the following official AMD ROCm documentation, "
            "answer the user's question.\n\n"
            "--- DOCUMENTATION CONTEXT ---\n"
            f"{doc_context}\n"
            "--- END CONTEXT ---\n\n"
            f"**User Question:** {message}\n\n"
            "Provide a clear, actionable answer grounded in the documentation. "
            "Cite the source documents."
        )

        # Build messages
        messages = [{"role": "system", "content": system_message}]
        for msg in self.conversation_history[-6:]:
            messages.append(msg)
        messages.append({"role": "user", "content": user_message})

        # Call LLM
        response_text = chat(messages=messages, model=self.model)

        # Handle tool calls
        tool_calls = self.tool_executor.parse_tool_calls(response_text)
        if tool_calls:
            # Execute tools (auto-approve in web UI for simplicity)
            tool_executor_auto = ToolExecutor(auto_approve=True)
            tool_results = []
            for tool_name in tool_calls:
                result_text = tool_executor_auto.execute_and_format(tool_name)
                tool_results.append(result_text)

            messages.append({"role": "assistant", "content": response_text})
            tool_context = "\n\n".join(tool_results)
            messages.append({
                "role": "user",
                "content": (
                    f"Here are the results of the diagnostic tools:\n\n"
                    f"{tool_context}\n\n"
                    f"Based on these results, provide your answer."
                ),
            })
            response_text = chat(messages=messages, model=self.model)

        # Update history
        self.conversation_history.append({"role": "user", "content": message})
        self.conversation_history.append({"role": "assistant", "content": response_text})

        return response_text, self.conversation_history, sources_md

    def _format_sources(self, results: List[Dict]) -> str:
        """Format retrieval results as markdown for the sources panel."""
        if not results:
            return "*No sources retrieved.*"

        lines = ["## 📄 Retrieved Sources\n"]
        for i, r in enumerate(results, 1):
            m = r.get('metadata', {})
            lines.append(f"### Source {i}")
            lines.append(f"**File:** `{m.get('source_repo', '?')}/{m.get('source_file', '?')}`")
            lines.append(f"**Type:** {m.get('doc_type', '?')} | **Section:** {m.get('section_title', '?')}")
            url = m.get('source_url', '')
            if url:
                lines.append(f"**URL:** [{url}]({url})")
            text_preview = r.get('text', '')[:200]
            lines.append(f"\n> {text_preview}...")
            lines.append("")

        return '\n'.join(lines)

    def get_status_md(self) -> str:
        """Return markdown for the hardware status panel."""
        if not self._initialized:
            return "⏳ *Not yet initialized. Send a message to start.*"

        env = self.env_raw
        lines = ["## 🖥️ System Status\n"]

        # Container
        container = env.get('container', {})
        if container.get('in_container'):
            ctype = container.get('container_type', 'unknown')
            lines.append(f"⚠️ **Running in {ctype} container**")

        # GPUs
        gpus = env.get('gpus', [])
        detected = [g for g in gpus if g.get('detected')]
        if detected:
            lines.append(f"**GPUs Detected:** {len(detected)}")
            for i, gpu in enumerate(detected):
                lines.append(f"- **GPU {i}:** {gpu.get('model', 'Unknown')}")
                if gpu.get('vram') and gpu['vram'] != 'Unknown':
                    lines.append(f"  - VRAM: {gpu['vram']}")
                if gpu.get('arch'):
                    lines.append(f"  - Arch: {gpu['arch']}")
            rocm_ver = detected[0].get('rocm_version', 'Unknown')
            lines.append(f"\n**ROCm Version:** {rocm_ver}")
            pytorch_ver = detected[0].get('pytorch_version')
            if pytorch_ver:
                lines.append(f"**PyTorch Version:** {pytorch_ver}")
        else:
            lines.append("**GPU:** No AMD GPU detected")

        # Software
        sw = env.get('software', {})
        lines.append(f"\n**OS:** {sw.get('os', 'Unknown')}")
        lines.append(f"**Python:** {sw.get('python_version', 'Unknown')}")

        frameworks = sw.get('frameworks', {})
        if frameworks:
            lines.append("\n**Frameworks:**")
            for name, ver in sorted(frameworks.items()):
                lines.append(f"- {name}: {ver}")

        return '\n'.join(lines)

    def clear_history(self):
        """Reset conversation history."""
        self.conversation_history.clear()
        return [], ""


# ──────────────────────────────────────────────────────────────────
# Gradio UI
# ──────────────────────────────────────────────────────────────────

def build_ui(db_path: str = 'data/chroma_db') -> gr.Blocks:
    """Build the Gradio Blocks interface."""

    agent = WebAgent(db_path=db_path)

    # Custom CSS for AMD-themed dark interface
    custom_css = """
    .gradio-container {
        font-family: 'Inter', 'Segoe UI', sans-serif;
    }
    #header {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
        color: #ed1c24;
        padding: 20px;
        border-radius: 10px;
        margin-bottom: 15px;
        text-align: center;
    }
    #header h1 {
        color: #ed1c24;
        margin: 0;
        font-size: 2em;
    }
    #header p {
        color: #ccc;
        margin: 5px 0 0 0;
    }
    .status-panel {
        background: #1a1a2e;
        border: 1px solid #333;
        border-radius: 8px;
        padding: 15px;
    }
    .sources-panel {
        max-height: 400px;
        overflow-y: auto;
    }
    footer { display: none !important; }
    """

    with gr.Blocks(
        title="ROCm-Pilot",
        css=custom_css,
        theme=gr.themes.Soft(
            primary_hue="red",
            secondary_hue="slate",
            neutral_hue="slate",
        ),
    ) as demo:

        # Header
        gr.HTML(
            '<div id="header">'
            '<h1>🚀 ROCm-Pilot</h1>'
            '<p>AI-Powered AMD Setup Assistant — Grounded in Official Documentation</p>'
            '</div>'
        )

        with gr.Row():
            # Main chat column
            with gr.Column(scale=3):
                chatbot = gr.Chatbot(
                    label="Chat",
                    height=500,
                    show_copy_button=True,
                    type="messages",
                )

                with gr.Row():
                    msg = gr.Textbox(
                        label="Ask about AMD ROCm, GPU setup, or AI frameworks...",
                        placeholder="e.g., How do I install PyTorch on MI300X?",
                        scale=4,
                        show_label=False,
                    )
                    submit_btn = gr.Button("Send", variant="primary", scale=1)

                with gr.Row():
                    clear_btn = gr.Button("🗑️ Clear History", size="sm")
                    sources_toggle = gr.Button("📄 Show Sources", size="sm")

            # Side panel
            with gr.Column(scale=1):
                status_md = gr.Markdown(
                    value="⏳ *Send a message to initialize and detect hardware.*",
                    label="System Status",
                    elem_classes=["status-panel"],
                )
                sources_md = gr.Markdown(
                    value="",
                    label="Retrieved Sources",
                    elem_classes=["sources-panel"],
                )

        # ── Event Handlers ──

        def user_submit(message, history):
            """Handle user message submission."""
            if not message.strip():
                return "", history, gr.update(), gr.update()

            # Add user message to display immediately
            history = history or []
            history.append({"role": "user", "content": message})

            # Get agent response
            response_text, _, sources = agent.respond(message, history)

            # Add assistant response
            history.append({"role": "assistant", "content": response_text})

            # Update status panel
            status = agent.get_status_md()

            return "", history, status, sources

        def handle_clear():
            """Handle clear history button."""
            agent.clear_history()
            return [], ""

        def handle_sources_toggle(history):
            """Show sources for the last user message."""
            if not history:
                return "*No messages yet.*"
            # Find last user message
            for msg in reversed(history):
                if msg.get('role') == 'user':
                    _, _, sources = agent.respond(msg['content'], [])
                    return sources
            return "*No user messages found.*"

        # Bind events
        msg.submit(
            user_submit,
            inputs=[msg, chatbot],
            outputs=[msg, chatbot, status_md, sources_md],
        )
        submit_btn.click(
            user_submit,
            inputs=[msg, chatbot],
            outputs=[msg, chatbot, status_md, sources_md],
        )
        clear_btn.click(
            handle_clear,
            outputs=[chatbot, sources_md],
        )
        sources_toggle.click(
            handle_sources_toggle,
            inputs=[chatbot],
            outputs=[sources_md],
        )

    return demo


# ──────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='ROCm-Pilot Web UI')
    parser.add_argument(
        '--db-path',
        default='data/chroma_db',
        help='Path to ChromaDB vector store',
    )
    parser.add_argument(
        '--port',
        type=int,
        default=7860,
        help='Port to serve on (default: 7860)',
    )
    parser.add_argument(
        '--share',
        action='store_true',
        help='Create a public Gradio share link',
    )
    args = parser.parse_args()

    demo = build_ui(db_path=args.db_path)
    demo.launch(
        server_name="0.0.0.0",
        server_port=args.port,
        share=args.share,
    )
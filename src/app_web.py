"""
ROCm-Pilot Web UI — Gradio-based chat interface.
Provides a web-based alternative to the CLI agent with:
  - Chat interface with message history
  - Hardware status panel
  - Source citation display
  - Tool execution approval
  - Real-time AMD GPU monitor (VRAM, utilization, processes)
  - LLM provider selector (Cloud / Local AMD GPU)
  - Remote machine diagnostics upload
"""

import os
import sys
import json
import logging
import gradio as gr
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Generator

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.env_detector import (
    detect_environment,
    format_env_context,
    detect_gpu_utilization,
    detect_gpu_processes,
    format_gpu_monitor,
)
from src.retriever import get_retriever, retrieve, format_context, smart_retrieve
from src.llm_provider import get_provider, BaseLLMProvider
from src.agent import (
    _check_api_key,
    ToolExecutor,
    SYSTEM_PROMPT_TEMPLATE,
)

logger = logging.getLogger("rocm_pilot.web")

# ──────────────────────────────────────────────────────────────────
# Diagnostic script content (served to user for copy-paste)
# ──────────────────────────────────────────────────────────────────

DIAGNOSE_SCRIPT_CMD = (
    "curl -s https://raw.githubusercontent.com/Pranavpaharia/ROCm-Pilot/"
    "main/src/diagnose_system.py | python3"
)

DIAGNOSE_SCRIPT_ALT = (
    "python3 diagnose_system.py"
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
        self.remote_env_context = ""   # Overridden by remote diagnostics
        self.conversation_history: List[Dict[str, str]] = []
        self.tool_executor = ToolExecutor(auto_approve=False)
        self.provider: Optional[BaseLLMProvider] = None
        self.provider_type = "cloud"
        self.model_name = "accounts/fireworks/models/deepseek-v4-pro"
        self.gpu_db = None
        self.gpu_context = ""
        self._initialized = False

    def initialize(self):
        """Lazy initialization — called on first message."""
        if self._initialized:
            return

        # Validate API key (only required for cloud mode)
        if self.provider_type == "cloud":
            _check_api_key()

        # Load vector store
        logger.info("Loading knowledge base...")
        self.collection = get_retriever(self.db_path)
        doc_count = self.collection.count()
        logger.info("Knowledge base loaded: %d chunks", doc_count)

        # Load embedding model
        try:
            from sentence_transformers import SentenceTransformer
            self.embedding_model = SentenceTransformer(
                'sentence-transformers/all-MiniLM-L6-v2',
            )
            logger.info("Embedding model loaded")
        except Exception:
            logger.warning("Embedding model unavailable — using ChromaDB defaults")

        # Detect environment
        logger.info("Detecting AMD hardware environment...")
        self.env_raw = detect_environment()
        self.env_context = format_env_context(self.env_raw)
        logger.info(self.env_context)

        # Initialize LLM provider
        self._init_provider()

        # Load structured GPU compatibility database
        try:
            from src.gpu_compat import load_gpu_database, get_gpu_by_detected_name, format_gpu_report
            self.gpu_db = load_gpu_database('data/gpu_database.json')
            logger.info("GPU compatibility database loaded for Web Agent")
            
            # If GPU detected locally, pre-inject compatibility info
            gpus = self.env_raw.get('gpus', [])
            gpu_reports = []
            for gpu in gpus:
                if not gpu.get('detected'):
                    continue
                gpu_name = gpu.get('model', '')
                gpu_info = get_gpu_by_detected_name(self.gpu_db, gpu_name)
                if gpu_info:
                    gfx_id = gpu_info.get('gfx_id', '')
                    report = format_gpu_report(self.gpu_db, gfx_id)
                    gpu_reports.append(report)
            if gpu_reports:
                self.gpu_context = (
                    "--- DETECTED GPU COMPATIBILITY INFO ---\n"
                    + "\n\n".join(gpu_reports)
                    + "\n--- END GPU INFO ---"
                )
        except Exception as e:
            logger.warning("Failed to load GPU database for Web Agent: %s", e)

        self._initialized = True

    def _init_provider(self):
        """Initialize the LLM provider based on current settings."""
        logger.info(
            "Initializing LLM provider: type=%s, model=%s",
            self.provider_type, self.model_name,
        )
        self.provider = get_provider(self.provider_type, self.model_name)

    def switch_provider(self, provider_type: str, model_name: str = ""):
        """Switch between Cloud and Local providers."""
        self.provider_type = provider_type
        if model_name:
            self.model_name = model_name
        elif provider_type == "local":
            self.model_name = os.environ.get(
                "LOCAL_MODEL_ID", "Qwen/Qwen2.5-7B-Instruct"
            )
        else:
            self.model_name = "accounts/fireworks/models/deepseek-v4-pro"
        self._init_provider()

    def set_remote_env(self, json_text: str) -> str:
        """Parse pasted diagnostic JSON and override environment context."""
        try:
            data = json.loads(json_text)
            # Build a context block from the remote data
            lines = ["=== REMOTE MACHINE ENVIRONMENT ==="]
            if 'os' in data:
                os_info = data['os']
                lines.append(f"OS: {os_info.get('pretty_name', os_info.get('name', 'Unknown'))}")
            if 'kernel' in data:
                lines.append(f"Kernel: {data['kernel']}")
            if 'python' in data:
                lines.append(f"Python: {data['python'].get('version', 'Unknown')}")
            if 'rocm' in data:
                lines.append(f"ROCm Version: {data['rocm'].get('version', 'Unknown')}")
            if 'gpus' in data:
                gpus = data['gpus']
                lines.append(f"GPUs Detected: {len(gpus)}")
                for i, gpu in enumerate(gpus):
                    lines.append(f"  GPU {i}: {gpu.get('model', 'Unknown')}")
                    if gpu.get('vram'):
                        lines.append(f"    VRAM: {gpu['vram']}")
                    if gpu.get('temperature'):
                        lines.append(f"    Temp: {gpu['temperature']}")
                    if gpu.get('architecture'):
                        lines.append(f"    Arch: {gpu['architecture']}")
            if 'pytorch' in data:
                pt = data['pytorch']
                lines.append(f"PyTorch: {pt.get('version', 'N/A')}")
                lines.append(f"  ROCm available: {pt.get('rocm_available', False)}")
                if pt.get('hip_version'):
                    lines.append(f"  HIP: {pt['hip_version']}")
            if 'packages' in data:
                pkg = data['packages']
                installed = {k: v for k, v in pkg.items() if v is not None}
                if installed:
                    lines.append("Installed Frameworks:")
                    for name, ver in sorted(installed.items()):
                        lines.append(f"  - {name}: {ver}")
            if 'container' in data:
                c = data['container']
                if c.get('in_container'):
                    lines.append(f"Warning: Running in {c.get('type', 'unknown')} container")
            lines.append("=== END REMOTE ENVIRONMENT ===")

            self.remote_env_context = '\n'.join(lines)
            return f"Remote environment loaded!\n\n```\n{self.remote_env_context}\n```"
        except json.JSONDecodeError as e:
            return f"Invalid JSON: {e}"
        except Exception as e:
            return f"Error parsing diagnostics: {e}"

    def _get_effective_env_context(self) -> str:
        """Return remote env context if set, otherwise server env context."""
        return self.remote_env_context or self.env_context

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

        # Smart Retrieve (structured + semantic)
        context = smart_retrieve(
            query=message,
            collection=self.collection,
            gpu_db=self.gpu_db,
            embedding_model=self.embedding_model,
            top_k=8,
        )

        # Build sources display
        sources_md = self._format_sources(context.get('results', []))

        # Assemble system message
        tool_descriptions = self.tool_executor.get_tool_descriptions()
        system_message = SYSTEM_PROMPT_TEMPLATE.format(
            tool_descriptions=tool_descriptions,
        )
        env_ctx = self._get_effective_env_context()
        if env_ctx:
            system_message += f"\n\n{env_ctx}"
        if self.gpu_context:
            system_message += f"\n\n{self.gpu_context}"

        # Build user message with structured + doc context
        context_sections = []
        if context['structured_data']:
            context_sections.append(
                "--- GPU COMPATIBILITY DATA (from structured database — high confidence) ---\n"
                f"{context['structured_data']}\n"
                "--- END GPU DATA ---"
            )
        if context['documentation']:
            context_sections.append(
                "--- DOCUMENTATION CONTEXT ---\n"
                f"{context['documentation']}\n"
                "--- END CONTEXT ---"
            )

        combined_context = "\n\n".join(context_sections) if context_sections else "No relevant documentation found."

        # Include verified source URLs
        source_urls_text = ""
        if context['source_urls']:
            source_urls_text = (
                "\n\nVerified source URLs for citation:\n"
                + "\n".join(f"- {url}" for url in context['source_urls'][:10])
            )

        user_message = (
            "Based on the following information, answer the user's question.\n"
            "Prefer the GPU COMPATIBILITY DATA section for version/compatibility facts.\n\n"
            f"{combined_context}"
            f"{source_urls_text}\n\n"
            f"**User Question:** {message}\n\n"
            "Provide a clear, actionable answer grounded in the provided data. "
            "Cite the source documents and include relevant URLs."
        )

        # Build messages
        messages = [{"role": "system", "content": system_message}]
        for msg in self.conversation_history[-6:]:
            messages.append(msg)
        messages.append({"role": "user", "content": user_message})

        # Call LLM
        response_text = self.provider.chat(messages=messages, stream=False)

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
            response_text = self.provider.chat(messages=messages, stream=False)

        # Update history
        self.conversation_history.append({"role": "user", "content": message})
        self.conversation_history.append({"role": "assistant", "content": response_text})

        return response_text, self.conversation_history, sources_md

    def respond_stream(
        self,
        message: str,
        history: List[Dict],
    ) -> Generator[Tuple[str, str], None, None]:
        """
        Stream the agent response chunk by chunk.
        Yields (current_response_text, sources_md).
        """
        self.initialize()

        if not message.strip():
            return

        # Smart Retrieve (structured + semantic)
        context = smart_retrieve(
            query=message,
            collection=self.collection,
            gpu_db=self.gpu_db,
            embedding_model=self.embedding_model,
            top_k=8,
        )
        sources_md = self._format_sources(context.get('results', []))

        # Assemble system message
        tool_descriptions = self.tool_executor.get_tool_descriptions()
        system_message = SYSTEM_PROMPT_TEMPLATE.format(
            tool_descriptions=tool_descriptions,
        )
        env_ctx = self._get_effective_env_context()
        if env_ctx:
            system_message += f"\n\n{env_ctx}"
        if self.gpu_context:
            system_message += f"\n\n{self.gpu_context}"

        # Build user message with structured + doc context
        context_sections = []
        if context['structured_data']:
            context_sections.append(
                "--- GPU COMPATIBILITY DATA (from structured database — high confidence) ---\n"
                f"{context['structured_data']}\n"
                "--- END GPU DATA ---"
            )
        if context['documentation']:
            context_sections.append(
                "--- DOCUMENTATION CONTEXT ---\n"
                f"{context['documentation']}\n"
                "--- END CONTEXT ---"
            )

        combined_context = "\n\n".join(context_sections) if context_sections else "No relevant documentation found."

        # Include verified source URLs
        source_urls_text = ""
        if context['source_urls']:
            source_urls_text = (
                "\n\nVerified source URLs for citation:\n"
                + "\n".join(f"- {url}" for url in context['source_urls'][:10])
            )

        user_message = (
            "Based on the following information, answer the user's question.\n"
            "Prefer the GPU COMPATIBILITY DATA section for version/compatibility facts.\n\n"
            f"{combined_context}"
            f"{source_urls_text}\n\n"
            f"**User Question:** {message}\n\n"
            "Provide a clear, actionable answer grounded in the provided data. "
            "Cite the source documents and include relevant URLs."
        )

        # Build messages
        messages = [{"role": "system", "content": system_message}]
        for msg in self.conversation_history[-6:]:
            messages.append(msg)
        messages.append({"role": "user", "content": user_message})

        # Call LLM with stream=True
        stream_generator = self.provider.chat(messages=messages, stream=True)
        
        response_text = ""
        for chunk in stream_generator:
            response_text += chunk
            yield response_text, sources_md

        # Handle tool calls (if any are emitted by the LLM)
        tool_calls = self.tool_executor.parse_tool_calls(response_text)
        if tool_calls:
            # Yield a message indicating that diagnostic tools are running
            yield response_text + "\n\n*Running diagnostic tools...*", sources_md
            
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
            
            # Stream the second stage response
            second_generator = self.provider.chat(messages=messages, stream=True)
            second_response_text = ""
            for chunk in second_generator:
                second_response_text += chunk
                yield second_response_text, sources_md
            
            response_text = second_response_text

        # Update history
        self.conversation_history.append({"role": "user", "content": message})
        self.conversation_history.append({"role": "assistant", "content": response_text})

    def _format_sources(self, results: List[Dict]) -> str:
        """Format retrieval results as markdown for the sources panel."""
        if not results:
            return "*No sources retrieved.*"

        lines = ["## Retrieved Sources\n"]
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
            return "Not yet initialized. Send a message to start."

        env = self.env_raw
        lines = ["## System Status\n"]

        # Provider info
        provider_label = "Cloud (Fireworks)" if self.provider_type == "cloud" else "Local AMD GPU"
        lines.append(f"**LLM Provider:** {provider_label}")
        lines.append(f"**Model:** `{self.model_name}`")
        lines.append("")

        # Container
        container = env.get('container', {})
        if container.get('in_container'):
            ctype = container.get('container_type', 'unknown')
            lines.append(f"**Running in {ctype} container**")

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

    def get_gpu_monitor_md(self) -> str:
        """Return live GPU monitor markdown."""
        utilization = detect_gpu_utilization()
        processes = detect_gpu_processes()
        return format_gpu_monitor(utilization, processes)

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
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

    .gradio-container {
        font-family: 'Inter', 'Segoe UI', sans-serif !important;
    }
    #header {
        background: linear-gradient(135deg, #0a0a1a 0%, #1a1a2e 30%, #16213e 60%, #0f3460 100%);
        color: #ed1c24;
        padding: 24px;
        border-radius: 12px;
        margin-bottom: 16px;
        text-align: center;
        border: 1px solid rgba(237, 28, 36, 0.2);
        box-shadow: 0 4px 20px rgba(237, 28, 36, 0.1);
    }
    #header h1 {
        color: #ed1c24;
        margin: 0;
        font-size: 2.2em;
        font-weight: 700;
        letter-spacing: -0.02em;
    }
    #header p {
        color: #94a3b8;
        margin: 8px 0 0 0;
        font-size: 0.95em;
    }
    .status-panel {
        background: linear-gradient(180deg, #0f0f23 0%, #1a1a2e 100%);
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 10px;
        padding: 16px;
    }
    .monitor-panel {
        background: linear-gradient(180deg, #0a1628 0%, #0f1d32 100%);
        border: 1px solid rgba(237, 28, 36, 0.15);
        border-radius: 10px;
        padding: 16px;
    }
    .sources-panel {
        max-height: 400px;
        overflow-y: auto;
    }
    .amd-badge {
        background: linear-gradient(135deg, #ed1c24, #c4161d);
        color: white;
        padding: 2px 8px;
        border-radius: 4px;
        font-size: 0.75em;
        font-weight: 600;
    }
    footer { display: none !important; }
    """

    with gr.Blocks(
        title="ROCm-Pilot -- AI-Powered AMD Setup Assistant",
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
            '<h1>ROCm-Pilot</h1>'
            '<p>AI-Powered AMD Setup Assistant -- Grounded in Official Documentation</p>'
            '<p style="margin-top:4px; font-size:0.8em; color:#64748b;">'
            'Powered by AMD Instinct GPU | ROCm | Cross-Encoder Reranking | Local LLM Inference'
            '</p>'
            '</div>'
        )

        with gr.Row():
            # Main chat column
            with gr.Column(scale=3):
                chatbot = gr.Chatbot(
                    label="Chat",
                    height=500,
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
                    clear_btn = gr.Button("Clear History", size="sm")
                    refresh_gpu_btn = gr.Button("Refresh GPU Monitor", size="sm")

            # Side panel
            with gr.Column(scale=1):

                # LLM Provider Selector
                with gr.Accordion("LLM Provider", open=True):
                    provider_radio = gr.Radio(
                        choices=["cloud", "local"],
                        value="cloud",
                        label="Inference Mode",
                        info="Cloud = Fireworks AI (DeepSeek) | Local = AMD GPU (Qwen 2.5)",
                    )
                    provider_status = gr.Markdown(
                        "Using **Cloud** provider (Fireworks AI)",
                    )

                # System Status
                status_md = gr.Markdown(
                    value="Send a message to initialize and detect hardware.",
                    label="System Status",
                    elem_classes=["status-panel"],
                )

                # GPU Monitor
                with gr.Accordion("AMD GPU Monitor", open=True):
                    gpu_monitor_md = gr.Markdown(
                        value="*Send a message or click Refresh to load GPU stats.*",
                        elem_classes=["monitor-panel"],
                    )

                # Remote Diagnostics
                with gr.Accordion("Remote Machine Diagnostics", open=False):
                    gr.Markdown(
                        "Run this command on your **local AMD machine** to "
                        "generate a diagnostic report, then paste the JSON output below.\n\n"
                        "```bash\n"
                        f"{DIAGNOSE_SCRIPT_ALT}\n"
                        "```\n\n"
                        "Or download and run:\n"
                        "```bash\n"
                        f"{DIAGNOSE_SCRIPT_CMD}\n"
                        "```"
                    )
                    remote_json_input = gr.Textbox(
                        label="Paste JSON output here",
                        placeholder='{"os": {...}, "gpus": [...], ...}',
                        lines=6,
                    )
                    remote_submit_btn = gr.Button(
                        "Load Remote Environment", size="sm", variant="secondary",
                    )
                    remote_status = gr.Markdown("")

        # Retrieved Sources Panel below the main layout
        with gr.Row():
            with gr.Column(scale=1):
                with gr.Accordion("Retrieved Source References", open=True):
                    sources_md = gr.Markdown(
                        value="*References will appear here after you send a message.*",
                        elem_classes=["sources-panel"],
                    )

        # AMD Compute Showcase banner
        gr.HTML(
            '<div style="background: linear-gradient(90deg, #0a0a1a, #1a1a2e); '
            'border: 1px solid rgba(237,28,36,0.15); border-radius: 8px; '
            'padding: 12px 20px; margin-top: 12px; text-align: center; '
            'color: #94a3b8; font-size: 0.85em;">'
            'AMD Compute Pipeline: '
            'GPU-accelerated Embeddings (sentence-transformers) | '
            'Cross-Encoder Reranking (ms-marco) | '
            'Local LLM Inference (Qwen 2.5) -- '
            'all running on <strong style="color: #ed1c24;">AMD Instinct</strong> via ROCm'
            '</div>'
        )

        # ── Event Handlers ──

        def user_submit(message, history):
            """Handle user message submission with streaming."""
            if not message.strip():
                yield "", history, gr.update(), gr.update(), gr.update()
                return

            # Add user message to display immediately
            history = history or []
            history.append({"role": "user", "content": message})

            # Yield first to clear textbox and show the user's message immediately!
            yield "", history, agent.get_status_md(), "", agent.get_gpu_monitor_md()

            # Add assistant placeholder response
            history.append({"role": "assistant", "content": ""})

            # Stream response
            for response_text, sources in agent.respond_stream(message, history[:-1]):
                history[-1]["content"] = response_text
                yield "", history, agent.get_status_md(), sources, agent.get_gpu_monitor_md()

        def handle_clear():
            """Handle clear history button."""
            agent.clear_history()
            return [], "", ""

        def handle_refresh_gpu():
            """Refresh GPU monitor data."""
            return agent.get_gpu_monitor_md()

        def handle_provider_switch(choice):
            """Switch LLM provider."""
            agent.switch_provider(choice)
            if choice == "local":
                return "Using **Local AMD GPU** provider (Qwen 2.5)"
            return "Using **Cloud** provider (Fireworks AI)"

        def handle_remote_submit(json_text):
            """Process pasted remote diagnostics JSON."""
            return agent.set_remote_env(json_text)

        # Bind events
        msg.submit(
            user_submit,
            inputs=[msg, chatbot],
            outputs=[msg, chatbot, status_md, sources_md, gpu_monitor_md],
        )
        submit_btn.click(
            user_submit,
            inputs=[msg, chatbot],
            outputs=[msg, chatbot, status_md, sources_md, gpu_monitor_md],
        )
        clear_btn.click(
            handle_clear,
            outputs=[chatbot, sources_md, gpu_monitor_md],
        )
        refresh_gpu_btn.click(
            handle_refresh_gpu,
            outputs=[gpu_monitor_md],
        )
        provider_radio.change(
            handle_provider_switch,
            inputs=[provider_radio],
            outputs=[provider_status],
        )
        remote_submit_btn.click(
            handle_remote_submit,
            inputs=[remote_json_input],
            outputs=[remote_status],
        )

    return demo


# ──────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    import argparse

    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(name)s] %(levelname)s: %(message)s',
    )

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
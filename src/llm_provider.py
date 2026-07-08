"""
LLM Provider Abstraction.
Supports switching between Cloud (Fireworks) and Local (HuggingFace) inference.
"""

import os
import logging
import threading
from typing import Generator, Union, List, Dict, Optional

logger = logging.getLogger("rocm_pilot.llm")


class BaseLLMProvider:
    def chat(self, messages: List[Dict], stream: bool = False) -> Union[str, Generator]:
        raise NotImplementedError


class FireworksProvider(BaseLLMProvider):
    def __init__(self, model: str):
        self.model = model
        from src.fireworks_client import chat
        self.chat_func = chat

    def chat(self, messages: List[Dict], stream: bool = False) -> Union[str, Generator]:
        return self.chat_func(messages=messages, model=self.model, stream=stream)


class LocalGPUProvider(BaseLLMProvider):
    """Local GPU inference provider using HuggingFace transformers.

    Supports both blocking and streaming generation via TextIteratorStreamer.
    """

    DEFAULT_MODEL_ID = os.environ.get("LOCAL_MODEL_ID", "Qwen/Qwen2.5-7B-Instruct")

    def __init__(self, model_id: Optional[str] = None):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

        model_id = model_id or self.DEFAULT_MODEL_ID
        logger.info("Initializing Local LLM on AMD GPU: %s", model_id)

        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.float16,
            device_map="auto",
        )
        self.pipe = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
        )

        # ── Verbose GPU diagnostics ──────────────────────────────────────
        self._log_gpu_info(torch)
        self._log_model_size()
        logger.info("Local LLM ready: %s", model_id)

    # ── internal helpers ─────────────────────────────────────────────────

    @staticmethod
    def _log_gpu_info(torch) -> None:
        """Log AMD GPU device name, VRAM usage, and HIP runtime version."""
        if not torch.cuda.is_available():
            logger.warning("No CUDA/ROCm device detected — running on CPU")
            return

        device = torch.cuda.current_device()
        name = torch.cuda.get_device_name(device)
        allocated = torch.cuda.memory_allocated(device) / (1024 ** 3)
        reserved = torch.cuda.memory_reserved(device) / (1024 ** 3)

        logger.info("GPU device : %s", name)
        logger.info("VRAM allocated : %.2f GiB", allocated)
        logger.info("VRAM reserved  : %.2f GiB", reserved)

        hip_version = getattr(torch.version, "hip", None)
        if hip_version:
            logger.info("HIP version    : %s", hip_version)
        else:
            logger.info("HIP version    : N/A (CUDA runtime)")

    def _log_model_size(self) -> None:
        """Log the total number of parameters in the loaded model."""
        total_params = sum(p.numel() for p in self.model.parameters())
        logger.info(
            "Model size     : %.2f B params (%.2f GiB @ fp16)",
            total_params / 1e9,
            total_params * 2 / (1024 ** 3),  # 2 bytes per fp16 param
        )

    # ── public API ───────────────────────────────────────────────────────

    def chat(
        self,
        messages: List[Dict],
        stream: bool = False,
        max_new_tokens: int = 512,
    ) -> Union[str, Generator]:
        """Generate a response from *messages*.

        Parameters
        ----------
        messages : list[dict]
            OpenAI-style chat messages.
        stream : bool
            If ``True`` return a token-by-token generator; otherwise return
            the completed string.
        max_new_tokens : int
            Maximum number of new tokens to generate.
        """
        prompt = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        if not stream:
            return self._generate_full(prompt, max_new_tokens)
        return self._generate_stream(prompt, max_new_tokens)

    # ── generation back-ends ─────────────────────────────────────────────

    def _generate_full(self, prompt: str, max_new_tokens: int) -> str:
        """Blocking generation — returns the complete assistant reply."""
        result = self.pipe(prompt, max_new_tokens=max_new_tokens, do_sample=True)
        return result[0]["generated_text"][len(prompt):]

    def _generate_stream(
        self, prompt: str, max_new_tokens: int
    ) -> Generator[str, None, None]:
        """Streaming generation via ``TextIteratorStreamer``.

        Tokenises the prompt, launches ``model.generate`` in a background
        thread, and yields decoded tokens as they become available.
        """
        import torch
        from transformers import TextIteratorStreamer

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)

        streamer = TextIteratorStreamer(
            self.tokenizer, skip_prompt=True, skip_special_tokens=True
        )

        generation_kwargs = {
            **inputs,
            "max_new_tokens": max_new_tokens,
            "do_sample": True,
            "streamer": streamer,
        }

        thread = threading.Thread(
            target=self.model.generate, kwargs=generation_kwargs, daemon=True
        )
        thread.start()

        for token_text in streamer:
            yield token_text

        thread.join()


# ── factory ──────────────────────────────────────────────────────────────


def get_provider(provider_type: str, model: Optional[str] = None) -> BaseLLMProvider:
    """Return the appropriate LLM provider.

    Parameters
    ----------
    provider_type : str
        ``"local"`` for on-device AMD GPU inference,
        anything else for the Fireworks cloud provider.
    model : str, optional
        Model identifier.  For *local* mode this overrides the default
        ``LOCAL_MODEL_ID`` env-var / ``Qwen/Qwen2.5-7B-Instruct`` fallback.
        For *cloud* mode this is forwarded to ``FireworksProvider``.
    """
    if provider_type == "local":
        return LocalGPUProvider(model_id=model)
    return FireworksProvider(model=model or "")

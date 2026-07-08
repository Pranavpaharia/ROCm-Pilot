"""
Fireworks AI API client.
Uses the OpenAI-compatible chat-completions endpoint.
"""

import os
from typing import Generator, Union

# Default model — Llama 3.1 70B is a strong general-purpose choice
DEFAULT_MODEL = "accounts/fireworks/models/llama-v3p1-70b-instruct"
FIREWORKS_BASE_URL = "https://api.fireworks.ai/inference/v1"


def get_client():
    """
    Create an OpenAI-compatible client pointed at Fireworks AI.

    Requires the FIREWORKS_API_KEY environment variable to be set.
    """
    from openai import OpenAI

    api_key = os.environ.get('FIREWORKS_API_KEY')
    if not api_key:
        raise ValueError(
            "FIREWORKS_API_KEY environment variable is not set!\n"
            "Set it with:\n"
            "  export FIREWORKS_API_KEY='your-key-here'\n"
            "Or in Python:\n"
            "  import os; os.environ['FIREWORKS_API_KEY'] = 'your-key-here'"
        )

    return OpenAI(
        base_url=FIREWORKS_BASE_URL,
        api_key=api_key,
    )


def chat(
    messages: list,
    model: str = DEFAULT_MODEL,
    temperature: float = 0.3,
    max_tokens: int = 2048,
    stream: bool = False,
) -> Union[str, Generator]:
    """
    Send a chat completion request to Fireworks AI.

    Args:
        messages: List of message dicts, each with 'role' and 'content'.
        model: Fireworks model identifier.
        temperature: Sampling temperature (lower = more focused / deterministic).
        max_tokens: Maximum number of tokens in the response.
        stream: If True, return a generator that yields response chunks.

    Returns:
        The full response string, or a generator of string chunks if stream=True.
    """
    client = get_client()

    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
        stream=stream,
    )

    if stream:
        return _stream_response(response)
    else:
        return response.choices[0].message.content


def _stream_response(response) -> Generator:
    """Yield content chunks from a streaming chat-completion response."""
    for chunk in response:
        delta = chunk.choices[0].delta
        if delta.content:
            yield delta.content


def test_connection(model: str = DEFAULT_MODEL) -> bool:
    """
    Quick connectivity test against the Fireworks AI API.

    Returns True on success, False on failure.
    """
    try:
        result = chat(
            messages=[
                {"role": "user", "content": "Respond with only the word 'hello'."}
            ],
            model=model,
            max_tokens=10,
        )
        print(f"✅ Fireworks AI connected — response: {result}")
        return True
    except Exception as e:
        print(f"❌ Fireworks AI connection failed: {e}")
        return False


if __name__ == '__main__':
    test_connection()

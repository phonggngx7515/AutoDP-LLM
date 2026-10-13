"""
API caller utilities
"""

import os
import time
from openai import OpenAI

try:
    from anthropic import Anthropic
except ImportError:  # Anthropic is only required for Anthropic/YesScale providers.
    Anthropic = None


def _read_price(name: str) -> float:
    """Read a USD-per-1M-tokens price from env. Returns 0.0 if unset or invalid."""
    raw = os.environ.get(name, "").strip()
    if not raw:
        return 0.0
    try:
        return float(raw)
    except ValueError:
        return 0.0


def _read_positive_int(name: str, default: int) -> int:
    """Read a positive integer setting and fail early on invalid configuration."""
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        value = int(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be a positive integer, got {raw!r}") from exc
    if value < 1:
        raise ValueError(f"{name} must be a positive integer, got {raw!r}")
    return value


def _normalize_provider(provider: str) -> str:
    """Map user-facing provider names to the API protocol used by the client."""
    normalized = provider.strip().lower().replace("_", "-")
    if normalized in {"openai", "openai-compatible"}:
        return "openai"
    if normalized in {"anthropic", "yescale"}:
        return "anthropic"
    raise ValueError(
        "LLM_PROVIDER must be one of: openai, openai-compatible, anthropic, "
        f"yescale; got {provider!r}"
    )


class APICaller:
    def __init__(self, base_url: str | None, api_key: str, model_name: str,
                 provider: str | None = None, max_tokens: int | None = None):
        configured_provider = provider or os.environ.get("LLM_PROVIDER", "openai")
        self.provider = _normalize_provider(configured_provider)

        if max_tokens is None:
            max_tokens = _read_positive_int("LLM_MAX_TOKENS", 8192)
        elif not isinstance(max_tokens, int) or isinstance(max_tokens, bool) or max_tokens < 1:
            raise ValueError("max_tokens must be a positive integer")
        self.max_tokens = max_tokens

        if self.provider == "anthropic":
            if Anthropic is None:
                raise ImportError(
                    "The anthropic package is required when LLM_PROVIDER is "
                    "anthropic or yescale. Install dependencies with "
                    "pip install -r requirements.txt."
                )
            self.client = Anthropic(base_url=base_url, api_key=api_key)
        else:
            self.client = OpenAI(base_url=base_url, api_key=api_key)

        self.model_name = model_name
        self.messages = []  # Initialize an empty list to store messages

        # Cumulative token usage across all calls on this instance
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0
        self.total_tokens = 0
        self.total_api_latency = 0.0
        self.call_count = 0

        # Pricing in USD per 1M tokens — set via env (.env)
        # Defaults to 0 so cost shows as $0.00 unless prices are configured.
        self.price_prompt_per_1m = _read_price("PRICE_PROMPT_PER_1M_USD")
        self.price_completion_per_1m = _read_price("PRICE_COMPLETION_PER_1M_USD")

    def call_llm(self, prompt, system_prompt=None, temperature=0.2, reset=False,
                 seed=42):

        # `reset=True` starts a fresh, isolated conversation for this call. This is
        # what lets each stage run as its own agent (own system prompt, no shared
        # history) instead of one long single-agent conversation. Cumulative token
        # usage is still tracked on the instance across all agents.
        if reset:
            self.messages = []

        # Add system prompt once (if provided at first call)
        if not self.messages and system_prompt:
            self.messages.append({"role": "system", "content": system_prompt})

        # Append the user message
        self.messages.append({"role": "user", "content": prompt})

        # Call LLM — this elapsed time is the API latency for this call.
        t0 = time.perf_counter()
        if self.provider == "anthropic":
            response = self._call_anthropic()
        else:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=self.messages,
                seed=seed,
                temperature=temperature,
            )
        api_latency = time.perf_counter() - t0

        # Track token usage and extract text from each provider's response shape.
        if self.provider == "anthropic":
            usage = getattr(response, "usage", None)
            prompt_tokens = sum(
                int(getattr(usage, field, 0) or 0)
                for field in (
                    "input_tokens",
                    "cache_creation_input_tokens",
                    "cache_read_input_tokens",
                )
            ) if usage else 0
            completion_tokens = int(getattr(usage, "output_tokens", 0) or 0) if usage else 0
            total_tokens = prompt_tokens + completion_tokens
            reply = "".join(
                str(getattr(block, "text", ""))
                for block in (getattr(response, "content", None) or [])
                if getattr(block, "type", None) == "text"
            )
        else:
            usage = getattr(response, "usage", None)
            prompt_tokens = getattr(usage, "prompt_tokens", 0) if usage else 0
            completion_tokens = getattr(usage, "completion_tokens", 0) if usage else 0
            total_tokens = getattr(usage, "total_tokens", 0) if usage else 0
            reply = response.choices[0].message.content

        self.total_prompt_tokens += prompt_tokens
        self.total_completion_tokens += completion_tokens
        self.total_tokens += total_tokens
        self.total_api_latency += api_latency
        self.call_count += 1

        call_cost = self._compute_cost(prompt_tokens, completion_tokens)
        print(
            f"[LLM call #{self.call_count}] api_latency={api_latency:.2f}s | "
            f"prompt={prompt_tokens} tok, completion={completion_tokens} tok, total={total_tokens} tok | "
            f"cost=${call_cost:.6f}"
        )

        # Save assistant reply in the provider-neutral conversation history.
        self.messages.append({"role": "assistant", "content": reply})

        return reply

    def _call_anthropic(self):
        """Call Anthropic Messages API (also used by YesScale's Anthropic endpoint)."""
        system_parts = [
            str(message["content"])
            for message in self.messages
            if message.get("role") == "system"
        ]
        anthropic_messages = [
            message for message in self.messages if message.get("role") != "system"
        ]
        request = {
            "model": self.model_name,
            "max_tokens": self.max_tokens,
            "messages": anthropic_messages,
        }
        if system_parts:
            request["system"] = "\n\n".join(system_parts)

        # Anthropic Messages has no seed parameter. Temperature is intentionally
        # omitted for compatibility with current SDK/model versions that removed it.
        return self.client.messages.create(**request)

    def _compute_cost(self, prompt_tokens: int, completion_tokens: int) -> float:
        return (
            prompt_tokens / 1_000_000.0 * self.price_prompt_per_1m
            + completion_tokens / 1_000_000.0 * self.price_completion_per_1m
        )

    def usage_summary(self) -> dict:
        return {
            "calls": self.call_count,
            "prompt_tokens": self.total_prompt_tokens,
            "completion_tokens": self.total_completion_tokens,
            "total_tokens": self.total_tokens,
            "api_latency_seconds": self.total_api_latency,
            "cost_usd": self._compute_cost(self.total_prompt_tokens, self.total_completion_tokens),
            "price_prompt_per_1m_usd": self.price_prompt_per_1m,
            "price_completion_per_1m_usd": self.price_completion_per_1m,
        }

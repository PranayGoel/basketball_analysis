"""
Provider-agnostic LLM client for the AI insight layer (narrative generation + the
tool-calling Q&A chat in game_qa.py).

Design note: OpenAI, Google Gemini, DeepSeek, and OpenRouter are all reachable
through the exact same `openai` SDK client shape (`OpenAI(api_key=..., base_url=...)`
+ `.chat.completions.create(...)`) -- switching providers is a config change, not a
per-provider branch. See PROVIDER_CONFIG below.

The `openai` package import is deferred into get_client() (not done at module import
time) specifically so the rest of this module -- and anything that depends on it via
dependency injection (pass a `client` object in, rather than constructing one
internally) -- can be unit-tested without the `openai` package installed at all. Only
get_client() itself needs the real package, and only when actually called.
"""

import os


# Model name is intentionally NOT hardcoded as a default here (beyond a documented
# fallback) -- DeepSeek's "deepseek-chat" alias is flagged in their docs for
# deprecation around 2026-07-24, so the active model should come from config
# (LLM_MODEL env var), not be baked into this module.
PROVIDER_CONFIG = {
    "openai": {
        "base_url": None,  # None => SDK default (https://api.openai.com/v1)
        "default_model": "gpt-4o-mini",
        "supports_strict_json_schema": True,
    },
    "gemini": {
        "base_url": "https://generativelanguage.googleapis.com/v1beta/openai/",
        "default_model": "gemini-flash-latest",
        # Real per Google's docs, but whether it works unchanged through the OpenAI
        # *compatibility* endpoint (vs. Gemini's native responseSchema field) wasn't
        # confirmed empirically -- treat as unverified until tested against a live key.
        "supports_strict_json_schema": True,
    },
    "deepseek": {
        "base_url": "https://api.deepseek.com",
        "default_model": "deepseek-chat",
        # DeepSeek's compat layer is JSON-mode-only per their docs at research time --
        # no confirmed json_schema/strict guarantee. Structured-output code must not
        # assume one for this provider.
        "supports_strict_json_schema": False,
    },
    "openrouter": {
        "base_url": "https://openrouter.ai/api/v1",
        # openai/gpt-oss-20b:free was confirmed via a live query of OpenRouter's own
        # /api/v1/models endpoint (not just docs prose) to expose BOTH
        # response_format/structured_outputs AND tools/tool_choice on the free tier --
        # the genuinely-free option with the least uncertainty for this codebase's
        # needs, unlike Gemini's compat layer (documented "beta", with a confirmed
        # bug on the 2.0 model series specifically) or Groq (whose docs explicitly
        # disallow combining strict JSON schema with tool calling in one request).
        "default_model": "openai/gpt-oss-20b:free",
        "supports_strict_json_schema": True,
    },
}


class UnknownProviderError(ValueError):
    pass


class MissingCredentialError(ValueError):
    pass


def resolve_provider_config(provider, api_key=None, model=None, base_url=None):
    """
    Resolve the (api_key, base_url, model, supports_strict_json_schema) tuple for a
    provider, applying env-var fallbacks and explicit overrides. Pure function --
    no network access, no SDK import -- fully unit-testable on its own.

    Args:
        provider (str): one of PROVIDER_CONFIG's keys ("openai", "gemini", "deepseek").
        api_key (str, optional): explicit API key; falls back to LLM_API_KEY env var.
        model (str, optional): explicit model name; falls back to LLM_MODEL env var,
            then the provider's documented default.
        base_url (str, optional): explicit base_url override; falls back to
            LLM_BASE_URL env var, then the provider's default.

    Returns:
        dict: {"api_key", "base_url", "model", "supports_strict_json_schema"}

    Raises:
        UnknownProviderError: if `provider` isn't in PROVIDER_CONFIG.
        MissingCredentialError: if no api_key is available from any source.
    """
    if provider not in PROVIDER_CONFIG:
        raise UnknownProviderError(
            f"Unknown LLM provider '{provider}'. Valid options: {sorted(PROVIDER_CONFIG)}"
        )

    config = PROVIDER_CONFIG[provider]
    resolved_key = api_key or os.environ.get("LLM_API_KEY")
    if not resolved_key:
        raise MissingCredentialError(
            f"No API key provided for provider '{provider}'. Pass api_key= or set LLM_API_KEY."
        )

    return {
        "api_key": resolved_key,
        "base_url": base_url or os.environ.get("LLM_BASE_URL") or config["base_url"],
        "model": model or os.environ.get("LLM_MODEL") or config["default_model"],
        "supports_strict_json_schema": config["supports_strict_json_schema"],
    }


def get_client(provider=None, api_key=None, model=None, base_url=None):
    """
    Construct a real OpenAI-SDK client configured for the given provider (defaulting
    to the LLM_PROVIDER env var). This is the one function in this module that needs
    the `openai` package installed -- kept separate so everything else can be tested
    without it.

    Returns:
        tuple: (client, resolved_config_dict) -- resolved_config_dict is the dict
        returned by resolve_provider_config, useful for callers that need to know
        e.g. whether the active provider supports strict JSON schema.
    """
    from openai import OpenAI  # deferred import -- see module docstring

    provider = provider or os.environ.get("LLM_PROVIDER", "openai")
    resolved = resolve_provider_config(provider, api_key=api_key, model=model, base_url=base_url)
    client = OpenAI(api_key=resolved["api_key"], base_url=resolved["base_url"])
    return client, resolved


def call_chat(client, model, messages, temperature=0, max_tokens=1000, **kwargs):
    """
    Thin wrapper around client.chat.completions.create(...) returning just the text
    content. Accepts any object shaped like the OpenAI SDK's client (duck typing) --
    tests can pass a fake client exposing the same `.chat.completions.create(...)`
    shape without needing the real `openai` package installed.
    """
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
        **kwargs,
    )
    return response.choices[0].message.content

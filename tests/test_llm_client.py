import os
import unittest
from unittest import mock

from llm_client import resolve_provider_config, UnknownProviderError, MissingCredentialError, call_chat
from tests.fakes import FakeClient, FakeResponse, FakeMessage


class TestResolveProviderConfig(unittest.TestCase):
    def test_unknown_provider_raises(self):
        with self.assertRaises(UnknownProviderError):
            resolve_provider_config("not-a-real-provider", api_key="k")

    def test_missing_api_key_raises(self):
        with mock.patch.dict(os.environ, {}, clear=True):
            with self.assertRaises(MissingCredentialError):
                resolve_provider_config("openai")

    def test_explicit_args_take_priority_over_env(self):
        with mock.patch.dict(os.environ, {"LLM_API_KEY": "env-key", "LLM_MODEL": "env-model"}):
            resolved = resolve_provider_config("openai", api_key="explicit-key", model="explicit-model")
        self.assertEqual(resolved["api_key"], "explicit-key")
        self.assertEqual(resolved["model"], "explicit-model")

    def test_env_fallback_used_when_no_explicit_args(self):
        with mock.patch.dict(os.environ, {"LLM_API_KEY": "env-key"}, clear=True):
            resolved = resolve_provider_config("openai")
        self.assertEqual(resolved["api_key"], "env-key")
        self.assertEqual(resolved["model"], "gpt-4o-mini")  # provider default

    def test_gemini_uses_openai_compat_base_url(self):
        resolved = resolve_provider_config("gemini", api_key="k")
        self.assertEqual(resolved["base_url"], "https://generativelanguage.googleapis.com/v1beta/openai/")

    def test_deepseek_does_not_claim_strict_json_schema_support(self):
        resolved = resolve_provider_config("deepseek", api_key="k")
        self.assertFalse(resolved["supports_strict_json_schema"])

    def test_openai_claims_strict_json_schema_support(self):
        resolved = resolve_provider_config("openai", api_key="k")
        self.assertTrue(resolved["supports_strict_json_schema"])

    def test_openrouter_uses_free_tier_default_model_and_base_url(self):
        resolved = resolve_provider_config("openrouter", api_key="k")
        self.assertEqual(resolved["base_url"], "https://openrouter.ai/api/v1")
        self.assertEqual(resolved["model"], "openai/gpt-oss-20b:free")
        self.assertTrue(resolved["supports_strict_json_schema"])


class TestCallChat(unittest.TestCase):
    def test_returns_message_content_from_a_duck_typed_client(self):
        client = FakeClient([FakeResponse(FakeMessage(content="hello"))])
        result = call_chat(client, "any-model", [{"role": "user", "content": "hi"}])
        self.assertEqual(result, "hello")
        self.assertEqual(client.calls[0]["model"], "any-model")


if __name__ == "__main__":
    unittest.main()

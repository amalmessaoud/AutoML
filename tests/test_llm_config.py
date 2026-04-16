# tests/test_llm_config.py
import os
from unittest.mock import patch

import pytest
from openai import OpenAI

from src.config.llm_config import GOOGLE_GEMINI_FLASH, GROQ_LLAMA_8B, GROQ_LLAMA_70B, LLMConfig


class TestLLMConfigInstantiation:
    def test_groq_8b_fields(self):
        assert GROQ_LLAMA_8B.provider == 'groq'
        assert GROQ_LLAMA_8B.model == 'llama-3.1-8b-instant'
        assert GROQ_LLAMA_8B.api_key_env_var == 'GROQ_API_KEY'

    def test_groq_70b_fields(self):
        assert GROQ_LLAMA_70B.provider == 'groq'
        assert GROQ_LLAMA_70B.model == 'llama-3.1-70b-versatile'

    def test_google_fields(self):
        assert GOOGLE_GEMINI_FLASH.provider == 'google'
        assert 'generativelanguage' in GOOGLE_GEMINI_FLASH.base_url

    def test_no_api_key_in_config_object(self):
        """The config object must never contain the actual key value."""
        config_dict = GROQ_LLAMA_8B.to_loggable_dict()
        # api_key_env_var should be the ENV VAR NAME, not a key value
        assert config_dict['api_key_env_var'] == 'GROQ_API_KEY'
        # Make sure no field contains anything that looks like a real API key
        for value in config_dict.values():
            assert not str(value).startswith('gsk_'), 'Real API key found in config!'
            assert not str(value).startswith('AIza'), 'Real API key found in config!'

    def test_switching_model_is_one_line(self):
        """Demonstrate that switching model is a single field change."""
        from dataclasses import replace

        fast = GROQ_LLAMA_8B
        strong = replace(fast, model='llama-3.1-70b-versatile')
        assert fast.model != strong.model
        assert fast.base_url == strong.base_url  # everything else stays the same


class TestBuildClient:
    def test_build_client_returns_openai_instance(self):
        with patch.dict(os.environ, {'GROQ_API_KEY': 'test-fake-key'}):
            client = GROQ_LLAMA_8B.build_client()
            assert isinstance(client, OpenAI)

    def test_build_client_raises_if_env_var_missing(self):
        config = LLMConfig(
            provider='groq',
            model='llama-3.1-8b-instant',
            base_url='https://api.groq.com/openai/v1',
            temperature=0.3,
            api_key_env_var='NONEXISTENT_KEY_12345',
        )
        with pytest.raises(EnvironmentError, match='NONEXISTENT_KEY_12345'):
            config.build_client()

    def test_google_client_builds_correctly(self):
        with patch.dict(os.environ, {'GOOGLE_API_KEY': 'test-fake-google-key'}):
            client = GOOGLE_GEMINI_FLASH.build_client()
            assert isinstance(client, OpenAI)

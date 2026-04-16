# src/config/llm_config.py
import os
from dataclasses import dataclass
from typing import Literal

from openai import OpenAI


@dataclass
class LLMConfig:
    provider: Literal['groq', 'google', 'ollama']
    model: str
    base_url: str
    temperature: float
    api_key_env_var: str  # Name of the env var, NEVER the key itself

    def build_client(self) -> OpenAI:
        """Build an OpenAI-compatible client from this config."""
        api_key = os.environ.get(self.api_key_env_var)
        if not api_key:
            raise OSError(
                f"Environment variable '{self.api_key_env_var}' is not set. "
                f'Add it to your .env file.'
            )
        return OpenAI(api_key=api_key, base_url=self.base_url)

    def to_loggable_dict(self) -> dict:
        """Serialize config for logging — api_key_env_var included, key value never."""
        return {
            'provider': self.provider,
            'model': self.model,
            'base_url': self.base_url,
            'temperature': self.temperature,
            'api_key_env_var': self.api_key_env_var,
        }


# --- Ready-made configs — import these in agent files ---

GROQ_LLAMA_8B = LLMConfig(
    provider='groq',
    model='llama-3.1-8b-instant',
    base_url='https://api.groq.com/openai/v1',
    temperature=0.3,
    api_key_env_var='GROQ_API_KEY',
)

GROQ_LLAMA_70B = LLMConfig(
    provider='groq',
    model='llama-3.1-70b-versatile',
    base_url='https://api.groq.com/openai/v1',
    temperature=0.3,
    api_key_env_var='GROQ_API_KEY',
)

GOOGLE_GEMINI_FLASH = LLMConfig(
    provider='google',
    model='gemini-2.0-flash',
    base_url='https://generativelanguage.googleapis.com/v1beta/openai/',
    temperature=0.3,
    api_key_env_var='GOOGLE_API_KEY',
)

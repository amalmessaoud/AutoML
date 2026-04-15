# src/agents/base_agent.py
from abc import ABC, abstractmethod

from src.config.llm_config import LLMConfig


class BaseAgent(ABC):
    """
    Abstract base for all LLM agents in the pipeline.

    Rules enforced here:
    - Every agent gets a LLMConfig — no hardcoded URLs or model names in subclasses
    - Every agent gets a shared logs list — append to it, never print directly
    - No agent stores the API key — LLMConfig.build_client() handles that at call time
    """

    def __init__(self, config: LLMConfig, logs: list[str]) -> None:
        self.config = config
        self.logs = logs

    def _log(self, message: str) -> None:
        self.logs.append(message)

    def _build_client(self):
        """Build the LLM client. Reads API key from env at call time."""
        return self.config.build_client()

    @abstractmethod
    def run(self, **kwargs):
        """Execute the agent's task. Subclasses define inputs/outputs."""
        ...
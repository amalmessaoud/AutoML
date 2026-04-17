# src/agents/base_agent.py
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

from src.config.llm_config import LLMConfig

if TYPE_CHECKING:
    from src.utils.cost_reporter import CostTracker


class BaseAgent(ABC):
    def __init__(
        self,
        config: LLMConfig,
        logs: list[str],
        cost_tracker: 'CostTracker | None' = None,
    ) -> None:
        self.config = config
        self.logs = logs
        self.cost_tracker = cost_tracker

    def _log(self, message: str) -> None:
        self.logs.append(message)

    def _build_client(self):
        return self.config.build_client()

    def _record_call(self, agent_name: str, prompt: str, completion: str) -> None:
        if self.cost_tracker is not None:
            self.cost_tracker.record_call(agent_name, prompt, completion)

    @abstractmethod
    def run(self, **kwargs): ...

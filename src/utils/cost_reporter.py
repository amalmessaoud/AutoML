# src/utils/cost_reporter.py
from dataclasses import dataclass

# Approximate token costs per model (input + output combined, per 1K tokens)
# Using Groq free tier — cost is 0 but we track token counts for the paper
# These are character-based estimates since Groq doesn't return token counts
# in streaming mode. 1 token ≈ 4 characters is the standard approximation.

CHARS_PER_TOKEN = 4


@dataclass
class AgentCallCost:
    agent_name: str
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


@dataclass
class RunCostReport:
    total_llm_calls: int
    total_prompt_tokens: int
    total_completion_tokens: int
    total_tokens: int
    calls_per_agent: dict[str, int]
    tokens_per_agent: dict[str, int]
    validator_triggered_replans: int
    execution_errors_total: int
    iterations: int

    def summary(self) -> str:
        lines = [
            f'  Total LLM calls:         {self.total_llm_calls}',
            f'  Total tokens (est.):     {self.total_tokens}',
            f'  Prompt tokens (est.):    {self.total_prompt_tokens}',
            f'  Completion tokens (est.): {self.total_completion_tokens}',
            f'  Validator re-plans:      {self.validator_triggered_replans}',
            f'  Execution errors:        {self.execution_errors_total}',
            f'  Iterations:              {self.iterations}',
            '  Per-agent calls:',
        ]
        for agent, count in self.calls_per_agent.items():
            tokens = self.tokens_per_agent.get(agent, 0)
            lines.append(f'    {agent}: {count} call(s), ~{tokens} tokens')
        return '\n'.join(lines)


def estimate_tokens(text: str) -> int:
    """Estimate token count from character count. 1 token ≈ 4 chars."""
    return max(1, len(text) // CHARS_PER_TOKEN)


class CostTracker:
    """
    Tracks LLM calls and token estimates throughout a pipeline run.
    Attach to orchestrator — call record_call() after each LLM interaction.
    """

    def __init__(self) -> None:
        self._calls: list[AgentCallCost] = []
        self._validator_replans: int = 0
        self._execution_errors: int = 0

    def record_call(
        self,
        agent_name: str,
        prompt: str,
        completion: str,
    ) -> None:
        prompt_tokens = estimate_tokens(prompt)
        completion_tokens = estimate_tokens(completion)
        self._calls.append(
            AgentCallCost(
                agent_name=agent_name,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=prompt_tokens + completion_tokens,
            )
        )

    def record_validator_replan(self) -> None:
        self._validator_replans += 1

    def record_execution_errors(self, count: int) -> None:
        self._execution_errors += count

    def build_report(self, iterations: int) -> RunCostReport:
        calls_per_agent: dict[str, int] = {}
        tokens_per_agent: dict[str, int] = {}

        for call in self._calls:
            calls_per_agent[call.agent_name] = calls_per_agent.get(call.agent_name, 0) + 1
            tokens_per_agent[call.agent_name] = (
                tokens_per_agent.get(call.agent_name, 0) + call.total_tokens
            )

        return RunCostReport(
            total_llm_calls=len(self._calls),
            total_prompt_tokens=sum(c.prompt_tokens for c in self._calls),
            total_completion_tokens=sum(c.completion_tokens for c in self._calls),
            total_tokens=sum(c.total_tokens for c in self._calls),
            calls_per_agent=calls_per_agent,
            tokens_per_agent=tokens_per_agent,
            validator_triggered_replans=self._validator_replans,
            execution_errors_total=self._execution_errors,
            iterations=iterations,
        )

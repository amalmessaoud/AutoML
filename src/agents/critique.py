# src/agents/critique.py
import json

from src.agents.base_agent import BaseAgent
from src.config.llm_config import LLMConfig
from src.schemas.plan import AutoMLPlan


class CritiqueAgent(BaseAgent):
    """
    Critique agent — LLM generates text critique and suggestion only.
    The solved boolean is NOT determined by the LLM (fixed in Day 4).
    For now: solved logic preserved as-is, will be replaced with deterministic
    threshold in Day 4 when constants.py is built.
    """

    def __init__(self, config: LLMConfig, logs: list[str]) -> None:
        super().__init__(config, logs)

    def run(self, plan: AutoMLPlan, results: dict, problem: str) -> dict:
        self._log("CritiqueAgent: starting evaluation.")

        valid_results = {k: v for k, v in results.items() if "error" not in v}
        if not valid_results:
            best_model = None
            best_score = 0.0
        else:
            best_model = max(valid_results, key=lambda k: valid_results[k]["mean_score"])
            best_score = valid_results[best_model]["mean_score"]

        prompt = f"""
You are an expert ML evaluator. Review this AutoML run.

Problem: {problem}

Plan Summary:
- Target: {plan.target_column}
- Metric: {plan.primary_metric}
- Preprocessing: {len(plan.preprocessing_steps)} steps
- Models: {', '.join([m.name for m in plan.models_to_try])}

Results:
Best Model: {best_model}
Best Score: {best_score:.4f}
Full Results: {json.dumps(results, indent=2)}

Output exactly this JSON:
{{
  "decision": "SOLVED" or "NOT_SOLVED",
  "reason": "brief explanation",
  "suggestion": "what to try next if NOT_SOLVED (or empty string)"
}}
"""

        client = self._build_client()
        self._log(f"CritiqueAgent: calling {self.config.provider}/{self.config.model}.")

        response = client.chat.completions.create(
            model=self.config.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=self.config.temperature,
            response_format={"type": "json_object"},
        )

        critique = json.loads(response.choices[0].message.content)
        self._log("CritiqueAgent: evaluation complete.")

        return {
            "solved": critique["decision"] == "SOLVED",  # NOTE: replaced Day 4
            "critique": critique["reason"],
            "suggestion": critique["suggestion"],
            "best_model": best_model,
            "best_score": best_score,
        }


# --- Convenience function for backward compatibility with orchestrator ---
def evaluate_results(
    plan: AutoMLPlan,
    results: dict,
    problem: str,
    config: LLMConfig = None,
    logs: list[str] = None,
) -> dict:
    from src.config.llm_config import GROQ_LLAMA_8B
    if config is None:
        config = GROQ_LLAMA_8B
    if logs is None:
        logs = []
    agent = CritiqueAgent(config=config, logs=logs)
    return agent.run(plan=plan, results=results, problem=problem)
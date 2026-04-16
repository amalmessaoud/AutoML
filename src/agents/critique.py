# src/agents/critique.py
import json

import numpy as np
import pandas as pd

from src.agents.base_agent import BaseAgent
from src.config.constants import compute_dummy_score, is_solved
from src.config.llm_config import LLMConfig
from src.schemas.plan import AutoMLPlan


class CritiqueAgent(BaseAgent):
    def __init__(self, config: LLMConfig, logs: list[str]) -> None:
        super().__init__(config, logs)

    def run(
        self,
        plan: AutoMLPlan,
        results: dict,
        problem: str,
        X: pd.DataFrame,
        y: np.ndarray,
    ) -> dict:
        self._log("CritiqueAgent: starting evaluation.")

        valid_results = {k: v for k, v in results.items() if "error" not in v}
        if not valid_results:
            best_model = None
            best_score = 0.0
        else:
            best_model = max(valid_results, key=lambda k: valid_results[k]["mean_score"])
            best_score = valid_results[best_model]["mean_score"]

        # --- Deterministic solved — never from LLM ---
        dummy_score = compute_dummy_score(X, y, plan.primary_metric, plan.random_seed)
        solved = is_solved(best_score, plan.primary_metric, dummy_score)

        self._log(
            f"CritiqueAgent: best={best_model} score={best_score:.4f} "
            f"dummy={dummy_score:.4f} ({'SOLVED' if solved else 'NOT_SOLVED'})"
        )

        prompt = f"""
You are an expert ML evaluator. Review this AutoML run and provide feedback.

Problem: {problem}

Plan:
- Target: {plan.target_column}
- Metric: {plan.primary_metric}
- Preprocessing steps: {len(plan.preprocessing_steps)}
- Models tried: {', '.join([m.name for m in plan.models_to_try])}

Results:
Best Model: {best_model}
Best Score: {best_score:.4f}
Dummy Baseline Score: {dummy_score:.4f}
Solved: {solved}
Full Results: {json.dumps(results, indent=2)}

Output exactly this JSON:
{{
  "reason": "brief explanation of the results",
  "suggestion": "what to try next to improve (or empty string if solved)"
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
            "solved": solved,
            "critique": critique["reason"],
            "suggestion": critique.get("suggestion", ""),
            "best_model": best_model,
            "best_score": best_score,
            "dummy_score": dummy_score,
        }


def evaluate_results(
    plan: AutoMLPlan,
    results: dict,
    problem: str,
    X: pd.DataFrame,
    y: np.ndarray,
    config: LLMConfig = None,
    logs: list[str] = None,
) -> dict:
    from src.config.llm_config import GROQ_LLAMA_8B
    if config is None:
        config = GROQ_LLAMA_8B
    if logs is None:
        logs = []
    agent = CritiqueAgent(config=config, logs=logs)
    return agent.run(plan=plan, results=results, problem=problem, X=X, y=y)
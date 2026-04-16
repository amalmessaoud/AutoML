# src/orchestrator.py
from dotenv import load_dotenv
load_dotenv()

from typing import Optional

from src.agents.analyzer import AnalyzerAgent
from src.agents.critique import CritiqueAgent
from src.agents.implementation import execute_plan
from src.config.llm_config import GROQ_LLAMA_8B, LLMConfig
from src.schemas.plan import AutoMLPlan, AttemptSummary
from src.utils.data_utils import generate_dataset_description


def run_automl_pipeline(
    csv_path: str,
    problem: str,
    max_iterations: int = 3,
    config: LLMConfig = None,
) -> dict:
    if config is None:
        config = GROQ_LLAMA_8B

    logs: list[str] = []
    previous_attempts: list[AttemptSummary] = []
    final_plan: Optional[AutoMLPlan] = None
    final_results: Optional[dict] = None
    final_evaluation: Optional[dict] = None

    analyzer = AnalyzerAgent(config=config, logs=logs)
    critique = CritiqueAgent(config=config, logs=logs)

    for iteration in range(1, max_iterations + 1):
        logs.append(f"==== ITERATION {iteration} ====")

        desc = generate_dataset_description(csv_path)

        plan = analyzer.run(
            dataset_description=desc,
            problem_description=problem,
            previous_attempts=previous_attempts,
        )
        final_plan = plan

        results, X, y_encoded = execute_plan(plan, csv_path, logs=logs)
        final_results = results

        evaluation = critique.run(
            plan=plan,
            results=results,
            problem=problem,
            X=X,
            y=y_encoded,
        )
        final_evaluation = evaluation

        errors = [
            f"{model}: {info['error']}"
            for model, info in results.items()
            if "error" in info
        ]

        previous_attempts.append(AttemptSummary(
            iteration=iteration,
            models_tried=[m.name for m in plan.models_to_try],
            best_score=evaluation["best_score"],
            metric=plan.primary_metric,
            failure_reason=evaluation["suggestion"] if not evaluation["solved"] else None,
            execution_errors=errors,
        ))

        if evaluation["solved"]:
            logs.append("Pipeline: problem solved — stopping.")
            break

        logs.append("Pipeline: not solved — memory updated for next iteration.")

    else:
        logs.append("Pipeline: max iterations reached.")

    return {
        "final_plan": final_plan,
        "final_results": final_results,
        "final_evaluation": final_evaluation,
        "logs": logs,
        "iterations": iteration,
    }
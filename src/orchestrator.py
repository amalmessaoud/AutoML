# src/orchestrator.py
from dotenv import load_dotenv
load_dotenv()

from typing import Optional

from src.agents.analyzer import AnalyzerAgent
from src.agents.critique import CritiqueAgent
from src.agents.implementation import execute_plan
from src.config.llm_config import GROQ_LLAMA_8B, LLMConfig
from src.schemas.plan import AutoMLPlan
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
    current_problem = problem
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
            problem_description=current_problem,
        )
        final_plan = plan

        results, X, y_encoded = execute_plan(plan, csv_path, logs=logs)
        final_results = results

        evaluation = critique.run(
            plan=plan,
            results=results,
            problem=current_problem,
            X=X,
            y=y_encoded,
        )
        final_evaluation = evaluation

        if evaluation["solved"]:
            logs.append("Pipeline: problem solved — stopping.")
            break

        logs.append("Pipeline: not solved — passing feedback to next iteration.")
        current_problem = f"{problem}\nPrevious attempt feedback: {evaluation['suggestion']}"

    else:
        logs.append("Pipeline: max iterations reached.")

    return {
        "final_plan": final_plan,
        "final_results": final_results,
        "final_evaluation": final_evaluation,
        "logs": logs,
        "iterations": iteration,
    }
# src/orchestrator.py
import pandas as pd
from dotenv import load_dotenv

from src.agents.analyzer import AnalyzerAgent
from src.agents.critique import CritiqueAgent
from src.agents.dataset_quality_agent import DatasetQualityAgent
from src.agents.implementation import execute_plan
from src.config.llm_config import GROQ_LLAMA_8B, LLMConfig
from src.schemas.plan import AttemptSummary, AutoMLPlan
from src.schemas.quality_report import DatasetQualityReport
from src.utils.data_utils import generate_dataset_description

load_dotenv()

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
    final_plan: AutoMLPlan | None = None
    final_results: dict | None = None
    final_evaluation: dict | None = None
    quality_report: DatasetQualityReport | None = None

    analyzer = AnalyzerAgent(config=config, logs=logs)
    critique = CritiqueAgent(config=config, logs=logs)

    # Load dataframe once — used for quality agent and passed to critique
    df = pd.read_csv(csv_path, sep=None, engine='python')

    for iteration in range(1, max_iterations + 1):
        logs.append(f'==== ITERATION {iteration} ====')

        desc = generate_dataset_description(csv_path)

        # Quality agent — use confirmed target after first iteration
        if iteration == 1:
            guessed_target = df.columns[-1]
        else:
            guessed_target = final_plan.target_column

        quality_agent = DatasetQualityAgent(logs=logs)
        quality_report = quality_agent.run(df, target_column=guessed_target)

        plan = analyzer.run(
            dataset_description=desc,
            problem_description=problem,
            previous_attempts=previous_attempts,
            quality_report=quality_report,
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

        errors = [f'{model}: {info["error"]}' for model, info in results.items() if 'error' in info]

        previous_attempts.append(
            AttemptSummary(
                iteration=iteration,
                models_tried=[m.name for m in plan.models_to_try],
                best_score=evaluation['best_score'],
                metric=plan.primary_metric,
                failure_reason=evaluation['suggestion'] if not evaluation['solved'] else None,
                execution_errors=errors,
            )
        )

        if evaluation['solved']:
            logs.append('Pipeline: problem solved — stopping.')
            break

        logs.append('Pipeline: not solved — memory updated for next iteration.')

    else:
        logs.append('Pipeline: max iterations reached.')

    return {
        'final_plan': final_plan,
        'final_results': final_results,
        'final_evaluation': final_evaluation,
        'quality_report': quality_report,
        'logs': logs,
        'iterations': iteration,
    }


if __name__ == '__main__':
    output = run_automl_pipeline(
        'data/iris.csv',
        'Predict the species of an iris flower.',
    )
    print('\n'.join(output['logs']))
    print(f'\nSolved: {output["final_evaluation"]["solved"]}')
    print(
        f'Best: {output["final_evaluation"]["best_model"]} — {output["final_evaluation"]["best_score"]:.4f}'
    )

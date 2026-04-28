# src/orchestrator.py

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer

from src.agents.analyzer import AnalyzerAgent
from src.agents.critique import CritiqueAgent
from src.agents.dataset_quality_agent import DatasetQualityAgent
from src.agents.explainability_agent import ExplainabilityAgent
from src.agents.implementation import (
    MODEL_MAP,
    _build_transformer_specs,
    _strip_target_from_steps,
    execute_plan,
)
from src.agents.plan_validator_agent import PlanValidatorAgent
from src.config.llm_config import GROQ_LLAMA_8B, LLMConfig
from src.schemas.explainability import ExplainabilityReport
from src.schemas.plan import AttemptSummary, AutoMLPlan
from src.schemas.quality_report import DatasetQualityReport
from src.schemas.validation_result import ValidationResult
from src.utils.cost_reporter import CostTracker
from src.utils.data_utils import generate_dataset_description

load_dotenv()


def _fit_best_model(
    final_plan: AutoMLPlan,
    csv_path: str,
    best_model_name: str,
    logs: list[str],
):
    """
    Fit the best model on the full dataset (no CV) and return
    (fitted_model, X_transformed, y_encoded, feature_names).
    Used exclusively by ExplainabilityAgent.
    """
    from sklearn.preprocessing import LabelEncoder

    df = pd.read_csv(csv_path, sep=None, engine='python')
    X = df.drop(columns=[final_plan.target_column])
    y = df[final_plan.target_column]

    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    plan_steps = _strip_target_from_steps(
        final_plan.preprocessing_steps, final_plan.target_column, logs
    )

    transformers = _build_transformer_specs(plan_steps, X)
    if transformers:
        preprocessor = ColumnTransformer(transformers=transformers, remainder='passthrough')
    else:
        preprocessor = FunctionTransformer()

    model_spec = next(m for m in final_plan.models_to_try if m.name == best_model_name)
    model = MODEL_MAP[best_model_name](**model_spec.hyperparameters)

    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', model),
    ])
    pipeline.fit(X, y_encoded)

    fitted_model = pipeline.named_steps['classifier']

    try:
        feature_names = list(
            pipeline.named_steps['preprocessor'].get_feature_names_out()
        )
    except Exception:
        feature_names = list(X.columns)

    X_transformed = pipeline.named_steps['preprocessor'].transform(X)

    return fitted_model, X_transformed, y_encoded, feature_names


def run_automl_pipeline(
    csv_path: str,
    problem: str,
    max_iterations: int = 3,
    config: LLMConfig = None,
    use_explainability: bool = True,
    log_callback=None,
) -> dict:
    if config is None:
        config = GROQ_LLAMA_8B

    logs: list[str] = []

    def _log(msg: str):
        logs.append(msg)
        if log_callback:
            log_callback(msg)

    previous_attempts: list[AttemptSummary] = []
    final_plan: AutoMLPlan | None = None
    final_results: dict | None = None
    final_evaluation: dict | None = None
    quality_report: DatasetQualityReport | None = None
    validation_result: ValidationResult | None = None
    explainability_report: ExplainabilityReport | None = None
    cost_tracker = CostTracker()

    analyzer = AnalyzerAgent(config=config, logs=logs, cost_tracker=cost_tracker)
    critique = CritiqueAgent(config=config, logs=logs, cost_tracker=cost_tracker)
    validator = PlanValidatorAgent(logs=logs)

    df = pd.read_csv(csv_path, sep=None, engine='python')

    for iteration in range(1, max_iterations + 1):
        _log(f'==== ITERATION {iteration} ====')

        desc = generate_dataset_description(csv_path)

        guessed_target = df.columns[-1] if iteration == 1 else final_plan.target_column
        quality_agent = DatasetQualityAgent(logs=logs)
        quality_report = quality_agent.run(df, target_column=guessed_target)

        plan = analyzer.run(
            dataset_description=desc,
            problem_description=problem,
            previous_attempts=previous_attempts,
            quality_report=quality_report,
        )

        validation_result = validator.run(plan, quality_report)
        if not validation_result.passed:
            _log(
                f'PlanValidatorAgent: issues found — requesting correction. '
                f'Issues: {validation_result.issues}'
            )
            cost_tracker.record_validator_replan()
            correction_problem = (
                f'{problem}\n'
                f'CORRECTION REQUIRED — your previous plan had these validation issues '
                f'that MUST be fixed:\n'
                + '\n'.join(f'- {issue}' for issue in validation_result.issues)
            )
            plan = analyzer.run(
                dataset_description=desc,
                problem_description=correction_problem,
                previous_attempts=previous_attempts,
                quality_report=quality_report,
            )
            validation_result = validator.run(plan, quality_report)
            _log(
                f'PlanValidatorAgent: after correction — '
                f'{"PASSED" if validation_result.passed else "STILL FAILING"}.'
            )

        final_plan = plan

        results, X, y_encoded = execute_plan(plan, csv_path, logs=logs)
        final_results = results

        errors = [
            f'{model}: {info["error"]}'
            for model, info in results.items()
            if 'error' in info
        ]
        cost_tracker.record_execution_errors(len(errors))

        evaluation = critique.run(
            plan=plan,
            results=results,
            problem=problem,
            X=X,
            y=y_encoded,
        )
        final_evaluation = evaluation

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
            _log('Pipeline: problem solved — stopping.')
            break

        _log('Pipeline: not solved — memory updated for next iteration.')

    else:
        _log('Pipeline: max iterations reached.')

    # --- Explainability ---
    if use_explainability and final_plan is not None and final_evaluation is not None:
        best_model_name = final_evaluation.get('best_model')
        if (
            best_model_name
            and best_model_name in final_results
            and 'error' not in final_results[best_model_name]
        ):
            _log(f'ExplainabilityAgent: fitting {best_model_name} on full dataset.')
            try:
                fitted_model, X_transformed, y_enc, feature_names = _fit_best_model(
                    final_plan, csv_path, best_model_name, logs
                )
                explain_agent = ExplainabilityAgent(
                    config=config,
                    logs=logs,
                    generate_summary=True,
                )
                explainability_report = explain_agent.run(
                    model=fitted_model,
                    X=X_transformed,
                    y=y_enc,
                    feature_names=feature_names,
                    random_state=final_plan.random_seed,
                )
                _log('ExplainabilityAgent: report generated successfully.')
            except Exception as e:
                _log(f'ExplainabilityAgent: failed — {e}')
        else:
            _log(
                f'ExplainabilityAgent: skipped — best model '
                f'"{best_model_name}" had execution errors.'
            )

    cost_report = cost_tracker.build_report(iterations=iteration)
    _log('==== COST REPORT ====')
    _log(cost_report.summary())

    return {
        'final_plan': final_plan,
        'final_results': final_results,
        'final_evaluation': final_evaluation,
        'quality_report': quality_report,
        'validation_result': validation_result,
        'explainability_report': explainability_report,
        'cost_report': cost_report,
        'logs': logs,
        'iterations': iteration,
    }


if __name__ == '__main__':
    output = run_automl_pipeline(
        'data/adult.csv',
        'Predict the income range of individuals.',
    )
    print('\n'.join(output['logs']))
    print(f'\nSolved: {output["final_evaluation"]["solved"]}')
    print(
        f'Best: {output["final_evaluation"]["best_model"]} — '
        f'{output["final_evaluation"]["best_score"]:.4f}'
    )
    if output['explainability_report']:
        r = output['explainability_report']
        print(f'\nExplainability ({r.method}):')
        for f in r.top_features:
            print(f'  {f.rank}. {f.feature_name}: {f.importance_score:.4f}')
        print(f'\nSummary: {r.decision_summary}')
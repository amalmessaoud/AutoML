# experiments/run_experiment.py
"""
Single-run experiment entry point.

Usage:
    python -m experiments.run_experiment --dataset iris --mode schema --runs 5
    python -m experiments.run_experiment --dataset adult --mode baseline --runs 5

Saves each run to:
    experiments/artifacts/{dataset_slug}/{mode}/run_{i}.json

Collects per-run metrics:
    plan_valid, execution_success, best_score, n_iterations,
    validator_issues_caught, duration_seconds, mode, dataset
"""

from __future__ import annotations

import argparse
import json
import os
import time
from typing import Literal

import pandas as pd
from dotenv import load_dotenv

from data.load_datasets import get_dataset
from src.agents.analyzer import AnalyzerAgent
from src.agents.analyzer_baseline import BaselineAnalyzerAgent, BaselinePlan
from src.agents.critique import CritiqueAgent
from src.agents.dataset_quality_agent import DatasetQualityAgent
from src.agents.implementation import execute_plan
from src.agents.plan_validator_agent import PlanValidatorAgent
from src.config.llm_config import LLMConfig,GOOGLE_GEMINI_FLASH, OPENROUTER_DEEPSEEK

import warnings
from sklearn.exceptions import ConvergenceWarning
warnings.filterwarnings("ignore", category=ConvergenceWarning)

GROQ_LLAMA_8B = LLMConfig(
    provider='groq',
    model='llama-3.3-70b-versatile',
    base_url='https://api.groq.com/openai/v1',
    temperature=0.3,
    api_key_env_var='GROQ_API_KEY',
)
from src.schemas.plan import AttemptSummary, AutoMLPlan
from src.schemas.run_artifact import AgentCall, IterationRecord, RunArtifact
from src.utils.cost_reporter import CostTracker
from src.utils.data_utils import generate_dataset_description

load_dotenv()

ARTIFACTS_DIR = os.path.join(os.path.dirname(__file__), "artifacts")
MAX_ITERATIONS = 3


# ---------------------------------------------------------------------------
# Per-run result — what gets written to the CSV
# ---------------------------------------------------------------------------

def _make_output_dir(dataset_slug: str, mode: str) -> str:
    path = os.path.join(ARTIFACTS_DIR, dataset_slug, mode)
    os.makedirs(path, exist_ok=True)
    return path


def _save_artifact(artifact: RunArtifact, path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        f.write(artifact.model_dump_json(indent=2))


# ---------------------------------------------------------------------------
# Schema mode pipeline (wraps existing orchestrator logic)
# ---------------------------------------------------------------------------

def _run_schema(
    dataset_slug: str,
    run_index: int,
) -> dict:
    """Run one schema-mode experiment. Returns a flat metrics dict."""
    from datetime import UTC, datetime

    dataset = get_dataset(dataset_slug)
    config = GROQ_LLAMA_8B
    logs: list[str] = []
    previous_attempts: list[AttemptSummary] = []
    cost_tracker = CostTracker()

    analyzer  = AnalyzerAgent(config=config, logs=logs, cost_tracker=cost_tracker)
    critique  = CritiqueAgent(config=config, logs=logs, cost_tracker=cost_tracker)
    validator = PlanValidatorAgent(logs=logs)

    df = pd.read_csv(dataset.csv_path, sep=None, engine="python")
    desc = generate_dataset_description(dataset.csv_path)

    iteration_records: list[IterationRecord] = []
    plan_valid        = False
    execution_success = False
    best_score        = 0.0
    validator_issues  = 0
    solved            = False
    t0                = time.time()

    for iteration in range(1, MAX_ITERATIONS + 1):
        logs.append(f"==== ITERATION {iteration} ====")
        agent_calls: list[AgentCall] = []
        iter_start = time.time()

        quality_agent  = DatasetQualityAgent(logs=logs)
        quality_report = quality_agent.run(df, target_column=dataset.target_column)

        # --- Analyzer ---
        t = time.time()
        try:
            plan = analyzer.run(
                dataset_description=desc,
                problem_description=dataset.problem_statement,
                previous_attempts=previous_attempts,
                quality_report=quality_report,
            )
            plan_valid = True
        except Exception as e:
            logs.append(f"AnalyzerAgent FAILED: {e}")
            break

        agent_calls.append(AgentCall(
            agent_name="AnalyzerAgent",
            input_summary=f"dataset={dataset_slug}, iter={iteration}",
            output_summary=f"models={[m.name for m in plan.models_to_try]}",
            duration_seconds=round(time.time() - t, 3),
            timestamp=datetime.now(UTC),
        ))

        # --- Validator ---
        t = time.time()
        validation = validator.run(plan, quality_report)
        validator_issues += len(validation.issues)

        if not validation.passed:
            logs.append(f"Validator issues: {validation.issues}")
            cost_tracker.record_validator_replan()
            correction = (
                f"{dataset.problem_statement}\n"
                "CORRECTION REQUIRED:\n"
                + "\n".join(f"- {i}" for i in validation.issues)
            )
            try:
                plan = analyzer.run(
                    dataset_description=desc,
                    problem_description=correction,
                    previous_attempts=previous_attempts,
                    quality_report=quality_report,
                )
            except Exception as e:
                logs.append(f"AnalyzerAgent retry FAILED: {e}")
                break
            validation = validator.run(plan, quality_report)

        agent_calls.append(AgentCall(
            agent_name="PlanValidatorAgent",
            input_summary=f"plan models={[m.name for m in plan.models_to_try]}",
            output_summary=f"passed={validation.passed}, issues={validation.issues}",
            duration_seconds=round(time.time() - t, 3),
            timestamp=datetime.now(UTC),
        ))

        # --- Implementation ---
        t = time.time()
        try:
            results, X, y_encoded = execute_plan(plan, dataset.csv_path, logs=logs)
            execution_success = any("error" not in info for info in results.values())
        except Exception as e:
            logs.append(f"execute_plan FAILED: {e}")
            results, X, y_encoded = {}, None, None

        errors = [
            f"{model}: {info['error']}"
            for model, info in results.items()
            if "error" in info
        ]
        cost_tracker.record_execution_errors(len(errors))

        agent_calls.append(AgentCall(
            agent_name="ImplementationAgent",
            input_summary=f"models={[m.name for m in plan.models_to_try]}",
            output_summary=f"errors={errors}",
            duration_seconds=round(time.time() - t, 3),
            timestamp=datetime.now(UTC),
        ))

        # --- Critique ---
        t = time.time()
        evaluation = critique.run(
            plan=plan,
            results=results,
            problem=dataset.problem_statement,
            X=X,
            y=y_encoded,
        )

        agent_calls.append(AgentCall(
            agent_name="CritiqueAgent",
            input_summary=f"results keys={list(results.keys())}",
            output_summary=f"solved={evaluation['solved']}, score={evaluation['best_score']:.4f}",
            duration_seconds=round(time.time() - t, 3),
            timestamp=datetime.now(UTC),
        ))

        best_score = evaluation["best_score"]
        solved     = evaluation["solved"]

        previous_attempts.append(AttemptSummary(
            iteration=iteration,
            models_tried=[m.name for m in plan.models_to_try],
            best_score=best_score,
            metric=plan.primary_metric,
            failure_reason=evaluation["suggestion"] if not solved else None,
            execution_errors=errors,
        ))

        iteration_records.append(IterationRecord(
            iteration_number=iteration,
            plan=plan,
            results=results,
            evaluation=evaluation["suggestion"],
            solved=solved,
            agent_calls=agent_calls,
        ))

        if solved:
            break

    duration = round(time.time() - t0, 3)
    cost_report = cost_tracker.build_report(iterations=len(iteration_records) or 1)

    artifact = RunArtifact(
        dataset_path=dataset.csv_path,
        problem=dataset.problem_statement,
        llm_config=config.to_loggable_dict(),
        random_seed=42,
        iterations=iteration_records,
        solved=solved,
        total_duration_seconds=duration,
    )

    out_dir  = _make_output_dir(dataset_slug, "schema")
    out_path = os.path.join(out_dir, f"run_{run_index}.json")
    _save_artifact(artifact, out_path)

    return {
        "run_id":                artifact.run_id,
        "dataset":               dataset_slug,
        "mode":                  "schema",
        "run_index":             run_index,
        "plan_valid":            int(plan_valid),
        "execution_success":     int(execution_success),
        "best_score":            best_score,
        "n_iterations":          len(iteration_records),
        "validator_issues_caught": validator_issues,
        "solved":                int(solved),
        "duration_seconds":      duration,
        "artifact_path":         out_path,
    }


# ---------------------------------------------------------------------------
# Baseline mode pipeline
# ---------------------------------------------------------------------------

def _run_baseline(
    dataset_slug: str,
    run_index: int,
) -> dict:
    """Run one baseline-mode experiment. Returns a flat metrics dict."""
    from datetime import UTC, datetime

    dataset = get_dataset(dataset_slug)
    config = GROQ_LLAMA_8B
    logs: list[str] = []
    previous_attempts: list[AttemptSummary] = []
    cost_tracker = CostTracker()

    analyzer = BaselineAnalyzerAgent(config=config, logs=logs, cost_tracker=cost_tracker)
    critique = CritiqueAgent(config=config, logs=logs, cost_tracker=cost_tracker)

    df   = pd.read_csv(dataset.csv_path, sep=None, engine="python")
    desc = generate_dataset_description(dataset.csv_path)

    iteration_records: list[IterationRecord] = []
    plan_valid        = False
    execution_success = False
    best_score        = 0.0
    solved            = False
    t0                = time.time()

    for iteration in range(1, MAX_ITERATIONS + 1):
        logs.append(f"==== ITERATION {iteration} ====")
        agent_calls: list[AgentCall] = []

        quality_agent  = DatasetQualityAgent(logs=logs)
        quality_report = quality_agent.run(df, target_column=dataset.target_column)

        # --- Baseline Analyzer ---
        t = time.time()
        baseline_plan: BaselinePlan = analyzer.run(
            dataset_description=desc,
            problem_description=dataset.problem_statement,
            previous_attempts=previous_attempts,
            quality_report=quality_report,
        )

        agent_calls.append(AgentCall(
            agent_name="BaselineAnalyzerAgent",
            input_summary=f"dataset={dataset_slug}, iter={iteration}",
            output_summary=(
                f"parse_success={baseline_plan.parse_success}, "
                f"models={baseline_plan.extracted_models}"
            ),
            duration_seconds=round(time.time() - t, 3),
            timestamp=datetime.now(UTC),
        ))

        if not baseline_plan.parse_success:
            logs.append("BaselineAnalyzerAgent: parse failed — skipping execution.")
            # Still record a stub iteration so the artifact is complete
            iteration_records.append(IterationRecord(
                iteration_number=iteration,
                plan=_stub_plan(dataset.target_column),
                results={},
                evaluation="plan_invalid",
                solved=False,
                agent_calls=agent_calls,
            ))
            break

        plan_valid = True

        # Convert BaselinePlan → AutoMLPlan for ImplementationAgent
        # Invalid model names pass through here — execution failure is the signal
        try:
            automl_plan = _baseline_to_automl_plan(baseline_plan, dataset.target_column)
        except Exception as e:
            logs.append(f"BaselinePlan conversion failed: {e}")
            break

        # --- Implementation (same agent as schema mode — fair comparison) ---
        t = time.time()
        try:
            results, X, y_encoded = execute_plan(automl_plan, dataset.csv_path, logs=logs)
            execution_success = any("error" not in info for info in results.values())
        except Exception as e:
            logs.append(f"execute_plan FAILED: {e}")
            results, X, y_encoded = {}, None, None

        errors = [
            f"{model}: {info['error']}"
            for model, info in results.items()
            if "error" in info
        ]
        cost_tracker.record_execution_errors(len(errors))

        agent_calls.append(AgentCall(
            agent_name="ImplementationAgent",
            input_summary=f"models={baseline_plan.extracted_models}",
            output_summary=f"errors={errors}",
            duration_seconds=round(time.time() - t, 3),
            timestamp=datetime.now(UTC),
        ))

        if not results or all("error" in info for info in results.values()):
            # All models failed — log and stop
            iteration_records.append(IterationRecord(
                iteration_number=iteration,
                plan=automl_plan,
                results=results,
                evaluation="execution_failed",
                solved=False,
                agent_calls=agent_calls,
            ))
            break

        # --- Critique ---
        t = time.time()
        evaluation = critique.run(
            plan=automl_plan,
            results=results,
            problem=dataset.problem_statement,
            X=X,
            y=y_encoded,
        )

        agent_calls.append(AgentCall(
            agent_name="CritiqueAgent",
            input_summary=f"results keys={list(results.keys())}",
            output_summary=f"solved={evaluation['solved']}, score={evaluation['best_score']:.4f}",
            duration_seconds=round(time.time() - t, 3),
            timestamp=datetime.now(UTC),
        ))

        best_score = evaluation["best_score"]
        solved     = evaluation["solved"]

        previous_attempts.append(AttemptSummary(
            iteration=iteration,
            models_tried=baseline_plan.extracted_models,
            best_score=best_score,
            metric=baseline_plan.extracted_metric,
            failure_reason=evaluation["suggestion"] if not solved else None,
            execution_errors=errors,
        ))

        iteration_records.append(IterationRecord(
            iteration_number=iteration,
            plan=automl_plan,
            results=results,
            evaluation=evaluation["suggestion"],
            solved=solved,
            agent_calls=agent_calls,
        ))

        if solved:
            break

    duration = round(time.time() - t0, 3)
    cost_tracker.build_report(iterations=len(iteration_records) or 1)

    artifact = RunArtifact(
        dataset_path=dataset.csv_path,
        problem=dataset.problem_statement,
        llm_config=config.to_loggable_dict(),
        random_seed=42,
        iterations=iteration_records,
        solved=solved,
        total_duration_seconds=duration,
    )

    out_dir  = _make_output_dir(dataset_slug, "baseline")
    out_path = os.path.join(out_dir, f"run_{run_index}.json")
    _save_artifact(artifact, out_path)

    return {
        "run_id":                artifact.run_id,
        "dataset":               dataset_slug,
        "mode":                  "baseline",
        "run_index":             run_index,
        "plan_valid":            int(plan_valid),
        "execution_success":     int(execution_success),
        "best_score":            best_score,
        "n_iterations":          len(iteration_records),
        "validator_issues_caught": 0,  # baseline has no validator
        "solved":                int(solved),
        "duration_seconds":      duration,
        "artifact_path":         out_path,
    }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _stub_plan(target_column: str) -> AutoMLPlan:
    """Minimal valid AutoMLPlan used when baseline parse fails."""
    from src.schemas.plan import ModelToTry
    return AutoMLPlan(
        target_column=target_column,
        primary_metric="accuracy",
        preprocessing_steps=[],
        models_to_try=[ModelToTry(name="LogisticRegression", hyperparameters={})],
        folds=5,
        random_seed=42,
        reasoning="stub — baseline parse failed",
    )


def _baseline_to_automl_plan(plan: BaselinePlan, target_column: str) -> AutoMLPlan:
    """
    Convert a BaselinePlan to AutoMLPlan for ImplementationAgent.

    Bad model names pass through — ImplementationAgent will raise, which
    gets caught and logged as an execution failure. That's intentional (Option A).
    """
    from src.schemas.plan import ModelToTry, PreprocessingStep

    models = [
        ModelToTry(name=name, hyperparameters={})
        for name in plan.extracted_models
    ] or [ModelToTry(name="LogisticRegression", hyperparameters={})]

    steps = []
    for op in plan.extracted_preprocessing:
        try:
            steps.append(PreprocessingStep(operation=op, columns=[]))
        except Exception:
            # Invalid operation name — skip, don't crash
            pass

    # Metric: if not in allowlist, fall back to accuracy
    allowed_metrics = {"accuracy", "f1_macro", "f1_weighted", "roc_auc", "balanced_accuracy"}
    metric = plan.extracted_metric if plan.extracted_metric in allowed_metrics else "accuracy"

    return AutoMLPlan(
        target_column=target_column,
        primary_metric=metric,
        preprocessing_steps=steps,
        models_to_try=models,
        folds=5,
        random_seed=42,
        reasoning="converted from BaselinePlan",
    )


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def run_experiment(
    dataset: str,
    mode: Literal["schema", "baseline"],
    n_runs: int = 5,
) -> list[dict]:
    """
    Run n_runs experiments on a single dataset in the given mode.
    Returns list of per-run metric dicts.
    """
    print(f"\n{'='*60}")
    print(f"Dataset: {dataset} | Mode: {mode} | Runs: {n_runs}")
    print(f"{'='*60}")

    runner = _run_schema if mode == "schema" else _run_baseline
    results = []

    for i in range(1, n_runs + 1):
        print(f"\n  Run {i}/{n_runs}...", end=" ", flush=True)
        try:
            metrics = runner(dataset, run_index=i)
            results.append(metrics)
            print(
                f"plan_valid={metrics['plan_valid']} "
                f"exec_ok={metrics['execution_success']} "
                f"score={metrics['best_score']:.4f} "
                f"iters={metrics['n_iterations']}"
            )
        except Exception as e:
            print(f"FAILED: {e}")
            results.append({
                "dataset": dataset, "mode": mode, "run_index": i,
                "plan_valid": 0, "execution_success": 0,
                "best_score": 0.0, "n_iterations": 0,
                "validator_issues_caught": 0, "solved": 0,
                "duration_seconds": 0.0, "artifact_path": "",
                "error": str(e),
            })

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run AutoML experiment")
    parser.add_argument("--dataset", required=True, help="Dataset slug (e.g. iris, adult)")
    parser.add_argument("--mode", required=True, choices=["schema", "baseline"])
    parser.add_argument("--runs", type=int, default=5)
    args = parser.parse_args()

    metrics_list = run_experiment(args.dataset, args.mode, args.runs)

    print("\nRun summary:")
    for m in metrics_list:
        print(
            f"  run {m['run_index']}: "
            f"plan_valid={m['plan_valid']} "
            f"exec={m['execution_success']} "
            f"score={m['best_score']:.4f}"
        )
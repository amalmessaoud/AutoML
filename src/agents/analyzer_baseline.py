# src/agents/analyzer_baseline.py
"""
Baseline AnalyzerAgent — JSON mode, no Pydantic validation.

This is the CONTROL CONDITION for the paper's core claim.

Mirrors Option 3: the expected structure is communicated via prompt description,
the LLM uses response_format={'type': 'json_object'}, but no programmatic schema
enforcement occurs. Consistent with prior work (AutoML-GPT, DS-Agent).

Key differences from AnalyzerAgent:
  - No Pydantic validation — json.loads() + manual .get() only
  - Wrong key names, missing fields, out-of-allowlist model names all pass through
  - ImplementationAgent will choke on bad model names → logged as execution failure
  - That failure IS the paper data point

Everything else is identical for a fair comparison:
  - Same response_format={'type': 'json_object'}
  - Same LLM, same base_url, same temperature
  - Same prompt intent (dataset/problem/memory/quality inputs)
  - Same _build_client() / _log() / _record_call() from BaseAgent
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field

from src.agents.base_agent import BaseAgent
from src.config.llm_config import LLMConfig
from src.schemas.plan import AttemptSummary
from src.schemas.quality_report import DatasetQualityReport
from src.utils.cost_reporter import CostTracker


# ---------------------------------------------------------------------------
# BaselinePlan — plain dataclass, NOT Pydantic. No validation on purpose.
# ---------------------------------------------------------------------------

@dataclass
class BaselinePlan:
    """
    Best-effort parse of a JSON LLM response with no schema enforcement.

    Fields are extracted with .get() — missing or malformed fields are kept
    as-is and passed downstream. ImplementationAgent failures from bad values
    are logged as execution failures, not caught here.

    parse_success=False only when the response is not valid JSON at all.
    """
    raw_text: str
    extracted_models: list[str] = field(default_factory=list)
    extracted_preprocessing: list[str] = field(default_factory=list)
    extracted_metric: str = "accuracy"
    target_column: str = ""
    parse_success: bool = False


# ---------------------------------------------------------------------------
# Parser — json.loads() + manual .get(), no validation
# ---------------------------------------------------------------------------

def parse_json_response(raw_text: str) -> BaselinePlan:
    """
    Parse a free-structure JSON response into a BaselinePlan.

    parse_success=False only on JSON decode failure.
    Structural issues (wrong keys, bad model names, missing fields) pass through —
    ImplementationAgent handles them downstream.
    """
    try:
        data = json.loads(raw_text)
    except json.JSONDecodeError:
        return BaselinePlan(raw_text=raw_text, parse_success=False)

    # Extract models — whatever the LLM called the key, try common variants
    models_raw = (
        data.get("models_to_try")
        or data.get("models")
        or data.get("classifiers")
        or []
    )
    # models_to_try may be list[dict] (like schema) or list[str]
    extracted_models: list[str] = []
    for m in models_raw:
        if isinstance(m, dict):
            name = m.get("name") or m.get("model") or m.get("classifier") or ""
            if name:
                extracted_models.append(name)
        elif isinstance(m, str):
            extracted_models.append(m)

    # Extract preprocessing — may be list[dict] or list[str]
    preprocessing_raw = (
        data.get("preprocessing_steps")
        or data.get("preprocessing")
        or []
    )
    extracted_preprocessing: list[str] = []
    for p in preprocessing_raw:
        if isinstance(p, dict):
            op = p.get("operation") or p.get("step") or p.get("name") or ""
            if op:
                extracted_preprocessing.append(op)
        elif isinstance(p, str):
            extracted_preprocessing.append(p)

    # Extract metric — no allowlist check, whatever the LLM says
    metric = (
        data.get("primary_metric")
        or data.get("metric")
        or "accuracy"
    )

    # Extract target column
    target = data.get("target_column") or data.get("target") or ""

    return BaselinePlan(
        raw_text=raw_text,
        extracted_models=extracted_models,
        extracted_preprocessing=extracted_preprocessing,
        extracted_metric=str(metric),
        target_column=str(target),
        parse_success=True,  # JSON decoded — structural issues handled downstream
    )


# ---------------------------------------------------------------------------
# BaselineAnalyzerAgent
# ---------------------------------------------------------------------------

class BaselineAnalyzerAgent(BaseAgent):
    """
    Drop-in replacement for AnalyzerAgent using JSON mode without Pydantic.

    Same constructor signature, same run() signature — experiment runner
    swaps between them with a single flag.
    """

    def __init__(
        self,
        config: LLMConfig,
        logs: list[str],
        cost_tracker: CostTracker | None = None,
    ) -> None:
        super().__init__(config, logs, cost_tracker)

    def run(
        self,
        dataset_description: str,
        problem_description: str,
        previous_attempts: list[AttemptSummary] | None = None,
        quality_report: DatasetQualityReport | None = None,
    ) -> BaselinePlan:
        self._log("BaselineAnalyzerAgent: starting plan generation.")

        prompt = self._build_prompt(
            dataset_description=dataset_description,
            problem_description=problem_description,
            previous_attempts=previous_attempts or [],
            quality_report=quality_report,
        )

        client = self._build_client()
        self._log(f"BaselineAnalyzerAgent: calling {self.config.provider}/{self.config.model}.")

        stream = client.chat.completions.create(
            model=self.config.model,
            messages=[{"role": "user", "content": prompt}],
            stream=True,
            temperature=self.config.temperature,
            response_format={"type": "json_object"},  # same as schema agent
            # Difference: no Pydantic validation after this
        )

        raw_output = ""
        for chunk in stream:
            if chunk.choices[0].delta.content:
                raw_output += chunk.choices[0].delta.content

        self._record_call("BaselineAnalyzerAgent", prompt, raw_output)

        plan = parse_json_response(raw_output)

        if plan.parse_success:
            self._log(
                f"BaselineAnalyzerAgent: JSON parsed — "
                f"models={plan.extracted_models}, metric={plan.extracted_metric}."
            )
        else:
            self._log(
                "BaselineAnalyzerAgent: JSON decode FAILED. Logging plan_valid=False."
            )

        return plan

    def _build_prompt(
        self,
        dataset_description: str,
        problem_description: str,
        previous_attempts: list[AttemptSummary],
        quality_report: DatasetQualityReport | None,
    ) -> str:
        # Memory block — identical logic to AnalyzerAgent
        memory_block = ""
        if previous_attempts:
            lines = ["PREVIOUS ATTEMPTS — LEARN FROM THESE:"]
            for a in previous_attempts:
                lines.append(
                    f"  Iteration {a.iteration}: tried {a.models_tried}, "
                    f"best {a.metric}={a.best_score:.4f}, "
                    f"reason not solved: {a.failure_reason or 'score below threshold'}"
                )
                if a.execution_errors:
                    lines.append(f"  EXECUTION ERRORS: {a.execution_errors}")
            lines.append("Choose DIFFERENT models and/or preprocessing than above.")
            memory_block = "\n".join(lines)

        # Quality block — identical to AnalyzerAgent
        quality_block = ""
        if quality_report is not None:
            quality_block = (
                f"DATASET QUALITY REPORT:\n"
                f"  Rows: {quality_report.n_rows}, Cols: {quality_report.n_cols}\n"
                f"  Imbalanced: {quality_report.imbalance_flag}\n"
                f"  Duplicate rate: {quality_report.duplicate_row_rate:.2%}\n"
                f"  Warnings:\n"
                + "\n".join(f"    - {w}" for w in quality_report.warnings)
                + "\nEnsure your plan addresses ALL warnings above."
            )

        # Prompt describes the expected JSON structure in plain text —
        # no response_format schema, no Pydantic — mirrors prior work style
        return f"""You are an expert machine learning engineer specializing in tabular classification.
Your task is to create an AutoML plan as a JSON object.

Dataset description:
{dataset_description}

Problem: {problem_description}

{memory_block}

{quality_block}

Return a JSON object with this structure:
{{
  "target_column": "the column to predict",
  "primary_metric": "one of: accuracy, f1_macro, f1_weighted, roc_auc, balanced_accuracy",
  "preprocessing_steps": [
    {{"operation": "impute_missing", "method": "median", "columns": ["col1"]}},
    {{"operation": "encode_categorical", "method": "one_hot", "columns": ["col2"]}},
    {{"operation": "scale_numeric", "method": "standard", "columns": ["col3"]}}
  ],
  "models_to_try": [
    {{"name": "RandomForestClassifier", "hyperparameters": {{}}}},
    {{"name": "LogisticRegression", "hyperparameters": {{}}}}
  ],
  "reasoning": "brief explanation"
}}

Use between 1 and 3 models. Output only the JSON object.
"""
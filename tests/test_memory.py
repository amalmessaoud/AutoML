# tests/test_memory.py
import json
import numpy as np
import pandas as pd
from unittest.mock import MagicMock, patch

from src.agents.analyzer import AnalyzerAgent
from src.config.llm_config import GROQ_LLAMA_8B
from src.schemas.plan import AttemptSummary


def _mock_analyzer_client(plan_json: str):
    chunk = MagicMock()
    chunk.choices[0].delta.content = plan_json
    stream = [chunk, MagicMock(choices=[MagicMock(delta=MagicMock(content=None))])]
    client = MagicMock()
    client.chat.completions.create.return_value = iter(stream)
    return client


def _make_valid_plan_json(models: list[str]) -> str:
    return json.dumps({
        "task_type": "classification",
        "target_column": "species",
        "primary_metric": "accuracy",
        "preprocessing_steps": [],
        "models_to_try": [{"name": m, "hyperparameters": {}} for m in models],
        "validation_method": "cross_validation",
        "folds": 5,
        "random_seed": 42,
        "reasoning": "Test plan.",
    })


class TestPlanMemory:
    def test_memory_block_injected_into_prompt(self):
        """When previous_attempts is non-empty, the prompt must mention those models."""
        previous = [
            AttemptSummary(
                iteration=1,
                models_tried=["LogisticRegression", "RandomForestClassifier"],
                best_score=0.72,
                metric="accuracy",
                failure_reason="score below threshold",
            )
        ]

        captured_prompt = {}

        def capture_create(**kwargs):
            captured_prompt["messages"] = kwargs["messages"]
            plan_json = _make_valid_plan_json(["XGBClassifier"])
            chunk = MagicMock()
            chunk.choices[0].delta.content = plan_json
            end = MagicMock()
            end.choices[0].delta.content = None
            return iter([chunk, end])

        mock_client = MagicMock()
        mock_client.chat.completions.create.side_effect = capture_create

        with patch.object(GROQ_LLAMA_8B, "build_client", return_value=mock_client):
            agent = AnalyzerAgent(config=GROQ_LLAMA_8B, logs=[])
            agent.run(
                dataset_description="4 numeric cols, 150 rows",
                problem_description="Predict species",
                previous_attempts=previous,
            )

        prompt_text = captured_prompt["messages"][0]["content"]
        assert "LogisticRegression" in prompt_text
        assert "RandomForestClassifier" in prompt_text
        assert "DO NOT REPEAT" in prompt_text or "PREVIOUS ATTEMPTS" in prompt_text

    def test_no_memory_block_on_first_iteration(self):
        """First iteration — prompt must NOT contain memory block."""
        captured_prompt = {}

        def capture_create(**kwargs):
            captured_prompt["messages"] = kwargs["messages"]
            plan_json = _make_valid_plan_json(["KNeighborsClassifier"])
            chunk = MagicMock()
            chunk.choices[0].delta.content = plan_json
            end = MagicMock()
            end.choices[0].delta.content = None
            return iter([chunk, end])

        mock_client = MagicMock()
        mock_client.chat.completions.create.side_effect = capture_create

        with patch.object(GROQ_LLAMA_8B, "build_client", return_value=mock_client):
            agent = AnalyzerAgent(config=GROQ_LLAMA_8B, logs=[])
            agent.run(
                dataset_description="4 numeric cols, 150 rows",
                problem_description="Predict species",
                previous_attempts=[],
            )

        prompt_text = captured_prompt["messages"][0]["content"]
        assert "PREVIOUS ATTEMPTS" not in prompt_text

    def test_attempt_summary_captures_correct_fields(self):
        summary = AttemptSummary(
            iteration=2,
            models_tried=["SVC", "XGBClassifier"],
            best_score=0.81,
            metric="f1_macro",
            failure_reason="score below threshold",
        )
        assert summary.iteration == 2
        assert "SVC" in summary.models_tried
        assert summary.best_score == 0.81
        assert summary.failure_reason is not None

    def test_attempt_summary_failure_reason_optional(self):
        summary = AttemptSummary(
            iteration=1,
            models_tried=["RandomForestClassifier"],
            best_score=0.95,
            metric="accuracy",
        )
        assert summary.failure_reason is None
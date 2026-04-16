# tests/test_run_artifact.py
from datetime import datetime, timezone

from src.schemas.plan import AutoMLPlan, ModelToTry, PreprocessingStep
from src.schemas.run_artifact import AgentCall, IterationRecord, RunArtifact


def _make_fake_plan() -> AutoMLPlan:
    return AutoMLPlan(
        task_type="classification",
        target_column="species",
        primary_metric="accuracy",
        preprocessing_steps=[
            PreprocessingStep(
                operation="scale_numeric",
                method="standard",
                columns=["sepal_length", "sepal_width"],
            )
        ],
        models_to_try=[
            ModelToTry(name="KNeighborsClassifier", hyperparameters={"n_neighbors": 5})
        ],
        validation_method="cross_validation",
        folds=5,
        random_seed=42,
        reasoning="Test plan for unit test.",
    )


def _make_fake_agent_call(name: str) -> AgentCall:
    return AgentCall(
        agent_name=name,
        input_summary="test input",
        output_summary="test output",
        duration_seconds=1.23,
        timestamp=datetime(2026, 1, 1, 12, 0, 0, tzinfo=timezone.utc),
    )


def _make_fake_run_artifact() -> RunArtifact:
    plan = _make_fake_plan()

    iteration = IterationRecord(
        iteration_number=1,
        plan=plan,
        results={
            "KNeighborsClassifier": {
                "mean_score": 0.973,
                "std_score": 0.025,
                "individual_scores": [1.0, 0.96, 0.93, 1.0, 0.96],
            }
        },
        evaluation="Good performance on balanced dataset.",
        solved=True,
        agent_calls=[
            _make_fake_agent_call("AnalyzerAgent"),
            _make_fake_agent_call("ImplementationAgent"),
            _make_fake_agent_call("CritiqueAgent"),
        ],
    )

    return RunArtifact(
        dataset_path="data/iris.csv",
        problem="Predict iris species.",
        llm_config={
            "provider": "groq",
            "model": "llama-3.1-8b-instant",
            "base_url": "https://api.groq.com/openai/v1",
            "temperature": 0.3,
            "api_key_env_var": "GROQ_API_KEY",
        },
        random_seed=42,
        iterations=[iteration],
        solved=True,
        total_duration_seconds=12.5,
    )


class TestRunArtifactSchema:
    def test_construct_full_artifact(self):
        artifact = _make_fake_run_artifact()
        assert artifact.schema_version == "1.0"
        assert len(artifact.iterations) == 1
        assert len(artifact.iterations[0].agent_calls) == 3

    def test_run_id_is_auto_generated(self):
        a1 = _make_fake_run_artifact()
        a2 = _make_fake_run_artifact()
        assert a1.run_id != a2.run_id  # uuid4 — must differ

    def test_schema_version_hardcoded(self):
        artifact = _make_fake_run_artifact()
        assert artifact.schema_version == "1.0"

    def test_no_api_key_in_llm_config(self):
        artifact = _make_fake_run_artifact()
        for value in artifact.llm_config.values():
            assert not str(value).startswith("gsk_"), "Real API key found in RunArtifact!"
            assert not str(value).startswith("AIza"), "Real API key found in RunArtifact!"

    def test_json_round_trip(self):
        """Serialize to JSON, deserialize back — must be structurally identical."""
        original = _make_fake_run_artifact()
        json_str = original.model_dump_json()
        restored = RunArtifact.model_validate_json(json_str)

        assert restored.run_id == original.run_id
        assert restored.dataset_path == original.dataset_path
        assert restored.solved == original.solved
        assert restored.schema_version == original.schema_version
        assert restored.iterations[0].iteration_number == 1
        assert restored.iterations[0].solved == True
        assert (
            restored.iterations[0].results["KNeighborsClassifier"]["mean_score"]
            == original.iterations[0].results["KNeighborsClassifier"]["mean_score"]
        )

    def test_round_trip_byte_identical(self):
        """JSON serialized twice from the same object must be identical."""
        original = _make_fake_run_artifact()
        json1 = original.model_dump_json()
        restored = RunArtifact.model_validate_json(json1)
        json2 = restored.model_dump_json()
        assert json1 == json2
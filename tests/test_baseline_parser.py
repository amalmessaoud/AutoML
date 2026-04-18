# tests/test_baseline_parser.py
"""
Tests for the baseline JSON parser.

Samples represent the realistic range of LLM outputs when given a JSON structure
description in the prompt but no programmatic schema enforcement.

Failure modes tested:
  - Valid JSON, correct structure (happy path)
  - Valid JSON, wrong key names (LLM ignored the prompt's field names)
  - Valid JSON, models as list[str] instead of list[dict]
  - Valid JSON, out-of-allowlist model name (passes here, fails in ImplementationAgent)
  - Valid JSON, missing models_to_try field entirely
  - Valid JSON, metric not in allowlist (passes here, caught nowhere — paper data point)
  - Valid JSON, empty models list
  - Valid JSON, extra unexpected fields (should be ignored cleanly)
  - Invalid JSON (decode failure → parse_success=False)
  - Valid JSON, minimal response matching prompt exactly

Run with: pytest tests/test_baseline_parser.py -v
"""

import json
import pytest

from src.agents.analyzer_baseline import BaselinePlan, parse_json_response


# ---------------------------------------------------------------------------
# 10 handwritten sample LLM responses
# ---------------------------------------------------------------------------

# 1 — Perfect structure, matches prompt exactly. Happy path.
SAMPLE_1 = json.dumps({
    "target_column": "income",
    "primary_metric": "f1_macro",
    "preprocessing_steps": [
        {"operation": "impute_missing", "method": "median", "columns": ["age", "hours_per_week"]},
        {"operation": "encode_categorical", "method": "one_hot", "columns": ["workclass", "occupation"]},
        {"operation": "scale_numeric", "method": "standard", "columns": ["age", "hours_per_week"]},
        {"operation": "handle_imbalance", "method": "oversample", "columns": []},
    ],
    "models_to_try": [
        {"name": "RandomForestClassifier", "hyperparameters": {}},
        {"name": "LogisticRegression", "hyperparameters": {"C": 1.0}},
        {"name": "XGBClassifier", "hyperparameters": {}},
    ],
    "reasoning": "Dataset is imbalanced, f1_macro appropriate.",
})

# 2 — LLM used wrong key names ("models" instead of "models_to_try", "metric" instead of "primary_metric")
SAMPLE_2 = json.dumps({
    "target_column": "class",
    "metric": "accuracy",
    "preprocessing": [
        {"operation": "scale_numeric", "method": "standard", "columns": ["f1", "f2"]},
    ],
    "models": [
        {"name": "RandomForestClassifier", "hyperparameters": {}},
        {"name": "SVC", "hyperparameters": {"kernel": "rbf"}},
    ],
    "reasoning": "Balanced dataset, accuracy fine.",
})

# 3 — Models as list[str] instead of list[dict] (LLM deviated from prompt structure)
SAMPLE_3 = json.dumps({
    "target_column": "species",
    "primary_metric": "accuracy",
    "preprocessing_steps": [],
    "models_to_try": ["LogisticRegression", "KNeighborsClassifier"],
    "reasoning": "Small clean dataset.",
})

# 4 — Out-of-allowlist model name. parse_success=True here, execution failure downstream.
SAMPLE_4 = json.dumps({
    "target_column": "target",
    "primary_metric": "roc_auc",
    "preprocessing_steps": [
        {"operation": "impute_missing", "method": "mean", "columns": ["col1"]},
    ],
    "models_to_try": [
        {"name": "XGBoost", "hyperparameters": {}},         # wrong — should be XGBClassifier
        {"name": "LightGBM", "hyperparameters": {}},        # wrong — should be LGBMClassifier
    ],
    "reasoning": "Boosting methods are strong on tabular data.",
})

# 5 — Missing models_to_try field entirely. parse_success=True, extracted_models=[].
SAMPLE_5 = json.dumps({
    "target_column": "class",
    "primary_metric": "f1_weighted",
    "preprocessing_steps": [
        {"operation": "encode_categorical", "method": "ordinal", "columns": ["cat_col"]},
    ],
    "reasoning": "I forgot to include the models field.",
})

# 6 — Metric not in allowlist ("f1" instead of "f1_macro"). Passes through unchecked.
SAMPLE_6 = json.dumps({
    "target_column": "income",
    "primary_metric": "f1",     # not in allowlist — schema agent would reject this
    "preprocessing_steps": [],
    "models_to_try": [
        {"name": "RandomForestClassifier", "hyperparameters": {}},
    ],
    "reasoning": "Short on details.",
})

# 7 — Empty models list. parse_success=True, extracted_models=[].
SAMPLE_7 = json.dumps({
    "target_column": "label",
    "primary_metric": "accuracy",
    "preprocessing_steps": [
        {"operation": "scale_numeric", "method": "standard", "columns": ["x1", "x2"]},
    ],
    "models_to_try": [],
    "reasoning": "Could not decide on models.",
})

# 8 — Extra unexpected fields (LLM added its own). Should be ignored cleanly.
SAMPLE_8 = json.dumps({
    "target_column": "species",
    "primary_metric": "accuracy",
    "confidence": "high",           # unexpected field
    "author": "llama-3.1",          # unexpected field
    "preprocessing_steps": [],
    "models_to_try": [
        {"name": "KNeighborsClassifier", "hyperparameters": {"n_neighbors": 3}},
    ],
    "notes": "Added extra fields the schema doesn't expect.",  # unexpected
    "reasoning": "Simple dataset.",
})

# 9 — Invalid JSON entirely. parse_success=False.
SAMPLE_9 = """{
    "target_column": "income",
    "models_to_try": [
        {"name": "RandomForestClassifier"
    ]
    -- forgot to close properly
"""

# 10 — Minimal valid response, only required fields, no preprocessing.
SAMPLE_10 = json.dumps({
    "target_column": "class",
    "primary_metric": "balanced_accuracy",
    "preprocessing_steps": [],
    "models_to_try": [
        {"name": "LGBMClassifier", "hyperparameters": {}},
    ],
    "reasoning": "",
})

ALL_SAMPLES = [
    SAMPLE_1, SAMPLE_2, SAMPLE_3, SAMPLE_4, SAMPLE_5,
    SAMPLE_6, SAMPLE_7, SAMPLE_8, SAMPLE_9, SAMPLE_10,
]

EXPECTED_SUCCESS = [
    True,   # 1 — perfect structure
    True,   # 2 — wrong key names, fallback keys handle it
    True,   # 3 — models as list[str]
    True,   # 4 — bad model names, passes through
    True,   # 5 — missing models field, extracted_models=[]
    True,   # 6 — bad metric, passes through
    True,   # 7 — empty models list
    True,   # 8 — extra fields ignored
    False,  # 9 — invalid JSON
    True,   # 10 — minimal valid
]


# ---------------------------------------------------------------------------
# Bulk success-rate test
# ---------------------------------------------------------------------------

def test_parse_success_rate():
    results = [parse_json_response(s).parse_success for s in ALL_SAMPLES]
    for i, (actual, expected) in enumerate(zip(results, EXPECTED_SUCCESS), start=1):
        assert actual == expected, (
            f"Sample {i}: expected parse_success={expected}, got {actual}"
        )
    rate = sum(results) / len(results)
    print(f"\nBaseline parse success rate: {rate:.0%} ({sum(results)}/{len(results)})")


# ---------------------------------------------------------------------------
# Sample 1 — perfect structure
# ---------------------------------------------------------------------------

class TestSample1:
    def test_parse_success(self):
        assert parse_json_response(SAMPLE_1).parse_success is True

    def test_extracts_three_models(self):
        plan = parse_json_response(SAMPLE_1)
        assert "RandomForestClassifier" in plan.extracted_models
        assert "LogisticRegression" in plan.extracted_models
        assert "XGBClassifier" in plan.extracted_models

    def test_extracts_preprocessing(self):
        plan = parse_json_response(SAMPLE_1)
        assert "impute_missing"     in plan.extracted_preprocessing
        assert "encode_categorical" in plan.extracted_preprocessing
        assert "scale_numeric"      in plan.extracted_preprocessing
        assert "handle_imbalance"   in plan.extracted_preprocessing

    def test_metric(self):
        assert parse_json_response(SAMPLE_1).extracted_metric == "f1_macro"

    def test_target_column(self):
        assert parse_json_response(SAMPLE_1).target_column == "income"

    def test_raw_text_preserved(self):
        assert parse_json_response(SAMPLE_1).raw_text == SAMPLE_1


# ---------------------------------------------------------------------------
# Sample 2 — wrong key names
# ---------------------------------------------------------------------------

class TestSample2:
    def test_parse_success(self):
        assert parse_json_response(SAMPLE_2).parse_success is True

    def test_models_extracted_via_fallback_key(self):
        plan = parse_json_response(SAMPLE_2)
        assert "RandomForestClassifier" in plan.extracted_models
        assert "SVC" in plan.extracted_models

    def test_metric_extracted_via_fallback_key(self):
        assert parse_json_response(SAMPLE_2).extracted_metric == "accuracy"

    def test_preprocessing_extracted_via_fallback_key(self):
        plan = parse_json_response(SAMPLE_2)
        assert "scale_numeric" in plan.extracted_preprocessing


# ---------------------------------------------------------------------------
# Sample 3 — models as list[str]
# ---------------------------------------------------------------------------

class TestSample3:
    def test_parse_success(self):
        assert parse_json_response(SAMPLE_3).parse_success is True

    def test_models_extracted_from_string_list(self):
        plan = parse_json_response(SAMPLE_3)
        assert "LogisticRegression" in plan.extracted_models
        assert "KNeighborsClassifier" in plan.extracted_models

    def test_empty_preprocessing(self):
        assert parse_json_response(SAMPLE_3).extracted_preprocessing == []


# ---------------------------------------------------------------------------
# Sample 4 — out-of-allowlist model names pass through
# ---------------------------------------------------------------------------

class TestSample4:
    def test_parse_success(self):
        # Parser doesn't validate allowlist — that's the point
        assert parse_json_response(SAMPLE_4).parse_success is True

    def test_bad_model_names_pass_through(self):
        plan = parse_json_response(SAMPLE_4)
        assert "XGBoost" in plan.extracted_models       # not XGBClassifier
        assert "LightGBM" in plan.extracted_models      # not LGBMClassifier

    def test_no_name_correction(self):
        # Confirm we do NOT silently correct to canonical names
        plan = parse_json_response(SAMPLE_4)
        assert "XGBClassifier" not in plan.extracted_models
        assert "LGBMClassifier" not in plan.extracted_models


# ---------------------------------------------------------------------------
# Sample 5 — missing models field
# ---------------------------------------------------------------------------

class TestSample5:
    def test_parse_success(self):
        assert parse_json_response(SAMPLE_5).parse_success is True

    def test_no_models_extracted(self):
        assert parse_json_response(SAMPLE_5).extracted_models == []

    def test_preprocessing_still_extracted(self):
        plan = parse_json_response(SAMPLE_5)
        assert "encode_categorical" in plan.extracted_preprocessing


# ---------------------------------------------------------------------------
# Sample 6 — metric not in allowlist
# ---------------------------------------------------------------------------

class TestSample6:
    def test_parse_success(self):
        assert parse_json_response(SAMPLE_6).parse_success is True

    def test_bad_metric_passes_through(self):
        # "f1" is not in the schema allowlist but parser doesn't check
        assert parse_json_response(SAMPLE_6).extracted_metric == "f1"

    def test_model_extracted(self):
        assert "RandomForestClassifier" in parse_json_response(SAMPLE_6).extracted_models


# ---------------------------------------------------------------------------
# Sample 7 — empty models list
# ---------------------------------------------------------------------------

class TestSample7:
    def test_parse_success(self):
        assert parse_json_response(SAMPLE_7).parse_success is True

    def test_empty_models(self):
        assert parse_json_response(SAMPLE_7).extracted_models == []

    def test_preprocessing_extracted(self):
        assert "scale_numeric" in parse_json_response(SAMPLE_7).extracted_preprocessing


# ---------------------------------------------------------------------------
# Sample 8 — extra unexpected fields ignored
# ---------------------------------------------------------------------------

class TestSample8:
    def test_parse_success(self):
        assert parse_json_response(SAMPLE_8).parse_success is True

    def test_model_extracted(self):
        assert "KNeighborsClassifier" in parse_json_response(SAMPLE_8).extracted_models

    def test_extra_fields_do_not_crash(self):
        plan = parse_json_response(SAMPLE_8)
        assert not hasattr(plan, "confidence")
        assert not hasattr(plan, "author")


# ---------------------------------------------------------------------------
# Sample 9 — invalid JSON
# ---------------------------------------------------------------------------

class TestSample9:
    def test_parse_fails(self):
        assert parse_json_response(SAMPLE_9).parse_success is False

    def test_no_models(self):
        assert parse_json_response(SAMPLE_9).extracted_models == []

    def test_raw_text_preserved(self):
        assert parse_json_response(SAMPLE_9).raw_text == SAMPLE_9


# ---------------------------------------------------------------------------
# Sample 10 — minimal valid
# ---------------------------------------------------------------------------

class TestSample10:
    def test_parse_success(self):
        assert parse_json_response(SAMPLE_10).parse_success is True

    def test_model_extracted(self):
        assert "LGBMClassifier" in parse_json_response(SAMPLE_10).extracted_models

    def test_metric(self):
        assert parse_json_response(SAMPLE_10).extracted_metric == "balanced_accuracy"

    def test_empty_preprocessing(self):
        assert parse_json_response(SAMPLE_10).extracted_preprocessing == []
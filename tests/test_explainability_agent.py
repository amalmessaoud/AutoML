# tests/test_explainability_agent.py
import numpy as np
import pytest
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from src.agents.explainability_agent import ExplainabilityAgent
from src.config.llm_config import LLMConfig
from src.schemas.explainability import ExplainabilityReport


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope='module')
def iris_data():
    data = load_iris()
    X = data.data
    y = data.target
    feature_names = list(data.feature_names)
    return X, y, feature_names


@pytest.fixture(scope='module')
def rf_model(iris_data):
    X, y, _ = iris_data
    model = RandomForestClassifier(n_estimators=50, random_state=42)
    model.fit(X, y)
    return model


@pytest.fixture(scope='module')
def lr_model(iris_data):
    X, y, _ = iris_data
    model = LogisticRegression(max_iter=200, random_state=42)
    model.fit(X, y)
    return model


@pytest.fixture
def agent(tmp_path):
    """Agent with LLM summary disabled — offline, deterministic."""
    config = LLMConfig(
        provider='groq',
        model='llama-3.3-70b-versatile',
        base_url='https://api.groq.com/openai/v1',
        temperature=0.3,
        api_key_env_var='GROQ_API_KEY',
    )
    logs = []
    return ExplainabilityAgent(config=config, logs=logs, generate_summary=False)


# ---------------------------------------------------------------------------
# Tree model tests (RandomForest — uses feature_importances_)
# ---------------------------------------------------------------------------

class TestTreeModel:
    def test_returns_explainability_report(self, agent, rf_model, iris_data):
        X, y, feature_names = iris_data
        report = agent.run(rf_model, X, y, feature_names)
        assert isinstance(report, ExplainabilityReport)

    def test_method_is_feature_importances(self, agent, rf_model, iris_data):
        X, y, feature_names = iris_data
        report = agent.run(rf_model, X, y, feature_names)
        assert report.method == 'feature_importances'

    def test_model_name_captured(self, agent, rf_model, iris_data):
        X, y, feature_names = iris_data
        report = agent.run(rf_model, X, y, feature_names)
        assert report.model_name == 'RandomForestClassifier'

    def test_top_3_features_returned(self, agent, rf_model, iris_data):
        """Iris has 4 features — RF should return all 4 (≤5), at least 3."""
        X, y, feature_names = iris_data
        report = agent.run(rf_model, X, y, feature_names)
        assert len(report.top_features) >= 3

    def test_scores_sum_approximately_one(self, agent, rf_model, iris_data):
        """Tree importances are normalised — all 4 iris features sum to 1.0."""
        X, y, feature_names = iris_data
        # We need ALL features, not just top 5 — but iris has 4, so top_features IS all
        report = agent.run(rf_model, X, y, feature_names)
        total = sum(f.importance_score for f in report.top_features)
        assert abs(total - 1.0) < 1e-6, f'Scores sum to {total}, expected ~1.0'

    def test_ranks_are_consecutive_from_one(self, agent, rf_model, iris_data):
        X, y, feature_names = iris_data
        report = agent.run(rf_model, X, y, feature_names)
        ranks = [f.rank for f in report.top_features]
        assert ranks == list(range(1, len(ranks) + 1))

    def test_scores_are_descending(self, agent, rf_model, iris_data):
        X, y, feature_names = iris_data
        report = agent.run(rf_model, X, y, feature_names)
        scores = [f.importance_score for f in report.top_features]
        assert scores == sorted(scores, reverse=True)

    def test_feature_names_are_subset_of_input(self, agent, rf_model, iris_data):
        X, y, feature_names = iris_data
        report = agent.run(rf_model, X, y, feature_names)
        for f in report.top_features:
            assert f.feature_name in feature_names

    def test_at_most_5_features(self, agent, rf_model, iris_data):
        X, y, feature_names = iris_data
        report = agent.run(rf_model, X, y, feature_names)
        assert len(report.top_features) <= 5

    def test_summary_skipped_offline(self, agent, rf_model, iris_data):
        X, y, feature_names = iris_data
        report = agent.run(rf_model, X, y, feature_names)
        assert '[summary skipped]' in report.decision_summary

    def test_logs_populated(self, rf_model, iris_data):
        X, y, feature_names = iris_data
        config = LLMConfig(
            provider='groq',
            model='llama-3.3-70b-versatile',
            base_url='https://api.groq.com/openai/v1',
            temperature=0.3,
            api_key_env_var='GROQ_API_KEY',
        )
        logs = []
        a = ExplainabilityAgent(config=config, logs=logs, generate_summary=False)
        a.run(rf_model, X, y, feature_names)
        assert any('ExplainabilityAgent' in msg for msg in logs)


# ---------------------------------------------------------------------------
# Non-tree model tests (LogisticRegression — uses permutation importance)
# ---------------------------------------------------------------------------

class TestNonTreeModel:
    def test_method_is_permutation(self, agent, lr_model, iris_data):
        X, y, feature_names = iris_data
        report = agent.run(lr_model, X, y, feature_names)
        assert report.method == 'permutation_importance'

    def test_model_name_captured(self, agent, lr_model, iris_data):
        X, y, feature_names = iris_data
        report = agent.run(lr_model, X, y, feature_names)
        assert report.model_name == 'LogisticRegression'

    def test_at_most_5_features(self, agent, lr_model, iris_data):
        X, y, feature_names = iris_data
        report = agent.run(lr_model, X, y, feature_names)
        assert len(report.top_features) <= 5

    def test_scores_non_negative(self, agent, lr_model, iris_data):
        X, y, feature_names = iris_data
        report = agent.run(lr_model, X, y, feature_names)
        for f in report.top_features:
            assert f.importance_score >= 0.0

    def test_ranks_consecutive(self, agent, lr_model, iris_data):
        X, y, feature_names = iris_data
        report = agent.run(lr_model, X, y, feature_names)
        ranks = [f.rank for f in report.top_features]
        assert ranks == list(range(1, len(ranks) + 1))

    def test_scores_descending(self, agent, lr_model, iris_data):
        X, y, feature_names = iris_data
        report = agent.run(lr_model, X, y, feature_names)
        scores = [f.importance_score for f in report.top_features]
        assert scores == sorted(scores, reverse=True)


# ---------------------------------------------------------------------------
# Edge case: dataset with >5 features — cap enforced
# ---------------------------------------------------------------------------

class TestTopNCap:
    def test_capped_at_5_with_many_features(self, agent):
        rng = np.random.default_rng(0)
        X = rng.random((200, 12))
        y = (X[:, 0] + X[:, 1] > 1.0).astype(int)
        feature_names = [f'feat_{i}' for i in range(12)]

        model = RandomForestClassifier(n_estimators=20, random_state=0)
        model.fit(X, y)

        report = agent.run(model, X, y, feature_names)
        assert len(report.top_features) == 5
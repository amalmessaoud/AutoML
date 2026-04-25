# tests/test_explainability_schema.py
import pytest
from pydantic import ValidationError

from src.schemas.explainability import ExplainabilityReport, FeatureImportance


def _make_feature(name: str, score: float, rank: int) -> FeatureImportance:
    return FeatureImportance(feature_name=name, importance_score=score, rank=rank)


def _valid_report(**overrides) -> dict:
    base = {
        'method': 'feature_importances',
        'top_features': [
            _make_feature('petal_length', 0.6, 1),
            _make_feature('petal_width', 0.3, 2),
            _make_feature('sepal_length', 0.1, 3),
        ],
        'decision_summary': 'The model relies primarily on petal dimensions.',
        'model_name': 'RandomForestClassifier',
    }
    base.update(overrides)
    return base


class TestFeatureImportance:
    def test_valid(self):
        f = _make_feature('petal_length', 0.5, 1)
        assert f.feature_name == 'petal_length'
        assert f.importance_score == 0.5
        assert f.rank == 1

    def test_negative_score_rejected(self):
        with pytest.raises(ValidationError):
            FeatureImportance(feature_name='x', importance_score=-0.1, rank=1)

    def test_rank_zero_rejected(self):
        with pytest.raises(ValidationError):
            FeatureImportance(feature_name='x', importance_score=0.5, rank=0)


class TestExplainabilityReport:
    def test_valid_report(self):
        report = ExplainabilityReport(**_valid_report())
        assert report.model_name == 'RandomForestClassifier'
        assert len(report.top_features) == 3
        assert report.top_features[0].rank == 1

    def test_too_many_features_rejected(self):
        features = [_make_feature(f'f{i}', 1.0 / (i + 1), i + 1) for i in range(6)]
        with pytest.raises(ValidationError, match='at most 5'):
            ExplainabilityReport(**_valid_report(top_features=features))

    def test_non_consecutive_ranks_rejected(self):
        features = [
            _make_feature('a', 0.6, 1),
            _make_feature('b', 0.3, 3),  # gap — rank 2 missing
        ]
        with pytest.raises(ValidationError, match='consecutive'):
            ExplainabilityReport(**_valid_report(top_features=features))

    def test_wrong_order_rejected(self):
        # scores not descending
        features = [
            _make_feature('a', 0.2, 1),
            _make_feature('b', 0.8, 2),
        ]
        with pytest.raises(ValidationError, match='descending'):
            ExplainabilityReport(**_valid_report(top_features=features))

    def test_single_feature_valid(self):
        features = [_make_feature('only_feature', 1.0, 1)]
        report = ExplainabilityReport(**_valid_report(top_features=features))
        assert len(report.top_features) == 1

    def test_method_field_stored(self):
        report = ExplainabilityReport(**_valid_report(method='permutation_importance'))
        assert report.method == 'permutation_importance'
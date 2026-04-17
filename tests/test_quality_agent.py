# tests/test_quality_agent.py
import pandas as pd

from src.agents.dataset_quality_agent import DatasetQualityAgent
from src.schemas.quality_report import DatasetQualityReport


def _make_agent() -> DatasetQualityAgent:
    return DatasetQualityAgent(logs=[])


class TestColumnQuality:
    def test_constant_column_detected(self):
        df = pd.DataFrame(
            {
                'feature': [1, 2, 3, 4, 5],
                'constant': [7, 7, 7, 7, 7],
                'target': [0, 1, 0, 1, 0],
            }
        )
        report = _make_agent().run(df, target_column='target')
        constant_col = next(c for c in report.columns if c.name == 'constant')
        assert constant_col.is_constant is True

    def test_non_constant_column(self):
        df = pd.DataFrame(
            {
                'feature': [1, 2, 3, 4, 5],
                'target': [0, 1, 0, 1, 0],
            }
        )
        report = _make_agent().run(df, target_column='target')
        feature_col = next(c for c in report.columns if c.name == 'feature')
        assert feature_col.is_constant is False

    def test_missing_rate_correct(self):
        df = pd.DataFrame(
            {
                'feature': [1.0, None, None, 4.0, None],
                'target': [0, 1, 0, 1, 0],
            }
        )
        report = _make_agent().run(df, target_column='target')
        feat_col = next(c for c in report.columns if c.name == 'feature')
        assert abs(feat_col.missing_rate - 0.6) < 0.01

    def test_zero_missing_rate(self):
        df = pd.DataFrame(
            {
                'feature': [1, 2, 3, 4, 5],
                'target': [0, 1, 0, 1, 0],
            }
        )
        report = _make_agent().run(df, target_column='target')
        feat_col = next(c for c in report.columns if c.name == 'feature')
        assert feat_col.missing_rate == 0.0


class TestImbalanceDetection:
    def test_imbalanced_target_flagged(self):
        df = pd.DataFrame(
            {
                'feature': range(100),
                'target': [0] * 95 + [1] * 5,
            }
        )
        report = _make_agent().run(df, target_column='target')
        assert report.imbalance_flag is True

    def test_balanced_target_not_flagged(self):
        df = pd.DataFrame(
            {
                'feature': range(100),
                'target': [0] * 50 + [1] * 50,
            }
        )
        report = _make_agent().run(df, target_column='target')
        assert report.imbalance_flag is False

    def test_borderline_imbalance(self):
        # Exactly 20% minority — should NOT flag (flag is strictly < 20%)
        df = pd.DataFrame(
            {
                'feature': range(100),
                'target': [0] * 80 + [1] * 20,
            }
        )
        report = _make_agent().run(df, target_column='target')
        assert report.imbalance_flag is False

    def test_imbalance_warning_in_warnings(self):
        df = pd.DataFrame(
            {
                'feature': range(100),
                'target': [0] * 95 + [1] * 5,
            }
        )
        report = _make_agent().run(df, target_column='target')
        assert any('imbalanced' in w.lower() for w in report.warnings)


class TestDuplicates:
    def test_duplicate_rate_correct(self):
        df = pd.DataFrame(
            {
                'feature': [1, 2, 3, 1, 2],
                'target': [0, 1, 0, 0, 1],
            }
        )
        report = _make_agent().run(df, target_column='target')
        assert abs(report.duplicate_row_rate - 0.4) < 0.01

    def test_no_duplicates(self):
        df = pd.DataFrame(
            {
                'feature': [1, 2, 3, 4, 5],
                'target': [0, 1, 0, 1, 0],
            }
        )
        report = _make_agent().run(df, target_column='target')
        assert report.duplicate_row_rate == 0.0


class TestReportStructure:
    def test_n_rows_n_cols_correct(self):
        df = pd.DataFrame(
            {
                'a': [1, 2, 3],
                'b': [4, 5, 6],
                'target': [0, 1, 0],
            }
        )
        report = _make_agent().run(df, target_column='target')
        assert report.n_rows == 3
        assert report.n_cols == 3

    def test_report_serializes_to_json(self):
        df = pd.DataFrame(
            {
                'feature': [1, 2, 3, 4, 5],
                'target': [0, 1, 0, 1, 0],
            }
        )
        report = _make_agent().run(df, target_column='target')
        json_str = report.model_dump_json()
        restored = DatasetQualityReport.model_validate_json(json_str)
        assert restored.n_rows == report.n_rows

# tests/test_cost_reporter.py
from src.utils.cost_reporter import CostTracker, estimate_tokens


class TestEstimateTokens:
    def test_empty_string(self):
        assert estimate_tokens('') == 1  # minimum 1

    def test_known_length(self):
        text = 'a' * 400  # 400 chars / 4 = 100 tokens
        assert estimate_tokens(text) == 100

    def test_longer_text(self):
        text = 'x' * 1000
        assert estimate_tokens(text) == 250


class TestCostTracker:
    def test_records_single_call(self):
        tracker = CostTracker()
        tracker.record_call('AnalyzerAgent', 'prompt ' * 100, 'output ' * 20)
        report = tracker.build_report(iterations=1)
        assert report.total_llm_calls == 1
        assert report.calls_per_agent['AnalyzerAgent'] == 1

    def test_records_multiple_agents(self):
        tracker = CostTracker()
        tracker.record_call('AnalyzerAgent', 'p' * 400, 'o' * 100)
        tracker.record_call('CritiqueAgent', 'p' * 200, 'o' * 80)
        report = tracker.build_report(iterations=1)
        assert report.total_llm_calls == 2
        assert report.calls_per_agent['AnalyzerAgent'] == 1
        assert report.calls_per_agent['CritiqueAgent'] == 1

    def test_token_counts_accumulate(self):
        tracker = CostTracker()
        tracker.record_call('AnalyzerAgent', 'a' * 400, 'b' * 400)  # 100 + 100 = 200
        tracker.record_call('AnalyzerAgent', 'a' * 400, 'b' * 400)  # 100 + 100 = 200
        report = tracker.build_report(iterations=1)
        assert report.total_tokens == 400
        assert report.calls_per_agent['AnalyzerAgent'] == 2

    def test_validator_replans_tracked(self):
        tracker = CostTracker()
        tracker.record_validator_replan()
        tracker.record_validator_replan()
        report = tracker.build_report(iterations=2)
        assert report.validator_triggered_replans == 2

    def test_execution_errors_tracked(self):
        tracker = CostTracker()
        tracker.record_execution_errors(3)
        report = tracker.build_report(iterations=1)
        assert report.execution_errors_total == 3

    def test_summary_is_string(self):
        tracker = CostTracker()
        tracker.record_call('AnalyzerAgent', 'prompt', 'output')
        report = tracker.build_report(iterations=1)
        assert isinstance(report.summary(), str)
        assert 'AnalyzerAgent' in report.summary()

    def test_empty_tracker_builds_report(self):
        tracker = CostTracker()
        report = tracker.build_report(iterations=0)
        assert report.total_llm_calls == 0
        assert report.total_tokens == 0

# tests/test_critique_determinism.py
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd

from src.agents.critique import CritiqueAgent
from src.config.constants import IMPROVEMENT_MARGIN, is_solved
from src.config.llm_config import GROQ_LLAMA_8B
from src.schemas.plan import AutoMLPlan, ModelToTry


def _make_plan(metric: str = 'accuracy') -> AutoMLPlan:
    return AutoMLPlan(
        task_type='classification',
        target_column='species',
        primary_metric=metric,
        preprocessing_steps=[],
        models_to_try=[ModelToTry(name='KNeighborsClassifier', hyperparameters={})],
        validation_method='cross_validation',
        folds=5,
        random_seed=42,
        reasoning='Test.',
    )


def _make_results(score: float) -> dict:
    return {
        'KNeighborsClassifier': {
            'mean_score': score,
            'std_score': 0.01,
            'individual_scores': [score] * 5,
        }
    }


def _make_iris_xy() -> tuple[pd.DataFrame, np.ndarray]:
    """Tiny balanced 3-class dataset — dummy score ~0.33."""
    rng = np.random.default_rng(42)
    X = pd.DataFrame(rng.random((150, 4)), columns=['a', 'b', 'c', 'd'])
    y = np.array([0] * 50 + [1] * 50 + [2] * 50)
    return X, y


def _mock_llm(text: str):
    msg = MagicMock()
    msg.content = text
    choice = MagicMock()
    choice.message = msg
    resp = MagicMock()
    resp.choices = [choice]
    client = MagicMock()
    client.chat.completions.create.return_value = resp
    return client


class TestIsSolvedLogic:
    def test_beats_dummy_and_floor(self):
        assert is_solved(0.95, 'accuracy', dummy_score=0.33) is True

    def test_beats_dummy_but_fails_floor(self):
        # dummy=0.10, score=0.25 — margin ok (+0.15) but below floor 0.60
        assert is_solved(0.25, 'accuracy', dummy_score=0.10) is False

    def test_above_floor_but_not_enough_margin(self):
        # dummy=0.80, score=0.85 — above floor but margin only +0.05 < 0.10
        assert is_solved(0.85, 'accuracy', dummy_score=0.80) is False

    def test_exactly_at_threshold_passes(self):
        # Use a dummy score high enough that margin+dummy clears the absolute floor
        dummy = 0.60
        score = dummy + IMPROVEMENT_MARGIN['accuracy']  # 0.70 — clears both margin and floor
        assert is_solved(score, 'accuracy', dummy_score=dummy) is True


class TestCritiqueDeterminism:
    def test_solved_identical_across_three_runs(self):
        plan = _make_plan('accuracy')
        results = _make_results(0.95)
        X, y = _make_iris_xy()

        responses = [
            '{"reason": "great", "suggestion": ""}',
            '{"reason": "excellent result", "suggestion": ""}',
            '{"reason": "very different wording", "suggestion": "nothing needed"}',
        ]

        solved_values = []
        for text in responses:
            with patch.object(GROQ_LLAMA_8B, 'build_client', return_value=_mock_llm(text)):
                agent = CritiqueAgent(config=GROQ_LLAMA_8B, logs=[])
                result = agent.run(plan=plan, results=results, problem='test', X=X, y=y)
                solved_values.append(result['solved'])

        assert len(set(solved_values)) == 1, f'solved varied: {solved_values}'

    def test_critique_text_may_vary(self):
        plan = _make_plan('accuracy')
        results = _make_results(0.95)
        X, y = _make_iris_xy()

        texts = [
            '{"reason": "first opinion", "suggestion": ""}',
            '{"reason": "completely different opinion", "suggestion": ""}',
        ]
        critiques = []
        for text in texts:
            with patch.object(GROQ_LLAMA_8B, 'build_client', return_value=_mock_llm(text)):
                agent = CritiqueAgent(config=GROQ_LLAMA_8B, logs=[])
                result = agent.run(plan=plan, results=results, problem='test', X=X, y=y)
                critiques.append(result['critique'])

        assert isinstance(critiques[0], str) and isinstance(critiques[1], str)

    def test_dummy_score_returned_in_result(self):
        plan = _make_plan('accuracy')
        results = _make_results(0.95)
        X, y = _make_iris_xy()

        with patch.object(
            GROQ_LLAMA_8B,
            'build_client',
            return_value=_mock_llm('{"reason": "ok", "suggestion": ""}'),
        ):
            agent = CritiqueAgent(config=GROQ_LLAMA_8B, logs=[])
            result = agent.run(plan=plan, results=results, problem='test', X=X, y=y)

        assert 'dummy_score' in result
        assert 0.0 <= result['dummy_score'] <= 1.0

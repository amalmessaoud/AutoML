# src/agents/explainability_agent.py
import json

import numpy as np
from sklearn.inspection import permutation_importance

from src.agents.base_agent import BaseAgent
from src.config.llm_config import LLMConfig
from src.schemas.explainability import ExplainabilityReport, FeatureImportance
from src.utils.cost_reporter import CostTracker

_TOP_N = 5
_TREE_MODELS = {
    'RandomForestClassifier',
    'ExtraTreesClassifier',
    'DecisionTreeClassifier',
    'GradientBoostingClassifier',
    'XGBClassifier',
    'LGBMClassifier',
    'CatBoostClassifier',
}


def _has_feature_importances(model) -> bool:
    return hasattr(model, 'feature_importances_')


def _extract_tree_importances(
    model, feature_names: list[str]
) -> list[tuple[str, float]]:
    """Return (name, score) pairs sorted descending. Scores sum to 1.0."""
    raw: np.ndarray = model.feature_importances_
    pairs = sorted(zip(feature_names, raw.tolist()), key=lambda x: x[1], reverse=True)
    return pairs[:_TOP_N]


def _extract_permutation_importances(
    model, X, y, feature_names: list[str], random_state: int = 42
) -> list[tuple[str, float]]:
    """Return (name, mean_importance) pairs sorted descending."""
    result = permutation_importance(
        model, X, y, n_repeats=10, random_state=random_state, scoring='accuracy'
    )
    means = result.importances_mean
    # Clip negatives to 0 — permutation importance can go slightly negative
    means = np.clip(means, 0, None)
    pairs = sorted(zip(feature_names, means.tolist()), key=lambda x: x[1], reverse=True)
    return pairs[:_TOP_N]


def _build_feature_list(pairs: list[tuple[str, float]]) -> list[FeatureImportance]:
    return [
        FeatureImportance(feature_name=name, importance_score=score, rank=rank)
        for rank, (name, score) in enumerate(pairs, start=1)
    ]


class ExplainabilityAgent(BaseAgent):
    def __init__(
        self,
        config: LLMConfig,
        logs: list[str],
        cost_tracker: 'CostTracker | None' = None,
        generate_summary: bool = True,
    ) -> None:
        super().__init__(config, logs, cost_tracker)
        self.generate_summary = generate_summary

    def run(
        self,
        model,
        X,
        y,
        feature_names: list[str],
        random_state: int = 42,
    ) -> ExplainabilityReport:
        model_name = type(model).__name__
        self._log(f'ExplainabilityAgent: explaining {model_name}.')

        # --- Choose method and extract importances ---
        if _has_feature_importances(model):
            method = 'feature_importances'
            pairs = _extract_tree_importances(model, feature_names)
            self._log('ExplainabilityAgent: using native feature_importances_.')
        else:
            method = 'permutation_importance'
            pairs = _extract_permutation_importances(model, X, y, feature_names, random_state)
            self._log('ExplainabilityAgent: using permutation importance.')

        top_features = _build_feature_list(pairs)

        # --- Decision summary ---
        if self.generate_summary:
            decision_summary = self._generate_summary(model_name, top_features)
        else:
            # Offline / test fallback — deterministic, no LLM call
            names = ', '.join(f.feature_name for f in top_features)
            decision_summary = f'[summary skipped] Top features: {names}.'

        report = ExplainabilityReport(
            method=method,
            top_features=top_features,
            decision_summary=decision_summary,
            model_name=model_name,
        )
        self._log('ExplainabilityAgent: report built successfully.')
        return report

    def _generate_summary(
        self, model_name: str, top_features: list[FeatureImportance]
    ) -> str:
        feature_lines = '\n'.join(
            f'  {f.rank}. {f.feature_name} (score: {f.importance_score:.4f})'
            for f in top_features
        )
        prompt = f"""You are a machine learning interpretability expert.
A {model_name} was trained on a tabular classification task.
Its top predictive features are:
{feature_lines}

Write a single concise paragraph (2-3 sentences) that explains what these features suggest
about how the model makes decisions. Start with "Given these top features, the model appears
to decide based on...". Be specific about the feature names. Do not mention scores."""

        client = self._build_client()
        self._log('ExplainabilityAgent: calling LLM for decision_summary.')

        stream = client.chat.completions.create(
            model=self.config.model,
            messages=[{'role': 'user', 'content': prompt}],
            stream=True,
            temperature=self.config.temperature,
        )

        summary = ''
        for chunk in stream:
            if chunk.choices[0].delta.content:
                summary += chunk.choices[0].delta.content

        self._record_call('ExplainabilityAgent', prompt, summary)
        return summary.strip()
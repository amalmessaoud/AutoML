# src/schemas/explainability.py
from pydantic import BaseModel, Field, model_validator


class FeatureImportance(BaseModel):
    feature_name: str = Field(..., description='Name of the feature')
    importance_score: float = Field(..., ge=0.0, description='Importance score (non-negative)')
    rank: int = Field(..., ge=1, description='Rank starting at 1 (1 = most important)')


class ExplainabilityReport(BaseModel):
    method: str = Field(
        ..., description='Method used: "feature_importances" or "permutation_importance"'
    )
    top_features: list[FeatureImportance] = Field(
        ..., description='Top features, max 5, ordered by rank'
    )
    decision_summary: str = Field(
        ..., description='LLM-generated natural language summary of model decision logic'
    )
    model_name: str = Field(..., description='Class name of the fitted model')

    @model_validator(mode='after')
    def validate_top_features(self):
        if len(self.top_features) > 5:
            raise ValueError('top_features must contain at most 5 entries')

        ranks = [f.rank for f in self.top_features]
        if sorted(ranks) != list(range(1, len(ranks) + 1)):
            raise ValueError('Ranks must be consecutive integers starting at 1')

        scores = [f.importance_score for f in self.top_features]
        if scores != sorted(scores, reverse=True):
            raise ValueError('top_features must be ordered by descending importance_score')

        return self
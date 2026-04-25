from typing import Literal

from pydantic import BaseModel, Field, model_validator


class PreprocessingStep(BaseModel):
    operation: str = Field(..., description='Operation name (will be normalized)')
    method: str | None = Field(
        default=None,
        description='Method for the operation (optional for some ops)',
    )

    columns: list[str] = Field(default_factory=list, description='Columns to apply to')


    @model_validator(mode='after')
    def normalize_and_validate_operation(self):
        op = self.operation.lower().strip()
        mapping = {
            'one_hot_encode': 'encode_categorical',
            'one_hot_encoding': 'encode_categorical',
            'onehot': 'encode_categorical',
            'ohe': 'encode_categorical',
            'impute': 'impute_missing',
            'imputation': 'impute_missing',
            'impute_missing_values': 'impute_missing',
            'scale': 'scale_numeric',
            'standardize': 'scale_numeric',
            'normalize': 'scale_numeric',
        }
        normalized = mapping.get(op, op)

        allowed = {
            'impute_missing',
            'encode_categorical',
            'scale_numeric',
            'drop_columns',
            'handle_imbalance',
        }
        if normalized not in allowed:
            raise ValueError(f'Invalid operation: {self.operation} (normalized to {normalized})')

        self.operation = normalized

        # Default method for encoding
        if self.operation == 'encode_categorical' and not self.method:
            self.method = 'one_hot'

        return self


class ModelToTry(BaseModel):
    name: str = Field(..., description='Name of the sklearn classifier')
    hyperparameters: dict[str, object] = Field(
        default_factory=dict, description='Initial hyperparameters'
    )


class AutoMLPlan(BaseModel):
    task_type: Literal['classification'] = Field('classification')
    target_column: str = Field(...)
    primary_metric: Literal[
        'accuracy', 'f1_macro', 'f1_weighted', 'roc_auc', 'balanced_accuracy'
    ] = Field(...)
    preprocessing_steps: list[PreprocessingStep] = Field(default_factory=list)
    models_to_try: list[ModelToTry] = Field(...)
    validation_method: Literal['cross_validation'] = Field('cross_validation')
    folds: int = Field(5, ge=1)
    random_seed: int = Field(42)
    reasoning: str = Field('')

    @model_validator(mode='after')
    def check_model_count(self):
        if not 1 <= len(self.models_to_try) <= 3:
            raise ValueError('Must suggest between 1 and 3 models')
        return self


class AttemptSummary(BaseModel):
    iteration: int
    models_tried: list[str]
    best_score: float
    metric: str
    failure_reason: str | None = None
    execution_errors: list[str] = []

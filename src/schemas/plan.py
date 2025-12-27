from typing import Literal

from pydantic import BaseModel, Field, validator


class PreprocessingStep(BaseModel):
    operation: Literal[
        'impute_missing', 'encode_categorical', 'scale_numeric', 'drop_columns', 'handle_imbalance'
    ] = Field(..., description='Standardized operation name')

    method: str = Field(..., description="Specific method, e.g. 'mean', 'one_hot', 'standard'")

    columns: list[str] = Field(..., description='Columns to apply to')

    @validator('operation', pre=True)
    def normalize_operation(cls, v):
        mapping = {
            'one_hot_encode': 'encode_categorical',
            'one_hot_encoding': 'encode_categorical',
            'impute': 'impute_missing',
            'imputation': 'impute_missing',
            'scale': 'scale_numeric',
            'standardize': 'scale_numeric',
        }
        return mapping.get(v.lower(), v)


class ModelToTry(BaseModel):
    name: str = Field(..., description='Name of the model to try')
    hyperparameters: dict[str, object] = Field(
        default_factory=dict,
        description='Initial hyperparameters; empty dict means use sklearn defaults',
    )


class AutoMLPlan(BaseModel):
    task_type: Literal['classification'] = Field(..., description='Type of machine learning task')
    target_column: str = Field(..., description='Name of the target column')
    primary_metric: Literal[
        'accuracy', 'f1_macro', 'f1_weighted', 'roc_auc', 'balanced_accuracy'
    ] = Field(..., description='Primary metric for model evaluation')
    preprocessing_steps: list[PreprocessingStep] = Field(
        default_factory=list, description='List of preprocessing steps (can be empty)'
    )
    models_to_try: list[ModelToTry] = Field(
        ..., description='List of models to try during training'
    )
    validation_method: Literal['cross_validation'] = Field(
        'cross_validation', description='Validation strategy - fixed to cross-validation'
    )
    folds: int = Field(5, ge=1, description='Number of folds for cross-validation')
    random_seed: int = Field(42, description='Fixed seed for reproducibility')

    reasoning: str = Field(
        '', description='Optional textual reasoning from the analyzer about plan choices'
    )

    # Enforce 1 to 3 models
    @validator('models_to_try')
    def check_model_count(cls, v):
        if not 1 <= len(v) <= 3:
            raise ValueError('Must suggest between 1 and 3 models')
        return v

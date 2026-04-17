# src/agents/implementation.py
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from lightgbm import LGBMClassifier
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    make_scorer,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    LabelEncoder,
    MinMaxScaler,
    OneHotEncoder,
    OrdinalEncoder,
    StandardScaler,
)
from sklearn.svm import SVC
from xgboost import XGBClassifier

from src.schemas.plan import AutoMLPlan, PreprocessingStep

MODEL_MAP = {
    'LogisticRegression': lambda **kw: LogisticRegression(max_iter=1000, **kw),
    'RandomForestClassifier': RandomForestClassifier,
    'XGBClassifier': XGBClassifier,
    'LGBMClassifier': LGBMClassifier,
    'CatBoostClassifier': CatBoostClassifier,
    'SVC': SVC,
    'KNeighborsClassifier': KNeighborsClassifier,
}

METRIC_MAP = {
    'accuracy': make_scorer(accuracy_score),
    'f1_macro': make_scorer(f1_score, average='macro'),
    'f1_weighted': make_scorer(f1_score, average='weighted'),
    'roc_auc': make_scorer(roc_auc_score, multi_class='ovr'),
    'balanced_accuracy': make_scorer(balanced_accuracy_score),
}


def _strip_target_from_steps(
    steps: list[PreprocessingStep],
    target_column: str,
    logs: list[str] | None,
) -> list[PreprocessingStep]:
    """
    Safety net: remove target column from any preprocessing step columns.
    This should have been caught by Rule 6 of the validator, but LLM is stochastic.
    """
    clean = []
    for step in steps:
        clean_cols = [c for c in step.columns if c != target_column]
        if len(clean_cols) != len(step.columns):
            if logs:
                logs.append(
                    f'ImplementationAgent: WARNING — removed target column '
                    f"'{target_column}' from step '{step.operation}'. "
                    f'Plan error not caught by validator.'
                )
        if clean_cols:
            clean.append(
                PreprocessingStep(
                    operation=step.operation,
                    method=step.method,
                    columns=clean_cols,
                )
            )
    return clean


def execute_plan(
    plan: AutoMLPlan,
    csv_path: str,
    logs: list[str] | None = None,
) -> tuple[dict[str, dict], pd.DataFrame, np.ndarray]:
    if logs is not None:
        logs.append('ImplementationAgent: starting execution.')

    df = pd.read_csv(csv_path, sep=None, engine='python')
    X = df.drop(columns=[plan.target_column])
    y = df[plan.target_column]

    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    # Safety: strip target column from any preprocessing steps
    plan_steps = _strip_target_from_steps(plan.preprocessing_steps, plan.target_column, logs)

    transformers = []
    has_imbalance_step = any(s.operation == 'handle_imbalance' for s in plan_steps)

    for step in plan_steps:
        if step.operation == 'handle_imbalance':
            continue
        elif step.operation == 'impute_missing':
            transformers.append(
                (
                    f'impute_{step.method}',
                    SimpleImputer(strategy=step.method),
                    step.columns,
                )
            )
        elif step.operation == 'encode_categorical':
            if step.method == 'one_hot':
                transformer = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
            elif step.method == 'ordinal':
                transformer = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
            else:
                raise ValueError(f'Unknown encode method: {step.method}')
            transformers.append(('encode', transformer, step.columns))
        elif step.operation == 'scale_numeric':
            if step.method == 'standard':
                transformer = StandardScaler()
            elif step.method == 'minmax':
                transformer = MinMaxScaler()
            else:
                raise ValueError(f'Unknown scale method: {step.method}')
            transformers.append(('scale', transformer, step.columns))
        elif step.operation == 'drop_columns':
            transformers.append(('drop', 'drop', step.columns))

    preprocessor = ColumnTransformer(transformers=transformers, remainder='passthrough')
    cv = StratifiedKFold(n_splits=plan.folds, shuffle=True, random_state=plan.random_seed)
    scorer = METRIC_MAP[plan.primary_metric]
    results = {}

    for model_info in plan.models_to_try:
        try:
            if logs is not None:
                logs.append(f'ImplementationAgent: running {model_info.name}.')
            model = MODEL_MAP[model_info.name](**model_info.hyperparameters)

            if has_imbalance_step:
                pipeline = ImbPipeline(
                    [
                        ('preprocessor', preprocessor),
                        ('smote', SMOTE(random_state=plan.random_seed)),
                        ('classifier', model),
                    ]
                )
            else:
                pipeline = Pipeline(
                    [
                        ('preprocessor', preprocessor),
                        ('classifier', model),
                    ]
                )

            scores = cross_val_score(
                pipeline, X, y_encoded, cv=cv, scoring=scorer, n_jobs=1, error_score='raise'
            )
            results[model_info.name] = {
                'mean_score': float(scores.mean()),
                'std_score': float(scores.std()),
                'individual_scores': [float(s) for s in scores],
            }
        except Exception as e:
            if logs is not None:
                logs.append(f'ImplementationAgent: ERROR in {model_info.name} — {str(e)}')
            results[model_info.name] = {'error': str(e)}

    if logs is not None:
        logs.append('ImplementationAgent: execution complete.')

    return results, X, y_encoded

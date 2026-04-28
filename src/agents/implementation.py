# src/agents/implementation.py
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from lightgbm import LGBMClassifier
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
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
    FunctionTransformer,
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
    'SVC': SVC,
    'KNeighborsClassifier': KNeighborsClassifier,
    'GradientBoostingClassifier': GradientBoostingClassifier,
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
    Steps with empty columns after stripping are kept — auto-detection handles them.
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
        clean.append(
            PreprocessingStep(
                operation=step.operation,
                method=step.method,
                columns=clean_cols,
            )
        )
    return clean


def _build_transformer_specs(
    plan_steps: list[PreprocessingStep],
    X: pd.DataFrame,
) -> list:
    """
    Build the transformers list for ColumnTransformer from plan steps.

    Key invariant: each column appears in exactly ONE transformer slot.
    When a column needs both imputation and encoding (common for categoricals),
    we chain them into a Pipeline so ColumnTransformer sees one entry per column group.
    """
    # Collect instructions per operation type
    impute_specs: dict[str, list[str]] = {}   # strategy -> cols
    encode_spec: tuple[str, list[str]] | None = None   # (method, cols)
    scale_spec: tuple[str, list[str]] | None = None    # (method, cols)
    drop_cols: list[str] = []

    for step in plan_steps:
        if step.operation == 'handle_imbalance':
            continue

        elif step.operation == 'impute_missing':
            cols = step.columns if step.columns else list(X.columns)
            strategy = step.method or 'mean'
            impute_specs.setdefault(strategy, []).extend(cols)

        elif step.operation == 'encode_categorical':
            cols = (
                step.columns if step.columns
                else list(X.select_dtypes(include='object').columns)
            )
            encode_spec = (step.method or 'one_hot', cols)

        elif step.operation == 'scale_numeric':
            cols = (
                step.columns if step.columns
                else list(X.select_dtypes(include='number').columns)
            )
            scale_spec = (step.method or 'standard', cols)

        elif step.operation == 'drop_columns':
            drop_cols.extend(step.columns)

    transformers = []
    all_handled: set[str] = set(drop_cols)

    # --- Categorical columns: impute → encode as a sub-pipeline ---
    if encode_spec is not None:
        enc_method, enc_cols = encode_spec
        enc_cols = [c for c in enc_cols if c in X.columns]

        if enc_cols:
            # Find impute strategy for these cols (most_frequent is correct for categoricals)
            cat_impute_strategy = None
            for strategy, cols in impute_specs.items():
                if any(c in enc_cols for c in cols):
                    cat_impute_strategy = strategy
                    break

            if enc_method == 'one_hot':
                encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
            else:
                encoder = OrdinalEncoder(
                    handle_unknown='use_encoded_value', unknown_value=-1
                )

            if cat_impute_strategy:
                cat_pipeline = Pipeline([
                    ('impute', SimpleImputer(strategy=cat_impute_strategy)),
                    ('encode', encoder),
                ])
            else:
                cat_pipeline = Pipeline([('encode', encoder)])

            transformers.append(('categorical', cat_pipeline, enc_cols))
            all_handled.update(enc_cols)

    # --- Numeric columns: impute → scale as a sub-pipeline ---
    if scale_spec is not None:
        scale_method, scale_cols = scale_spec
        scale_cols = [c for c in scale_cols if c in X.columns and c not in all_handled]

        if scale_cols:
            num_impute_strategy = None
            for strategy, cols in impute_specs.items():
                if any(c in scale_cols for c in cols):
                    num_impute_strategy = strategy
                    break

            scaler = StandardScaler() if scale_method == 'standard' else MinMaxScaler()

            if num_impute_strategy:
                num_pipeline = Pipeline([
                    ('impute', SimpleImputer(strategy=num_impute_strategy)),
                    ('scale', scaler),
                ])
            else:
                num_pipeline = Pipeline([('scale', scaler)])

            transformers.append(('numeric', num_pipeline, scale_cols))
            all_handled.update(scale_cols)

    # --- Impute-only columns (numeric, no scale step) ---
    for strategy, cols in impute_specs.items():
        impute_only = [c for c in cols if c in X.columns and c not in all_handled]
        if impute_only:
            transformers.append((
                f'impute_only_{strategy}',
                SimpleImputer(strategy=strategy),
                impute_only,
            ))
            all_handled.update(impute_only)

    # --- Drop explicitly requested columns ---
    if drop_cols:
        valid_drop = [c for c in drop_cols if c in X.columns]
        if valid_drop:
            transformers.append(('drop_explicit', 'drop', valid_drop))

    # --- Safety net: drop any remaining object columns not handled ---
    unhandled_obj = [
        c for c in X.select_dtypes(include='object').columns
        if c not in all_handled
    ]
    if unhandled_obj:
        transformers.append(('drop_unhandled_strings', 'drop', unhandled_obj))

    return transformers

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

    plan_steps = _strip_target_from_steps(plan.preprocessing_steps, plan.target_column, logs)
    has_imbalance_step = any(s.operation == 'handle_imbalance' for s in plan_steps)

    cv = StratifiedKFold(n_splits=plan.folds, shuffle=True, random_state=plan.random_seed)
    scorer = METRIC_MAP[plan.primary_metric]
    results = {}

    for model_info in plan.models_to_try:
        try:
            if logs is not None:
                logs.append(f'ImplementationAgent: running {model_info.name}.')

            model = MODEL_MAP[model_info.name](**model_info.hyperparameters)

            # Build a fresh preprocessor per model — avoids state leakage
            # across multiple cross_val_score calls sharing the same instance.
            transformers = _build_transformer_specs(plan_steps, X)
            if transformers:
                preprocessor = ColumnTransformer(
                    transformers=transformers, remainder='passthrough'
                )
            else:
                preprocessor = FunctionTransformer()  # identity pass-through

            if has_imbalance_step:
                pipeline = ImbPipeline([
                    ('preprocessor', preprocessor),
                    ('smote', SMOTE(random_state=plan.random_seed)),
                    ('classifier', model),
                ])
            else:
                pipeline = Pipeline([
                    ('preprocessor', preprocessor),
                    ('classifier', model),
                ])

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
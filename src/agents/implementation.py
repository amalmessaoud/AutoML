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

from src.schemas.plan import AutoMLPlan

# Mapping for preprocessing
PREPROCESS_MAP = {
    'impute_missing': {
        'mean': SimpleImputer(strategy='mean'),
        'median': SimpleImputer(strategy='median'),
        'most_frequent': SimpleImputer(strategy='most_frequent'),
    },
    'encode_categorical': {
        'one_hot': OneHotEncoder(handle_unknown='ignore'),
        'ordinal': OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1),
    },
    'scale_numeric': {'standard': StandardScaler(), 'minmax': MinMaxScaler()},
    'drop_columns': lambda: 'drop',  # Special for dropping
    'handle_imbalance': SMOTE,  # Add imblearn if needed
}

# Mapping for models
MODEL_MAP = {
    'LogisticRegression': lambda **kwargs: LogisticRegression(max_iter=1000, **kwargs),
    'RandomForestClassifier': RandomForestClassifier,
    'XGBClassifier': XGBClassifier,
    'LGBMClassifier': LGBMClassifier,
    'CatBoostClassifier': CatBoostClassifier,
    'SVC': SVC,
    'KNeighborsClassifier': KNeighborsClassifier,
}

# Mapping for metrics (make_scorer)
METRIC_MAP = {
    'accuracy': make_scorer(accuracy_score),
    'f1_macro': make_scorer(f1_score, average='macro'),
    'f1_weighted': make_scorer(f1_score, average='weighted'),
    'roc_auc': make_scorer(roc_auc_score, multi_class='ovr'),
    'balanced_accuracy': make_scorer(balanced_accuracy_score),
}


def execute_plan(
    plan: AutoMLPlan, csv_path: str, logs: list[str] = None
) -> dict[str, dict[str, float]]:
    """
    Execute the AutoMLPlan on the CSV and return CV metrics for each model.
    """
    # Load data
    if logs is not None:
        logs.append('Starting Implementation Agent...')
    df = pd.read_csv(csv_path,sep=None, engine="python")
    X = df.drop(columns=[plan.target_column])
    y = df[plan.target_column]

    # Encode target labels to 0,1,2,... (required for XGBoost and multi-class metrics)
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    # Build preprocessing transformers from plan
    transformers = []
    
    has_imbalance_step = any(step.operation == 'handle_imbalance' for step in plan.preprocessing_steps)

    for step in plan.preprocessing_steps:
        if step.operation == 'handle_imbalance':
            continue  # Skip — handled separately
        if step.operation == 'impute_missing':
            strategy = step.method
            transformer = SimpleImputer(strategy=strategy)
            transformers.append((f'impute_{strategy}', transformer, step.columns))

        elif step.operation == 'encode_categorical':
            if step.method == 'one_hot':
                transformer = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
            elif step.method == 'ordinal':
                transformer = OrdinalEncoder(
                    handle_unknown_value='use_encoded_value', unknown_value=-1
                )
            else:
                raise ValueError(f'Unknown method for encode_categorical: {step.method}')
            transformers.append(('encode', transformer, step.columns))

        elif step.operation == 'scale_numeric':
            if step.method == 'standard':
                transformer = StandardScaler()
            elif step.method == 'minmax':
                transformer = MinMaxScaler()
            else:
                raise ValueError(f'Unknown method for scale_numeric: {step.method}')
            transformers.append(('scale', transformer, step.columns))

        elif step.operation == 'drop_columns':
            transformers.append(('drop', 'drop', step.columns))


    # Build preprocessing pipeline (expand this)
    preprocessor = ColumnTransformer(transformers=transformers, remainder='passthrough')

    results = {}

    cv = StratifiedKFold(n_splits=plan.folds, shuffle=True, random_state=plan.random_seed)
    scorer = METRIC_MAP[plan.primary_metric]

    for model_info in plan.models_to_try:
        try:
            model = MODEL_MAP[model_info.name](**model_info.hyperparameters)
            if logs is not None:
                logs.append(f'Running {model_info.name}...')

            if has_imbalance_step:
                # Use imblearn Pipeline and add SMOTE at the end of preprocessing
                pipeline = ImbPipeline([
                    ('preprocessor', preprocessor),
                    ('smote', SMOTE(random_state=plan.random_seed)),
                    ('classifier', model)
                ])
            else:
                pipeline = Pipeline([
                    ('preprocessor', preprocessor),
                    ('classifier', model)
                ])

            scores = cross_val_score(
                pipeline, X, y_encoded, cv=cv, scoring=scorer, n_jobs=-1, error_score='raise'
            )

            results[model_info.name] = {
                'mean_score': float(scores.mean()),
                'std_score': float(scores.std()),
                'individual_scores': [float(s) for s in scores],
            }
        except Exception as e:
            results[model_info.name] = {'error': str(e)}

    if logs is not None:
        logs.append('Implementation finished execution.')
    return results


# Quick test
if __name__ == '__main__':
    # Hardcode a plan for Iris (replace with your generated one)
    # plan = AutoMLPlan(
    #     task_type="classification",
    #     target_column="class",
    #     primary_metric="accuracy",
    #     preprocessing_steps=[
    #         PreprocessingStep(operation="scale_numeric", method="standard", columns=["sepal_length", "sepal_width", "petal_length", "petal_width"])
    #     ],
    #     models_to_try=[
    #         ModelToTry(name="KNeighborsClassifier", hyperparameters={"n_neighbors": 5}),
    #         ModelToTry(name="SVC", hyperparameters={"kernel": "linear"})
    #     ],
    #     reasoning="Test plan"
    # )
    from src.schemas.plan import AutoMLPlan, ModelToTry, PreprocessingStep

    # plan = AutoMLPlan(
    #     task_type="classification",
    #     target_column="income",
    #     primary_metric="f1_macro",
    #     preprocessing_steps=[
    #         PreprocessingStep(
    #             operation="impute_missing",
    #             method="mean",
    #             columns=[
    #                 "age",
    #                 "fnlwgt",
    #                 "educational-num",
    #                 "capital-gain",
    #                 "capital-loss",
    #                 "hours-per-week",
    #             ],
    #         ),
    #         PreprocessingStep(
    #             operation="encode_categorical",
    #             method="one_hot",
    #             columns=[
    #                 "workclass",
    #                 "education",
    #                 "marital-status",
    #                 "occupation",
    #                 "relationship",
    #                 "race",
    #                 "gender",
    #                 "native-country",
    #             ],
    #         ),
    #         PreprocessingStep(
    #             operation="scale_numeric",
    #             method="standard",
    #             columns=[
    #                 "age",
    #                 "fnlwgt",
    #                 "educational-num",
    #                 "capital-gain",
    #                 "capital-loss",
    #                 "hours-per-week",
    #             ],
    #         ),
    #     ],
    #     models_to_try=[
    #         ModelToTry(name="LogisticRegression", hyperparameters={"C": 0.1, "penalty": "l2"}),
    #         ModelToTry(name="RandomForestClassifier", hyperparameters={"n_estimators": 100, "max_depth": 5}),
    #         ModelToTry(name="XGBClassifier", hyperparameters={"max_depth": 5, "learning_rate": 0.1}),
    #     ],
    #     validation_method="cross_validation",
    #     folds=5,
    #     random_seed=42,
    #     reasoning=(
    #         "Dataset has missing values and categorical columns. "
    #         "Imputation and encoding are necessary for numeric and categorical columns respectively. "
    #         "Models chosen for their performance on classification tasks."
    #     ),
    # )

    # plan = AutoMLPlan(
    #     task_type='classification',
    #     target_column='y',
    #     primary_metric='f1_macro',
    #     preprocessing_steps=[
    #         PreprocessingStep(
    #             operation='impute_missing',
    #             method='mean',
    #             columns=[
    #                 'age',
    #                 'balance',
    #                 'day',
    #                 'duration',
    #                 'campaign',
    #                 'pdays',
    #                 'previous',
    #             ],
    #         ),
    #         PreprocessingStep(
    #             operation='encode_categorical',
    #             method='one_hot',
    #             columns=[
    #                 'job',
    #                 'marital',
    #                 'education',
    #                 'default',
    #                 'housing',
    #                 'loan',
    #                 'contact',
    #                 'month',
    #                 'poutcome',
    #             ],
    #         ),
    #     ],
    #     models_to_try=[
    #         ModelToTry(name='LogisticRegression', hyperparameters={}),
    #         ModelToTry(name='RandomForestClassifier', hyperparameters={}),
    #         ModelToTry(name='XGBClassifier', hyperparameters={}),
    #     ],
    #     validation_method='cross_validation',
    #     folds=5,
    #     random_seed=42,
    #     reasoning=(
    #         'Dataset has categorical columns and a potential target column. '
    #         'We will encode the categorical columns and use a combination of '
    #         'logistic regression, random forest, and XGBoost for classification.'
    #     ),
    # )
    #     plan = AutoMLPlan(
    #     task_type="classification",
    #     target_column="class",
    #     primary_metric="accuracy",
    #     preprocessing_steps=[
    #         PreprocessingStep(
    #             operation="scale_numeric",
    #             method="standard",
    #             columns=["sepal_length", "sepal_width", "petal_length", "petal_width"]
    #         )
    #     ],
    #     models_to_try=[
    #         ModelToTry(name="KNeighborsClassifier", hyperparameters={"n_neighbors": 5}),
    #         ModelToTry(name="SVC", hyperparameters={"kernel": "linear"})
    #     ],
    #     validation_method="cross_validation",
    #     folds=5,
    #     random_seed=42,
    #     reasoning="Dataset is numeric with no missing values or categoricals. Simple features suggest KNN and SVM as good starters for multi-class."
    # )

    plan = AutoMLPlan(
        task_type='classification',
        target_column='income',
        primary_metric='f1_macro',
        preprocessing_steps=[
            PreprocessingStep(
                operation='encode_categorical',
                method='one_hot',
                columns=[
                    'workclass',
                    'education',
                    'marital-status',
                    'occupation',
                    'relationship',
                    'race',
                    'gender',
                    'native-country',
                ],
            ),
            PreprocessingStep(
                operation='impute_missing',
                method='mean',
                columns=[
                    'age',
                    'fnlwgt',
                    'educational-num',
                    'capital-gain',
                    'capital-loss',
                    'hours-per-week',
                ],
            ),
        ],
        models_to_try=[
            ModelToTry(name='LogisticRegression', hyperparameters={}),
            ModelToTry(name='RandomForestClassifier', hyperparameters={}),
            ModelToTry(name='XGBClassifier', hyperparameters={}),
        ],
        validation_method='cross_validation',
        folds=5,
        random_seed=42,
        reasoning=(
            'Dataset has missing values and categorical columns. '
            'One-hot encoding and mean imputation are used for preprocessing. '
            'Simple models are tried first due to the simplicity of the dataset.'
        ),
    )

    plan = AutoMLPlan(
        task_type='classification',
        target_column='Quality',
        primary_metric='accuracy',
        preprocessing_steps=[],
        models_to_try=[
            ModelToTry(name='LogisticRegression', hyperparameters={}),
            ModelToTry(name='RandomForestClassifier', hyperparameters={}),
            ModelToTry(name='XGBClassifier', hyperparameters={}),
        ],
        validation_method='cross_validation',
        folds=5,
        random_seed=42,
        reasoning=(
            'Dataset is numeric with no missing values or categorical features. '
            'Simple baseline models are suitable starting points for multi-class classification.'
        ),
    )

    csv_path = 'data/wine.csv'

    results = execute_plan(plan, csv_path)
    print(results)

import json

from openai import OpenAI

from src.schemas.plan import AutoMLPlan, ModelToTry

# Groq setup — just change this block when you want to switch back
client = OpenAI(
    api_key='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',  # ← paste your key
    base_url='https://api.groq.com/openai/v1',
)


model = 'llama-3.1-8b-instant'  # or "llama-3.1-8b-instant" if you want speed


def evaluate_results(plan: AutoMLPlan, results: dict, problem: str, logs: list[str] = None) -> dict:
    """
    Critique the execution results and decide if the problem is solved.

    Input:
    - plan: The AutoMLPlan used
    - results: Dict from Implementation Agent (model -> mean_score, std_score)
    - problem: Original problem description

    Output:
    - dict with 'solved': bool, 'critique': str, 'suggestion': str, 'best_model': str, 'best_score': float
    """
    # Find best model
    if logs is not None:
        logs.append('Starting Critique Agent...')
    valid_results = {k: v for k, v in results.items() if 'error' not in v}
    if not valid_results:
        best_model = None
        best_score = 0.0
    else:
        best_model = max(valid_results, key=lambda k: valid_results[k]['mean_score'])
        best_score = valid_results[best_model]['mean_score']

    # Build prompt for LLM critique
    prompt = f"""
You are an expert ML evaluator. Review this AutoML run and decide if the problem is solved.

Problem: {problem}

Plan Summary:
- Target: {plan.target_column}
- Metric: {plan.primary_metric}
- Preprocessing: {len(plan.preprocessing_steps)} steps
- Models: {', '.join([m.name for m in plan.models_to_try])}

Results Summary:
Best Model: {best_model}
Best Score: {best_score:.4f}
Full Results: {json.dumps(results, indent=2)}

Critique Guidelines:
- Is the best score good? (e.g., >0.85 excellent, >0.75 good, <0.75 poor for f1_macro/accuracy)
- Does it solve the problem? (consider data size, imbalance, etc.)
- If not, suggest improvements (e.g., different models, add imbalance handling)
- For f1_macro on imbalanced data: >0.75 excellent, >0.65 good, <0.65 poor

Output exactly this JSON:
{{
  "decision": "SOLVED" or "NOT_SOLVED",
  "reason": "brief explanation",
  "suggestion": "what to try next if NOT_SOLVED (or empty string)"
}}
"""

    response = client.chat.completions.create(
        model=model,
        messages=[{'role': 'user', 'content': prompt}],
        temperature=0.3,
        response_format={'type': 'json_object'},
    )

    critique = json.loads(response.choices[0].message.content)
    if logs is not None:
        logs.append('Critique finished evaluation.')

    return {
        'solved': critique['decision'] == 'SOLVED',
        'critique': critique['reason'],
        'suggestion': critique['suggestion'],
        'best_model': best_model,
        'best_score': best_score,
    }


# Quick test
if __name__ == '__main__':
    # Sample plan and results from your previous run
    # sample_plan = AutoMLPlan(
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

    # sample_plan = AutoMLPlan(
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

    #   sample_plan = AutoMLPlan(
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

    #   sample_plan = AutoMLPlan(
    #     task_type="classification",
    #     target_column="income",
    #     primary_metric="f1_macro",
    #     preprocessing_steps=[
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
    #                 "native-country"
    #             ]
    #         ),
    #         PreprocessingStep(
    #             operation="impute_missing",
    #             method="mean",
    #             columns=[
    #                 "age",
    #                 "fnlwgt",
    #                 "educational-num",
    #                 "capital-gain",
    #                 "capital-loss",
    #                 "hours-per-week"
    #             ]
    #         )
    #     ],
    #     models_to_try=[
    #         ModelToTry(name="LogisticRegression", hyperparameters={}),
    #         ModelToTry(name="RandomForestClassifier", hyperparameters={}),
    #         ModelToTry(name="XGBClassifier", hyperparameters={})
    #     ],
    #     validation_method="cross_validation",
    #     folds=5,
    #     random_seed=42,
    #     reasoning=(
    #         "Dataset has missing values and categorical columns. "
    #         "One-hot encoding and mean imputation are used for preprocessing. "
    #         "Simple models are tried first due to the simplicity of the dataset."
    #     )
    # )

    sample_plan = AutoMLPlan(
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

    # sample_results = {
    #     "LogisticRegression": {"mean_score": 0.79, "std_score": 0.01},
    #     "RandomForestClassifier": {"mean_score": 0.85, "std_score": 0.02},
    #     "XGBClassifier": {"mean_score": 0.87, "std_score": 0.01}
    # }

    # sample_results = {'LogisticRegression': {'mean_score': 0.6772971179659897, 'std_score': 0.009014814100217925, 'individual_scores': [0.6913465414070009, 0.6789259020744297, 0.6784422498171514, 0.6745188942963134, 0.6632520022350531]}, 'RandomForestClassifier': {'mean_score': 0.6508132147174284, 'std_score': 0.03405699013814008, 'individual_scores': [0.7053762440703191, 0.608929208929209, 0.6454000370704036, 0.6696780696780696, 0.6246825138391403]}, 'XGBClassifier': {'mean_score': 0.6976142624967716, 'std_score': 0.012819516708138535, 'individual_scores': [0.7108995655507283, 0.6789772727272727, 0.7104500891265597, 0.6866230974978744, 0.7011212875814234]}}

    # sample_results = {'KNeighborsClassifier': {'mean_score': 0.9733333333333334, 'std_score': 0.02494438257849294, 'individual_scores': [1.0, 0.9666666666666667, 0.9333333333333333, 1.0, 0.9666666666666667]}, 'SVC': {'mean_score': 0.9666666666666666, 'std_score': 0.05163977794943222, 'individual_scores': [1.0, 1.0, 0.8666666666666667, 1.0, 0.9666666666666667]}}

    # sample_results = {'LogisticRegression': {'mean_score': 0.7664137459105994, 'std_score': 0.005826986089087369, 'individual_scores': [0.7653110817230833, 0.769951399473814, 0.7607765180077048, 0.7757327816400138, 0.7602969487083817]}, 'RandomForestClassifier': {'mean_score': 0.7881997898464366, 'std_score': 0.0028077552519606415, 'individual_scores': [0.7864580904066942, 0.7923999891021042, 0.7861700922765154, 0.7906808153260729, 0.7852899621207964]}, 'XGBClassifier': {'mean_score': 0.8167298032631367, 'std_score': 0.003289151117209349, 'individual_scores': [0.8138504006823895, 0.8211490303966793, 0.8134688666155365, 0.8202509256448222, 0.814929792976256]}}

    sample_results = {
        'LogisticRegression': {
            'mean_score': 0.943968253968254,
            'std_score': 0.030290439217464302,
            'individual_scores': [
                0.9444444444444444,
                0.8888888888888888,
                0.9722222222222222,
                0.9428571428571428,
                0.9714285714285714,
            ],
        },
        'RandomForestClassifier': {
            'mean_score': 0.9828571428571429,
            'std_score': 0.022857142857142864,
            'individual_scores': [1.0, 1.0, 1.0, 0.9714285714285714, 0.9428571428571428],
        },
        'XGBClassifier': {
            'mean_score': 0.9717460317460318,
            'std_score': 0.03130084676673201,
            'individual_scores': [
                0.9722222222222222,
                1.0,
                0.9722222222222222,
                1.0,
                0.9142857142857143,
            ],
        },
    }

    # problem = "Predict whether a client will subscribe to a term deposit."
    # problem = "Predict the species of an iris flower (setosa, versicolor, virginica) from its sepal and petal measurements."
    problem = 'Predict the quality class of wine from its chemical properties.'
    # problem = "Predict whether an individual earns more than $50K per yearbased on their demographic and employment information."

    eval = evaluate_results(sample_plan, sample_results, problem)
    print(json.dumps(eval, indent=2))

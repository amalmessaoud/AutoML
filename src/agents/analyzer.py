import json

from openai import OpenAI

from src.schemas.plan import AutoMLPlan

# Groq setup — just change this block when you want to switch back
client = OpenAI(
    api_key='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',  # ← paste your key
    base_url='https://api.groq.com/openai/v1',
)

# Model name — choose one:
# "llama-3.1-8b-instant"   ← fast, good
# "llama-3.1-70b-versatile" ← much stronger reasoning (recommended for planning!)
model = 'llama-3.1-8b-instant'  # or "llama-3.1-8b-instant" if you want speed


def generate_automl_plan(
    dataset_description: str, problem_description: str, model: str = model, logs: list[str] = None
) -> AutoMLPlan:
    if logs is not None:
        logs.append('Starting Analyzer Agent...')
    example_json = {
        'task_type': 'classification',
        'target_column': 'species',
        'primary_metric': 'accuracy',
        'preprocessing_steps': [
            {
                'operation': 'scale_numeric',
                'method': 'standard',
                'columns': ['sepal_length', 'sepal_width', 'petal_length', 'petal_width'],
            }
        ],
        'models_to_try': [
            {'name': 'KNeighborsClassifier', 'hyperparameters': {'n_neighbors': 5}},
            {'name': 'SVC', 'hyperparameters': {'kernel': 'linear'}},
        ],
        'validation_method': 'cross_validation',
        'folds': 5,
        'random_seed': 42,
        'reasoning': 'Dataset is numeric with no missing values or categoricals. Simple features suggest KNN and SVM as good starters for multi-class.',
    }

    example_str = json.dumps(example_json, indent=2)

    prompt = f"""
You are an expert machine learning engineer specializing in tabular classification.
Your task is to create a complete, executable AutoML plan in strict JSON format.

Dataset description (PAY CLOSE ATTENTION TO TARGET VALUE COUNTS FOR IMBALANCE):
{dataset_description}

Problem: {problem_description}

You MUST output ONLY a valid JSON object that exactly matches the AutoMLPlan schema.

HERE IS THE EXACT STRUCTURE YOU MUST FOLLOW:

{example_str}

CRITICAL METRIC SELECTION RULE — FOLLOW THIS FIRST:
1. Look at the target column value counts in the dataset description.
2. If the classes are imbalanced (one class >70% or <30% of data), you MUST use primary_metric "f1_macro" or "balanced_accuracy".
3. If classes are balanced, use "accuracy".
4. Never use "accuracy" on imbalanced data — it is misleading.

Other STRICT RULES:
- Use ONLY the field names shown above
- task_type must always be "classification"
- For encode_categorical, method must be "one_hot" or "ordinal" (never "most_frequent" or "mean")
- For impute_missing: method 'mean' or 'median' for numeric, 'most_frequent' for categorical
- If target is imbalanced, ALSO add "handle_imbalance" with method "oversample"
- primary_metric must be one of: accuracy, f1_macro, f1_weighted, roc_auc, balanced_accuracy
- preprocessing_steps can be empty
- models_to_try: exactly 1 to 3 models
- Allowed model names: LogisticRegression, RandomForestClassifier, XGBClassifier, LGBMClassifier, CatBoostClassifier, SVC, KNeighborsClassifier
- hyperparameters: dict, empty means use defaults
- validation_method, folds, random_seed: use values from example
- reasoning: short text explaining choices
- Output ONLY the JSON. No extra text, no markdown.

Now generate the plan:
"""

    print('🤖 Analyzer Agent thinking... (streaming output)\n')

    stream = client.chat.completions.create(
        model=model,
        messages=[{'role': 'user', 'content': prompt}],
        stream=True,
        temperature=0.3,
        response_format={'type': 'json_object'},  # ← Forces valid JSON from Groq!
    )

    raw_output = ''
    for chunk in stream:
        if chunk.choices[0].delta.content:
            content = chunk.choices[0].delta.content
            print(content, end='', flush=True)
            raw_output += content

    print('\n\n✅ Analyzer finished thinking!\n')

    try:
        plan = AutoMLPlan.model_validate_json(raw_output)
        print('\n🎉 SUCCESS! Valid AutoML Plan generated:')
        print(plan.model_dump_json(indent=2))
        if logs is not None:
            logs.append('Analyzer finished generating plan.')
        return plan
    except Exception as e:
        print('\n❌ Validation failed:', e)
        print('\nRaw LLM output was:')
        print(raw_output)
        raise e


if __name__ == '__main__':
    from src.utils.data_utils import generate_dataset_description

    # Path to your downloaded CSV
    csv_path = 'data/wine.csv'  # Make sure this matches your file location

    dataset_desc = generate_dataset_description(csv_path)
    # problem = "Predict whether an individual earns more than $50K per yearbased on their demographic and employment information."
    # problem = 'Predict whether a client will subscribe to a term deposit.'
    # problem = "Predict the species of an iris flower (setosa, versicolor, virginica) from its sepal and petal measurements."
    problem = 'Predict the quality class of wine from its chemical properties.'

    print('=== GENERATED DATASET DESCRIPTION ===\n')
    print(dataset_desc)
    print('\n=== END DESCRIPTION ===\n')

    try:
        plan = generate_automl_plan(dataset_desc, problem)
        print('\n✅ Valid structured plan:')
        print(plan.model_dump_json(indent=2))
    except Exception as e:
        print('\n❌ Validation failed:', e)

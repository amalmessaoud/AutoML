# src/agents/analyzer.py
import json

from src.agents.base_agent import BaseAgent
from src.config.llm_config import LLMConfig
from src.schemas.plan import AutoMLPlan


class AnalyzerAgent(BaseAgent):
    """
    LLM-driven planning agent.
    Receives dataset description + problem statement, outputs a validated AutoMLPlan.
    """

    def __init__(self, config: LLMConfig, logs: list[str]) -> None:
        super().__init__(config, logs)

    def run(self, dataset_description: str, problem_description: str) -> AutoMLPlan:
        self._log("AnalyzerAgent: starting plan generation.")

        example_json = {
            "task_type": "classification",
            "target_column": "species",
            "primary_metric": "accuracy",
            "preprocessing_steps": [
                {
                    "operation": "scale_numeric",
                    "method": "standard",
                    "columns": ["sepal_length", "sepal_width", "petal_length", "petal_width"],
                }
            ],
            "models_to_try": [
                {"name": "KNeighborsClassifier", "hyperparameters": {"n_neighbors": 5}},
                {"name": "SVC", "hyperparameters": {"kernel": "linear"}},
            ],
            "validation_method": "cross_validation",
            "folds": 5,
            "random_seed": 42,
            "reasoning": "Dataset is numeric with no missing values or categoricals.",
        }

        prompt = f"""
You are an expert machine learning engineer specializing in tabular classification.
Your task is to create a complete, executable AutoML plan in strict JSON format.

Dataset description (PAY CLOSE ATTENTION TO TARGET VALUE COUNTS FOR IMBALANCE):
{dataset_description}

Problem: {problem_description}

You MUST output ONLY a valid JSON object that exactly matches the AutoMLPlan schema.

HERE IS THE EXACT STRUCTURE YOU MUST FOLLOW:
{json.dumps(example_json, indent=2)}

CRITICAL METRIC SELECTION RULE — FOLLOW THIS FIRST:
1. Look at the target column value counts in the dataset description.
2. If the classes are imbalanced (one class >70% or <30% of data), you MUST use primary_metric "f1_macro" or "balanced_accuracy".
3. If classes are balanced, use "accuracy".
4. Never use "accuracy" on imbalanced data — it is misleading.

STRICT RULES:
- task_type must always be "classification"
- For encode_categorical: method must be "one_hot" or "ordinal"
- For impute_missing: method "mean" or "median" for numeric, "most_frequent" for categorical
- If target is imbalanced, ALSO add "handle_imbalance" with method "oversample"
- primary_metric must be one of: accuracy, f1_macro, f1_weighted, roc_auc, balanced_accuracy
- preprocessing_steps can be empty list
- models_to_try: exactly 1 to 3 models
- Allowed model names: LogisticRegression, RandomForestClassifier, XGBClassifier, LGBMClassifier, CatBoostClassifier, SVC, KNeighborsClassifier
- hyperparameters: dict, empty means use defaults
- Output ONLY the JSON. No extra text, no markdown.

Now generate the plan:
"""

        client = self._build_client()

        self._log(f"AnalyzerAgent: calling {self.config.provider}/{self.config.model}.")

        stream = client.chat.completions.create(
            model=self.config.model,
            messages=[{"role": "user", "content": prompt}],
            stream=True,
            temperature=self.config.temperature,
            response_format={"type": "json_object"},
        )

        raw_output = ""
        for chunk in stream:
            if chunk.choices[0].delta.content:
                raw_output += chunk.choices[0].delta.content

        try:
            plan = AutoMLPlan.model_validate_json(raw_output)
            self._log("AnalyzerAgent: plan validated successfully.")
            return plan
        except Exception as e:
            self._log(f"AnalyzerAgent: validation failed — {e}")
            raise


# --- Convenience function for backward compatibility with orchestrator ---
def generate_automl_plan(
    dataset_description: str,
    problem_description: str,
    config: LLMConfig = None,
    logs: list[str] = None,
) -> AutoMLPlan:
    from src.config.llm_config import GROQ_LLAMA_8B
    if config is None:
        config = GROQ_LLAMA_8B
    if logs is None:
        logs = []
    agent = AnalyzerAgent(config=config, logs=logs)
    return agent.run(
        dataset_description=dataset_description,
        problem_description=problem_description,
    )
# src/agents/plan_validator_agent.py
from src.schemas.plan import AutoMLPlan
from src.schemas.quality_report import DatasetQualityReport
from src.schemas.validation_result import ValidationResult

ENSEMBLE_MODELS = {
    'RandomForestClassifier',
    'XGBClassifier',
    'LGBMClassifier',
    'CatBoostClassifier',
}


class PlanValidatorAgent:
    """
    Pure logic — no LLM.
    Validates an AutoMLPlan against the DatasetQualityReport before execution.
    Each rule violation is a paper data point.
    """

    def __init__(self, logs: list[str]) -> None:
        self.logs = logs

    def _log(self, message: str) -> None:
        self.logs.append(message)

    def run(
        self,
        plan: AutoMLPlan,
        quality_report: DatasetQualityReport,
    ) -> ValidationResult:
        self._log('PlanValidatorAgent: starting validation.')
        issues: list[str] = []
        warnings: list[str] = []

        operations = [step.operation for step in plan.preprocessing_steps]

        # --- Rule 1: impute_missing must come before scale_numeric ---
        if 'impute_missing' in operations and 'scale_numeric' in operations:
            impute_idx = operations.index('impute_missing')
            scale_idx = operations.index('scale_numeric')
            if impute_idx > scale_idx:
                issues.append(
                    'Rule1: impute_missing must appear before scale_numeric in '
                    'preprocessing_steps — scaling before imputation propagates NaN.'
                )

        # --- Rule 2: imbalanced target requires handle_imbalance step ---
        if quality_report.imbalance_flag and 'handle_imbalance' not in operations:
            issues.append(
                'Rule2: dataset is imbalanced (minority class < 20%) but plan has no '
                'handle_imbalance step — add SMOTE or class weighting.'
            )

        # --- Rule 3: accuracy metric on imbalanced data is misleading ---
        if quality_report.imbalance_flag and plan.primary_metric == 'accuracy':
            warnings.append(
                "Rule3: primary_metric is 'accuracy' but dataset is imbalanced — "
                'accuracy is misleading here, prefer f1_macro or balanced_accuracy.'
            )

        # --- Rule 4: ensemble models on very small datasets risk overfitting ---
        if quality_report.n_rows < 500:
            selected_ensembles = [m.name for m in plan.models_to_try if m.name in ENSEMBLE_MODELS]
            if selected_ensembles:
                warnings.append(
                    f'Rule4: dataset has only {quality_report.n_rows} rows — '
                    f'ensemble models {selected_ensembles} risk overfitting on small data.'
                )

        # --- Rule 5: leakage columns not dropped and not used in preprocessing ---
        if quality_report.potential_leakage_columns:
            all_referenced_cols = []
            for step in plan.preprocessing_steps:
                all_referenced_cols.extend(step.columns)

            not_addressed = [
                col
                for col in quality_report.potential_leakage_columns
                if col not in all_referenced_cols  # not dropped AND not referenced
            ]
            if not_addressed:
                issues.append(
                    f'Rule5: potential leakage columns {not_addressed} are not in any '
                    f'drop_columns step — this will cause data leakage.'
                )

        # --- Rule 6: target column must not appear in any preprocessing step ---
        for step in plan.preprocessing_steps:
            if plan.target_column in step.columns:
                issues.append(
                    f"Rule6: target column '{plan.target_column}' appears in "
                    f"preprocessing step '{step.operation}' — target is never a feature, "
                    f'remove it from the columns list.'
                )
        
        # Rule 7: object-dtype columns present but no encode_categorical step
        object_cols = [c.name for c in quality_report.columns if c.dtype == 'object' 
                    and c.name != plan.target_column]
        if object_cols:
            has_encode = any(s.operation == 'encode_categorical' for s in plan.preprocessing_steps)
            if not has_encode:
                issues.append(
                    f"Rule7: columns {object_cols[:3]} have dtype=object but no "
                    f"encode_categorical step found — all models will fail with "
                    f"'could not convert string to float'."
                )
        
        passed = len(issues) == 0

        self._log(
            f'PlanValidatorAgent: {"PASSED" if passed else "FAILED"} — '
            f'{len(issues)} issue(s), {len(warnings)} warning(s).'
        )

        return ValidationResult(passed=passed, issues=issues, warnings=warnings)

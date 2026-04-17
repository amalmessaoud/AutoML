# tests/test_plan_validator.py
from src.agents.plan_validator_agent import PlanValidatorAgent
from src.schemas.plan import AutoMLPlan, ModelToTry, PreprocessingStep
from src.schemas.quality_report import ColumnQuality, DatasetQualityReport


def _make_agent() -> PlanValidatorAgent:
    return PlanValidatorAgent(logs=[])


def _make_quality_report(
    imbalance_flag: bool = False,
    n_rows: int = 1000,
    potential_leakage_columns: list[str] = None,
) -> DatasetQualityReport:
    return DatasetQualityReport(
        n_rows=n_rows,
        n_cols=5,
        columns=[
            ColumnQuality(
                name='f1', missing_rate=0.0, dtype='float64', n_unique=100, is_constant=False
            )
        ],
        class_distribution={'0': 0.8, '1': 0.2} if not imbalance_flag else {'0': 0.95, '1': 0.05},
        imbalance_flag=imbalance_flag,
        duplicate_row_rate=0.0,
        potential_leakage_columns=potential_leakage_columns,  # now always a list
        warnings=[],
    )


def _make_plan(
    preprocessing_steps: list[PreprocessingStep] = None,
    models: list[str] = ['LogisticRegression'],
    metric: str = 'accuracy',
) -> AutoMLPlan:
    return AutoMLPlan(
        task_type='classification',
        target_column='target',
        primary_metric=metric,
        preprocessing_steps=preprocessing_steps,
        models_to_try=[ModelToTry(name=m, hyperparameters={}) for m in models],
        validation_method='cross_validation',
        folds=5,
        random_seed=42,
        reasoning='Test.',
    )


class TestRule1ImputeBeforeScale:
    def test_impute_after_scale_is_issue(self):
        plan = _make_plan(
            preprocessing_steps=[
                PreprocessingStep(operation='scale_numeric', method='standard', columns=['a']),
                PreprocessingStep(operation='impute_missing', method='mean', columns=['a']),
            ]
        )
        result = _make_agent().run(plan, _make_quality_report())
        assert not result.passed
        assert any('Rule1' in i for i in result.issues)

    def test_impute_before_scale_passes(self):
        plan = _make_plan(
            preprocessing_steps=[
                PreprocessingStep(operation='impute_missing', method='mean', columns=['a']),
                PreprocessingStep(operation='scale_numeric', method='standard', columns=['a']),
            ]
        )
        result = _make_agent().run(plan, _make_quality_report())
        assert not any('Rule1' in i for i in result.issues)


class TestRule2ImbalanceHandling:
    def test_imbalanced_no_smote_is_issue(self):
        plan = _make_plan(metric='f1_macro')
        result = _make_agent().run(plan, _make_quality_report(imbalance_flag=True))
        assert not result.passed
        assert any('Rule2' in i for i in result.issues)

    def test_imbalanced_with_smote_passes(self):
        plan = _make_plan(
            preprocessing_steps=[
                PreprocessingStep(operation='handle_imbalance', method='oversample', columns=[])
            ],
            metric='f1_macro',
        )
        result = _make_agent().run(plan, _make_quality_report(imbalance_flag=True))
        assert not any('Rule2' in i for i in result.issues)


class TestRule3AccuracyOnImbalanced:
    def test_accuracy_on_imbalanced_is_warning(self):
        plan = _make_plan(
            preprocessing_steps=[
                PreprocessingStep(operation='handle_imbalance', method='oversample', columns=[])
            ],
            metric='accuracy',
        )
        result = _make_agent().run(plan, _make_quality_report(imbalance_flag=True))
        assert any('Rule3' in w for w in result.warnings)

    def test_f1_macro_on_imbalanced_no_warning(self):
        plan = _make_plan(
            preprocessing_steps=[
                PreprocessingStep(operation='handle_imbalance', method='oversample', columns=[])
            ],
            metric='f1_macro',
        )
        result = _make_agent().run(plan, _make_quality_report(imbalance_flag=True))
        assert not any('Rule3' in w for w in result.warnings)


class TestRule4SmallDatasetEnsemble:
    def test_ensemble_on_small_data_is_warning(self):
        plan = _make_plan(models=['RandomForestClassifier'])
        result = _make_agent().run(plan, _make_quality_report(n_rows=200))
        assert any('Rule4' in w for w in result.warnings)

    def test_ensemble_on_large_data_no_warning(self):
        plan = _make_plan(models=['RandomForestClassifier'])
        result = _make_agent().run(plan, _make_quality_report(n_rows=1000))
        assert not any('Rule4' in w for w in result.warnings)

    def test_simple_model_on_small_data_no_warning(self):
        plan = _make_plan(models=['LogisticRegression'])
        result = _make_agent().run(plan, _make_quality_report(n_rows=200))
        assert not any('Rule4' in w for w in result.warnings)


class TestRule5LeakageColumns:
    def test_leakage_column_not_dropped_is_issue(self):
        plan = _make_plan()
        result = _make_agent().run(
            plan, _make_quality_report(potential_leakage_columns=['suspicious_col'])
        )
        assert not result.passed
        assert any('Rule5' in i for i in result.issues)

    def test_leakage_column_dropped_passes(self):
        plan = _make_plan(
            preprocessing_steps=[
                PreprocessingStep(
                    operation='drop_columns', method='drop', columns=['suspicious_col']
                )
            ]
        )
        result = _make_agent().run(
            plan, _make_quality_report(potential_leakage_columns=['suspicious_col'])
        )
        assert not any('Rule5' in i for i in result.issues)


class TestRule6TargetInPreprocessing:
    def test_target_in_preprocessing_is_issue(self):
        plan = _make_plan(
            preprocessing_steps=[
                PreprocessingStep(
                    operation='encode_categorical',
                    method='one_hot',
                    columns=['target'],  # target column — wrong
                )
            ]
        )
        result = _make_agent().run(plan, _make_quality_report())
        assert not result.passed
        assert any('Rule6' in i for i in result.issues)

    def test_target_not_in_preprocessing_passes(self):
        plan = _make_plan(
            preprocessing_steps=[
                PreprocessingStep(
                    operation='scale_numeric',
                    method='standard',
                    columns=['f1'],  # feature column — correct
                )
            ]
        )
        result = _make_agent().run(plan, _make_quality_report())
        assert not any('Rule6' in i for i in result.issues)


class TestCleanPlan:
    def test_clean_plan_passes_with_empty_issues(self):
        plan = _make_plan()
        result = _make_agent().run(plan, _make_quality_report())
        assert result.passed
        assert result.issues == []

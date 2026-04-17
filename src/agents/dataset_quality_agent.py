# src/agents/dataset_quality_agent.py
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

from src.schemas.quality_report import ColumnQuality, DatasetQualityReport


class DatasetQualityAgent:
    """
    Pure pandas — no LLM.
    Runs before the Analyzer and produces a DatasetQualityReport.
    The report's warnings list is injected into the Analyzer prompt.
    """

    def __init__(self, logs: list[str]) -> None:
        self.logs = logs

    def _log(self, message: str) -> None:
        self.logs.append(message)

    def run(self, df: pd.DataFrame, target_column: str) -> DatasetQualityReport:
        self._log('DatasetQualityAgent: starting analysis.')

        n_rows, n_cols = df.shape
        warnings: list[str] = []

        # --- Column-level quality ---
        columns = []
        for col in df.columns:
            missing_rate = float(df[col].isna().mean())
            dtype = str(df[col].dtype)
            n_unique = int(df[col].nunique())
            is_constant = n_unique == 1

            if is_constant:
                warnings.append(
                    f"Column '{col}' is constant (all values identical) — consider dropping it."
                )
            if missing_rate > 0.3:
                warnings.append(
                    f"Column '{col}' has {missing_rate:.0%} missing values — imputation required."
                )
            elif missing_rate > 0:
                warnings.append(f"Column '{col}' has {missing_rate:.1%} missing values.")

            columns.append(
                ColumnQuality(
                    name=col,
                    missing_rate=missing_rate,
                    dtype=dtype,
                    n_unique=n_unique,
                    is_constant=is_constant,
                )
            )

        # --- Class distribution ---
        target_counts = df[target_column].value_counts(normalize=True)
        class_distribution = {str(k): round(float(v), 4) for k, v in target_counts.items()}
        imbalance_flag = bool(target_counts.min() < 0.20)
        if imbalance_flag:
            minority = target_counts.idxmin()
            warnings.append(
                f"Target is imbalanced — class '{minority}' is only "
                f'{target_counts.min():.1%} of data. Use f1_macro or '
                f'balanced_accuracy, and add handle_imbalance step.'
            )

        # --- Duplicate rows ---
        duplicate_row_rate = float(df.duplicated().mean())
        if duplicate_row_rate > 0.01:
            warnings.append(f'{duplicate_row_rate:.1%} of rows are duplicates.')

        # --- Leakage heuristic ---
        potential_leakage_columns: list[str] = []
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        numeric_cols = [c for c in numeric_cols if c != target_column]

        if df[target_column].dtype in [np.int64, np.float64, int, float]:
            target_series = df[target_column]
        else:
            le = LabelEncoder()
            target_series = pd.Series(
                le.fit_transform(df[target_column].astype(str)),
                index=df.index,
            )

        for col in numeric_cols:
            try:
                corr = abs(df[col].corr(target_series))
                if corr > 0.95:
                    potential_leakage_columns.append(col)
                    warnings.append(
                        f"Column '{col}' has correlation {corr:.3f} with target "
                        f'— possible data leakage, consider dropping.'
                    )
            except Exception:
                pass

        if not warnings:
            warnings.append('No major data quality issues detected.')

        self._log(f'DatasetQualityAgent: found {len(warnings)} warning(s).')

        return DatasetQualityReport(
            n_rows=n_rows,
            n_cols=n_cols,
            columns=columns,
            class_distribution=class_distribution,
            imbalance_flag=imbalance_flag,
            duplicate_row_rate=duplicate_row_rate,
            potential_leakage_columns=potential_leakage_columns,
            warnings=warnings,
        )

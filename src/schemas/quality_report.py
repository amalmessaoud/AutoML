# src/schemas/quality_report.py
from pydantic import BaseModel


class ColumnQuality(BaseModel):
    name: str
    missing_rate: float
    dtype: str
    n_unique: int
    is_constant: bool


class DatasetQualityReport(BaseModel):
    n_rows: int
    n_cols: int
    columns: list[ColumnQuality]
    class_distribution: dict[str, float]
    imbalance_flag: bool
    duplicate_row_rate: float
    potential_leakage_columns: list[str]
    warnings: list[str]

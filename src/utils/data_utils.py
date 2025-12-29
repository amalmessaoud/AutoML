import json

import pandas as pd


def generate_dataset_description(csv_path: str) -> str:
    """
    Load a CSV and generate a detailed, LLM-friendly description.
    """
    df = pd.read_csv(csv_path)
    df = df.replace('?', pd.NA)

    # Basic info
    shape = df.shape
    total_missing = df.isnull().sum().sum()
    missing_per_column = df.isnull().sum()
    missing_per_column = missing_per_column[missing_per_column > 0]

    # Data types summary
    dtypes_summary = df.dtypes.value_counts()

    # Numeric columns stats
    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
    numeric_desc = df[numeric_cols].describe() if len(numeric_cols) > 0 else None

    # Categorical columns top values and cardinality
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns
    cat_top_values = {}  # ← Moved here (before any use)
    cat_cardinality = {}

    for col in categorical_cols:
        top = df[col].value_counts().head(5)
        cat_top_values[col] = top.to_dict()
        cat_cardinality[col] = df[col].nunique()

    # Target column guess
    potential_targets = [
        col for col in df.columns if col.lower() in ['income', 'target', 'label', 'y', 'class']
    ]
    target_hint = potential_targets[0] if potential_targets else None

    # Build description
    description_parts = [
        f'Dataset shape: {shape[0]} rows, {shape[1]} columns',
        f'Total missing values: {total_missing}',
    ]

    if len(missing_per_column) > 0:
        description_parts.append('Columns with missing values:')
        for col, count in missing_per_column.items():
            pct = count / len(df) * 100
            description_parts.append(f'  - {col}: {count} ({pct:.1f}%)')

    description_parts.append('\nCategorical columns cardinality:')
    description_parts.append(json.dumps(cat_cardinality, indent=2))

    description_parts.append('\nData types summary:')
    for dtype, count in dtypes_summary.items():
        description_parts.append(f'  - {dtype}: {count} columns')

    if numeric_desc is not None and not numeric_desc.empty:
        description_parts.append('\nNumeric columns summary:')
        description_parts.append(numeric_desc.round(2).to_string())

    if cat_top_values:
        description_parts.append('\nTop values in categorical columns (top 5):')
        description_parts.append(json.dumps(cat_top_values, indent=2))

    if target_hint:
        unique = df[target_hint].nunique()
        description_parts.append(
            f"\nPotential target column detected: '{target_hint}' with {unique} unique values"
        )

    description_parts.append('\nTask: Tabular classification')

    return '\n'.join(description_parts)


# Quick test
if __name__ == '__main__':
    desc = generate_dataset_description('data/wine.csv')
    print(desc)

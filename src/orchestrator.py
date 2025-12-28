import json

import pandas as pd


def main():
    print('Orchestrator running...')


def generate_dataset_description(csv_path: str) -> str:
    df = pd.read_csv(csv_path)

    # Basic info
    shape = df.shape
    dtypes = df.dtypes.to_string()
    missing = df.isnull().sum().to_string()

    # Numeric stats
    describe = df.describe().to_string()

    # Categorical value counts (shortened to top 5 per col)
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns
    value_counts = {}
    for col in categorical_cols:
        counts = df[col].value_counts().head(5).to_string()  # Limit to top 5
        value_counts[col] = counts

    # Combine
    description = f"""
Dataset shape: {shape}
Data types:
{dtypes}

Missing values per column:
{missing}

Summary statistics for numeric columns:
{describe}

Top value counts for categorical columns:
{json.dumps(value_counts, indent=2)}
"""

    return description


if __name__ == '__main__':
    main()

# data/load_datasets.py
"""
Downloads 10 OpenML datasets, saves to data/raw/, returns
(df, target_column, problem_statement) tuples for the experiment runner.
"""
import os
from dataclasses import dataclass

import pandas as pd

# Optional: install openml if not present
try:
    import openml
except ImportError:
    raise ImportError("Run: uv pip install openml")

RAW_DIR = os.path.join(os.path.dirname(__file__), "raw")
os.makedirs(RAW_DIR, exist_ok=True)

# Registry of all 10 datasets
# (openml_id, dataset_id_slug, target_column, problem_statement)
DATASET_REGISTRY = [
    (
        61,
        "iris",
        "class",
        "Predict the species of an iris flower from sepal and petal measurements.",
    ),
    (
        187,
        "wine",
        "class",
        "Predict the wine cultivar class from chemical analysis measurements.",
    ),
    (
        15,
        "breast_cancer",
        "Class",
        "Predict whether a breast tumor is malignant or benign from cell nucleus features.",
    ),
    (
        37,
        "diabetes",
        "class",
        "Predict whether a patient has diabetes based on diagnostic measurements.",
    ),
    (
        31,
        "german_credit",
        "class",
        "Predict whether a loan applicant is a good or bad credit risk.",
    ),
    (
        54,
        "vehicle",
        "Class",
        "Predict the type of vehicle silhouette from shape features.",
    ),
    (
        40975,
        "car_evaluation",
        "class",
        "Predict the acceptability of a car based on price, comfort, and safety attributes.",
    ),
    (
        1590,
        "adult",
        "class",
        "Predict whether an individual earns more than $50K per year from census data.",
    ),
    (
        1461,
        "bank_marketing",
        "Class",
        "Predict whether a client will subscribe to a term deposit based on marketing call data.",
    ),
    (
        40691,
        "jungle_chess",
        "class",
        "Predict the outcome of a Jungle Chess game position.",
    ),
]


@dataclass
class DatasetInfo:
    dataset_id: str          # slug used as folder/file name
    openml_id: int
    csv_path: str
    target_column: str
    problem_statement: str
    n_rows: int
    n_cols: int


def download_all(force: bool = False) -> list[DatasetInfo]:
    """
    Download all datasets from OpenML and save to data/raw/.
    Returns list of DatasetInfo for the experiment runner.
    Set force=True to re-download even if file exists.
    """
    results = []

    for openml_id, slug, target_col, problem_stmt in DATASET_REGISTRY:
        csv_path = os.path.join(RAW_DIR, f"{slug}.csv")

        if os.path.exists(csv_path) and not force:
            print(f"  [{slug}] already exists — skipping download.")
            df = pd.read_csv(csv_path)
        else:
            print(f"  [{slug}] downloading from OpenML (id={openml_id})...")
            try:
                dataset = openml.datasets.get_dataset(
                    openml_id,
                    download_data=True,
                    download_qualities=False,
                    download_features_meta_data=False,
                )
                X, y, _, _ = dataset.get_data(target=dataset.default_target_attribute)
                df = X.copy()
                df[target_col] = y.values
                df.to_csv(csv_path, index=False)
                print(f"  [{slug}] saved to {csv_path} ({len(df)} rows).")
            except Exception as e:
                print(f"  [{slug}] FAILED: {e}")
                continue

        results.append(DatasetInfo(
            dataset_id=slug,
            openml_id=openml_id,
            csv_path=csv_path,
            target_column=target_col,
            problem_statement=problem_stmt,
            n_rows=len(df),
            n_cols=len(df.columns),
        ))

    return results


def get_dataset(slug: str) -> DatasetInfo:
    """Get a single DatasetInfo by slug. Downloads if not cached."""
    match = [d for d in DATASET_REGISTRY if d[1] == slug]
    if not match:
        raise ValueError(f"Unknown dataset slug: {slug}. "
                         f"Available: {[d[1] for d in DATASET_REGISTRY]}")
    openml_id, slug, target_col, problem_stmt = match[0]
    csv_path = os.path.join(RAW_DIR, f"{slug}.csv")

    if not os.path.exists(csv_path):
        print(f"[{slug}] not found locally — downloading...")
        download_all()

    df = pd.read_csv(csv_path)
    return DatasetInfo(
        dataset_id=slug,
        openml_id=openml_id,
        csv_path=csv_path,
        target_column=target_col,
        problem_statement=problem_stmt,
        n_rows=len(df),
        n_cols=len(df.columns),
    )


if __name__ == "__main__":
    print("Downloading all datasets...\n")
    datasets = download_all()
    print(f"\nDownloaded {len(datasets)} datasets:\n")
    for d in datasets:
        print(f"  {d.dataset_id:20s} {d.n_rows:6d} rows  {d.n_cols:3d} cols  → {d.csv_path}")
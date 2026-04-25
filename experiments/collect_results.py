# experiments/collect_results.py
"""
Reads all run_*.json artifacts and outputs results/results_table.csv.

CSV columns:
    run_id, dataset, mode, run_index, plan_valid, execution_success,
    best_score, n_iterations, validator_issues_caught, solved, duration_seconds

Usage:
    python -m experiments.collect_results

Expects artifacts at:
    experiments/artifacts/{dataset}/{mode}/run_{i}.json
"""

from __future__ import annotations

import json
import os

import pandas as pd

ARTIFACTS_DIR = os.path.join(os.path.dirname(__file__), "artifacts")
RESULTS_DIR   = os.path.join(os.path.dirname(__file__), "..", "results")
OUTPUT_CSV    = os.path.join(RESULTS_DIR, "results_table.csv")

COLUMNS = [
    "run_id",
    "dataset",
    "mode",
    "run_index",
    "plan_valid",
    "execution_success",
    "best_score",
    "n_iterations",
    "validator_issues_caught",
    "solved",
    "duration_seconds",
]


def _parse_artifact(path: str, dataset: str, mode: str, run_index: int) -> dict:
    """Extract flat metrics from a RunArtifact JSON file."""
    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    iterations = data.get("iterations", [])

    # plan_valid: at least one iteration has a non-stub plan
    # (stub plans have reasoning == "stub — baseline parse failed")
    plan_valid = int(any(
        it.get("plan", {}).get("reasoning", "") != "stub — baseline parse failed"
        for it in iterations
    ))

    # execution_success: at least one model succeeded in at least one iteration
    execution_success = 0
    for it in iterations:
        results = it.get("results", {})
        if any("error" not in info for info in results.values()):
            execution_success = 1
            break

    # best_score: max across all iterations
    best_score = 0.0
    for it in iterations:
        results = it.get("results", {})
        for info in results.values():
            score = info.get("mean_score", 0.0)
            if score > best_score:
                best_score = score

    # stored in agent_calls output_summary — parse from string
    # validator_issues_caught: count issues from PlanValidatorAgent output_summary
    validator_issues = 0
    for it in iterations:
        for call in it.get("agent_calls", []):
            if call.get("agent_name") == "PlanValidatorAgent":
                summary = call.get("output_summary", "")
                # format: "passed=False, issues=[...]"  or  "passed=True, issues=[]"
                if "passed=False" in summary:
                    # count comma-separated items inside issues=[...]
                    try:
                        start = summary.index("issues=[") + len("issues=[")
                        end   = summary.rindex("]")
                        inner = summary[start:end].strip()
                        if inner:
                            # each issue is a quoted string — count by splitting on '", "'
                            validator_issues += inner.count('Rule')
                    except Exception:
                        validator_issues += 1  # at least one issue caused the failure

    return {
        "run_id":                  data.get("run_id", ""),
        "dataset":                 dataset,
        "mode":                    mode,
        "run_index":               run_index,
        "plan_valid":              plan_valid,
        "execution_success":       execution_success,
        "best_score":              round(best_score, 6),
        "n_iterations":            len(iterations),
        "validator_issues_caught": validator_issues,
        "solved":                  int(data.get("solved", False)),
        "duration_seconds":        round(data.get("total_duration_seconds", 0.0), 3),
    }


def collect() -> pd.DataFrame:
    """Walk artifacts dir, parse every run JSON, return DataFrame."""
    rows: list[dict] = []

    if not os.path.isdir(ARTIFACTS_DIR):
        print(f"No artifacts directory found at {ARTIFACTS_DIR}")
        return pd.DataFrame(columns=COLUMNS)

    for dataset in sorted(os.listdir(ARTIFACTS_DIR)):
        dataset_dir = os.path.join(ARTIFACTS_DIR, dataset)
        if not os.path.isdir(dataset_dir):
            continue

        for mode in sorted(os.listdir(dataset_dir)):
            mode_dir = os.path.join(dataset_dir, mode)
            if not os.path.isdir(mode_dir):
                continue

            for filename in sorted(os.listdir(mode_dir)):
                if not filename.startswith("run_") or not filename.endswith(".json"):
                    continue

                run_index = int(filename.replace("run_", "").replace(".json", ""))
                path      = os.path.join(mode_dir, filename)

                try:
                    row = _parse_artifact(path, dataset, mode, run_index)
                    rows.append(row)
                    print(f"  parsed {dataset}/{mode}/{filename}")
                except Exception as e:
                    print(f"  FAILED {dataset}/{mode}/{filename}: {e}")

    df = pd.DataFrame(rows, columns=COLUMNS)
    df = df.sort_values(["dataset", "mode", "run_index"]).reset_index(drop=True)
    return df


if __name__ == "__main__":
    os.makedirs(RESULTS_DIR, exist_ok=True)

    print("Collecting results from artifacts...\n")
    df = collect()

    if df.empty:
        print("No artifacts found — run experiments first.")
    else:
        df.to_csv(OUTPUT_CSV, index=False)
        print(f"\nWrote {len(df)} rows to {OUTPUT_CSV}")
        print(f"\nDatasets:  {df['dataset'].nunique()}")
        print(f"Modes:     {df['mode'].unique().tolist()}")
        print(f"Runs/cell: {df.groupby(['dataset','mode']).size().value_counts().to_dict()}")
        print(f"\nPreview:\n{df.head(10).to_string(index=False)}")
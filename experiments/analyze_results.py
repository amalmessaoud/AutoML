# experiments/analyze_results.py
"""
Loads results/results_table.csv and computes the paper's Table 1.

Metrics computed per dataset × mode:
  - Plan Validity Rate       % runs where plan_valid=1
  - Execution Success Rate   % runs where execution_success=1
  - Mean Score ± Std         mean and std of best_score across runs
  - Solve Rate               % runs where solved=1
  - Mean Iterations          mean n_iterations
  - Validator Issues Caught  sum of validator_issues_caught (schema only)

Summary row aggregates across all datasets.

Usage:
    python -m experiments.analyze_results
"""

from __future__ import annotations

import os
import pandas as pd
import numpy as np

RESULTS_CSV = os.path.join(os.path.dirname(__file__), '..', 'results', 'results_table.csv')


def load() -> pd.DataFrame:
    df = pd.read_csv(RESULTS_CSV)
    assert set(['dataset', 'mode', 'plan_valid', 'execution_success',
                'best_score', 'n_iterations', 'validator_issues_caught',
                'solved']).issubset(df.columns), "results_table.csv missing expected columns"
    return df


def compute_table(df: pd.DataFrame) -> pd.DataFrame:
    rows = []

    datasets = sorted(df['dataset'].unique())
    modes = ['schema', 'baseline']

    for dataset in datasets:
        for mode in modes:
            sub = df[(df['dataset'] == dataset) & (df['mode'] == mode)]
            if sub.empty:
                continue

            n = len(sub)
            # Only count scores from successful runs for mean/std
            successful_scores = sub[sub['execution_success'] == 1]['best_score']

            rows.append({
                'dataset':              dataset,
                'mode':                 mode,
                'n_runs':               n,
                'plan_validity_%':      round(sub['plan_valid'].mean() * 100, 1),
                'exec_success_%':       round(sub['execution_success'].mean() * 100, 1),
                'mean_score':           round(successful_scores.mean(), 4) if len(successful_scores) > 0 else float('nan'),
                'std_score':            round(successful_scores.std(), 4) if len(successful_scores) > 1 else float('nan'),
                'solve_rate_%':         round(sub['solved'].mean() * 100, 1),
                'mean_iterations':      round(sub['n_iterations'].mean(), 2),
                'validator_issues':     int(sub['validator_issues_caught'].sum()),
            })

    return pd.DataFrame(rows)


def compute_summary(table: pd.DataFrame) -> pd.DataFrame:
    """Aggregate row across all datasets per mode."""
    rows = []
    for mode in ['schema', 'baseline']:
        sub = table[table['mode'] == mode]
        if sub.empty:
            continue
        rows.append({
            'dataset':          'ALL (mean)',
            'mode':             mode,
            'n_runs':           int(sub['n_runs'].sum()),
            'plan_validity_%':  round(sub['plan_validity_%'].mean(), 1),
            'exec_success_%':   round(sub['exec_success_%'].mean(), 1),
            'mean_score':       round(sub['mean_score'].mean(skipna=True), 4),
            'std_score':        round(sub['std_score'].mean(skipna=True), 4),
            'solve_rate_%':     round(sub['solve_rate_%'].mean(), 1),
            'mean_iterations':  round(sub['mean_iterations'].mean(), 2),
            'validator_issues': int(sub['validator_issues'].sum()),
        })
    return pd.DataFrame(rows)


def compute_variance_ratio(df: pd.DataFrame) -> pd.DataFrame:
    """
    Per dataset: baseline_std / schema_std on best_score.
    Ratio > 1 means schema is more consistent. This is Table 2 in the paper.
    """
    rows = []
    for dataset in sorted(df['dataset'].unique()):
        for mode in ['schema', 'baseline']:
            sub = df[(df['dataset'] == dataset) & (df['mode'] == mode)]
            successful = sub[sub['execution_success'] == 1]['best_score']
            std = successful.std() if len(successful) > 1 else float('nan')
            rows.append({'dataset': dataset, 'mode': mode, 'score_std': std})

    pivot = pd.DataFrame(rows).pivot(index='dataset', columns='mode', values='score_std')
    pivot['variance_ratio (baseline_std/schema_std)'] = round(
        pivot['baseline'] / pivot['schema'], 3
    )
    return pivot


def print_tables(table: pd.DataFrame, summary: pd.DataFrame, variance: pd.DataFrame) -> None:
    separator = '-' * 100

    print('\n' + '=' * 100)
    print('TABLE 1 — Per-dataset results (copy-paste into paper)')
    print('=' * 100)

    # Print schema and baseline side by side per dataset
    datasets = sorted(table['dataset'].unique())
    header = (
        f"{'Dataset':<18} {'Mode':<10} {'PlanValid%':>11} {'ExecOK%':>9} "
        f"{'MeanScore':>10} {'StdScore':>9} {'SolveRate%':>11} "
        f"{'MeanIter':>9} {'ValidIssues':>12}"
    )
    print(header)
    print(separator)

    for dataset in datasets:
        for mode in ['schema', 'baseline']:
            row = table[(table['dataset'] == dataset) & (table['mode'] == mode)]
            if row.empty:
                continue
            r = row.iloc[0]
            print(
                f"{r['dataset']:<18} {r['mode']:<10} "
                f"{r['plan_validity_%']:>11} {r['exec_success_%']:>9} "
                f"{r['mean_score']:>10.4f} {r['std_score']:>9.4f} "
                f"{r['solve_rate_%']:>11} {r['mean_iterations']:>9.2f} "
                f"{r['validator_issues']:>12}"
            )
        print(separator)

    print('\nSUMMARY (mean across datasets):')
    print(separator)
    for _, r in summary.iterrows():
        print(
            f"{r['dataset']:<18} {r['mode']:<10} "
            f"{r['plan_validity_%']:>11} {r['exec_success_%']:>9} "
            f"{r['mean_score']:>10.4f} {r['std_score']:>9.4f} "
            f"{r['solve_rate_%']:>11} {r['mean_iterations']:>9.2f} "
            f"{r['validator_issues']:>12}"
        )

    print('\n' + '=' * 100)
    print('TABLE 2 — Score variance ratio (baseline_std / schema_std)')
    print('Higher = schema is more consistent than baseline')
    print('=' * 100)
    print(variance.to_string())

    print('\n' + '=' * 100)
    print('KEY NUMBERS FOR PAPER')
    print('=' * 100)
    schema_row  = summary[summary['mode'] == 'schema'].iloc[0]
    baseline_row = summary[summary['mode'] == 'baseline'].iloc[0]

    print(f"  Plan Validity Rate:      schema={schema_row['plan_validity_%']}%  "
          f"baseline={baseline_row['plan_validity_%']}%  "
          f"Δ={schema_row['plan_validity_%'] - baseline_row['plan_validity_%']:+.1f}pp")
    print(f"  Exec Success Rate:       schema={schema_row['exec_success_%']}%  "
          f"baseline={baseline_row['exec_success_%']}%  "
          f"Δ={schema_row['exec_success_%'] - baseline_row['exec_success_%']:+.1f}pp")
    print(f"  Solve Rate:              schema={schema_row['solve_rate_%']}%  "
          f"baseline={baseline_row['solve_rate_%']}%  "
          f"Δ={schema_row['solve_rate_%'] - baseline_row['solve_rate_%']:+.1f}pp")
    print(f"  Mean Score:              schema={schema_row['mean_score']:.4f}  "
          f"baseline={baseline_row['mean_score']:.4f}  "
          f"Δ={schema_row['mean_score'] - baseline_row['mean_score']:+.4f}")
    print(f"  Total Validator Issues:  {schema_row['validator_issues']} caught by schema validator")
    print(f"  Mean Iterations:         schema={schema_row['mean_iterations']:.2f}  "
          f"baseline={baseline_row['mean_iterations']:.2f}")


if __name__ == '__main__':
    df = load()
    table    = compute_table(df)
    summary  = compute_summary(table)
    variance = compute_variance_ratio(df)

    print_tables(table, summary, variance)

    # Save tables for paper
    results_dir = os.path.join(os.path.dirname(__file__), '..', 'results')
    table.to_csv(os.path.join(results_dir, 'table1.csv'), index=False)
    variance.to_csv(os.path.join(results_dir, 'table2.csv'))
    print(f"\nSaved table1.csv and table2.csv to results/")
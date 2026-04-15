from typing import Dict, List, Optional
from src.utils.data_utils import generate_dataset_description
from src.agents.analyzer import generate_automl_plan
from src.agents.implementation import execute_plan
from src.agents.critique import evaluate_results
from src.schemas.plan import AutoMLPlan
from dotenv import load_dotenv
load_dotenv()


def run_automl_pipeline(csv_path: str, problem: str, max_iterations: int = 3) -> Dict:
    """
    Run the full AutoML pipeline with critique loop.

    Returns: {
        'final_plan': AutoMLPlan,
        'final_results': Dict from implementation,
        'final_evaluation': Dict from critique,
        'logs': List[str],
        'iterations': int
    }
    """
    logs: List[str] = []
    current_problem = problem
    final_plan: Optional[AutoMLPlan] = None
    final_results: Optional[Dict] = None
    final_evaluation: Optional[Dict] = None

    for iteration in range(1, max_iterations + 1):
        logs.append(f'==== ITERATION {iteration} ====')

        # 1. Generate description
        desc = generate_dataset_description(csv_path)
        logs.append('Generated dataset description.')

        # 2. Generate plan
        plan = generate_automl_plan(desc, current_problem, logs=logs)
        final_plan = plan
        logs.append('Generated AutoML plan.')

        # 3. Execute plan
        results = execute_plan(plan, csv_path, logs=logs)
        final_results = results
        logs.append('Executed plan and got results.')

        # 4. Evaluate results
        evaluation = evaluate_results(plan, results, current_problem, logs=logs)
        final_evaluation = evaluation
        logs.append('Evaluated results.')

        # Check if solved
        if evaluation['solved']:
            logs.append('Problem solved — ending pipeline.')
            return {
                'final_plan': final_plan,
                'final_results': final_results,
                'final_evaluation': final_evaluation,
                'logs': logs,
                'iterations': iteration,
            }

        # Not solved — prepare feedback for next iteration
        logs.append('Not solved — preparing feedback for retry.')
        feedback = evaluation['suggestion']
        current_problem = f'{problem}\nPrevious attempt feedback: {feedback}'

    logs.append('Max iterations reached without solution.')
    return {
        'final_plan': final_plan,
        'final_results': final_results,
        'final_evaluation': final_evaluation,
        'logs': logs,
        'iterations': max_iterations,
    }


# Quick test
if __name__ == '__main__':
    csv_path = 'data/bank.csv'
    problem = 'Predict whether a client will subscribe to a term deposit.'
    output = run_automl_pipeline(csv_path, problem)
    print('\nFinal Logs:\n')
    print('\n'.join(output['logs']))
    print(
        f'\nBest Model: {output["final_evaluation"]["best_model"]} '
        f'with score {output["final_evaluation"]["best_score"]:.4f}'
    )
    print(f'Solved: {output["final_evaluation"]["solved"]}')

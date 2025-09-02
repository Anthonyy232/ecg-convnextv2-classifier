"""
Hyperparameter Optimization (HPO) for the ECG classification model.
"""

import os
import optuna
import argparse
from datetime import datetime

from config import *
from training_pipeline import train_model


def objective(trial: optuna.trial.Trial) -> float:
    """
    The objective function for Optuna to optimize.

    This function defines the hyperparameter search space and executes a
    training run for each trial, returning the performance metric to be maximized.

    Args:
        trial: An Optuna trial object.

    Returns:
        The validation F1-score achieved in the trial.
    """
    hyperparams = {
        'learning_rate': trial.suggest_float('learning_rate', 1e-5, 1e-2, log=True),
        'batch_size': trial.suggest_categorical('batch_size', [8, 16, 32, 64]),
        'focal_gamma': trial.suggest_float('focal_gamma', 1.0, 3.5),
        'epochs': HPO_MAX_EPOCHS
    }
    
    print("\n" + "="*50)
    print(f"STARTING HPO TRIAL {trial.number}")
    print(f"Hyperparameters: {hyperparams}")
    print("="*50)

    try:
        f1_score = train_model(hyperparams=hyperparams, return_f1=True, trial_num=trial.number)
        print(f"\n--- HPO Trial {trial.number} Finished. Score (Mean F1): {f1_score:.4f} ---")
        return f1_score
    except Exception as e:
        print(f"Trial {trial.number} failed with error: {e}")
        return 0.0


def run_hpo(study_name: str, n_trials: int = 20):
    """
    Initializes and runs the hyperparameter optimization study.

    Args:
        study_name: The name for the study. This will be used for the database file
                    to allow for resuming.
        n_trials: The total number of optimization trials to run.
    """
    os.makedirs(RESULTS_DIRECTORY, exist_ok=True)
    
    # Use a fixed database file based on the study name.
    storage_name = f"sqlite:///{os.path.join(RESULTS_DIRECTORY, f'{study_name}.db')}"
    
    print(f"Using study name: '{study_name}'")
    print(f"Using storage: '{storage_name}'")

    study = optuna.create_study(
        direction='maximize',
        study_name=study_name,
        storage=storage_name,
        load_if_exists=True
    )
    
    # Enqueue the default parameters only if it's a brand new study.
    if len(study.trials) == 0:
        print("This is a new study. Enqueuing default parameters as the first trial.")
        study.enqueue_trial({
            'learning_rate': LEARNING_RATE,
            'batch_size': BATCH_SIZE,
            'focal_gamma': FOCAL_LOSS_GAMMA
        })
    else:
        print(f"Resuming study. Already completed {len(study.trials)} trials.")
        
    # Calculate remaining trials
    remaining_trials = n_trials - len(study.trials)
    if remaining_trials <= 0:
        print(f"Study has already completed {len(study.trials)} trials, which meets or exceeds the target of {n_trials}. Nothing to do.")
        return
        
    print(f"Starting HPO study. Running {remaining_trials} more trials to reach a total of {n_trials}.")
    study.optimize(objective, n_trials=remaining_trials)
    
    print("\nHYPERPARAMETER OPTIMIZATION COMPLETE")
    print(f"Best trial F1-score: {study.best_value:.4f}")
    print("Best hyperparameters:")
    for key, value in study.best_params.items():
        print(f"  {key}: {value}")
    
    # Save results to a file, overwriting with the latest best results.
    results_file = os.path.join(RESULTS_DIRECTORY, f"hpo_results_{study_name}.txt")
    with open(results_file, 'w') as f:
        f.write(f"Study: {study_name}\n")
        f.write(f"Best F1-score: {study.best_value}\n")
        f.write("Best hyperparameters:\n")
        for key, value in study.best_params.items():
            f.write(f"  {key}: {value}\n")
    print(f"Results saved to: {results_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Hyperparameter Optimization for ECG Model.")
    parser.add_argument(
        "--study-name", 
        type=str, 
        default="ecg_hpo_study", 
        help="The name of the Optuna study. Used for resuming."
    )
    parser.add_argument(
        "--n-trials", 
        type=int, 
        default=10, 
        help="The total number of trials to run for the study."
    )
    args = parser.parse_args()
    
    run_hpo(study_name=args.study_name, n_trials=args.n_trials)
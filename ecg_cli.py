"""
Command-line interface for the ECG arrhythmia detection system.
"""

import argparse
import logging
import os
import subprocess
import sys

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def install_dependencies() -> int:
    """Installs required Python dependencies from requirements.txt."""
    logger.info("Installing dependencies...")
    result = subprocess.run(
        [sys.executable, "-m", "pip", "install", "-r", "requirements.txt"],
        check=False
    )
    return result.returncode


def run_installation_test() -> int:
    """Executes the system verification and integration tests."""
    logger.info("Running installation verification tests...")
    result = subprocess.run(
        [sys.executable, "system_verification.py"],
        cwd=os.path.dirname(__file__),
        check=False
    )
    return result.returncode


def run_model_training() -> int:
    """Executes the model training pipeline."""
    logger.info("Starting model training...")
    src_dir = os.path.join(os.path.dirname(__file__), "src")
    
    # Set CUDA debugging flags for improved stability in case of GPU errors.
    env = os.environ.copy()
    env['CUDA_LAUNCH_BLOCKING'] = '1'
    env['TORCH_USE_CUDA_DSA'] = '1'
    
    result = subprocess.run(
        [sys.executable, "training_pipeline.py"],
        cwd=src_dir,
        env=env,
        check=False
    )
    return result.returncode


def run_hyperparameter_optimization() -> int:
    """Executes the hyperparameter optimization process using Optuna."""
    logger.info("Starting hyperparameter optimization...")
    src_dir = os.path.join(os.path.dirname(__file__), "src")

    env = os.environ.copy()
    env['CUDA_LAUNCH_BLOCKING'] = '1'
    env['TORCH_USE_CUDA_DSA'] = '1'
    
    result = subprocess.run(
        [sys.executable, "hpo.py"],
        cwd=src_dir,
        env=env,
        check=False
    )
    return result.returncode


def run_model_evaluation() -> int:
    """Executes the evaluation script on the trained model."""
    logger.info("Running model evaluation on the test set...")
    src_dir = os.path.join(os.path.dirname(__file__), "src")
    
    result = subprocess.run(
        [sys.executable, "evaluate.py"],
        cwd=src_dir,
        check=False
    )
    return result.returncode


def main():
    """Parses command-line arguments and executes the corresponding command."""
    parser = argparse.ArgumentParser(
        description="ECG Arrhythmia Detection System CLI.",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "command",
        choices=["install", "test", "train", "eval", "hpo"],
        help="The command to execute."
    )
    
    args = parser.parse_args()
    
    command_map = {
        "install": install_dependencies,
        "test": run_installation_test,
        "train": run_model_training,
        "eval": run_model_evaluation,
        "hpo": run_hyperparameter_optimization
    }
    
    try:
        exit_code = command_map[args.command]()
        sys.exit(exit_code)
    except KeyboardInterrupt:
        logger.info("Operation cancelled by user.")
        sys.exit(1)
    except Exception as e:
        logger.error(f"An unexpected error occurred while running command '{args.command}': {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
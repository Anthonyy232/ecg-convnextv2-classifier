"""
Installation verification and system integration tests.
"""

import logging
import os
import sys
import warnings
import numpy as np
import torch

warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from config import NUM_CLASSES, S_TRANSFORM_HEIGHT, WINDOW_SIZE_SAMPLES
from model import create_ecg_model
from signal_processing import perform_s_transform
from dataset import ECGDataset

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def test_dependency_imports() -> bool:
    """Validates that all critical dependencies can be imported."""
    logger.info("Testing dependency imports...")
    required_modules = ['torch', 'numpy', 'scipy', 'sklearn', 'tqdm', 'matplotlib']
    try:
        for module_name in required_modules:
            __import__(module_name)
        logger.info("✓ All required dependencies imported successfully.")
        return True
    except ImportError as e:
        logger.error(f"✗ Missing dependency: {e}")
        return False


def test_s_transform_functionality() -> bool:
    """Validates the S-Transform signal processing functionality."""
    logger.info("Testing S-Transform signal processing...")
    try:
        synthetic_signal = np.random.randn(WINDOW_SIZE_SAMPLES)
        transform_result = perform_s_transform(
            synthetic_signal, 0, S_TRANSFORM_HEIGHT - 1, 100
        )
        
        expected_shape = (S_TRANSFORM_HEIGHT, WINDOW_SIZE_SAMPLES)
        assert transform_result.shape == expected_shape, \
            f"Shape mismatch: expected {expected_shape}, got {transform_result.shape}"
        
        logger.info(f"✓ S-Transform output shape validated: {transform_result.shape}")
        return True
    except Exception as e:
        logger.error(f"✗ S-Transform test failed: {e}")
        return False


def test_data_pipeline_integration() -> bool:
    """Validates the data pipeline from dataset creation to model forward pass."""
    logger.info("Testing data pipeline integration...")
    try:
        # Test Dataset
        batch_size = 10
        synthetic_signals = np.random.randn(
            batch_size, 2, S_TRANSFORM_HEIGHT, WINDOW_SIZE_SAMPLES
        ).astype(np.float32)
        synthetic_labels = np.random.randint(0, NUM_CLASSES, batch_size).astype(np.int64)
        
        dataset = ECGDataset(synthetic_signals, synthetic_labels)
        sample_tensor, _ = dataset[0]
        
        expected_tensor_shape = (2, S_TRANSFORM_HEIGHT, WINDOW_SIZE_SAMPLES)
        assert sample_tensor.shape == expected_tensor_shape, \
            f"Dataset output shape mismatch: expected {expected_tensor_shape}, got {sample_tensor.shape}"
        logger.info(f"✓ Dataset output validated: {sample_tensor.shape}")
        
        # Test Model Forward Pass
        model = create_ecg_model(NUM_CLASSES)
        model.eval()
        
        test_batch_size = 4
        test_input = torch.randn(test_batch_size, 2, S_TRANSFORM_HEIGHT, WINDOW_SIZE_SAMPLES)
        with torch.no_grad():
            model_output = model(test_input)
        
        expected_output_shape = (test_batch_size, NUM_CLASSES)
        assert model_output.shape == expected_output_shape, \
            f"Model output shape mismatch: expected {expected_output_shape}, got {model_output.shape}"
        logger.info(f"✓ Model forward pass validated: {model_output.shape}")
        return True
    except Exception as e:
        logger.error(f"✗ Data pipeline test failed: {e}")
        return False


def run_system_validation():
    """Executes all validation tests and provides a summary."""
    logger.info("Starting ECG System Installation Validation")
    logger.info("=" * 50)
    
    tests = [
        test_dependency_imports,
        test_s_transform_functionality,
        test_data_pipeline_integration
    ]
    
    passed_count = sum(1 for test in tests if test())
    total_count = len(tests)
    
    logger.info("-" * 50)
    if passed_count == total_count:
        logger.info(f"✅ All {total_count} validation tests passed.")
        logger.info("System is ready for operation.")
        return True
    else:
        failed_count = total_count - passed_count
        logger.error(f"❌ {failed_count} out of {total_count} tests failed.")
        logger.error("Please review errors and reinstall dependencies if needed.")
        return False


if __name__ == "__main__":
    is_successful = run_system_validation()
    sys.exit(0 if is_successful else 1)
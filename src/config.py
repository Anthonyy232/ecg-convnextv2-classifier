"""
Configuration settings for the ECG arrhythmia detection system.
"""

import os
import torch

# =============================================================================
# Project Structure
# =============================================================================
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIRECTORY = os.path.join(PROJECT_ROOT, "models")
BEST_MODEL_PATH = os.path.join(MODELS_DIRECTORY, "best_model.pth")
RESULTS_DIRECTORY = os.path.join(PROJECT_ROOT, "results")
DATA_DIRECTORY = os.path.join(PROJECT_ROOT, "data")

# =============================================================================
# Dataset Configuration
# =============================================================================
DATASET_NAME = 'mitdb'
RAW_SAMPLING_RATE = 360  # Original sampling rate of the MIT-BIH database (Hz).
TARGET_SAMPLING_RATE = 100  # Target sampling rate after resampling (Hz).
WINDOW_DURATION_SECONDS = 10  # Duration of each ECG segment.
WINDOW_SIZE_SAMPLES = WINDOW_DURATION_SECONDS * TARGET_SAMPLING_RATE

# AAMI standard patient splits for training and testing.
TRAINING_PATIENTS = [
    '101', '106', '108', '109', '112', '114', '115', '116', '118', '119',
    '122', '124', '201', '203', '205', '207', '208', '209', '215', '220',
    '223', '230'
]
TESTING_PATIENTS = [
    '100', '103', '105', '111', '113', '117', '121', '123', '200', '202',
    '210', '212', '213', '214', '219', '221', '222', '228', '231', '232',
    '233', '234'
]

# Patient splits designed for rare disease handling.
CORE_TRAINING_PATIENTS = [
    '101', '106', '108', '109', '112', '114', '115', '116', '118', '119',
    '122', '201', '203', '205', '209', '215', '220', '230'
]
RARE_CLASS_PATIENTS = ['207', '208']  # Patients known to contain rare classes.
STANDARD_VALIDATION_PATIENTS = ['124', '223']

# =============================================================================
# Signal Processing Parameters
# =============================================================================
LOW_PASS_CUTOFF_HZ = 30
BUTTERWORTH_FILTER_ORDER = 4
TARVAINEN_SMOOTHING_LAMBDA = 200  # For baseline wander removal.
S_TRANSFORM_FREQ_MIN_HZ = 0
S_TRANSFORM_FREQ_MAX_HZ = 40
S_TRANSFORM_HEIGHT = S_TRANSFORM_FREQ_MAX_HZ - S_TRANSFORM_FREQ_MIN_HZ + 1
S_TRANSFORM_WIDTH = WINDOW_SIZE_SAMPLES

# =============================================================================
# Class Configuration (AAMI Standard)
# =============================================================================
CLASS_LABELS = {'N': 0, 'S': 1, 'V': 2, 'F': 3, 'Q': 4}
CLASS_NAMES = ['Normal', 'Supraventricular', 'Ventricular', 'Fusion', 'Unknown']
NUM_CLASSES = len(CLASS_NAMES)

# =============================================================================
# Training and Model Configuration
# =============================================================================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 16
MAX_EPOCHS = 30
LEARNING_RATE = 2e-5
RANDOM_SEED = 42
EARLY_STOPPING_PATIENCE = 10
GRADIENT_CLIP_NORM = 0.5
USE_CROSS_VALIDATION = True
CROSS_VALIDATION_FOLDS = 4

# Hyperparameter Optimization (HPO) settings.
HPO_MAX_EPOCHS = 10

# Focal Loss parameters for handling class imbalance.
FOCAL_LOSS_GAMMA = 2.0
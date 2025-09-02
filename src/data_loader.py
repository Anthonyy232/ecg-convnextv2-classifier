"""
Data loading strategy to ensure rare classes are represented in splits.
"""

import numpy as np
import os
from sklearn.model_selection import train_test_split

from config import *
from preprocess import create_segmented_windows


def load_rare_disease_split(data_dir: str) -> tuple:
    """
    Creates train/validation splits with special handling for rare classes.

    Args:
        data_dir: The directory to cache the processed data.

    Returns:
        A tuple (X_train, y_train, X_val, y_val).
    """
    # Load core training and standard validation patients.
    X_train_core, y_train_core = create_segmented_windows(CORE_TRAINING_PATIENTS)
    X_val_standard, y_val_standard = create_segmented_windows(STANDARD_VALIDATION_PATIENTS)

    # Process and split patients with rare classes.
    X_train_rare_list, y_train_rare_list = [], []
    X_val_rare_list, y_val_rare_list = [], []

    for patient_id in RARE_CLASS_PATIENTS:
        X_patient, y_patient = create_segmented_windows([patient_id])
        
        # Stratified split of this patient's windows.
        if len(np.unique(y_patient)) > 1:
            X_p_train, X_p_val, y_p_train, y_p_val = train_test_split(
                X_patient, y_patient, test_size=0.3, random_state=RANDOM_SEED, stratify=y_patient
            )
        else: # Cannot stratify if only one class is present.
            X_p_train, X_p_val, y_p_train, y_p_val = train_test_split(
                X_patient, y_patient, test_size=0.3, random_state=RANDOM_SEED
            )
        
        X_train_rare_list.append(X_p_train)
        y_train_rare_list.append(y_p_train)
        X_val_rare_list.append(X_p_val)
        y_val_rare_list.append(y_p_val)

    # Combine all data segments.
    X_train = np.concatenate([X_train_core] + X_train_rare_list, axis=0)
    y_train = np.concatenate([y_train_core] + y_train_rare_list, axis=0)
    X_val = np.concatenate([X_val_standard] + X_val_rare_list, axis=0)
    y_val = np.concatenate([y_val_standard] + y_val_rare_list, axis=0)

    # Cache the generated splits for future runs.
    os.makedirs(data_dir, exist_ok=True)
    np.savez_compressed(os.path.join(data_dir, "train_rare_aware.npz"), x=X_train, y=y_train)
    np.savez_compressed(os.path.join(data_dir, "val_rare_aware.npz"), x=X_val, y=y_val)
    
    return X_train, y_train, X_val, y_val


def load_rare_aware_cached(data_dir: str) -> tuple or None:
    """
    Loads cached rare-disease-aware splits if they exist.

    Args:
        data_dir: The directory where data splits are cached.

    Returns:
        A tuple (X_train, y_train, X_val, y_val) if cache exists, otherwise None.
    """
    train_cache = os.path.join(data_dir, "train_rare_aware.npz")
    val_cache = os.path.join(data_dir, "val_rare_aware.npz")
    
    if os.path.exists(train_cache) and os.path.exists(val_cache):
        with np.load(train_cache) as data:
            X_train, y_train = data['x'], data['y']
        with np.load(val_cache) as data:
            X_val, y_val = data['x'], data['y']
        return X_train, y_train, X_val, y_val
    return None
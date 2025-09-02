"""
Data preprocessing pipeline for ECG signals from the MIT-BIH Arrhythmia Database.
"""

import os
import numpy as np
import warnings
from typing import List, Tuple
from scipy.signal import butter, filtfilt, resample_poly

try:
    import wfdb
    WFDB_AVAILABLE = True
except ImportError:
    WFDB_AVAILABLE = False
    warnings.warn("The 'wfdb' library is not installed. Cannot load MIT-BIH data.")

from config import *
from signal_processing import perform_s_transform


def load_mitbih_data(patient_ids: List[str]) -> List[Tuple[str, np.ndarray, object]]:
    """
    Loads raw signal and annotation data for specified MIT-BIH patient IDs.

    Args:
        patient_ids: A list of patient record IDs (e.g., ['100', '101']).

    Returns:
        A list of tuples, each containing (record_name, signal, annotations).

    Raises:
        ImportError: If the 'wfdb' library is not available.
        ValueError: If no data could be loaded for any of the specified patients.
    """
    if not WFDB_AVAILABLE:
        raise ImportError("wfdb library is required. Please install it via 'pip install wfdb'.")
    
    data = []
    for patient_id in patient_ids:
        try:
            record = wfdb.rdrecord(patient_id, pn_dir=DATASET_NAME)
            annotation = wfdb.rdann(patient_id, 'atr', pn_dir=DATASET_NAME)
            
            # Extract the MLII signal (typically channel 0).
            if record.p_signal is not None and record.p_signal.shape[1] > 0:
                signal_data = record.p_signal[:, 0]
                data.append((patient_id, signal_data, annotation))
            else:
                warnings.warn(f"No valid signal found for patient {patient_id}.")
        except Exception as e:
            warnings.warn(f"Failed to load data for patient {patient_id}: {e}")
            
    if not data:
        raise ValueError("No data could be loaded. Check patient IDs and network connection.")
    return data


def apply_filter(signal: np.ndarray, cutoff: float, fs: int, btype: str) -> np.ndarray:
    """Applies a Butterworth filter to a signal."""
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    b, a = butter(BUTTERWORTH_FILTER_ORDER, normal_cutoff, btype=btype, analog=False)
    return filtfilt(b, a, signal)


def detrend_signal(signal: np.ndarray) -> np.ndarray:
    """
    Removes baseline wander using the Tarvainen method with a high-pass filter fallback.
    
    For memory efficiency on very long signals, this function falls back to a
    simpler high-pass filter.
    """
    if len(signal) > 20000:
        return apply_filter(signal, 0.5, TARGET_SAMPLING_RATE, 'high')
    
    try:
        from scipy.sparse import csc_matrix, diags
        from scipy.sparse.linalg import spsolve
        
        n = len(signal)
        diagonals = [np.ones(n-2), -2*np.ones(n-2), np.ones(n-2)]
        D = diags(diagonals, [0, 1, 2], shape=(n-2, n), format='csc')
        H = csc_matrix(np.eye(n)) + TARVAINEN_SMOOTHING_LAMBDA * (D.T @ D)
        baseline = spsolve(H, signal)
        return signal - baseline
    except Exception as e:
        warnings.warn(f"Tarvainen detrending failed: {e}. Using high-pass filter fallback.")
        return apply_filter(signal, 0.5, TARGET_SAMPLING_RATE, 'high')


def normalize_signal(signal: np.ndarray) -> np.ndarray:
    """Applies Min-Max normalization to scale the signal to the [0, 1] range."""
    min_val, max_val = np.min(signal), np.max(signal)
    return (signal - min_val) / (max_val - min_val) if max_val > min_val else np.zeros_like(signal)


def map_beat_to_aami(beat_symbol: str) -> str:
    """Maps an MIT-BIH beat annotation to its corresponding AAMI standard class."""
    if beat_symbol in ['N', 'L', 'R', 'e', 'j']: return 'N'
    if beat_symbol in ['A', 'a', 'J', 'S']: return 'S'
    if beat_symbol in ['V', 'E']: return 'V'
    if beat_symbol in ['F']: return 'F'
    return 'Q'


def get_window_label(annotations, start_sample: int, end_sample: int) -> str:
    """Determines the label for a window by finding the most frequent beat type within it."""
    window_beats = [
        map_beat_to_aami(annotations.symbol[i])
        for i, sample in enumerate(annotations.sample)
        if start_sample <= sample < end_sample
    ]
    return max(set(window_beats), key=window_beats.count) if window_beats else 'Q'


def create_segmented_windows(patient_ids: List[str]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Processes raw ECG data for a list of patients into S-Transformed windows and labels.

    Args:
        patient_ids: List of patient IDs to process.

    Returns:
        A tuple containing:
        - X (np.ndarray): The S-Transformed data of shape [num_windows, 2, height, width].
        - y (np.ndarray): The corresponding labels.
    """
    patient_data = load_mitbih_data(patient_ids)
    all_windows, all_labels = [], []

    for record_name, raw_signal, annotations in patient_data:
        # Resample signal to the target frequency.
        resampled = resample_poly(raw_signal, TARGET_SAMPLING_RATE, RAW_SAMPLING_RATE)
        
        # Apply filters and detrending.
        filtered = apply_filter(resampled, LOW_PASS_CUTOFF_HZ, TARGET_SAMPLING_RATE, 'low')
        detrended = detrend_signal(filtered)
        
        # Segment the signal into windows.
        num_windows = len(detrended) // WINDOW_SIZE_SAMPLES
        for i in range(num_windows):
            start = i * WINDOW_SIZE_SAMPLES
            end = start + WINDOW_SIZE_SAMPLES
            window = detrended[start:end]
            
            # Normalize and compute S-Transform.
            normalized_window = normalize_signal(window)
            st_result = perform_s_transform(
                normalized_window, S_TRANSFORM_FREQ_MIN_HZ, S_TRANSFORM_FREQ_MAX_HZ, TARGET_SAMPLING_RATE
            )
            st_tensor = np.stack([st_result.real, st_result.imag], axis=0)
            
            # Determine the window's label.
            start_raw = int(start * (RAW_SAMPLING_RATE / TARGET_SAMPLING_RATE))
            end_raw = int(end * (RAW_SAMPLING_RATE / TARGET_SAMPLING_RATE))
            label = get_window_label(annotations, start_raw, end_raw)
            
            all_windows.append(st_tensor)
            all_labels.append(CLASS_LABELS[label])

    return np.array(all_windows, dtype=np.float32), np.array(all_labels, dtype=np.int64)
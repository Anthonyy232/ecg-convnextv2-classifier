"""
PyTorch Dataset implementation for ECG time-frequency data.
"""

import logging
import numpy as np
import torch
from torch.utils.data import Dataset
from typing import Tuple, Dict

from config import S_TRANSFORM_HEIGHT, WINDOW_SIZE_SAMPLES, CLASS_NAMES

logger = logging.getLogger(__name__)


class ECGDataset(Dataset):
    """
    A PyTorch Dataset for S-Transform processed ECG data.

    Args:
        signals (np.ndarray): Pre-processed S-Transform data of shape
                              [num_samples, 2, height, width].
        labels (np.ndarray): Corresponding classification labels.
    """
    def __init__(self, signals: np.ndarray, labels: np.ndarray):
        self.signals = signals
        self.labels = labels
        self._validate_inputs()
        logger.info(f"Dataset initialized with {len(self)} samples.")

    def _validate_inputs(self):
        """Ensures input data dimensions and consistency are correct."""
        if len(self.signals) != len(self.labels):
            raise ValueError("Signal and label counts must match.")
        if self.signals.ndim != 4 or self.signals.shape[1:] != (2, S_TRANSFORM_HEIGHT, WINDOW_SIZE_SAMPLES):
            raise ValueError(f"Invalid signal shape: {self.signals.shape}")

    def __len__(self) -> int:
        """Returns the total number of samples in the dataset."""
        return len(self.signals)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Retrieves a single sample from the dataset.

        Args:
            index: The index of the sample to retrieve.

        Returns:
            A tuple containing the signal tensor and its corresponding label tensor.
        """
        signal_tensor = torch.from_numpy(self.signals[index]).float()
        label_tensor = torch.tensor(self.labels[index], dtype=torch.long)
        return signal_tensor, label_tensor

    def get_class_distribution(self) -> Dict[str, int]:
        """
        Calculates and returns the class distribution of the dataset.

        Returns:
            A dictionary mapping class names to their counts.
        """
        unique, counts = np.unique(self.labels, return_counts=True)
        return {CLASS_NAMES[i]: count for i, count in zip(unique, counts)}
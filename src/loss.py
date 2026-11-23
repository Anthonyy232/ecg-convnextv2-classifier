"""
Loss functions for ECG arrhythmia classification.
"""

import logging
import torch
import torch.nn as nn
import torch.nn.functional as F

from config import NUM_CLASSES

logger = logging.getLogger(__name__)


class FocalLoss(nn.Module):
    """
    Focal Loss for addressing class imbalance in classification tasks.

    Args:
        alpha (torch.Tensor, optional): A tensor of weights for each class.
        gamma (float): The focusing parameter. Higher values give more weight
                       to hard-to-classify examples. Defaults to 2.0.
        reduction (str): Specifies the reduction to apply to the output:
                         'none' | 'mean' | 'sum'. Defaults to 'mean'.
    """
    def __init__(self, alpha=None, gamma: float = 2.0, reduction: str = 'mean'):
        super().__init__()
        self.gamma = gamma
        self.reduction = reduction
        self.alpha = alpha

    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Computes the focal loss.

        Args:
            predictions: Model predictions of shape [batch_size, num_classes].
            targets: Ground truth labels of shape [batch_size].

        Returns:
            The computed focal loss value.
        """
        # Ensure targets are within valid range
        if targets.max() >= predictions.size(1) or targets.min() < 0:
            raise ValueError(f"Target labels must be in range [0, {predictions.size(1)-1}], "
                           f"but got range [{targets.min()}, {targets.max()}]")
        
        ce_loss = F.cross_entropy(predictions, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = (1 - pt) ** self.gamma * ce_loss

        if self.alpha is not None:
            if self.alpha.device != predictions.device:
                self.alpha = self.alpha.to(predictions.device)
            
            # Ensure alpha has the correct size for all possible classes
            if self.alpha.size(0) != predictions.size(1):
                raise ValueError(f"Alpha tensor size ({self.alpha.size(0)}) must match "
                               f"number of classes ({predictions.size(1)})")
            
            alpha_t = self.alpha[targets]
            focal_loss = alpha_t * focal_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        if self.reduction == 'sum':
            return focal_loss.sum()
        return focal_loss
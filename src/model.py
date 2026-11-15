"""
ConvNeXt v2 architecture adapted for ECG arrhythmia detection.

LayerNorm, GlobalResponseNorm, ConvNeXtBlock and the ConvNeXtV2 backbone are
adapted from https://github.com/facebookresearch/ConvNeXt-V2 under the MIT
licence below; DropPath follows the drop_path of timm
(https://github.com/huggingface/pytorch-image-models, Apache-2.0).

    Copyright (c) Meta Platforms, Inc. and affiliates.

    Permission is hereby granted, free of charge, to any person obtaining a copy
    of this software and associated documentation files (the "Software"), to deal
    in the Software without restriction, including without limitation the rights
    to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
    copies of the Software, and to permit persons to whom the Software is
    furnished to do so, subject to the following conditions:

    The above copyright notice and this permission notice shall be included in all
    copies or substantial portions of the Software.

    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
    IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
    FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
    AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
    LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
    OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
    SOFTWARE.
"""

import logging
import torch
import torch.nn as nn
import torch.nn.functional as F

from config import NUM_CLASSES

logger = logging.getLogger(__name__)


class LayerNorm(nn.Module):
    """
    A LayerNorm implementation that supports both channels-last and channels-first
    data formats for 2D data.
    """
    def __init__(self, normalized_shape: int, eps: float = 1e-6, data_format: str = "channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if data_format not in ["channels_last", "channels_first"]:
            raise ValueError(f"Unsupported data_format: {data_format}")
        self.normalized_shape_tuple = (normalized_shape,)

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        if self.data_format == "channels_last":
            return F.layer_norm(input_tensor, self.normalized_shape_tuple, self.weight, self.bias, self.eps)
        
        # Calculate mean and variance across the channel dimension
        mean = input_tensor.mean(dim=1, keepdim=True)
        variance = (input_tensor - mean).pow(2).mean(dim=1, keepdim=True)
        
        normalized_tensor = (input_tensor - mean) / torch.sqrt(variance + self.eps)
        
        # The weight and bias are shaped for broadcasting over spatial dimensions (H, W)
        return self.weight.view(-1, 1, 1) * normalized_tensor + self.bias.view(-1, 1, 1)


class GlobalResponseNorm(nn.Module):
    """
    Global Response Normalization (GRN) layer.
    This layer is designed to enhance feature contrast and selectivity by normalizing
    feature maps based on their global L2-norm.
    """
    def __init__(self, num_channels: int):
        super().__init__()
        # Learnable parameters for scaling and shifting
        self.gamma = nn.Parameter(torch.zeros(1, 1, 1, num_channels))
        self.beta = nn.Parameter(torch.zeros(1, 1, 1, num_channels))

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        # Calculate the L2-norm for each feature map across spatial dimensions (H, W)
        # Shape: (N, H, W, C) -> (N, 1, 1, C)
        spatial_norm = torch.norm(input_tensor, p=2, dim=(1, 2), keepdim=True)
        
        # Calculate the mean of these norms across the channel dimension
        # Shape: (N, 1, 1, C) -> (N, 1, 1, 1)
        global_norm_mean = spatial_norm.mean(dim=-1, keepdim=True)
        
        norm_ratio = spatial_norm / (global_norm_mean + 1e-6)
        
        # Apply the GRN transformation
        calibrated_features = self.gamma * (input_tensor * norm_ratio) + self.beta + input_tensor
        return calibrated_features


class DropPath(nn.Module):
    """
    Stochastic depth implementation (Drop Path).
    Randomly drops entire paths (samples in a batch) during training to act as regularization.
    """
    def __init__(self, drop_probability: float = 0.0):
        super().__init__()
        self.drop_probability = drop_probability

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        if self.drop_probability == 0.0 or not self.training:
            return input_tensor
            
        keep_probability = 1.0 - self.drop_probability
        # Shape for broadcasting to the input tensor's dimensions
        random_tensor_shape = (input_tensor.shape[0],) + (1,) * (input_tensor.ndim - 1)
        
        # Create a binary mask to drop entire samples
        random_tensor = torch.rand(random_tensor_shape, dtype=input_tensor.dtype, device=input_tensor.device)
        random_tensor.add_(keep_probability).floor_()
        
        # Apply the mask and scale by keep_probability to maintain expected value
        return input_tensor.div(keep_probability) * random_tensor


class ConvNeXtBlock(nn.Module):
    """
    A single block of the ConvNeXt v2 architecture adapted for ECG time-frequency analysis.
    Features asymmetric depthwise convolutions optimized for ECG's frequency-time characteristics.
    """
    def __init__(self, num_channels: int, drop_path_rate: float = 0.0):
        super().__init__()
        # Asymmetric depthwise convolutions for ECG time-frequency data
        # Frequency dimension: smaller kernel (3) since we have only 41 freq bins
        # Time dimension: larger kernel (15) to capture temporal ECG patterns
        self.depthwise_conv_freq = nn.Conv2d(
            num_channels, num_channels, 
            kernel_size=(3, 1), padding=(1, 0), groups=num_channels
        )
        self.depthwise_conv_time = nn.Conv2d(
            num_channels, num_channels, 
            kernel_size=(1, 15), padding=(0, 7), groups=num_channels
        )
        self.norm = LayerNorm(num_channels, eps=1e-6)
        
        # Inverted bottleneck: Pointwise convolutions to expand and contract channels
        self.pointwise_conv1 = nn.Linear(num_channels, 4 * num_channels)
        self.activation = nn.GELU()
        self.grn = GlobalResponseNorm(4 * num_channels)
        self.pointwise_conv2 = nn.Linear(4 * num_channels, num_channels)
        
        # Regularization
        self.drop_path = DropPath(drop_path_rate) if drop_path_rate > 0.0 else nn.Identity()

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        residual = input_tensor
        
        # Main path with asymmetric convolutions
        # First apply frequency-wise convolution, then time-wise
        features = self.depthwise_conv_freq(input_tensor)
        features = self.depthwise_conv_time(features)
        
        features = features.permute(0, 2, 3, 1)  # (N, C, H, W) -> (N, H, W, C) for LayerNorm
        features = self.norm(features)
        features = self.pointwise_conv1(features)
        features = self.activation(features)
        features = self.grn(features)
        features = self.pointwise_conv2(features)
        features = features.permute(0, 3, 1, 2)  # (N, H, W, C) -> (N, C, H, W)
        
        # Add residual connection
        output_tensor = residual + self.drop_path(features)
        return output_tensor


class ConvNeXtV2(nn.Module):
    """
    ConvNeXt v2 model adapted for ECG time-frequency analysis.

    Args:
        input_channels: Number of input channels (e.g., 2 for real/imaginary parts).
        num_classes: Number of output arrhythmia classes.
        stage_depths: List specifying the number of blocks in each of the 4 stages.
        stage_dims: List specifying the number of channels at each of the 4 stages.
        drop_path_rate: Stochastic depth rate for regularization.
    """
    def __init__(self, input_channels=2, num_classes=NUM_CLASSES, stage_depths=[2, 2, 6, 2],
                 stage_dims=[64, 128, 256, 512], drop_path_rate=0.2):
        super().__init__()
        
        self.downsample_layers = nn.ModuleList()
        stem = nn.Sequential(
            nn.Conv2d(input_channels, stage_dims[0], kernel_size=(4, 7), stride=(2, 4), padding=(1, 3)),
            LayerNorm(stage_dims[0], eps=1e-6, data_format="channels_first")
        )
        self.downsample_layers.append(stem)
        
        # Add a downsampling layer before each subsequent stage
        for i in range(3):
            downsample_layer = nn.Sequential(
                LayerNorm(stage_dims[i], eps=1e-6, data_format="channels_first"),
                nn.Conv2d(stage_dims[i], stage_dims[i+1], kernel_size=2, stride=2),
            )
            self.downsample_layers.append(downsample_layer)

        # --- ConvNeXt Stages ---
        self.stages = nn.ModuleList()
        total_blocks = sum(stage_depths)
        drop_path_rates = [rate.item() for rate in torch.linspace(0, drop_path_rate, total_blocks)]
        
        block_idx = 0
        for i in range(4):
            stage_blocks = []
            for _ in range(stage_depths[i]):
                stage_blocks.append(ConvNeXtBlock(stage_dims[i], drop_path_rates[block_idx]))
                block_idx += 1
            self.stages.append(nn.Sequential(*stage_blocks))

        final_dim = stage_dims[-1]
        self.final_norm = LayerNorm(final_dim, eps=1e-6)
        
        # ECG-specific adaptive pooling to preserve temporal information
        # Use adaptive pooling to handle variable sequence lengths gracefully
        self.adaptive_freq_pool = nn.AdaptiveAvgPool2d((1, None))  # Pool frequency dimension only
        self.temporal_attention = nn.Sequential(
            nn.Conv1d(final_dim, final_dim // 4, kernel_size=1),
            nn.GELU(),
            nn.Conv1d(final_dim // 4, 1, kernel_size=1),
            nn.Sigmoid()
        )
        
        self.dropout = nn.Dropout(0.3)
        self.head = nn.Linear(final_dim, num_classes)
        
        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module):
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            nn.init.trunc_normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        """
        Performs a forward pass through the network.

        Args:
            input_tensor: Input tensor of shape [batch, channels, height, width].

        Returns:
            Classification logits of shape [batch, num_classes].
        """
        features = input_tensor
        for i in range(4):
            features = self.downsample_layers[i](features)
            features = self.stages[i](features)
        
        # ECG-specific pooling: preserve temporal dimension for attention
        # Pool frequency dimension to [batch, channels, 1, time]
        freq_pooled = self.adaptive_freq_pool(features)  # [batch, channels, 1, time]
        freq_pooled = freq_pooled.squeeze(2)  # [batch, channels, time]
        
        # Apply temporal attention to weight important time segments
        attention_weights = self.temporal_attention(freq_pooled)  # [batch, 1, time]
        attended_features = freq_pooled * attention_weights  # [batch, channels, time]
        
        # Global average pooling over time dimension
        pooled_features = attended_features.mean(dim=-1)  # [batch, channels]
        pooled_features = self.final_norm(pooled_features)
        
        dropped_features = self.dropout(pooled_features)
        logits = self.head(dropped_features)
        return logits


def create_ecg_model(num_classes: int = NUM_CLASSES) -> ConvNeXtV2:
    """
    Factory function to create and return an initialized ConvNeXt v2 model.

    Args:
        num_classes: The number of output classes for the model.

    Returns:
        An initialized ConvNeXtV2 model instance.
    """
    logger.info(f"Creating ConvNeXt v2 model with {num_classes} classes.")
    model = ConvNeXtV2(num_classes=num_classes)
    
    # Log the number of trainable parameters
    num_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Model created with {num_trainable_params:,} trainable parameters.")
    
    return model
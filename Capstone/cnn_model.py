"""
Custom Convolutional Neural Network (CNN) Architecture & Utilities
-------------------------------------------------------------------
Provides PyTorch CNN building blocks, custom feature extractor backbones,
and evaluation metric helpers for computer vision tasks.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Tuple


class ConvBlock(nn.Module):
    """Standard Convolutional Block: Conv2d -> BatchNorm2d -> SiLU/ReLU activation."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3, stride: int = 1, padding: int = 1):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.bn(self.conv(x)))


class CustomCNNBackbone(nn.Module):
    """
    Deep Convolutional Neural Network (CNN) Feature Extractor
    ---------------------------------------------------------
    Multi-scale convolutional backbone with residual connections for rich spatial 
    feature representation and high-accuracy object recognition.
    """

    def __init__(self, in_channels: int = 3, num_classes: int = 80):
        super().__init__()
        
        # Stem Layer
        self.stem = ConvBlock(in_channels, 32, kernel_size=3, stride=2, padding=1)  # H/2, W/2
        
        # Stage 1
        self.stage1 = nn.Sequential(
            ConvBlock(32, 64, kernel_size=3, stride=2, padding=1),                 # H/4, W/4
            ConvBlock(64, 64, kernel_size=3, stride=1, padding=1),
        )
        
        # Stage 2
        self.stage2 = nn.Sequential(
            ConvBlock(64, 128, kernel_size=3, stride=2, padding=1),                # H/8, W/8
            ConvBlock(128, 128, kernel_size=3, stride=1, padding=1),
        )
        
        # Stage 3
        self.stage3 = nn.Sequential(
            ConvBlock(128, 256, kernel_size=3, stride=2, padding=1),               # H/16, W/16
            ConvBlock(256, 256, kernel_size=3, stride=1, padding=1),
        )

        # Global Feature Aggregation & Classifier
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.SiLU(),
            nn.Linear(128, num_classes)
        )

    def extract_features(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Extract multi-scale CNN feature maps (P3, P4, P5)."""
        c1 = self.stage1(self.stem(x))
        c2 = self.stage2(c1)
        c3 = self.stage3(c2)
        return c1, c2, c3

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, _, c3 = self.extract_features(x)
        pooled = self.global_pool(c3)
        flat = torch.flatten(pooled, 1)
        return self.fc(flat)


def get_model_summary(model: nn.Module, input_size: Tuple[int, int, int, int] = (1, 3, 640, 640)) -> Dict[str, Any]:
    """Computes total parameter count and layer summary for a PyTorch CNN model."""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    dummy_input = torch.zeros(input_size)
    model.eval()
    with torch.no_grad():
        out = model(dummy_input)

    return {
        "total_parameters": total_params,
        "trainable_parameters": trainable_params,
        "output_shape": list(out.shape),
    }


if __name__ == "__main__":
    cnn = CustomCNNBackbone(num_classes=80)
    summary = get_model_summary(cnn)
    print("CNN Architecture Initialized:")
    print(f" - Total Parameters: {summary['total_parameters']:,}")
    print(f" - Trainable Parameters: {summary['trainable_parameters']:,}")
    print(f" - Output Shape: {summary['output_shape']}")

"""
Weight initialization utilities.
"""

import math
import torch
import torch.nn as nn


def orthogonal_init_(module: nn.Module, gain: float = 1.0) -> None:
    """
    Orthogonal initialization for weights.
    
    Args:
        module: Module to initialize
        gain: Scaling factor
    """
    if isinstance(module, (nn.Linear, nn.Conv2d, nn.Conv1d)):
        nn.init.orthogonal_(module.weight, gain=gain)
        if module.bias is not None:
            nn.init.constant_(module.bias, 0)


def xavier_init_(module: nn.Module, gain: float = 1.0) -> None:
    """
    Xavier/Glorot initialization for weights.
    
    Args:
        module: Module to initialize
        gain: Scaling factor
    """
    if isinstance(module, (nn.Linear, nn.Conv2d, nn.Conv1d)):
        nn.init.xavier_uniform_(module.weight, gain=gain)
        if module.bias is not None:
            nn.init.constant_(module.bias, 0)


def kaiming_init_(module: nn.Module, nonlinearity: str = "relu") -> None:
    """
    Kaiming/He initialization for weights.
    
    Args:
        module: Module to initialize
        nonlinearity: Nonlinearity type
    """
    if isinstance(module, (nn.Linear, nn.Conv2d, nn.Conv1d)):
        nn.init.kaiming_normal_(module.weight, nonlinearity=nonlinearity)
        if module.bias is not None:
            nn.init.constant_(module.bias, 0)


def init_module(module: nn.Module, init_type: str = "orthogonal", gain: float = 1.0) -> None:
    """
    Initialize a module with specified method.
    
    Args:
        module: Module to initialize
        init_type: Initialization type ('orthogonal', 'xavier', 'kaiming')
        gain: Scaling factor
    """
    if init_type == "orthogonal":
        orthogonal_init_(module, gain)
    elif init_type == "xavier":
        xavier_init_(module, gain)
    elif init_type == "kaiming":
        kaiming_init_(module, gain)
    else:
        raise ValueError(f"Unknown init_type: {init_type}")

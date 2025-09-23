# -*-coding:utf8-*-

import torch
import torch.nn as nn
from typing import Optional


class CompliedLoss(nn.Module):
    """Combined loss function for CSReL."""
    
    def __init__(self, ce_factor: float = 1.0, mse_factor: float = 0.0, reduction: str = 'mean'):
        super(CompliedLoss, self).__init__()
        self.ce_factor = ce_factor
        self.mse_factor = mse_factor
        self.reduction = reduction
        
        self.ce_loss = nn.CrossEntropyLoss(reduction='none')
        self.mse_loss = nn.MSELoss(reduction='none')
    
    def forward(self, outputs: torch.Tensor, targets: torch.Tensor, logits: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward pass of the loss function."""
        
        # Cross-entropy loss
        ce_loss = self.ce_loss(outputs, targets)
        
        # MSE loss (if logits are provided)
        mse_loss = torch.tensor(0.0, device=outputs.device)
        if logits is not None and self.mse_factor > 0:
            mse_loss = self.mse_loss(outputs, logits)
            mse_loss = torch.mean(mse_loss, dim=1)  # Average over classes
        
        # Combined loss
        total_loss = self.ce_factor * ce_loss + self.mse_factor * mse_loss
        
        if self.reduction == 'mean':
            return torch.mean(total_loss)
        elif self.reduction == 'sum':
            return torch.sum(total_loss)
        else:
            return total_loss


class KDCrossEntropyLoss(nn.Module):
    """Knowledge Distillation Cross Entropy Loss."""
    
    def __init__(self, temperature: float = 4.0, alpha: float = 0.7, reduction: str = 'mean'):
        super(KDCrossEntropyLoss, self).__init__()
        self.temperature = temperature
        self.alpha = alpha
        self.reduction = reduction
        
        self.ce_loss = nn.CrossEntropyLoss(reduction='none')
        self.kl_loss = nn.KLDivLoss(reduction='none')
    
    def forward(self, outputs: torch.Tensor, targets: torch.Tensor, teacher_outputs: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward pass of the KD loss function."""
        
        # Hard target loss
        hard_loss = self.ce_loss(outputs, targets)
        
        # Soft target loss (if teacher outputs are provided)
        soft_loss = torch.tensor(0.0, device=outputs.device)
        if teacher_outputs is not None:
            soft_targets = torch.softmax(teacher_outputs / self.temperature, dim=1)
            soft_outputs = torch.log_softmax(outputs / self.temperature, dim=1)
            soft_loss = self.kl_loss(soft_outputs, soft_targets)
            soft_loss = torch.sum(soft_loss, dim=1) * (self.temperature ** 2)
        
        # Combined loss
        total_loss = self.alpha * soft_loss + (1 - self.alpha) * hard_loss
        
        if self.reduction == 'mean':
            return torch.mean(total_loss)
        elif self.reduction == 'sum':
            return torch.sum(total_loss)
        else:
            return total_loss
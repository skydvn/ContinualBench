"""
CSReL (Coreset Selection for Continual Learning) core utilities.
Adapted from CSReL-Coreset-CL implementation for mammoth framework.
"""

import torch
import torch.nn as nn
import numpy as np
import copy
from typing import Dict, List, Tuple, Optional, Any
from torch.utils.data import DataLoader, TensorDataset


class CompliedLoss(nn.Module):
    """Combined loss function for CSReL (CE + MSE)."""
    
    def __init__(self, ce_factor=1.0, mse_factor=0.0, reduction='mean'):
        super().__init__()
        self.ce_factor = ce_factor
        self.mse_factor = mse_factor
        self.reduction = reduction
        self.ce_loss = nn.CrossEntropyLoss(reduction='none')
        self.mse_loss = nn.MSELoss(reduction='none')
    
    def forward(self, x, y, logits=None):
        ce_loss = self.ce_loss(x, y)
        
        if self.mse_factor > 0 and logits is not None:
            mse_loss = self.mse_loss(x, logits)
            total_loss = self.ce_factor * ce_loss + self.mse_factor * mse_loss
        else:
            total_loss = self.ce_factor * ce_loss
        
        if self.reduction == 'mean':
            return total_loss.mean()
        elif self.reduction == 'sum':
            return total_loss.sum()
        else:
            return total_loss


def compute_reference_losses(model, data_loader, loss_params, device):
    """
    Compute reference losses for all samples in the data loader.
    
    Args:
        model: Neural network model
        data_loader: Data loader containing samples
        loss_params: Loss function parameters
        device: Device to compute on
    
    Returns:
        Dictionary mapping sample IDs to their reference losses
    """
    model.eval()
    loss_fn = CompliedLoss(
        ce_factor=loss_params['ce_factor'], 
        mse_factor=loss_params['mse_factor'], 
        reduction='none'
    )
    
    ref_losses = {}
    
    with torch.no_grad():
        for batch in data_loader:
            if len(batch) == 3:
                inputs, targets, sample_ids = batch
                logits = None
            else:
                inputs, targets, sample_ids, logits = batch
            
            inputs = inputs.to(device)
            targets = targets.to(device)
            
            if logits is not None:
                logits = logits.to(device)
            
            outputs = model(inputs)
            losses = loss_fn(outputs, targets, logits)
            
            for i, sample_id in enumerate(sample_ids):
                ref_losses[sample_id.item()] = losses[i].item()
    
    return ref_losses


def select_by_loss_diff(ref_losses, candidates_data, candidates_labels, model, 
                       incremental_size, loss_params, device, class_sizes=None):
    """
    Select samples based on loss difference (CSReL core selection method).
    
    Args:
        ref_losses: Dictionary of reference losses for each sample
        candidates_data: Candidate data tensor [N, ...]
        candidates_labels: Candidate labels tensor [N]
        model: Neural network model
        incremental_size: Number of samples to select
        loss_params: Loss function parameters
        device: Device to compute on
        class_sizes: Optional class size constraints
    
    Returns:
        Tuple of (selected_indices, loss_differences)
    """
    model.eval()
    loss_fn = CompliedLoss(
        ce_factor=loss_params['ce_factor'], 
        mse_factor=loss_params['mse_factor'], 
        reduction='none'
    )
    
    loss_diffs = {}
    batch_size = 32
    
    with torch.no_grad():
        for i in range(0, len(candidates_data), batch_size):
            end_idx = min(i + batch_size, len(candidates_data))
            batch_data = candidates_data[i:end_idx].to(device)
            batch_labels = candidates_labels[i:end_idx].to(device)
            
            outputs = model(batch_data)
            losses = loss_fn(outputs, batch_labels)
            
            for j in range(len(batch_data)):
                sample_idx = i + j
                sample_id = sample_idx  # Use index as ID
                
                if sample_id in ref_losses:
                    current_loss = losses[j].item()
                    ref_loss = ref_losses[sample_id]
                    loss_diffs[sample_id] = current_loss - ref_loss
                else:
                    # If no reference loss, use current loss as difference
                    loss_diffs[sample_id] = losses[j].item()
    
    # Sort by loss difference (descending - higher difference = more informative)
    sorted_loss_diffs = sorted(loss_diffs.items(), key=lambda x: x[1], reverse=True)
    
    # Apply class balancing if specified
    if class_sizes is not None:
        selected_indices = class_balanced_selection(
            sorted_loss_diffs, candidates_labels, class_sizes, incremental_size
        )
    else:
        selected_indices = [item[0] for item in sorted_loss_diffs[:incremental_size]]
    
    return selected_indices, loss_diffs


def class_balanced_selection(sorted_loss_diffs, labels, class_sizes, total_size):
    """
    Select samples with class balancing constraints.
    
    Args:
        sorted_loss_diffs: List of (sample_id, loss_diff) sorted by loss difference
        labels: Label tensor for all samples
        class_sizes: Dictionary of class_id -> max_samples_per_class
        total_size: Total number of samples to select
    
    Returns:
        List of selected sample indices
    """
    selected_indices = []
    class_counts = {cls_id: 0 for cls_id in class_sizes.keys()}
    
    for sample_id, loss_diff in sorted_loss_diffs:
        if len(selected_indices) >= total_size:
            break
            
        class_id = labels[sample_id].item()
        
        if class_id in class_counts and class_counts[class_id] < class_sizes[class_id]:
            selected_indices.append(sample_id)
            class_counts[class_id] += 1
    
    return selected_indices


def create_data_loader(data, labels, sample_ids=None, batch_size=32, shuffle=False):
    """
    Create a data loader from numpy arrays.
    
    Args:
        data: Input data array
        labels: Label array
        sample_ids: Optional sample ID array
        batch_size: Batch size for data loader
        shuffle: Whether to shuffle the data
    
    Returns:
        DataLoader object
    """
    if sample_ids is None:
        sample_ids = np.arange(len(data))
    
    dataset = TensorDataset(
        torch.tensor(data, dtype=torch.float32),
        torch.tensor(labels, dtype=torch.long),
        torch.tensor(sample_ids, dtype=torch.long)
    )
    
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def train_reference_model(model, train_loader, epochs, lr, device, loss_params):
    """
    Train a reference model for loss computation.
    
    Args:
        model: Model to train
        train_loader: Training data loader
        epochs: Number of training epochs
        lr: Learning rate
        device: Device to train on
        loss_params: Loss function parameters
    
    Returns:
        Trained model
    """
    model.train()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    loss_fn = CompliedLoss(
        ce_factor=loss_params['ce_factor'], 
        mse_factor=loss_params['mse_factor']
    )
    
    for epoch in range(epochs):
        for batch in train_loader:
            if len(batch) == 3:
                inputs, targets, _ = batch
            else:
                inputs, targets, _, _ = batch
            
            inputs = inputs.to(device)
            targets = targets.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = loss_fn(outputs, targets)
            loss.backward()
            optimizer.step()
    
    return model


def get_class_distribution(labels):
    """
    Get class distribution from labels.
    
    Args:
        labels: Label tensor or array
    
    Returns:
        Dictionary mapping class_id to count
    """
    if isinstance(labels, torch.Tensor):
        labels = labels.numpy()
    
    unique, counts = np.unique(labels, return_counts=True)
    return dict(zip(unique, counts))


def make_class_sizes(class_distribution, total_size, num_classes):
    """
    Create balanced class size constraints.
    
    Args:
        class_distribution: Dictionary of class_id -> count
        total_size: Total number of samples to select
        num_classes: Number of classes
    
    Returns:
        Dictionary of class_id -> max_samples_per_class
    """
    base_size = total_size // num_classes
    remainder = total_size % num_classes
    
    class_sizes = {}
    for i in range(num_classes):
        class_sizes[i] = base_size + (1 if i < remainder else 0)
    
    return class_sizes

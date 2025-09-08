import torch
import numpy as np
from typing import Optional, Dict, Any


def ocs_rank_indices(eg: torch.Tensor, tau: float = 0.0, ref_grads: Optional[torch.Tensor] = None) -> np.ndarray:
    """
    Rank candidate indices using OCS measure: mean_sim - mean_div + tau * coreset_aff.
    
    Args:
        eg: Per-example gradients [N, P]
        tau: Affinity weight for coreset affinity term
        ref_grads: Reference gradients from coreset [P] or None
    
    Returns:
        Numpy array of indices sorted descending by the OCS measure
    """
    if eg.shape[0] == 0:
        return np.array([], dtype=np.int64)
    
    # Mean gradient across examples
    g = torch.mean(eg, dim=0)  # [P]
    ng = torch.norm(g) + 1e-12
    neg = torch.norm(eg, dim=1) + 1e-12  # [N]
    
    # Similarity to mean gradient
    mean_sim = torch.matmul(eg, g) / (neg * ng)  # [N]
    
    # Diversity term (cross-example similarity)
    neg_col = neg.unsqueeze(1)  # [N,1]
    denom = torch.clamp(neg_col @ neg_col.t(), min=1e-6)
    cross_div = (eg @ eg.t()) / denom  # [N,N]
    mean_div = torch.mean(cross_div, dim=0)  # [N]
    
    # Affinity to reference gradients (optional)
    coreset_aff = 0.0
    if ref_grads is not None:
        ref_ng = torch.norm(ref_grads) + 1e-12
        coreset_aff = torch.matmul(eg, ref_grads) / (neg * ref_ng)
    
    # OCS measure: similarity - diversity + affinity
    measure = mean_sim - mean_div + float(tau) * (coreset_aff if isinstance(coreset_aff, torch.Tensor) else 0.0)
    _, sorted_idx = torch.sort(measure, descending=True)
    
    return sorted_idx.detach().cpu().numpy()


def class_fair_mask(Y: torch.Tensor, sorted_idx: np.ndarray, total_slots: int, classes_per_task: int) -> torch.Tensor:
    """
    Build a boolean mask over candidates so that selection respects a class-fair cap.
    
    Args:
        Y: Target labels tensor
        sorted_idx: Sorted indices from OCS ranking
        total_slots: Total number of slots to fill
        classes_per_task: Number of classes per task
    
    Returns:
        Boolean mask indicating which samples to select
    """
    if total_slots <= 0 or len(sorted_idx) == 0:
        return torch.zeros(len(Y), dtype=torch.bool)
    
    # Calculate fair distribution per class
    per_class = max(1, total_slots // classes_per_task)
    extra = total_slots - per_class * classes_per_task
    
    selected_mask = torch.zeros(len(Y), dtype=torch.bool)
    class_counts = {int(c): 0 for c in Y.unique().tolist()}
    taken = 0
    
    for idx in sorted_idx:
        cls = int(Y[idx])
        cap = per_class + (1 if extra > 0 else 0)
        
        if class_counts.get(cls, 0) < cap:
            selected_mask[idx] = True
            class_counts[cls] = class_counts.get(cls, 0) + 1
            taken += 1
            
            if class_counts[cls] == cap and extra > 0:
                extra -= 1
                
            if taken >= total_slots:
                break
    
    return selected_mask


def uniform_selection(Y: torch.Tensor, total_slots: int) -> torch.Tensor:
    """
    Uniform random selection of samples.
    
    Args:
        Y: Target labels tensor
        total_slots: Total number of slots to fill
    
    Returns:
        Boolean mask indicating which samples to select
    """
    if total_slots <= 0:
        return torch.zeros(len(Y), dtype=torch.bool)
    
    indices = torch.randperm(len(Y))[:total_slots]
    mask = torch.zeros(len(Y), dtype=torch.bool)
    mask[indices] = True
    
    return mask


def entropy_selection(model: torch.nn.Module, data: torch.Tensor, Y: torch.Tensor, 
                     total_slots: int, device: torch.device) -> torch.Tensor:
    """
    Select samples based on prediction entropy (uncertainty-based selection).
    
    Args:
        model: Neural network model
        data: Input data tensor
        Y: Target labels tensor
        total_slots: Total number of slots to fill
        device: Device to compute on
    
    Returns:
        Boolean mask indicating which samples to select
    """
    if total_slots <= 0:
        return torch.zeros(len(Y), dtype=torch.bool)
    
    model.eval()
    with torch.no_grad():
        data = data.to(device)
        outputs = model(data)
        probs = torch.softmax(outputs, dim=1)
        entropy = -torch.sum(probs * torch.log(probs + 1e-8), dim=1)
        
        # Select samples with highest entropy
        _, indices = torch.sort(entropy, descending=True)
        mask = torch.zeros(len(Y), dtype=torch.bool)
        mask[indices[:total_slots]] = True
    
    return mask


def hardest_selection(model: torch.nn.Module, data: torch.Tensor, Y: torch.Tensor, 
                     total_slots: int, device: torch.device) -> torch.Tensor:
    """
    Select samples that are hardest to classify (highest loss).
    
    Args:
        model: Neural network model
        data: Input data tensor
        Y: Target labels tensor
        total_slots: Total number of slots to fill
        device: Device to compute on
    
    Returns:
        Boolean mask indicating which samples to select
    """
    if total_slots <= 0:
        return torch.zeros(len(Y), dtype=torch.bool)
    
    model.eval()
    with torch.no_grad():
        data = data.to(device)
        Y = Y.to(device)
        outputs = model(data)
        criterion = torch.nn.CrossEntropyLoss(reduction='none')
        losses = criterion(outputs, Y)
        
        # Select samples with highest loss
        _, indices = torch.sort(losses, descending=True)
        mask = torch.zeros(len(Y), dtype=torch.bool)
        mask[indices[:total_slots]] = True
    
    return mask


def select_samples_by_strategy(strategy: str, model: Optional[torch.nn.Module], 
                              data: Optional[torch.Tensor], Y: torch.Tensor, 
                              sorted_idx: Optional[np.ndarray], total_slots: int, 
                              classes_per_task: int, device: Optional[torch.device] = None,
                              **kwargs) -> torch.Tensor:
    """
    Select samples using the specified strategy.
    
    Args:
        strategy: Selection strategy ('ocs', 'uniform', 'entropy', 'hardest', 'class_fair')
        model: Neural network model (required for some strategies)
        data: Input data tensor (required for some strategies)
        Y: Target labels tensor
        sorted_idx: Sorted indices from OCS ranking (required for 'ocs' strategy)
        total_slots: Total number of slots to fill
        classes_per_task: Number of classes per task
        device: Device to compute on (required for some strategies)
        **kwargs: Additional arguments for specific strategies
    
    Returns:
        Boolean mask indicating which samples to select
    """
    if strategy == 'ocs':
        if sorted_idx is None:
            raise ValueError("sorted_idx is required for 'ocs' strategy")
        return class_fair_mask(Y, sorted_idx, total_slots, classes_per_task)
    
    elif strategy == 'uniform':
        return uniform_selection(Y, total_slots)
    
    elif strategy == 'entropy':
        if model is None or data is None or device is None:
            raise ValueError("model, data, and device are required for 'entropy' strategy")
        return entropy_selection(model, data, Y, total_slots, device)
    
    elif strategy == 'hardest':
        if model is None or data is None or device is None:
            raise ValueError("model, data, and device are required for 'hardest' strategy")
        return hardest_selection(model, data, Y, total_slots, device)
    
    elif strategy == 'class_fair':
        if sorted_idx is None:
            raise ValueError("sorted_idx is required for 'class_fair' strategy")
        return class_fair_mask(Y, sorted_idx, total_slots, classes_per_task)
    
    else:
        raise ValueError(f"Unknown selection strategy: {strategy}") 
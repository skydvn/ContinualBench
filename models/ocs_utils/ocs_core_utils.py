"""
Core utilities from OCS implementation for mammoth framework.
Adapted from OCS/core/utils.py and OCS/core/autograd_hacks.py
"""

import torch
import torch.nn as nn
import numpy as np
from typing import List, Optional


def flatten_grads(m, numpy_output=False, bias=True, only_linear=False):
    """Flatten gradients from model parameters."""
    total_grads = []
    for name, param in m.named_parameters():
        if only_linear:
            if (bias or not 'bias' in name) and 'linear' in name:
                total_grads.append(param.grad.detach().view(-1))
        else:
            if (bias or not 'bias' in name) and not 'bn' in name and not 'IC' in name:
                try:
                    total_grads.append(param.grad.detach().view(-1))
                except AttributeError:
                    pass
    total_grads = torch.cat(total_grads)
    if numpy_output:
        return total_grads.cpu().detach().numpy()
    return total_grads


def compute_and_flatten_example_grads(m, criterion, data, target, task_id=None):
    """Compute per-example gradients and flatten them."""
    _eg = []
    criterion2 = nn.CrossEntropyLoss(reduction='none')
    m.eval()
    m.zero_grad()
    
    # Get device from model
    device = next(m.parameters()).device
    
    # Handle task_id parameter if model expects it
    if task_id is not None:
        pred = m(data, task_id)
    else:
        pred = m(data)
    
    loss = criterion2(pred, target)
    for idx in range(len(data)):
        loss[idx].backward(retain_graph=True)
        _g = flatten_grads(m, numpy_output=True)
        _eg.append(torch.Tensor(_g).to(device))
        m.zero_grad()
    return torch.stack(_eg)


def sample_selection(g, eg, config, ref_grads=None, attn=None):
    """
    Original OCS sample selection function from train_methods_cifar.py.
    
    Args:
        g: Mean gradient [P]
        eg: Per-example gradients [N, P]
        config: Configuration dict with 'tau' parameter
        ref_grads: Reference gradients [P] (optional)
        attn: Attention weights (unused)
    
    Returns:
        Sorted indices by OCS measure
    """
    # Ensure all tensors are on the same device
    device = g.device
    eg = eg.to(device)
    if ref_grads is not None:
        ref_grads = ref_grads.to(device)
    
    ng = torch.norm(g)
    neg = torch.norm(eg, dim=1)
    mean_sim = torch.matmul(g, eg.t()) / torch.maximum(ng * neg, torch.ones_like(neg) * 1e-6)
    negd = torch.unsqueeze(neg, 1)

    cross_div = torch.matmul(eg, eg.t()) / torch.maximum(torch.matmul(negd, negd.t()), torch.ones_like(negd) * 1e-6)
    mean_div = torch.mean(cross_div, 0)

    coreset_aff = 0.
    if ref_grads is not None:
        ref_ng = torch.norm(ref_grads)
        coreset_aff = torch.matmul(ref_grads, eg.t()) / torch.maximum(ref_ng * neg, torch.ones_like(neg) * 1e-6)

    measure = mean_sim - mean_div + config['tau'] * coreset_aff
    _, u_idx = torch.sort(measure, descending=True)
    return u_idx.cpu().numpy()


def classwise_fair_selection(task, cand_target, sorted_index, num_per_label, config, is_shuffle=True):
    """
    Class-wise fair selection from original OCS implementation.
    
    Args:
        task: Current task number
        cand_target: Candidate target labels
        sorted_index: Sorted indices from sample selection
        num_per_label: Number of samples per label
        config: Configuration dict
        is_shuffle: Whether to shuffle the selection
    
    Returns:
        Selected indices
    """
    num_examples_per_task = config['memory_size'] // task
    num_examples_per_class = num_examples_per_task // config['n_classes']
    num_residuals = num_examples_per_task - num_examples_per_class * config['n_classes']
    residuals = np.sum([(num_examples_per_class - n_c) * (num_examples_per_class > n_c) for n_c in num_per_label])
    num_residuals += residuals

    # Get the number of coreset instances per class
    while True:
        n_less_sample_class = np.sum([(num_examples_per_class > n_c) for n_c in num_per_label])
        num_class = (config['n_classes'] - n_less_sample_class)
        if (num_residuals // num_class) > 0:
            num_examples_per_class += (num_residuals // num_class)
            num_residuals -= (num_residuals // num_class) * num_class
        else:
            break
    
    # Get best coresets per class
    selected = []
    target_tid = np.floor(max(cand_target) / config['n_classes'])

    for j in range(config['n_classes']):
        position = np.squeeze((cand_target[sorted_index] == j + (target_tid * config['n_classes'])).nonzero())
        if position.numel() > 1:
            selected.append(position[:num_examples_per_class])
        elif position.numel() == 0:
            continue
        else:
            selected.append([position])
    
    # Fill rest space as best residuals
    selected = np.concatenate(selected)
    unselected = np.array(list(set(np.arange(num_examples_per_task)) ^ set(selected)))
    final_num_residuals = num_examples_per_task - len(selected)
    best_residuals = unselected[:final_num_residuals]
    selected = np.concatenate([selected, best_residuals])

    if is_shuffle:
        np.random.shuffle(selected)

    return sorted_index[selected.astype(int)]


def get_coreset_loss(model, ref_data, config):
    """Compute coreset loss for reference gradient computation."""
    criterion = nn.CrossEntropyLoss()
    data, target, task_id = ref_data
    data = data.to(next(model.parameters()).device)
    target = target.to(next(model.parameters()).device)
    
    if task_id is not None:
        pred = model(data, task_id)
    else:
        pred = model(data)
    
    return criterion(pred, target)


def reconstruct_coreset_loader2(config, coreset_data, num_tasks):
    """Reconstruct coreset loader for reference gradient computation."""
    from torch.utils.data import DataLoader, TensorDataset
    
    all_data = []
    all_targets = []
    all_task_ids = []
    
    for tid in range(1, num_tasks + 1):
        if tid in coreset_data and 'train' in coreset_data[tid]:
            data = coreset_data[tid]['train'].data
            targets = coreset_data[tid]['train'].targets
            
            all_data.append(data)
            all_targets.append(targets)
            all_task_ids.append(torch.full((len(data),), tid, dtype=torch.long))
    
    if not all_data:
        return DataLoader(TensorDataset(torch.empty(0), torch.empty(0), torch.empty(0)), batch_size=1)
    
    combined_data = torch.cat(all_data, dim=0)
    combined_targets = torch.cat(all_targets, dim=0)
    combined_task_ids = torch.cat(all_task_ids, dim=0)
    
    dataset = TensorDataset(combined_data, combined_targets, combined_task_ids)
    return DataLoader(dataset, batch_size=config.get('batch_size', 32), shuffle=True)


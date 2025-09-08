import torch
from typing import List, Optional


def compute_per_example_grads(model: torch.nn.Module, data: torch.Tensor, target: torch.Tensor, 
                             device: torch.device, batch_size: int = 32) -> torch.Tensor:
    """
    Compute per-example gradients efficiently using batch processing.
    
    Args:
        model: Neural network model
        data: Input data tensor [N, ...]
        target: Target labels tensor [N]
        device: Device to compute on
        batch_size: Batch size for gradient computation (to avoid memory issues)
    
    Returns:
        Per-example gradients tensor [N, P] where P is the flattened parameter space
    """
    criterion = torch.nn.CrossEntropyLoss(reduction='none').to(device)
    model.train(False)
    eg_vectors: List[torch.Tensor] = []

    data = data.to(device)
    target = target.to(device)
    
    # Process in batches to avoid memory issues
    num_samples = data.shape[0]
    for i in range(0, num_samples, batch_size):
        end_idx = min(i + batch_size, num_samples)
        batch_data = data[i:end_idx]
        batch_target = target[i:end_idx]
        
        # Compute gradients for this batch
        batch_eg = _compute_batch_gradients(model, criterion, batch_data, batch_target, device)
        eg_vectors.append(batch_eg)
    
    model.zero_grad(set_to_none=True)
    return torch.cat(eg_vectors, dim=0)


def _compute_batch_gradients(model: torch.nn.Module, criterion: torch.nn.Module, 
                           data: torch.Tensor, target: torch.Tensor, device: torch.device) -> torch.Tensor:
    """
    Compute per-example gradients for a batch of samples.
    """
    batch_size = data.shape[0]
    eg_vectors: List[torch.Tensor] = []

    logits = model(data)
    losses = criterion(logits, target)

    for i in range(batch_size):
        model.zero_grad(set_to_none=True)
        losses[i].backward(retain_graph=True)
        
        grads_flat: List[torch.Tensor] = []
        for name, p in model.named_parameters():
            if p.grad is None:
                continue
            # Skip batch norm running statistics
            if 'running_mean' in name or 'running_var' in name:
                continue
            grads_flat.append(p.grad.detach().view(-1))
        
        if len(grads_flat) == 0:
            # Handle case where no gradients are computed
            eg_vectors.append(torch.zeros(1, device=device))
        else:
            eg_vectors.append(torch.cat(grads_flat))
    
    return torch.stack(eg_vectors, dim=0)


def compute_reference_gradients(model: torch.nn.Module, data: torch.Tensor, target: torch.Tensor, 
                               device: torch.device) -> Optional[torch.Tensor]:
    """
    Compute reference gradients from a coreset of data.
    
    Args:
        model: Neural network model
        data: Coreset data tensor
        target: Coreset labels tensor
        device: Device to compute on
    
    Returns:
        Flattened reference gradients tensor or None if computation fails
    """
    if len(data) == 0:
        return None
        
    try:
        model.train()
        model.zero_grad(set_to_none=True)
        
        data = data.to(device)
        target = target.to(device)
        
        criterion = torch.nn.CrossEntropyLoss()
        outputs = model(data)
        loss = criterion(outputs, target)
        loss.backward()
        
        # Collect gradients
        ref_grads = []
        for name, param in model.named_parameters():
            if param.grad is not None and 'running_mean' not in name and 'running_var' not in name:
                ref_grads.append(param.grad.detach().view(-1))
        
        if ref_grads:
            return torch.cat(ref_grads)
        else:
            return None
            
    except Exception as e:
        print(f"Warning: Failed to compute reference gradients: {e}")
        return None
    finally:
        model.zero_grad(set_to_none=True) 
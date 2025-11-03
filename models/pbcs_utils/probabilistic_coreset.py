"""
Probabilistic Coreset Selection utilities for Bilevel Optimization.
Adapted from Probabilistic-Bilevel-Coreset-Selection for mammoth framework.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
import copy
from typing import Tuple, List, Optional


class ProbabilisticCoresetSelector:
    """
    Probabilistic coreset selection using bilevel optimization.
    Implements the core selection logic from the original paper.
    """
    
    def __init__(self, device='cuda', K=5, outer_lr=0.1, inner_lr=0.005, 
                 max_outer_iter=100, epoch_converge=50, coreset_size=100,
                 use_variance_reduction=True, clip_grad=True, clip_constant=3.0,
                 inner_optim='sgd', inner_wd=0.0, div_tol=9.0):
        self.device = device
        self.K = K  # Number of samples per outer iteration
        self.outer_lr = outer_lr
        self.inner_lr = inner_lr
        self.max_outer_iter = max_outer_iter
        self.epoch_converge = epoch_converge
        self.coreset_size = coreset_size
        self.use_variance_reduction = use_variance_reduction
        self.clip_grad = clip_grad
        self.clip_constant = clip_constant
        self.inner_optim = inner_optim  # 'sgd' or 'adam'
        self.inner_wd = inner_wd  # Weight decay for inner optimizer
        self.div_tol = div_tol  # Divergence tolerance
    
    def _create_model_copy(self, model: nn.Module) -> nn.Module:
        """
        Create a safe copy of the model for training.
        This method avoids issues with deepcopy and model initialization.
        """
        try:
            # First try deepcopy
            model_copy = copy.deepcopy(model)
            return model_copy.to(self.device)
        except Exception as e:
            print(f"Deepcopy failed: {e}, using state dict method")
            try:
                # For ResNet and similar models, try to get the original arguments
                model_class = model.__class__
                
                # Check if it's a ResNet or similar model that needs specific args
                if hasattr(model, 'inplanes') and hasattr(model, 'num_classes'):
                    # This looks like a ResNet, try to extract the necessary parameters
                    try:
                        # Try to get the original arguments from the model's attributes
                        inplanes = getattr(model, 'inplanes', 64)
                        num_classes = getattr(model, 'num_classes', 1000)
                        
                        # Create ResNet with minimal required arguments
                        if 'ResNet' in model_class.__name__:
                            model_copy = model_class(num_classes=num_classes)
                        else:
                            model_copy = model_class()
                    except Exception:
                        # If that fails, try with no arguments
                        model_copy = model_class()
                else:
                    # For other models, try with no arguments
                    model_copy = model_class()
                
                # Load the state dict
                model_copy.load_state_dict(model.state_dict())
                return model_copy.to(self.device)
                
            except Exception as e2:
                print(f"State dict copy failed: {e2}, using direct reference")
                # Last resort: return the original model (not ideal but better than crashing)
                return model.to(self.device)
        
    def obtain_mask(self, scores: torch.Tensor, min_selections: int = 1) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Sample binary mask from Bernoulli distribution and compute REINFORCE gradients.
        Ensures at least min_selections samples are selected to avoid zero-selection issues.
        
        Args:
            scores: Selection probabilities [N] or [H, W] for pixel-level selection
            min_selections: Minimum number of samples to select (default: 1)
            
        Returns:
            subnet: Binary mask sampled from Bernoulli distribution
            grad: REINFORCE gradients for score updates
        """
        subnet = (torch.rand_like(scores) < scores).float()
        
        # Ensure minimum selections if needed
        if min_selections > 0 and subnet.sum().item() < min_selections:
            # Fallback: select top-k by scores to ensure minimum
            num_samples = len(scores) if len(scores.shape) == 1 else scores.numel()
            k = min(min_selections, num_samples)
            # Get top-k indices by scores
            if len(scores.shape) == 1:
                _, top_indices = torch.topk(scores, k)
                subnet = torch.zeros_like(scores)
                subnet[top_indices] = 1.0
            else:
                # For multi-dimensional (pixel-level), flatten and select
                scores_flat = scores.flatten()
                _, top_indices = torch.topk(scores_flat, k)
                subnet = torch.zeros_like(scores_flat)
                subnet[top_indices] = 1.0
                subnet = subnet.view(scores.shape)
        
        eps = 1e-20
        grad = (subnet - scores) / ((scores + eps) * (1 - scores + eps))
        return subnet, grad
    
    def constrain_scores(self, scores: torch.Tensor, target_size: int) -> torch.Tensor:
        """
        Constrain scores to maintain target coreset size.
        
        Args:
            scores: Selection probabilities
            target_size: Target number of selected samples
            
        Returns:
            Constrained scores
        """
        with torch.no_grad():
            v = self._solve_v_total(scores, target_size)
            scores.sub_(v).clamp_(0, 1)
        return scores
    
    def _solve_v_total(self, weight: torch.Tensor, subset: int) -> float:
        """Solve for threshold value to maintain subset size."""
        k = subset
        a, b = 0, 0
        b = max(b, weight.max().item())
        
        def f(v):
            s = (weight - v).clamp(0, 1).sum()
            return s - k
            
        if f(0) < 0:
            return 0
            
        itr = 0
        while True:
            itr += 1
            v = (a + b) / 2
            obj = f(v)
            if abs(obj) < 1e-3 or itr > 20:
                break
            if obj < 0:
                b = v
            else:
                a = v
        v = max(0, v)
        return v
    
    def calculate_grad_vr(self, scores: torch.Tensor, fn_list: List[float], 
                           grad_list: List[torch.Tensor], fn_avg: float):
        """Calculate gradients with variance reduction."""
        # Initialize grad if None (after zero_grad())
        if scores.grad is None:
            scores.grad = torch.zeros_like(scores)
        for i in range(self.K):
            scores.grad.data += 1 / (self.K-1) * (fn_list[i]-fn_avg) * grad_list[i]
    
    def calculate_grad(self, scores: torch.Tensor, fn_list: List[float], 
                      grad_list: List[torch.Tensor]):
        """Calculate standard gradients."""
        # Initialize grad if None (after zero_grad())
        if scores.grad is None:
            scores.grad = torch.zeros_like(scores)
        for i in range(self.K):
            scores.grad.data += 1/self.K * fn_list[i] * grad_list[i]
    
    def train_to_converge(self, model: nn.Module, x_coreset: torch.Tensor, 
                         y_coreset: torch.Tensor, epoch_converge: int = None) -> Tuple[nn.Module, float, float, float, bool]:
        """
        Train model to convergence on coreset data.
        Matches original implementation: processes all coreset data at once (no inner batching).
        
        Args:
            model: Model to train
            x_coreset: Coreset input data (already selected by indices)
            y_coreset: Coreset labels (already selected by indices)
            epoch_converge: Number of epochs to train
            
        Returns:
            trained_model: Trained model
            loss: Final loss
            acc1: Top-1 accuracy
            acc5: Top-5 accuracy
            diverged: Whether training diverged
        """
        if epoch_converge is None:
            epoch_converge = self.epoch_converge
            
        # Create a proper copy of the model for training
        model_copy = self._create_model_copy(model)
        
        # Create optimizer based on inner_optim setting (matching original)
        if self.inner_optim == 'sgd':
            optimizer = torch.optim.SGD(model_copy.parameters(), lr=self.inner_lr, 
                                       momentum=0.9, weight_decay=self.inner_wd)
        else:
            optimizer = torch.optim.Adam(model_copy.parameters(), lr=self.inner_lr, 
                                        weight_decay=self.inner_wd)
        
        # Process all coreset data at once (no batching in inner loop, matching original)
        data, target = x_coreset.to(self.device), y_coreset.to(self.device)
        diverged = False
        
        for i in range(epoch_converge):
            # Cosine annealing learning rate (matching original)
            lr = 0.5 * (1 + np.cos(np.pi * i / epoch_converge)) * self.inner_lr
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
                
            optimizer.zero_grad()
            output = model_copy(data)  # Forward pass on all coreset data at once
            loss = F.cross_entropy(output, target)
            loss.backward()
            optimizer.step()
            
            if i % 50 == 0:  # Matching original print frequency
                print(f"{i}th iter, inner loss {loss.item():.4f}")
                
        # Check for divergence (matching original)
        if math.isnan(loss.item()) or loss.item() > self.div_tol:
            diverged = True
            
        # Calculate accuracy (matching original implementation)
        with torch.no_grad():
            _, pred = output.topk(5, 1, True, True)
            pred = pred.t()
            correct = pred.eq(target.view(1, -1).expand_as(pred))
            acc1 = correct[:1].flatten().float().sum(0, keepdim=True).mul_(100.0 / target.size(0)).item()
            acc5 = correct[:5].flatten().float().sum(0, keepdim=True).mul_(100.0 / target.size(0)).item()
            
        return model_copy, loss.item(), acc1, acc5, diverged
    
    def get_loss_on_full_data(self, model: nn.Module, X: torch.Tensor, Y: torch.Tensor) -> float:
        """
        Calculate loss on full dataset.
        Matches original implementation: processes all data at once.
        """
        with torch.no_grad():
            data, target = X.to(self.device), Y.to(self.device)
            output = model(data)  # Process all data at once (matching original)
            loss = F.cross_entropy(output, target)
        return loss.item()
    
    def select_coreset(self, model: nn.Module, X: torch.Tensor, Y: torch.Tensor, 
                      task_id: int = 0) -> Tuple[np.ndarray, torch.Tensor, nn.Module]:
        """
        Main coreset selection using probabilistic bilevel optimization.
        Matches original implementation: uses index-based selection instead of mask-based.
        
        Args:
            model: Model to use for selection
            X: Input data [N, ...]
            Y: Labels [N]
            task_id: Task identifier
            
        Returns:
            selected_indices: Indices of selected samples (numpy array, matching original)
            selection_scores: Learned selection probabilities (torch tensor)
            selected_model: Best model from optimization (nn.Module)
        """
        print(f"Starting probabilistic coreset selection for task {task_id}")
        print(f"Data shape: {X.shape}, Target coreset size: {self.coreset_size}")
        
        # Ensure data is on the correct device
        X = X.to(self.device)
        Y = Y.to(self.device)
        print(f"Data moved to device: {X.device}")
        
        # Initialize selection scores (matching original)
        num_samples = X.shape[0]
        pr_target = self.coreset_size / num_samples
        scores = torch.full([num_samples], pr_target, dtype=torch.float, 
                           requires_grad=True, device=self.device)
        scores_opt = torch.optim.Adam([scores], lr=self.outer_lr)
        scores.grad = torch.zeros_like(scores)
        
        print(f"Initial selection probability: {pr_target:.4f}")
        
        # Outer optimization loop (matching original flow)
        for outer_iter in range(self.max_outer_iter):
            print(f"Outer iteration {outer_iter}/{self.max_outer_iter}")
            
            # Cosine annealing for outer learning rate (matching original)
            lr = 0.5 * (1 + np.cos(np.pi * outer_iter / self.max_outer_iter)) * self.outer_lr
            for param_group in scores_opt.param_groups:
                param_group['lr'] = lr
            
            fn_list = []
            grad_list = []
            fn_avg = 0
            all_models = []
            
            # Sample K models (matching original)
            for i in range(self.K):
                diverged = True
                attempts = 0
                max_attempts = 5
                indices_np = None
                trained_model = None
                grad = None
                
                while diverged and attempts < max_attempts:
                    # Sample binary mask and get REINFORCE gradients (matching original)
                    # Ensure at least 1 sample is selected to avoid zero-selection issues
                    min_selections = max(1, self.coreset_size // 10)  # At least 10% of target, but min 1
                    subnet, grad = self.obtain_mask(scores, min_selections=min_selections)
                    # Append grad immediately (matching original: always append K grads)
                    grad_list.append(grad)
                    
                    # Detach subnet and convert to indices (matching original index-based selection)
                    subnet_detached = subnet.detach()
                    indices = torch.nonzero(subnet_detached.squeeze())
                    if len(indices.shape) > 1:
                        indices = indices.reshape(len(indices))
                    indices_np = indices.cpu().numpy().flatten()
                    
                    # Select data by indexing (matching original) instead of masking
                    if len(indices_np) == 0:
                        # If still no samples selected (shouldn't happen with min_selections > 0, but handle it)
                        print(f"Warning: No samples selected in attempt {attempts+1}, using top-k fallback")
                        # Fallback: select top-k by scores
                        k = max(1, min(self.coreset_size, len(scores) // 2))
                        _, top_indices = torch.topk(scores, k)
                        indices_np = top_indices.cpu().numpy().flatten()
                        # Recompute grad for top-k selection
                        subnet_fallback = torch.zeros_like(scores)
                        subnet_fallback[top_indices] = 1.0
                        eps = 1e-20
                        grad_list[-1] = (subnet_fallback - scores) / ((scores + eps) * (1 - scores + eps))
                    
                    x_coreset = X[indices_np]
                    y_coreset = Y[indices_np]
                    
                    # Train model on coreset (matching original: all data at once, no inner batching)
                    trained_model, loss, acc1, acc5, diverged = self.train_to_converge(
                        model, x_coreset, y_coreset
                    )
                    
                    attempts += 1
                    if diverged:
                        print(f"Training diverged, attempt {attempts}/{max_attempts}, loss {loss:.4f}")
                        # Remove the grad we added since training diverged and we'll retry
                        if len(grad_list) > i:
                            grad_list.pop()
                
                # Always append loss value (matching original: always append K losses)
                if not diverged and indices_np is not None and len(indices_np) > 0 and trained_model is not None:
                    all_models.append(trained_model)
                    # Evaluate on full dataset (matching original: all data at once)
                    loss_full = self.get_loss_on_full_data(trained_model, X, Y)
                    fn_list.append(loss_full)
                    fn_avg += loss_full / self.K
                    print(f"Sample {i}: Full loss = {loss_full:.4f}")
                    # Clean up GPU memory (matching original)
                    del trained_model
                    torch.cuda.empty_cache() if torch.cuda.is_available() else None
                else:
                    # Handle failure case - use dummy loss (matching original behavior)
                    fn_list.append(1000.0)
                    fn_avg += 1000.0 / self.K
                    print(f"Sample {i}: Failed, using dummy loss")
                
                # Ensure we have exactly one grad for this iteration
                # If we don't have one yet (e.g., all attempts failed), add a zero grad
                if len(grad_list) <= i:
                    grad_list.append(torch.zeros_like(scores))
            
            # Update scores (matching original)
            scores_opt.zero_grad()
            
            # Ensure grad_list and fn_list have matching lengths
            if len(grad_list) != len(fn_list):
                print(f"Warning: Mismatch in grad_list ({len(grad_list)}) and fn_list ({len(fn_list)}) lengths")
                # Trim to minimum length
                min_len = min(len(grad_list), len(fn_list))
                grad_list = grad_list[:min_len]
                fn_list = fn_list[:min_len]
            
            if len(fn_list) > 0:
                if self.use_variance_reduction:
                    self.calculate_grad_vr(scores, fn_list, grad_list, fn_avg)
                else:
                    self.calculate_grad(scores, fn_list, grad_list)
            
            if self.clip_grad:
                torch.nn.utils.clip_grad_norm_([scores], self.clip_constant)
            
            scores_opt.step()
            
            # Constrain scores to maintain coreset size (matching original)
            self.constrain_scores(scores, self.coreset_size)
            
            # Select best model (matching original)
            if fn_list and len(all_models) > 0:
                best_idx = np.argmin(fn_list)
                best_model = all_models[best_idx] if best_idx < len(all_models) else model
                print(f"Best model loss: {fn_list[best_idx]:.4f}, avg: {sum(fn_list)/self.K:.4f}")
            else:
                best_model = model
            
            print(f"Average function value: {fn_avg:.4f}")
        
        # Final selection using learned probabilities (matching original)
        # Sample multiple times and pick best (matching original implementation)
        sample_times = 1  # Can be increased for more robust final selection
        best_loss = float('inf')
        best_indices = None
        
        # Ensure we get at least coreset_size samples in final selection
        min_final_selections = max(1, self.coreset_size)
        
        for sample_idx in range(sample_times):
            # Sample with minimum guarantee to avoid zero-selection issues
            final_subnet, _ = self.obtain_mask(scores, min_selections=min_final_selections)
            indices = torch.nonzero(final_subnet.squeeze())
            if len(indices.shape) > 1:
                indices = indices.reshape(len(indices))
            indices_np = indices.cpu().numpy().flatten()
            
            # Ensure we have enough samples
            if len(indices_np) >= min_final_selections:
                x_coreset = X[indices_np]
                y_coreset = Y[indices_np]
                # Train final model and evaluate
                trained_model, loss, acc1, acc5, _ = self.train_to_converge(
                    model, x_coreset, y_coreset
                )
                loss_full = self.get_loss_on_full_data(trained_model, X, Y)
                
                if loss_full < best_loss:
                    best_loss = loss_full
                    best_indices = indices_np
                    best_model = trained_model
                del trained_model
                torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
        # If we got valid indices, use them; otherwise use top-k by scores
        if best_indices is not None and len(best_indices) >= min_final_selections:
            selected_indices = best_indices.reshape(len(best_indices))
            print(f"Final selected {len(selected_indices)} samples out of {num_samples}")
            print(f"Selection ratio: {len(selected_indices)/num_samples:.4f}")
        else:
            # Fallback: select top-k by scores (ensure we get coreset_size samples)
            k = min(max(self.coreset_size, 1), len(scores))
            _, top_indices = torch.topk(scores, k)
            selected_indices = top_indices.cpu().numpy().flatten()
            print(f"Fallback: Selected {len(selected_indices)} samples using top-k (requested {self.coreset_size})")
        
        return selected_indices, scores, best_model


class PixelLevelCoresetSelector(ProbabilisticCoresetSelector):
    """
    Pixel-level coreset selection for image data.
    Extends the base selector for pixel-wise selection.
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.selection_level = 'pixel'
    
    def select_coreset(self, model: nn.Module, X: torch.Tensor, Y: torch.Tensor, 
                      task_id: int = 0) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Pixel-level coreset selection.
        
        Args:
            model: Model to use for selection
            X: Input images [N, C, H, W]
            Y: Labels [N]
            task_id: Task identifier
            
        Returns:
            selected_indices: Indices of selected samples
            selection_scores: Learned pixel selection probabilities [H, W]
            selected_model: Best model from optimization
        """
        print(f"Starting pixel-level coreset selection for task {task_id}")
        print(f"Image shape: {X.shape}, Target coreset size: {self.coreset_size}")
        
        # Ensure data is on the correct device
        X = X.to(self.device)
        Y = Y.to(self.device)
        print(f"Data moved to device: {X.device}")
        
        # Initialize pixel-level selection scores
        if len(X.shape) == 4:  # [N, C, H, W]
            H, W = X.shape[2], X.shape[3]
            pr_target = self.coreset_size / (H * W)
            scores = torch.full([H, W], pr_target, dtype=torch.float, 
                               requires_grad=True, device=self.device)
        else:
            raise ValueError("Expected 4D tensor for pixel-level selection")
            
        scores_opt = torch.optim.Adam([scores], lr=self.outer_lr)
        
        print(f"Initial pixel selection probability: {pr_target:.4f}")
        
        # Outer optimization loop
        for outer_iter in range(self.max_outer_iter):
            print(f"Outer iteration {outer_iter}/{self.max_outer_iter}")
            
            # Cosine annealing for outer learning rate
            lr = 0.5 * (1 + np.cos(np.pi * outer_iter / self.max_outer_iter)) * self.outer_lr
            for param_group in scores_opt.param_groups:
                param_group['lr'] = lr
            
            fn_list = []
            grad_list = []
            fn_avg = 0
            all_models = []
            
            # Sample K models
            for i in range(self.K):
                diverged = True
                attempts = 0
                max_attempts = 5
                
                while diverged and attempts < max_attempts:
                    # Sample pixel mask (pixel-level uses mask-based, not index-based)
                    subnet, grad = self.obtain_mask(scores)
                    grad_list.append(grad)
                    
                    # Apply pixel mask to all images (pixel-level selection uses masking)
                    subnet_expanded = subnet.unsqueeze(0).unsqueeze(0).expand_as(X)
                    x_coreset = X * subnet_expanded
                    y_coreset = Y
                    
                    # Train model on masked data (matching original: all data at once)
                    trained_model, loss, acc1, acc5, diverged = self.train_to_converge(
                        model, x_coreset, y_coreset
                    )
                    
                    attempts += 1
                    if diverged:
                        print(f"Training diverged, attempt {attempts}/{max_attempts}, loss {loss:.4f}")
                
                if not diverged:
                    all_models.append(trained_model)
                    # Evaluate on full dataset
                    loss_full = self.get_loss_on_full_data(trained_model, X, Y)
                    fn_list.append(loss_full)
                    fn_avg += loss_full / self.K
                    print(f"Sample {i}: Full loss = {loss_full:.4f}")
                else:
                    print(f"Failed to train model for sample {i}")
                    fn_list.append(1000.0)
                    fn_avg += 1000.0 / self.K
            
            # Update scores
            scores_opt.zero_grad()
            
            if self.use_variance_reduction:
                self.calculate_grad_vr(scores, fn_list, grad_list, fn_avg)
            else:
                self.calculate_grad(scores, fn_list, grad_list)
            
            if self.clip_grad:
                torch.nn.utils.clip_grad_norm_([scores], self.clip_constant)
            
            scores_opt.step()
            
            # Constrain scores to maintain coreset size
            self.constrain_scores(scores, self.coreset_size)
            
            # Select best model
            if fn_list:
                best_idx = np.argmin(fn_list)
                best_model = all_models[best_idx] if best_idx < len(all_models) else model
                print(f"Best model loss: {fn_list[best_idx]:.4f}")
            else:
                best_model = model
            
            print(f"Average function value: {fn_avg:.4f}")
            
            # Early stopping if converged
            if outer_iter > 10 and abs(fn_avg - np.mean(fn_list[-5:])) < 1e-4:
                print("Converged early")
                break
        
        # For pixel-level selection, we return the pixel mask instead of sample indices
        final_subnet = (torch.rand_like(scores) < scores).float()
        selected_pixels = final_subnet
        
        print(f"Selected {selected_pixels.sum().item()} pixels out of {scores.numel()}")
        print(f"Selection ratio: {selected_pixels.sum().item()/scores.numel():.4f}")
        
        return selected_pixels, scores, best_model





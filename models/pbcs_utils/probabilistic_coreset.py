"""
Probabilistic Coreset Selection utilities for Bilevel Optimization.
Adapted from Probabilistic-Bilevel-Coreset-Selection for mammoth framework.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from typing import Tuple, List, Optional


class ProbabilisticCoresetSelector:
    """
    Probabilistic coreset selection using bilevel optimization.
    Implements the core selection logic from the original paper.
    """
    
    def __init__(self, device='cuda', K=5, outer_lr=0.1, inner_lr=0.005, 
                 max_outer_iter=100, epoch_converge=50, coreset_size=100,
                 use_variance_reduction=True, clip_grad=True, clip_constant=3.0):
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
        
    def obtain_mask(self, scores: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Sample binary mask from Bernoulli distribution and compute REINFORCE gradients.
        
        Args:
            scores: Selection probabilities [N] or [H, W] for pixel-level selection
            
        Returns:
            subnet: Binary mask sampled from Bernoulli distribution
            grad: REINFORCE gradients for score updates
        """
        subnet = (torch.rand_like(scores) < scores).float()
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
        for i in range(self.K):
            scores.grad.data += 1 / (self.K-1) * (fn_list[i]-fn_avg) * grad_list[i]
    
    def calculate_grad(self, scores: torch.Tensor, fn_list: List[float], 
                      grad_list: List[torch.Tensor]):
        """Calculate standard gradients."""
        for i in range(self.K):
            scores.grad.data += 1/self.K * fn_list[i] * grad_list[i]
    
    def train_to_converge(self, model: nn.Module, x_coreset: torch.Tensor, 
                         y_coreset: torch.Tensor, optimizer, epoch_converge: int = None) -> Tuple[nn.Module, float, float, float, bool]:
        """
        Train model to convergence on coreset data.
        
        Args:
            model: Model to train
            x_coreset: Coreset input data
            y_coreset: Coreset labels
            optimizer: Optimizer for training
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
            
        model_copy = model
        data, target = x_coreset.to(self.device), y_coreset.to(self.device)
        diverged = False
        
        for i in range(epoch_converge):
            # Cosine annealing learning rate
            lr = 0.5 * (1 + np.cos(np.pi * i / epoch_converge)) * self.inner_lr
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
                
            optimizer.zero_grad()
            output = model_copy(data)
            loss = F.cross_entropy(output, target)
            loss.backward()
            optimizer.step()
            
            if i % 10 == 0:
                print(f"Epoch {i}, loss: {loss.item():.4f}")
                
        if math.isnan(loss.item()) or loss.item() > 10.0:  # Divergence threshold
            diverged = True
            
        # Calculate accuracy
        with torch.no_grad():
            _, pred = output.topk(5, 1, True, True)
            pred = pred.t()
            correct = pred.eq(target.view(1, -1).expand_as(pred))
            acc1 = correct[:1].flatten().float().sum(0, keepdim=True).mul_(100.0 / target.size(0)).item()
            acc5 = correct[:5].flatten().float().sum(0, keepdim=True).mul_(100.0 / target.size(0)).item()
            
        return model_copy, loss.item(), acc1, acc5, diverged
    
    def get_loss_on_full_data(self, model: nn.Module, X: torch.Tensor, Y: torch.Tensor) -> float:
        """Calculate loss on full dataset."""
        with torch.no_grad():
            data, target = X.to(self.device), Y.to(self.device)
            output = model(data)
            loss = F.cross_entropy(output, target)
        return loss.item()
    
    def select_coreset(self, model: nn.Module, X: torch.Tensor, Y: torch.Tensor, 
                      task_id: int = 0) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Main coreset selection using probabilistic bilevel optimization.
        
        Args:
            model: Model to use for selection
            X: Input data [N, ...]
            Y: Labels [N]
            task_id: Task identifier
            
        Returns:
            selected_indices: Indices of selected samples
            selection_scores: Learned selection probabilities
            selected_model: Best model from optimization
        """
        print(f"Starting probabilistic coreset selection for task {task_id}")
        print(f"Data shape: {X.shape}, Target coreset size: {self.coreset_size}")
        
        # Initialize selection scores
        num_samples = X.shape[0]
        pr_target = self.coreset_size / num_samples
        scores = torch.full([num_samples], pr_target, dtype=torch.float, 
                           requires_grad=True, device=self.device)
        scores_opt = torch.optim.Adam([scores], lr=self.outer_lr)
        
        print(f"Initial selection probability: {pr_target:.4f}")
        
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
                    # Sample binary mask
                    subnet, grad = self.obtain_mask(scores)
                    grad_list.append(grad)
                    
                    # Apply mask to data
                    if len(X.shape) == 4:  # Image data
                        subnet_expanded = subnet.view(-1, 1, 1, 1).expand_as(X)
                        x_coreset = X * subnet_expanded
                    else:  # Vector data
                        subnet_expanded = subnet.view(-1, 1).expand_as(X)
                        x_coreset = X * subnet_expanded
                    
                    y_coreset = Y
                    
                    # Train model on coreset
                    model_copy = model.__class__(**model.__dict__).to(self.device)
                    model_copy.load_state_dict(model.state_dict())
                    optimizer = torch.optim.Adam(model_copy.parameters(), lr=self.inner_lr)
                    
                    trained_model, loss, acc1, acc5, diverged = self.train_to_converge(
                        model_copy, x_coreset, y_coreset, optimizer
                    )
                    
                    attempts += 1
                    if diverged:
                        print(f"Training diverged, attempt {attempts}/{max_attempts}")
                
                if not diverged:
                    all_models.append(trained_model)
                    # Evaluate on full dataset
                    loss_full = self.get_loss_on_full_data(trained_model, X, Y)
                    fn_list.append(loss_full)
                    fn_avg += loss_full / self.K
                    print(f"Sample {i}: Full loss = {loss_full:.4f}")
                else:
                    print(f"Failed to train model for sample {i}")
                    # Use a dummy loss if all attempts failed
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
        
        # Final selection using learned probabilities
        final_subnet = (torch.rand_like(scores) < scores).float()
        selected_indices = torch.nonzero(final_subnet).squeeze()
        
        if len(selected_indices.shape) == 0:
            selected_indices = selected_indices.unsqueeze(0)
        
        print(f"Selected {len(selected_indices)} samples out of {num_samples}")
        print(f"Selection ratio: {len(selected_indices)/num_samples:.4f}")
        
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
                    # Sample pixel mask
                    subnet, grad = self.obtain_mask(scores)
                    grad_list.append(grad)
                    
                    # Apply pixel mask to all images
                    subnet_expanded = subnet.unsqueeze(0).unsqueeze(0).expand_as(X)
                    x_coreset = X * subnet_expanded
                    y_coreset = Y
                    
                    # Train model on masked data
                    model_copy = model.__class__(**model.__dict__).to(self.device)
                    model_copy.load_state_dict(model.state_dict())
                    optimizer = torch.optim.Adam(model_copy.parameters(), lr=self.inner_lr)
                    
                    trained_model, loss, acc1, acc5, diverged = self.train_to_converge(
                        model_copy, x_coreset, y_coreset, optimizer
                    )
                    
                    attempts += 1
                    if diverged:
                        print(f"Training diverged, attempt {attempts}/{max_attempts}")
                
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

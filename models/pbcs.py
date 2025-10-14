"""
Probabilistic Bilevel Coreset Selection (PBCS) for Continual Learning.
Adapted from Probabilistic-Bilevel-Coreset-Selection for mammoth framework.

This implementation integrates probabilistic coreset selection with continual learning
using bilevel optimization to select the most informative samples for replay.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import copy
from typing import Tuple, List, Optional

from models.utils.continual_model import ContinualModel
from utils.args import add_rehearsal_args, ArgumentParser
from utils.buffer import Buffer
from models.pbcs_utils.probabilistic_coreset import ProbabilisticCoresetSelector, PixelLevelCoresetSelector


class PBCS(ContinualModel):
    """
    Probabilistic Bilevel Coreset Selection for Continual Learning.
    
    This model uses probabilistic bilevel optimization to select the most informative
    samples for experience replay, improving the efficiency of continual learning.
    """
    
    NAME = 'pbcs'
    COMPATIBILITY = ['class-il', 'domain-il', 'task-il', 'general-continual']
    
    @staticmethod
    def get_parser(parser) -> ArgumentParser:
        """
        Returns an ArgumentParser object with predefined arguments for the PBCS model.
        """
        add_rehearsal_args(parser)
        
        # PBCS-specific arguments
        parser.add_argument('--pbcs_K', type=int, default=5, 
                           help='Number of samples per outer iteration')
        parser.add_argument('--pbcs_outer_lr', type=float, default=0.1,
                           help='Learning rate for outer optimization')
        parser.add_argument('--pbcs_inner_lr', type=float, default=0.005,
                           help='Learning rate for inner optimization')
        parser.add_argument('--pbcs_max_outer_iter', type=int, default=50,
                           help='Maximum outer iterations')
        parser.add_argument('--pbcs_epoch_converge', type=int, default=30,
                           help='Epochs for inner optimization convergence')
        parser.add_argument('--pbcs_use_vr', action='store_true',
                           help='Use variance reduction')
        parser.add_argument('--pbcs_clip_grad', action='store_true',
                           help='Clip gradients')
        parser.add_argument('--pbcs_clip_constant', type=float, default=3.0,
                           help='Gradient clipping constant')
        parser.add_argument('--pbcs_selection_type', type=str, default='sample',
                           choices=['sample', 'pixel'],
                           help='Type of selection: sample-level or pixel-level')
        parser.add_argument('--pbcs_coreset_ratio', type=float, default=0.1,
                           help='Ratio of data to select for coreset')
        
        return parser
    
    def __init__(self, backbone, loss, args, transform, dataset=None):
        """
        Initialize PBCS model.
        
        Args:
            backbone: Neural network backbone
            loss: Loss function
            args: Command line arguments
            transform: Data transformations
            dataset: Dataset object
        """
        super(PBCS, self).__init__(backbone, loss, args, transform, dataset=dataset)
        
        # Initialize buffer
        self.buffer = Buffer(self.args.buffer_size)
        
        # PBCS parameters
        self.K = getattr(args, 'pbcs_K', 5)
        self.outer_lr = getattr(args, 'pbcs_outer_lr', 0.1)
        self.inner_lr = getattr(args, 'pbcs_inner_lr', 0.005)
        self.max_outer_iter = getattr(args, 'pbcs_max_outer_iter', 50)
        self.epoch_converge = getattr(args, 'pbcs_epoch_converge', 30)
        self.use_vr = getattr(args, 'pbcs_use_vr', False)
        self.clip_grad = getattr(args, 'pbcs_clip_grad', False)
        self.clip_constant = getattr(args, 'pbcs_clip_constant', 3.0)
        self.selection_type = getattr(args, 'pbcs_selection_type', 'sample')
        self.coreset_ratio = getattr(args, 'pbcs_coreset_ratio', 0.1)
        
        # Initialize coreset selector
        if self.selection_type == 'pixel':
            self.coreset_selector = PixelLevelCoresetSelector(
                device=self.device,
                K=self.K,
                outer_lr=self.outer_lr,
                inner_lr=self.inner_lr,
                max_outer_iter=self.max_outer_iter,
                epoch_converge=self.epoch_converge,
                coreset_size=int(self.coreset_ratio * args.buffer_size),
                use_variance_reduction=self.use_vr,
                clip_grad=self.clip_grad,
                clip_constant=self.clip_constant
            )
        else:
            self.coreset_selector = ProbabilisticCoresetSelector(
                device=self.device,
                K=self.K,
                outer_lr=self.outer_lr,
                inner_lr=self.inner_lr,
                max_outer_iter=self.max_outer_iter,
                epoch_converge=self.epoch_converge,
                coreset_size=int(self.coreset_ratio * args.buffer_size),
                use_variance_reduction=self.use_vr,
                clip_grad=self.clip_grad,
                clip_constant=self.clip_constant
            )
        
        # Store task data for coreset selection
        self.task_data = []
        self.task_labels = []
        
        print(f"PBCS initialized with {self.selection_type}-level selection")
        print(f"Coreset size: {self.coreset_selector.coreset_size}")
        print(f"Buffer size: {self.args.buffer_size}")
    
    def observe(self, inputs, labels, not_aug_inputs, epoch=None):
        """
        Observe a batch of data and perform training with replay.
        
        Args:
            inputs: Current batch inputs
            labels: Current batch labels
            not_aug_inputs: Non-augmented inputs
            epoch: Current epoch number
            
        Returns:
            loss: Training loss
        """
        real_batch_size = inputs.shape[0]
        
        # Store current task data
        self.task_data.append(not_aug_inputs.cpu())
        self.task_labels.append(labels.cpu())
        
        # Get replay data from buffer
        if not self.buffer.is_empty():
            buf_inputs, buf_labels = self.buffer.get_data(
                self.args.minibatch_size, transform=self.transform
            )
            buf_inputs= buf_inputs.to(self.device)
            buf_labels= buf_labels.to(self.device)
            inputs = torch.cat((inputs, buf_inputs))
            labels = torch.cat((labels, buf_labels))
        
        # Forward pass
        outputs = self.net(inputs)
        loss = self.loss(outputs, labels)
        
        # Backward pass
        self.opt.zero_grad()
        loss.backward()
        self.opt.step()
        
        return loss.item()
    
    def end_task(self, dataset):
        """
        Called at the end of each task to perform coreset selection.
        
        Args:
            dataset: Current dataset
        """
        print(f"End of task {self.current_task}: Performing coreset selection")
        
        if len(self.task_data) == 0:
            print("No task data available for coreset selection")
            return
        
        # Concatenate all task data
        task_inputs = torch.cat(self.task_data, dim=0)
        task_labels = torch.cat(self.task_labels, dim=0)
        
        print(f"Task data shape: {task_inputs.shape}, Labels: {task_labels.shape}")
        
        # Determine coreset size based on buffer capacity
        remaining_slots = self.args.buffer_size - len(self.buffer)
        if remaining_slots <= 0:
            print("Buffer is full, skipping coreset selection")
            return
        
        # Calculate coreset size for this task
        coreset_size = min(remaining_slots, int(self.coreset_ratio * len(task_inputs)))
        self.coreset_selector.coreset_size = coreset_size
        
        print(f"Selecting {coreset_size} samples from {len(task_inputs)} task samples")
        
        try:
            # Perform coreset selection
            if self.selection_type == 'pixel':
                # Pixel-level selection
                selected_pixels, selection_scores, best_model = self.coreset_selector.select_coreset(
                    self.net, task_inputs, task_labels, self.current_task
                )
                
                # Apply pixel mask to select samples
                if len(task_inputs.shape) == 4:  # [N, C, H, W]
                    pixel_mask = selected_pixels.unsqueeze(0).unsqueeze(0).expand_as(task_inputs)
                    masked_inputs = task_inputs * pixel_mask
                    
                    # Select samples based on pixel importance
                    pixel_importance = selected_pixels.sum(dim=(1, 2)) if len(selected_pixels.shape) == 3 else selected_pixels.sum()
                    _, selected_indices = torch.topk(pixel_importance, coreset_size)
                    
                else:
                    print("Pixel-level selection requires 4D input tensors")
                    return
                    
            else:
                # Sample-level selection
                selected_indices, selection_scores, best_model = self.coreset_selector.select_coreset(
                    self.net, task_inputs, task_labels, self.current_task
                )
            
            # Add selected samples to buffer
            if len(selected_indices) > 0:
                selected_inputs = task_inputs[selected_indices]
                selected_labels = task_labels[selected_indices]

                
                # Add to buffer
                self.buffer.add_data(
                    examples=selected_inputs.to(self.device),
                    labels=selected_labels.to(self.device)
                )
                
                print(f"Added {len(selected_inputs)} samples to buffer")
                print(f"Buffer now contains {len(self.buffer)} samples")
                
                # Update model with best model from optimization
                if best_model is not None:
                    self.net.load_state_dict(best_model.state_dict())
                    print("Updated model with best model from coreset selection")
            
        except Exception as e:
            print(f"Coreset selection failed: {e}")
            print("Falling back to random selection")
            
            # Fallback to random selection
            num_samples = min(coreset_size, len(task_inputs))
            random_indices = torch.randperm(len(task_inputs))[:num_samples]
            selected_inputs = task_inputs[random_indices]
            selected_labels = task_labels[random_indices]
            
            self.buffer.add_data(
                examples=selected_inputs.to(self.device),
                labels=selected_labels.to(self.device)
            )
            
            print(f"Added {len(selected_inputs)} randomly selected samples to buffer")
        
        # Clear task data
        self.task_data = []
        self.task_labels = []
        
        print(f"Task {self.current_task} completed")
    
    def begin_task(self, dataset):
        """
        Called at the beginning of each task.
        
        Args:
            dataset: Current dataset
        """
        print(f"Beginning task {self.current_task}")
        
        # Clear task data for new task
        self.task_data = []
        self.task_labels = []
    
    def get_debug_iters(self):
        """Return number of iterations for debugging."""
        return 3
    
    def get_optimizer(self):
        """Return optimizer for training."""
        return torch.optim.Adam(self.net.parameters(), lr=self.args.lr)
    
    def load_buffer(self, buffer):
        """Load buffer from checkpoint."""
        self.buffer = buffer
        print(f"Loaded buffer with {len(self.buffer)} samples")
    
    def get_buffer(self):
        """Get current buffer."""
        return self.buffer
    
    def get_selection_scores(self):
        """Get learned selection scores (for analysis)."""
        if hasattr(self.coreset_selector, 'last_scores'):
            return self.coreset_selector.last_scores
        return None
    
    def get_selection_statistics(self):
        """Get selection statistics for analysis."""
        stats = {
            'buffer_size': len(self.buffer),
            'current_task': self.current_task,
            'selection_type': self.selection_type,
            'coreset_ratio': self.coreset_ratio,
            'K': self.K,
            'outer_lr': self.outer_lr,
            'inner_lr': self.inner_lr,
            'max_outer_iter': self.max_outer_iter,
            'epoch_converge': self.epoch_converge,
            'use_variance_reduction': self.use_vr,
            'clip_grad': self.clip_grad
        }
        return stats

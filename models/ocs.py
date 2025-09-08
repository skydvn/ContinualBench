"""
This module implements the Online Continual Learning with Selective Memory (OCS) method.
OCS uses gradient-based sample selection to maintain a coreset of representative examples
for rehearsal-based continual learning.

Reference: "Online Continual Learning with Selective Memory Replay" (ICLR 2021)
"""

import torch
import numpy as np
import copy
from argparse import ArgumentParser
from typing import Optional, Tuple

from models.utils.continual_model import ContinualModel
from utils.buffer import Buffer
from utils.args import add_rehearsal_args
from models.ocs_utils.ocs_core_utils import (
    sample_selection, classwise_fair_selection, compute_and_flatten_example_grads,
    get_coreset_loss, reconstruct_coreset_loader2
)


class OCS(ContinualModel):
    """
    Online Continual Learning with Selective Memory (OCS).
    
    OCS maintains a coreset of representative examples by:
    1. Computing per-example gradients for candidate samples
    2. Ranking samples using OCS measure: mean_sim - mean_div + tau * coreset_aff
    3. Performing class-fair selection to maintain balanced representation
    4. Using the coreset for rehearsal during training
    """
    NAME = 'ocs'
    COMPATIBILITY = ['class-il', 'domain-il', 'task-il', 'general-continual']

    @staticmethod
    def get_parser(parser: ArgumentParser) -> ArgumentParser:
        # Standard rehearsal arguments (buffer_size, minibatch_size, ...)
        add_rehearsal_args(parser)
        # OCS-specific arguments
        parser.add_argument('--ocs_tau', type=float, default=1000.0,
                            help='Affinity weight for OCS selection (tau term)')
        parser.add_argument('--ocs_ref_hyp', type=float, default=0.5,
                            help='Reference hypothesis weight for coreset loss')
        parser.add_argument('--ocs_batch_size', type=int, default=10,
                            help='Batch size for OCS sample selection')
        parser.add_argument('--ocs_r2c_iter', type=int, default=100,
                            help='Number of random-to-coreset iterations')
        parser.add_argument('--ocs_is_r2c', action='store_true', default=True,
                            help='Enable random-to-coreset warmup')
        parser.add_argument('--ocs_select_type', type=str, default='ocs_select',
                            help='Selection type for coreset building (ocs, uniform, entropy, hardest)')
        parser.add_argument('--ocs_grad_batch_size', type=int, default=16,
                            help='Batch size for gradient computation (reduced for memory)')
        return parser

    def __init__(self, backbone, loss, args, transform, dataset=None):
        super().__init__(backbone, loss, args, transform, dataset=dataset)
        self.buffer = Buffer(self.args.buffer_size, device=self.device)
        
        # OCS-specific parameters
        self.tau = getattr(args, 'ocs_tau', 1000.0)
        self.ref_hyp = getattr(args, 'ocs_ref_hyp', 0.5)
        self.ocs_batch_size = getattr(args, 'ocs_batch_size', 10)
        self.r2c_iter = getattr(args, 'ocs_r2c_iter', 100)
        self.is_r2c = getattr(args, 'ocs_is_r2c', True)
        self.select_type = getattr(args, 'ocs_select_type', 'ocs_select')
        self.grad_batch_size = getattr(args, 'ocs_grad_batch_size', 32)
        
        # Internal state
        self.ref_grads = None
        self.ocs_iteration = 0  # Custom counter for OCS random-to-coreset iterations
        
        # Configuration for OCS core functions
        self.ocs_config = {
            'tau': self.tau,
            'memory_size': self.args.buffer_size,
            'n_classes': self.cpt,
            'batch_size': self.ocs_batch_size,
            'r2c_iter': self.r2c_iter,
            'is_r2c': self.is_r2c,
            'select_type': self.select_type,
            'ref_hyp': self.ref_hyp
        }

    def observe(self, inputs, labels, not_aug_inputs, epoch=None):
        """
        OCS training step with coreset rehearsal and gradient-based sample selection.
        Based on the original train_ocs_single_step from train_methods_cifar.py.
        """
        real_batch_size = inputs.shape[0]
        self.opt.zero_grad()
        
        # Check if this is random start (first few iterations of first step)
        is_rand_start = (self.ocs_iteration < self.r2c_iter and self.is_r2c)
        
        # Get rehearsal data from buffer for reference gradient computation
        ref_grads = None
        if not self.buffer.is_empty() and self.current_task > 0:
            buf_inputs, buf_labels = self.buffer.get_data(
                self.args.minibatch_size, transform=self.transform, device=self.device
            )
            
            # Compute reference gradients from coreset using a copy to avoid interfering with main training
            model_copy = copy.deepcopy(self.net)
            model_copy.eval()
            
            # Ensure gradients are enabled for the model copy
            for param in model_copy.parameters():
                param.requires_grad_(True)
            
            model_copy.zero_grad(set_to_none=True)
            criterion = torch.nn.CrossEntropyLoss()
            ref_pred = model_copy(buf_inputs)
            ref_loss = criterion(ref_pred, buf_labels)
            ref_loss.backward()
            ref_grads = self._flatten_grads_from_model(model_copy)
            model_copy.zero_grad(set_to_none=True)
            
            # Clean up model copy and intermediate tensors
            del model_copy, ref_pred, ref_loss
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
        # Sample selection for current batch
        if is_rand_start:
            # Random selection for warmup
            size = min(len(inputs), self.ocs_batch_size)
            pick = torch.randperm(len(inputs))[:size]
        else:
            # OCS-based selection
            self.net.eval()
            criterion = torch.nn.CrossEntropyLoss()
            _eg = compute_and_flatten_example_grads(self.net, criterion, inputs, labels)
            _g = torch.mean(_eg, 0)
            sorted_idx = sample_selection(_g, _eg, self.ocs_config, ref_grads=ref_grads)
            pick = torch.from_numpy(sorted_idx[:self.ocs_batch_size]).to(self.device)
            
            # Clean up gradient computation tensors
            del _eg, _g
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
        # Training step
        self.net.train()
        self.opt.zero_grad()
        
        # Forward pass on selected samples
        selected_inputs = inputs[pick]
        selected_labels = labels[pick]
        outputs = self.net(selected_inputs)
        loss = self.loss(outputs, selected_labels)
        
        # Add reference loss if available
        if ref_grads is not None and not self.buffer.is_empty():
            ref_loss = get_coreset_loss(self.net, (buf_inputs, buf_labels, None), self.ocs_config)
            loss += self.ref_hyp * ref_loss
        
        loss.backward()
        self.opt.step()

        # Add current samples to buffer (will be trimmed at end_task)
        self.buffer.add_data(examples=not_aug_inputs,
                             labels=labels[:real_batch_size])
        
        # Clean up tensors and memory
        del selected_inputs, selected_labels, outputs
        if ref_grads is not None:
            del ref_grads
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
        # Increment OCS iteration counter
        self.ocs_iteration += 1
        return loss.item()

    def _flatten_grads(self):
        """Flatten gradients from model parameters."""
        total_grads = []
        for name, param in self.net.named_parameters():
            if param.grad is not None and not 'bn' in name and not 'IC' in name:
                total_grads.append(param.grad.detach().view(-1))
        return torch.cat(total_grads) if total_grads else torch.tensor([])
    
    def _flatten_grads_from_model(self, model):
        """Flatten gradients from a specific model's parameters."""
        total_grads = []
        for name, param in model.named_parameters():
            if param.grad is not None and not 'bn' in name and not 'IC' in name:
                total_grads.append(param.grad.detach().view(-1))
        return torch.cat(total_grads) if total_grads else torch.tensor([])
    
    def _cleanup_memory(self):
        """Clean up GPU memory."""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()


    def _fair_reduce_existing_buffer(self) -> None:
        """
        Fairly reduce existing buffer to make space for new task samples.
        """
        if self.current_task <= 0 or len(self.buffer) == 0:
            return
            
        # Fairly keep the same number of examples per seen class
        examples_per_class = self.args.buffer_size // (self.cpt * self.current_task)
        buf_x, buf_y = self.buffer.get_all_data()
        self.buffer.empty()
        
        for cid in buf_y.unique():
            idx = (buf_y == cid)
            keep = min(idx.sum().item(), examples_per_class)
            if keep > 0:
                self.buffer.add_data(examples=buf_x[idx][:keep], labels=buf_y[idx][:keep])

    def end_task(self, dataset):
        """
        End of task processing: perform OCS sample selection and update coreset.
        Based on the original select_coreset function from train_methods_cifar.py.
        """
        
        # 1) Reduce existing buffer fairly across seen classes
        self._fair_reduce_existing_buffer()

        # 2) Gather current task candidates
        xs, ys = [], []
        for batch in dataset.train_loader:
            _, y, not_aug = batch[0], batch[1], batch[2]
            xs.append(not_aug.cpu())
            ys.append(y.cpu())
            
        if len(xs) == 0:
            return
            
        X = torch.cat(xs, dim=0)
        Y = torch.cat(ys, dim=0)

        # 3) Check remaining buffer space
        remaining = self.args.buffer_size - len(self.buffer)
        if remaining <= 0:
            return

        # 4) Perform OCS sample selection
        if self.select_type == 'ocs_select':
            # Create a copy of the model for gradient computation to avoid interfering with main training
            model_copy = copy.deepcopy(self.net)
            model_copy.eval()
            
            # Ensure gradients are enabled for the model copy
            for param in model_copy.parameters():
                param.requires_grad_(True)
            
            # Compute per-example gradients for OCS selection using the copy
            criterion = torch.nn.CrossEntropyLoss()
            _eg = compute_and_flatten_example_grads(model_copy, criterion, X.to(self.device), Y.to(self.device))
            _g = torch.mean(_eg, 0)
            sorted_idx = sample_selection(_g, _eg, self.ocs_config)
            
            # Class-wise fair selection
            num_per_label = [len((Y == (jj + self.cpt * (self.current_task - 1))).nonzero()) for jj in range(self.cpt)]
            selected = classwise_fair_selection(
                self.current_task, Y.numpy(), sorted_idx, num_per_label, self.ocs_config, is_shuffle=True
            )
            
            # Add selected samples to buffer
            sel_x = X[selected].to(self.device)
            sel_y = Y[selected].to(self.device)
            
            if sel_x.shape[0] > 0:
                self.buffer.add_data(examples=sel_x, labels=sel_y)
            
            # Clean up model copy and large tensors
            del model_copy, _eg, _g, X, Y
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
        # Reset OCS iteration counter for new task
        self.ocs_iteration = 0

    def begin_task(self, dataset):
        """
        Initialize task-specific variables.
        """
        super().begin_task(dataset)
        # Reset OCS iteration counter for new task
        self.ocs_iteration = 0 

"""
Enhanced CSReL (Coreset Selection for Continual Learning) implementation for mammoth framework.
Based on the CSReL-Coreset-CL implementation with full integration.

Reference: "CSReL: Coreset Selection for Continual Learning" (ICLR 2022)
"""

import torch
import numpy as np
import copy
import os
import random
from argparse import ArgumentParser
from typing import Optional, Dict, Any, List, Tuple

from models.utils.continual_model import ContinualModel
from utils.buffer import Buffer
from utils.args import add_rehearsal_args
from models.csrel_utils.csrel_coreset_functions import (
    select_by_loss_diff, get_class_dic, get_subset_by_id, make_class_sizes
)
from models.csrel_utils.csrel_selection_agent import CSReLSelectionAgent
from models.csrel_utils.csrel_coreset_buffer import CSReLCoresetBuffer
from models.csrel_utils.csrel_loss_functions import CompliedLoss


class CSReL(ContinualModel):
   
    NAME = 'csrel'
    COMPATIBILITY = ['class-il', 'domain-il', 'task-il', 'general-continual']

    @staticmethod
    def get_parser(parser: ArgumentParser) -> ArgumentParser:
        # Standard rehearsal arguments
        add_rehearsal_args(parser)
        
        # CSReL-specific arguments
        parser.add_argument('--csrel_ref_epochs', type=int, default=10,
                           help='Number of epochs to train reference model')
        parser.add_argument('--csrel_ref_lr', type=float, default=0.01,
                           help='Learning rate for reference model training')
        parser.add_argument('--csrel_ce_factor', type=float, default=1.0,
                           help='Cross-entropy loss factor')
        parser.add_argument('--csrel_mse_factor', type=float, default=0.0,
                           help='MSE loss factor for logit matching')
        parser.add_argument('--csrel_class_balance', action='store_true', default=True,
                           help='Enable class-balanced selection')
        parser.add_argument('--csrel_batch_size', type=int, default=32,
                           help='Batch size for reference model training')
        parser.add_argument('--csrel_selection_steps', type=int, default=1,
                           help='Number of selection steps (for incremental selection)')
        parser.add_argument('--csrel_use_coreset_buffer', action='store_true', default=True,
                           help='Use CSReL coreset buffer instead of standard buffer')
        parser.add_argument('--csrel_buffer_path', type=str, default='./csrel_buffer',
                           help='Path for CSReL buffer storage')
        return parser

    def __init__(self, backbone, loss, args, transform, dataset=None):
        super().__init__(backbone, loss, args, transform, dataset=dataset)
        print("Device:", self.device)

        self.net = self.net.to(self.device)
        
        # CSReL-specific parameters
        self.ref_epochs = getattr(args, 'csrel_ref_epochs', 10)
        self.ref_lr = getattr(args, 'csrel_ref_lr', 0.01)
        self.ce_factor = getattr(args, 'csrel_ce_factor', 1.0)
        self.mse_factor = getattr(args, 'csrel_mse_factor', 0.0)
        self.class_balance = getattr(args, 'csrel_class_balance', True)
        self.batch_size = getattr(args, 'csrel_batch_size', 32)
        self.selection_steps = getattr(args, 'csrel_selection_steps', 1)
        self.use_coreset_buffer = getattr(args, 'csrel_use_coreset_buffer', True)
        self.buffer_path = getattr(args, 'csrel_buffer_path', './csrel_buffer')
        
        # Loss parameters for CSReL
        self.loss_params = {
            'ce_factor': self.ce_factor,
            'mse_factor': self.mse_factor
        }
        # Internal state
        self.task_samples = {}  # Store samples per task
        self.ref_models = {}    # Store reference models per task
        self.task_cnts = []     # Task sample counts for buffer management
        self._dimension_fix_applied = False  # Track if dimension fix has been applied
        
        # Selection agent parameters
        self.model_params = {
            'model_type': 'resnet',  # This should be determined from backbone
            'num_class': getattr(dataset, 'N_CLASSES_PER_TASK', 10) * getattr(dataset, 'N_TASKS', 5),
            'use_bn': True,
            'num_blocks': [2, 2, 2, 2]  # ResNet-18 configuration
        }
        
        self.selection_params = {
            'selection_steps': self.selection_steps,
            'cur_train_lr': self.ref_lr,
            'cur_train_steps': self.ref_epochs,
            'class_balance': self.class_balance,
            'ref_train_params': {
                'epochs': self.ref_epochs,
                'lr': self.ref_lr,
                'batch_size': self.batch_size,
                'ce_factor': self.ce_factor,
                'mse_factor': self.mse_factor
            },
            'loss_params': self.loss_params
        }

        # Initialize buffer
        if self.use_coreset_buffer:
            # Use CSReL coreset buffer
            self._init_coreset_buffer()
        else:
            # Use standard mammoth buffer
            self.buffer = Buffer(self.args.buffer_size, device=self.device)
        
        

    def _init_coreset_buffer(self):
        """Initialize CSReL coreset buffer."""
        # Create task dictionary for class distribution
        self.task_dic = {}
        if hasattr(self.dataset, 'N_TASKS') and hasattr(self.dataset, 'N_CLASSES_PER_TASK'):
            for i in range(self.dataset.N_TASKS):
                start_class = i * self.dataset.N_CLASSES_PER_TASK
                end_class = (i + 1) * self.dataset.N_CLASSES_PER_TASK
                self.task_dic[i] = list(range(start_class, end_class))
        
        self.coreset_buffer = CSReLCoresetBuffer(
            local_path=self.buffer_path,
            model_params=self.model_params,
            transforms=self.transform,
            selection_params=self.selection_params,
            buffer_size=self.args.buffer_size,
            use_cuda=True,
            task_dic=self.task_dic,
            seed=getattr(self.args, 'seed', 42)
        )

    def observe(self, inputs, labels, not_aug_inputs, epoch=None):
        """
        CSReL training step with coreset rehearsal.
        """
        real_batch_size = inputs.shape[0]
        self.opt.zero_grad()

        inputs = inputs.to(self.device)
        labels = labels.to(self.device)

        # Get rehearsal data from buffer
        if self.use_coreset_buffer:
            if not self.coreset_buffer.is_empty():
                # Get data from CSReL buffer
                buf_data = list(self.coreset_buffer.get_data())
                if buf_data and len(buf_data) > 0:
                    # Filter out empty data
                    valid_buf_data = [data for data in buf_data if len(data) >= 2 and data[0].numel() > 0]
                    
                    if valid_buf_data:
                        buf_inputs, buf_labels = valid_buf_data[0]  # Get first valid task data
                        if len(valid_buf_data) > 1:
                            # Concatenate all valid task data
                            all_buf_inputs = [buf_inputs]
                            all_buf_labels = [buf_labels]
                            for task_data in valid_buf_data[1:]:
                                all_buf_inputs.append(task_data[0])
                                all_buf_labels.append(task_data[1])
                            buf_inputs = torch.cat(all_buf_inputs, dim=0)
                            buf_labels = torch.cat(all_buf_labels, dim=0)
                        
                        # Fix tensor dimension mismatch (optimized)
                        if buf_inputs.dim() == 5:
                            # Remove extra dimension: [batch, 1, channels, height, width] -> [batch, channels, height, width]
                            buf_inputs = buf_inputs.squeeze(1)
                            if not self._dimension_fix_applied:
                                print("✅ CSReL: Fixed tensor dimension mismatch (5D -> 4D) for buffer data")
                                self._dimension_fix_applied = True
                        elif buf_inputs.dim() == 3:
                            # Add channel dimension: [batch, height, width] -> [batch, 1, height, width]
                            buf_inputs = buf_inputs.unsqueeze(1)
                            if not self._dimension_fix_applied:
                                print("✅ CSReL: Fixed tensor dimension mismatch (3D -> 4D) for buffer data")
                                self._dimension_fix_applied = True
                        
                        # Ensure both tensors have same number of dimensions
                        if inputs.dim() != buf_inputs.dim():
                            if buf_inputs.dim() > inputs.dim():
                                # Remove extra dimensions from buf_inputs
                                for _ in range(buf_inputs.dim() - inputs.dim()):
                                    buf_inputs = buf_inputs.squeeze(0)
                            else:
                                # Add missing dimensions to buf_inputs
                                for _ in range(inputs.dim() - buf_inputs.dim()):
                                    buf_inputs = buf_inputs.unsqueeze(0)
                        
                        inputs = torch.cat((inputs, buf_inputs.to(self.device)))
                        labels = torch.cat((labels, buf_labels.to(self.device)))
        else:
            # Use standard buffer
            if not self.buffer.is_empty():
                buf_inputs, buf_labels = self.buffer.get_data(
                    self.args.minibatch_size, transform=self.transform, device=self.device
                )
                inputs = torch.cat((inputs, buf_inputs))
                labels = torch.cat((labels, buf_labels))

        # Forward pass and compute loss
        outputs = self.net(inputs)
        loss = self.loss(outputs, labels)
        loss.backward()
        self.opt.step()

        # Store current task samples for reference loss computation
        self._store_task_samples(not_aug_inputs, labels[:real_batch_size])

        return loss.item()

    def _store_task_samples(self, inputs, labels):
        """Store samples from current task for reference loss computation."""
        if self.current_task not in self.task_samples:
            self.task_samples[self.current_task] = {'inputs': [], 'labels': []}
        
        self.task_samples[self.current_task]['inputs'].append(inputs.cpu())
        self.task_samples[self.current_task]['labels'].append(labels.cpu())

    def _train_reference_model(self, task_data, task_labels):
        """Train a reference model on current task data."""
        # Create a copy of the current model
        ref_model = copy.deepcopy(self.net)
        ref_model.train()
        ref_model = ref_model.to(self.device)
        # Convert to numpy for CSReL functions (ensure tensors are on CPU first)
        if isinstance(task_data, torch.Tensor):
            task_data_np = task_data.cpu().numpy()
        else:
            task_data_np = task_data
        
        if isinstance(task_labels, torch.Tensor):
            task_labels_np = task_labels.cpu().numpy()
        else:
            task_labels_np = task_labels
        
        # Create selection agent for reference model training
        selection_agent = CSReLSelectionAgent(
            local_path=os.path.join(self.buffer_path, f'task_{self.current_task}'),
            transforms=self.transform,
            init_size=0,
            selection_steps=1,
            cur_train_lr=self.ref_lr,
            cur_train_steps=self.ref_epochs,
            use_cuda=torch.cuda.is_available(),
            eval_mode='none',
            early_stop=-1,
            eval_steps=100,
            model_params=self.model_params,
            ref_train_params=self.selection_params['ref_train_params'],
            seed=getattr(self.args, 'seed', 42),
            ref_model=ref_model,
            class_balance=self.class_balance,
            only_new_data=True,
            loss_params=self.loss_params
        )
        
        # Train reference model
        selection_agent.train_ref_model(
            x=task_data_np,
            y=task_labels_np,
            verbose=True
        )
        
        return selection_agent.ref_model

    def _select_coreset_samples_csrel(self, task_data, task_labels, remaining_slots):
        """Select coreset samples using CSReL-Coreset-CL method."""
        if remaining_slots <= 0:
            return []
        
        # Convert to numpy for CSReL functions (ensure tensors are on CPU first)
        if isinstance(task_data, torch.Tensor):
            task_data_np = task_data.cpu().numpy()
        else:
            task_data_np = task_data
        
        if isinstance(task_labels, torch.Tensor):
            task_labels_np = task_labels.cpu().numpy()
        else:
            task_labels_np = task_labels
        
        # Create selection agent
        selection_agent = CSReLSelectionAgent(
            local_path=os.path.join(self.buffer_path, f'task_{self.current_task}'),
            transforms=self.transform,
            init_size=0,
            selection_steps=self.selection_steps,
            cur_train_lr=self.ref_lr,
            cur_train_steps=self.ref_epochs,
            use_cuda=True,
            eval_mode='none',
            early_stop=-1,
            eval_steps=100,
            model_params=self.model_params,
            ref_train_params=self.selection_params['ref_train_params'],
            seed=getattr(self.args, 'seed', 42),
            ref_model=None,  # Will be set during training
            class_balance=self.class_balance,
            only_new_data=True,
            loss_params=self.loss_params
        )
        
        # Set current model as reference and ensure it's on the correct device
        selection_agent.ref_model = copy.deepcopy(self.net)
        selection_agent.ref_model = selection_agent.ref_model.to(self.device)
        
        # Perform incremental selection
        selected_data = selection_agent.incremental_selection(
            x=task_data_np,
            y=task_labels_np,
            select_size=remaining_slots,
            verbose=True
        )
        
        # Extract selected indices
        selected_indices = []
        for data_item in selected_data:
            if len(data_item) >= 3:
                selected_indices.append(data_item[0])  # First element is the ID
        
        # Clean up
        selection_agent.clear_path()
        del selection_agent
        
        return selected_indices

    def _fair_reduce_existing_buffer(self):
        """Fairly reduce existing buffer to make space for new task samples."""
        if self.current_task <= 0:
            return
        
        if self.use_coreset_buffer:
            # CSReL buffer handles this internally
            return
        else:
            # Standard buffer reduction
            if len(self.buffer) == 0:
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

    def begin_task(self, dataset):
        """Initialize task-specific variables."""
        super().begin_task(dataset)
        # Task samples will be collected during observe()

    def end_task(self, dataset):
        """End of task processing with CSReL selection."""
        print("Debug: Collect new data for new task end task")
        
        # 1) Gather current task data
        if self.current_task in self.task_samples:
            task_inputs = torch.cat(self.task_samples[self.current_task]['inputs'], dim=0)
            task_labels = torch.cat(self.task_samples[self.current_task]['labels'], dim=0)
        else:
            # Fallback: gather from dataset
            xs, ys = [], []
            for batch in dataset.train_loader:
                
                _, y, not_aug = batch[0], batch[1], batch[2]
                xs.append(not_aug.cpu())
                ys.append(y.cpu())
            task_inputs = torch.cat(xs, dim=0)
            task_labels = torch.cat(ys, dim=0)
        print("Debug: finished train_loader batch")

        # Update task counts
        self.task_cnts.append(len(task_inputs))

        if self.use_coreset_buffer:
            # Use CSReL coreset buffer
            self._update_coreset_buffer(task_inputs, task_labels, dataset)
        else:
            # Use standard buffer with CSReL selection
            self._update_standard_buffer(task_inputs, task_labels)
        print("Debug: Finished end task")

        # Clean up task samples
        if self.current_task in self.task_samples:
            del self.task_samples[self.current_task]
        
        # Ensure model is on correct device after coreset selection
        self.net = self.net.to(self.device)
        
        
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

    def _update_coreset_buffer(self, task_inputs, task_labels, dataset):
        print("Update CSReL coreset buffer.")
        # Convert to numpy (ensure tensors are on CPU first)
        task_data_np = task_inputs.cpu().numpy()
        task_labels_np = task_labels.cpu().numpy()
        
        # Set current model for reference and ensure it's on the correct device
        model_copy = copy.deepcopy(self.net)
        model_copy = model_copy.to(self.device)
        self.coreset_buffer.set_current_model(model_copy)
        
        # Update buffer using CSReL method
        self.coreset_buffer.update_buffer(
            task_cnts=self.task_cnts,
            task_id=self.current_task,
            cur_x=task_data_np,
            cur_y=task_labels_np,
            full_cur_x=task_data_np,  # Use same data for full and current
            full_cur_y=task_labels_np
        )

    def _update_standard_buffer(self, task_inputs, task_labels):
        print("Update standard buffer with CSReL selection.")
        # 1) Reduce existing buffer fairly across seen classes
        self._fair_reduce_existing_buffer()

        # 2) Check remaining buffer space
        remaining = self.args.buffer_size - len(self.buffer)
        if remaining <= 0:
            return

        # 3) Perform CSReL coreset selection
        selected_indices = self._select_coreset_samples_csrel(task_inputs, task_labels, remaining)
        
        if len(selected_indices) > 0:
            # Add selected samples to buffer
            sel_x = task_inputs[selected_indices].to(self.device)
            sel_y = task_labels[selected_indices].to(self.device)
            self.buffer.add_data(examples=sel_x, labels=sel_y)


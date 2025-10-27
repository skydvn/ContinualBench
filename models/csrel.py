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
import pickle
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
import torchvision.transforms as tv_transforms


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
        parser.add_argument('--csrel_class_balance', action='store_true', default=False,
                           help='Enable class-balanced selection')
        parser.add_argument('--csrel_batch_size', type=int, default=32,
                           help='Batch size for reference model training')
        parser.add_argument('--csrel_selection_steps', type=int, default=1,
                           help='Number of selection steps (for incremental selection)')
        parser.add_argument('--csrel_use_coreset_buffer', action='store_true', default=False,
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
        self.class_balance = getattr(args, 'csrel_class_balance', False)
        self.batch_size = getattr(args, 'csrel_batch_size', 32)
        self.selection_steps = getattr(args, 'csrel_selection_steps', 1)
        self.use_coreset_buffer = getattr(args, 'csrel_use_coreset_buffer', False)
        self.buffer_path = getattr(args, 'csrel_buffer_path', './csrel_buffer')
        
        # Loss parameters for CSReL
        self.loss_params = {
            'ce_factor': self.ce_factor,
            'mse_factor': self.mse_factor
        }
        # Internal state (memory-efficient)
        self.task_data_files = {}  # Store file paths instead of data (memory-efficient)
        self.ref_models = {}       # Store reference models per task
        self.task_cnts = []        # Task sample counts for buffer management
        self._dimension_fix_applied = False  # Track if dimension fix has been applied
        self.chunk_size = getattr(args, 'csrel_chunk_size', 1000)  # Chunk size for processing
        
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
        """Store samples from current task using memory-efficient file-based storage."""
        if self.current_task not in self.task_data_files:
            # Create temporary file for this task
            temp_file = os.path.join(self.buffer_path, f'task_{self.current_task}_data.pkl')
            self.task_data_files[self.current_task] = temp_file
            
            # Initialize file
            with open(temp_file, 'wb') as f:
                pass  # Create empty file
        
        # Append data to file (chunked writing for memory efficiency)
        with open(self.task_data_files[self.current_task], 'ab') as f:
            # Convert to numpy for efficient storage
            inputs_np = inputs.cpu().numpy()
            labels_np = labels.cpu().numpy()
            
            # Store in chunks to avoid memory spikes
            for i in range(0, len(inputs_np), self.chunk_size):
                chunk_inputs = inputs_np[i:i+self.chunk_size]
                chunk_labels = labels_np[i:i+self.chunk_size]
                pickle.dump((chunk_inputs, chunk_labels), f)

    def _load_task_samples_efficient(self, task_id):
        """Load task samples from file (memory-efficient)."""
        if task_id not in self.task_data_files:
            return None, None
        
        temp_file = self.task_data_files[task_id]
        if not os.path.exists(temp_file):
            return None, None
        
        all_inputs = []
        all_labels = []
        
        # Load data in chunks
        with open(temp_file, 'rb') as f:
            while True:
                try:
                    chunk_inputs, chunk_labels = pickle.load(f)
                    all_inputs.append(chunk_inputs)
                    all_labels.append(chunk_labels)
                except EOFError:
                    break
        
        if all_inputs:
            inputs = np.concatenate(all_inputs, axis=0)
            labels = np.concatenate(all_labels, axis=0)
            return torch.from_numpy(inputs), torch.from_numpy(labels)
        
        return None, None

    def _make_data_loader_csrel_style(self, x, y, fname, batch_size, id_list=None, id2logit=None, extra_data=None):
        """
        Make data loader using CSReL-Coreset-CL file-based approach (memory-efficient).
        Based on the original CSReL-Coreset-CL make_data_loader function.
        """
        from torch.utils.data import DataLoader
        import torchvision.transforms as transforms
        
        data_size = x.shape[0]
        data_file = os.path.join(self.buffer_path, fname)
        
        # Create file-based storage (CSReL-Coreset-CL style)
        with open(data_file, 'wb') as fw:
            for i in range(data_size):
                if self.transform is not None:
                    # Convert to PIL Image for transforms
                    to_pil = transforms.ToPILImage()
                    sp = to_pil(torch.tensor(x[i], dtype=torch.float32).clone().detach())
                else:
                    sp = torch.tensor(x[i], dtype=torch.float32).clone().detach()
                
                if id_list is not None:
                    data = [id_list[i], sp, int(y[i])]
                    if id2logit is not None:
                        data.append(id2logit[id_list[i]])
                else:
                    data = [i, sp, int(y[i])]
                    if id2logit is not None:
                        data.append(id2logit[i])
                pickle.dump(data, fw)
            
            if extra_data is not None:
                for di in extra_data:
                    if len(di) == 4 and id2logit is None:
                        pickle.dump(di[:3], fw)
                    else:
                        pickle.dump(di, fw)
        
        # Create dataset from file (memory-efficient)
        class FileBasedDataset:
            def __init__(self, data_file, transforms=None):
                self.data_file = data_file
                self.transforms = transforms
                self.to_tensor = tv_transforms.ToTensor()
            
            def __len__(self):
                # Count samples in file
                count = 0
                with open(self.data_file, 'rb') as f:
                    while True:
                        try:
                            pickle.load(f)
                            count += 1
                        except EOFError:
                            break
                return count
            
            def __getitem__(self, idx):
                # Load specific sample from file
                with open(self.data_file, 'rb') as f:
                    for i, data in enumerate(self._load_samples(f)):
                        if i == idx:
                            if len(data) == 3:
                                d_id, sp, lab = data
                            elif len(data) == 4:
                                d_id, sp, lab, logit = data
                            else:
                                raise ValueError('Invalid data format')
                            
                            # Apply transforms if provided
                            if self.transforms is not None:
                                try:
                                    sp = self.transforms(sp)
                                except Exception as e:
                                    # If transforms fail, convert to tensor first
                                    if not isinstance(sp, torch.Tensor):
                                        sp = self.to_tensor(sp)
                                    sp = self.transforms(sp)
                            else:
                                sp = self.to_tensor(sp)
                            
                            return sp, lab
            
            def _load_samples(self, f):
                """Generator to load samples from file."""
                while True:
                    try:
                        yield pickle.load(f)
                    except EOFError:
                        break
        
        # Create dataset and data loader
        dataset = FileBasedDataset(data_file, transforms=self.transform)
        data_loader = DataLoader(dataset, batch_size=batch_size, drop_last=False, shuffle=True)
        
        return data_loader, data_file

    def _compute_reference_losses_file_based(self, model, data_loader):
        """Compute reference losses using file-based data loader (memory-efficient)."""
        model.eval()
        loss_fn = CompliedLoss(
            ce_factor=self.ce_factor,
            mse_factor=self.mse_factor,
            reduction='none'
        )
        
        losses = {}
        
        with torch.no_grad():
            for batch_idx, (sps, labs) in enumerate(data_loader):
                device = next(model.parameters()).device
                sps = sps.to(device)
                labs = labs.to(device)
                # Fix tensor shape if needed (remove extra dimensions)
                if len(sps.shape) == 5:  # [batch, 1, channels, height, width]
                    sps = sps.squeeze(1)  # Remove the extra dimension
                
                outputs = model(sps)
                batch_losses = loss_fn(outputs, labs)
                
                for i, loss in enumerate(batch_losses):
                    sample_id = batch_idx * data_loader.batch_size + i
                    losses[sample_id] = loss.item()
        
        return losses

    def _select_samples_by_loss_diff(self, loss_diffs, labels, num_to_select):
        """Select samples based on loss differences (memory-efficient)."""
        if self.class_balance:
            return self._class_balanced_selection_by_loss_diff(loss_diffs, labels, num_to_select)
        else:
            return self._top_k_selection_by_loss_diff(loss_diffs, num_to_select)

    def _class_balanced_selection_by_loss_diff(self, loss_diffs, labels, num_to_select):
        """Class-balanced selection based on loss differences."""
        unique_classes = np.unique(labels)
        samples_per_class = num_to_select // len(unique_classes)
        remainder = num_to_select % len(unique_classes)
        
        # Sort by loss difference (descending)
        sorted_diffs = sorted(loss_diffs.items(), key=lambda x: x[1], reverse=True)
        
        selected_indices = []
        class_counts = {cls: 0 for cls in unique_classes}
        
        for sample_id, loss_diff in sorted_diffs:
            if len(selected_indices) >= num_to_select:
                break
                
            class_id = labels[sample_id]
            max_per_class = samples_per_class + (1 if class_counts[class_id] < remainder else 0)
            
            if class_counts[class_id] < max_per_class:
                selected_indices.append(sample_id)
                class_counts[class_id] += 1
        
        return selected_indices

    def _top_k_selection_by_loss_diff(self, loss_diffs, k):
        """Top-k selection based on loss differences."""
        sorted_diffs = sorted(loss_diffs.items(), key=lambda x: x[1], reverse=True)
        return [item[0] for item in sorted_diffs[:k]]

    def _train_reference_model(self, task_data, task_labels):
        """Train a reference model using CSReL-Coreset-CL file-based approach (memory-efficient)."""
        # Create a copy of the current model
        ref_model = copy.deepcopy(self.net)
        ref_model.train()
        ref_model = ref_model.to(self.device)
        
        # Clear GPU cache before training
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # Convert to numpy for CSReL functions (ensure tensors are on CPU first)
        if isinstance(task_data, torch.Tensor):
            task_data_np = task_data.cpu().numpy()
        else:
            task_data_np = task_data
        
        if isinstance(task_labels, torch.Tensor):
            task_labels_np = task_labels.cpu().numpy()
        else:
            task_labels_np = task_labels
        
        # Use CSReL-Coreset-CL approach: create data loader with file-based storage
        train_loader, data_file = self._make_data_loader_csrel_style(
            x=task_data_np,
            y=task_labels_np,
            fname=f'ref_train_task_{self.current_task}.pkl',
            batch_size=self.batch_size
        )
        
        # Train reference model using the data loader
        optimizer = torch.optim.SGD(ref_model.parameters(), lr=self.ref_lr)
        criterion = CompliedLoss(
            ce_factor=self.ce_factor,
            mse_factor=self.mse_factor,
            reduction='mean'
        )
        
        # Training loop
        for epoch in range(self.ref_epochs):
            
            epoch_loss = 0.0
            for batch_idx, (sps, labs) in enumerate(train_loader):
                # Ensure all tensors are on the same device as the model
                device = next(ref_model.parameters()).device
                sps = sps.to(device)
                labs = labs.to(device)
                if len(sps.shape) == 5:  # [batch, 1, channels, height, width]
                    sps = sps.squeeze(1)
               
                optimizer.zero_grad()
                outputs = ref_model(sps)
                loss = criterion(outputs, labs)
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
            
            if epoch % 1 == 0:
                print(f"     Reference Epoch {epoch+1}/{self.ref_epochs}, Loss: {epoch_loss / len(train_loader):.4f}")
        
        # Set model to eval mode
        ref_model.eval()
        
        # Clean up temporary file (CSReL-Coreset-CL style)
        if os.path.exists(data_file):
            os.remove(data_file)
        
        # Clear memory after training
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        return ref_model

    def _compute_reference_losses_simple(self, model, task_data, task_labels):
        """Compute reference losses using simple approach (no data loader)."""
        model.eval()
        loss_fn = CompliedLoss(
            ce_factor=self.ce_factor,
            mse_factor=self.mse_factor,
            reduction='none'
        )
        
        losses = {}
        
        with torch.no_grad():
            # Process data in batches to avoid memory issues
            batch_size = 32
            for i in range(0, len(task_data), batch_size):
                batch_data = task_data[i:i+batch_size]
                batch_labels = task_labels[i:i+batch_size]
                
                # Ensure tensors are on the correct device
                device = next(model.parameters()).device
                batch_data = batch_data.to(device)
                batch_labels = batch_labels.to(device)
                
                # Fix tensor shape if needed
                if len(batch_data.shape) == 5:  # [batch, 1, channels, height, width]
                    batch_data = batch_data.squeeze(1)
                elif len(batch_data.shape) == 6:  # [batch, 1, 1, channels, height, width]
                    batch_data = batch_data.squeeze(1).squeeze(1)
                
                outputs = model(batch_data)
                batch_losses = loss_fn(outputs, batch_labels)
                
                for j, loss in enumerate(batch_losses):
                    sample_id = i + j
                    losses[sample_id] = loss.item()
        
        return losses

    def _select_coreset_samples_csrel(self, task_data, task_labels, remaining_slots):
        """Select coreset samples using CSReL-Coreset-CL iterative method (exact implementation)."""
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
        
        print(f"Using CSReL-Coreset-CL iterative selection for {remaining_slots} samples...")
        
        # Train reference model
        ref_model = self._train_reference_model(task_data, task_labels)
        
        # Compute reference losses using simple approach
        ref_losses = self._compute_reference_losses_simple(ref_model, task_data, task_labels)
        
        # Initialize iterative selection (CSReL-Coreset-CL style)
        all_selected_ids = set()
        incremental_size = max(1, remaining_slots // self.selection_steps) if self.selection_steps > 1 else remaining_slots
        
        # Create class distribution for balanced selection
        class_ids = {}
        if self.class_balance:
            unique_classes = np.unique(task_labels_np)
            for cls in unique_classes:
                class_ids[cls] = set()
                class_mask = (task_labels_np == cls)
                class_indices = np.where(class_mask)[0]
                for idx in class_indices:
                    class_ids[cls].add(idx)
        
        # Full IDs list
        full_ids = list(range(len(task_data_np)))
        
        # Iterative selection loop (CSReL-Coreset-CL style)
        while len(all_selected_ids) < remaining_slots:
            # Create ID pool from unselected IDs (CSReL-Coreset-CL style)
            id_pool = set()
            for d_id in full_ids:
                if d_id not in all_selected_ids:
                    id_pool.add(d_id)
            
            if not id_pool:
                break  # No more samples to select
            
            # Get subset by IDs using CSReL-Coreset-CL style
            rand_data = self._get_subset_by_id(task_data_np, task_labels_np, list(id_pool))
            print(f"Debug: Got {len(rand_data)} samples from subset")
            
            # Compute loss differences for subset using CSReL-Coreset-CL style
            current_incremental_size = min(incremental_size, remaining_slots - len(all_selected_ids))
            print(f"Debug: Selecting {current_incremental_size} samples from subset")
            
            selected_data, _ = self._select_by_loss_diff(
                ref_loss_dic=ref_losses,
                rand_data=rand_data,
                model=self.net,
                incremental_size=current_incremental_size,
                class_sizes=self._make_class_sizes(class_ids, current_incremental_size) if self.class_balance else None
            )
            
            print(f"Debug: Selected {len(selected_data)} samples from subset")
            
            # Add selected samples to all_selected_ids (CSReL-Coreset-CL style)
            flg_add = False
            for di in selected_data:
                d_id = int(di[0])
                lab = int(di[2])
                if d_id not in all_selected_ids:
                    all_selected_ids.add(d_id)
                    flg_add = True
                if self.class_balance:
                    class_ids[lab].add(d_id)
            
            print(f"CSReL iterative selection: {len(all_selected_ids)}/{remaining_slots} samples selected")
            
            if not flg_add:
                break  # No new samples added, stop selection
        
        # Convert to list and return
        selected_indices = list(all_selected_ids)
        
        # Clean up models
        del ref_model
        
        # Clear memory after selection
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        print(f"CSReL iterative selection completed: {len(selected_indices)} samples selected")
        return selected_indices

    def _select_coreset_samples_with_agent(self, task_data, task_labels, remaining_slots):
        """Select coreset samples using CSReLSelectionAgent with memory-efficient file-based approach."""
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
        
        # Use CSReL-Coreset-CL iterative selection method
        print(f"Using CSReL-Coreset-CL iterative selection for {remaining_slots} samples...")
        try:
            selected_indices = self._select_coreset_samples_csrel(task_data, task_labels, remaining_slots)
            print(f"CSReL selected {len(selected_indices)} samples out of {len(task_data_np)}")
            
        except Exception as e:
            print(f"Warning: CSReL iterative selection failed, falling back to random selection: {e}")
            import traceback
            traceback.print_exc()
            # Fallback to random selection
            import random
            selected_indices = random.sample(range(len(task_data_np)), min(remaining_slots, len(task_data_np)))
        
        # Clear memory after selection
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        return selected_indices

    def _get_subset_by_id(self, x, y, ids, transforms=None, id_list=None, id2logit=None):
        """Get subset by IDs (CSReL-Coreset-CL style)."""
        selected_data = []
        if id_list is None:
            d_pos = ids
        else:
            d_pos = []
            id_pool = set(ids)
            for i, d_id in enumerate(id_list):
                if d_id in id_pool:
                    d_pos.append(i)
        
        for pi in d_pos:
            if transforms is None:
                sp = torch.tensor(x[pi], dtype=torch.float32).clone().detach()
            else:
                sp = transforms(torch.tensor(x[pi], dtype=torch.float32).clone().detach())
            
            if id_list is None:
                d_id = pi
            else:
                d_id = id_list[pi]
            
            data = [d_id, sp, int(y[pi])]
            if id2logit is not None:
                data.append(id2logit[d_id])
            selected_data.append(data)
        
        return selected_data

    def _select_by_loss_diff(self, ref_loss_dic, rand_data, model, incremental_size, class_sizes=None):
        """Select by loss difference (CSReL-Coreset-CL style)."""
        model.eval()
        if torch.cuda.is_available():
            model.cuda()
        
        loss_fn = CompliedLoss(
            ce_factor=self.ce_factor,
            mse_factor=self.mse_factor,
            reduction='none'
        )
        
        loss_diffs = {}
        id2pos = {}
        batch_ids = []
        batch_sps = []
        batch_labs = []
        
        with torch.no_grad():
            for i, di in enumerate(rand_data):
                if len(di) == 4:
                    d_id, sp, lab, logit = di
                else:
                    d_id, sp, lab = di
                    logit = None
                
                id2pos[d_id] = i
                batch_ids.append(d_id)
                batch_sps.append(sp)
                batch_labs.append(int(lab))
                
                if i % 32 == 0 or i == len(rand_data) - 1:
                    sps = torch.stack(batch_sps)
                    labs = torch.tensor(batch_labs, dtype=torch.long)
                    
                    if torch.cuda.is_available():
                        sps = sps.cuda()
                        labs = labs.cuda()
                    
                    # Fix tensor shape if needed
                    if len(sps.shape) == 5:  # [batch, 1, channels, height, width]
                        sps = sps.squeeze(1)
                    elif len(sps.shape) == 6:  # [batch, 1, 1, channels, height, width]
                        sps = sps.squeeze(1).squeeze(1)
                    
                    loss = loss_fn(model(sps), labs)
                    loss = loss.clone().detach()
                    if torch.cuda.is_available():
                        loss = loss.cpu()
                    loss = loss.numpy()
                    
                    for j in range(len(batch_labs)):
                        did = batch_ids[j]
                        loss_dif = float(loss[j] - ref_loss_dic[did])
                        loss_diffs[did] = loss_dif
                    
                    batch_ids.clear()
                    batch_sps.clear()
                    batch_labs.clear()
        
        # Sort by loss difference and select
        sorted_loss_diffs = sorted(loss_diffs.items(), key=lambda x: x[1], reverse=True)
        print(f"Debug: Sorted {len(sorted_loss_diffs)} loss differences, selecting {incremental_size}")
        
        selected_data = []
        id2loss_dif = {}
        class_cnt = {}
        
        if class_sizes is not None:
            for ci in class_sizes.keys():
                class_cnt[ci] = 0
            print(f"Debug: Class sizes: {class_sizes}")
        
        for i in range(len(sorted_loss_diffs)):
            d_id = sorted_loss_diffs[i][0]
            pos = id2pos[d_id]
            di = rand_data[pos]
            
            if class_sizes is not None:
                lab = int(di[2])
                if class_cnt[lab] == class_sizes[lab]:
                    print(f"Debug: Skipping sample {d_id} (class {lab} full)")
                    continue
                else:
                    class_cnt[lab] += 1
                    print(f"Debug: Adding sample {d_id} (class {lab}, count: {class_cnt[lab]})")
            
            new_di = copy.deepcopy(di)
            selected_data.append(new_di)
            id2loss_dif[d_id] = sorted_loss_diffs[i][1]
            
            if len(selected_data) == incremental_size:
                break
        
        print(f"Debug: Final selection: {len(selected_data)} samples")
        
        if torch.cuda.is_available():
            model.cpu()
        
        return selected_data, id2loss_dif

    def _make_class_sizes(self, class_ids, incremental_size):
        """Make class sizes for balanced selection."""
        class_cnts = {}
        max_cnt = -1
        for ci in class_ids.keys():
            class_cnts[ci] = len(class_ids[ci])
            if class_cnts[ci] > max_cnt:
                max_cnt = class_cnts[ci]
        
        class_sizes = {}
        for ci in class_ids.keys():
            if class_cnts[ci] == 0:
                class_sizes[ci] = 0
            else:
                class_sizes[ci] = max(1, int(incremental_size * class_cnts[ci] / max_cnt))
        
        return class_sizes

    def _select_samples_by_loss_diff_simple(self, loss_diffs, labels, num_to_select):
        """Select samples based on loss differences (simple approach)."""
        if self.class_balance:
            return self._class_balanced_selection_by_loss_diff(loss_diffs, labels, num_to_select)
        else:
            return self._top_k_selection_by_loss_diff(loss_diffs, num_to_select)

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
        """End of task processing with memory-efficient CSReL selection."""
        print("Debug: Memory-efficient CSReL end task processing")
        
        # 1) Load current task data from file (memory-efficient)
        task_inputs, task_labels = self._load_task_samples_efficient(self.current_task)
        
        if task_inputs is None:
            # Fallback: gather from dataset
            xs, ys = [], []
            for batch in dataset.train_loader:
                _, y, not_aug = batch[0], batch[1], batch[2]
                xs.append(not_aug.cpu())
                ys.append(y.cpu())
            task_inputs = torch.cat(xs, dim=0)
            task_labels = torch.cat(ys, dim=0)
        print("Debug: Loaded task data efficiently")

        # Update task counts
        self.task_cnts.append(len(task_inputs))

        if self.use_coreset_buffer:
            # Use CSReL coreset buffer
            self._update_coreset_buffer(task_inputs, task_labels, dataset)
        else:
            # Use standard buffer with CSReL selection
            self._update_standard_buffer(task_inputs, task_labels)
        print("Debug: Finished memory-efficient end task")
        # Clean the files
        if self.current_task in self.task_data_files:
            temp_file = self.task_data_files[self.current_task]
            if os.path.exists(temp_file):
                os.remove(temp_file)
            del self.task_data_files[self.current_task]
        
        # Ensure model is on correct device after coreset selection
        self.net = self.net.to(self.device)
        
        # Clear memory
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _update_coreset_buffer(self, task_inputs, task_labels, dataset):
        print("Update CSReL coreset buffer using CSReLSelectionAgent.")
        # Convert to numpy (ensure tensors are on CPU first)
        task_data_np = task_inputs.cpu().numpy()
        task_labels_np = task_labels.cpu().numpy()
        
        # Set current model for reference and ensure it's on the correct device
        model_copy = copy.deepcopy(self.net)
        model_copy = model_copy.to(self.device)
        self.coreset_buffer.set_current_model(model_copy)
        
        # Update buffer using CSReL method (this internally uses CSReLSelectionAgent)
        self.coreset_buffer.update_buffer(
            task_cnts=self.task_cnts,
            task_id=self.current_task,
            cur_x=task_data_np,
            cur_y=task_labels_np,
            full_cur_x=task_data_np,  # Use same data for full and current
            full_cur_y=task_labels_np
        )
        
        # Clear memory after buffer update
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _update_standard_buffer(self, task_inputs, task_labels):
        print("Update standard buffer with CSReL selection using CSReLSelectionAgent.")
        # 1) Reduce existing buffer fairly across seen classes
        self._fair_reduce_existing_buffer()

        # 2) Check remaining buffer space
        remaining = self.args.buffer_size - len(self.buffer)
        if remaining <= 0:
            return

        # 3) Use CSReLSelectionAgent for coreset selection (memory-efficient)
        print(f"CSReL: Selecting {remaining} samples from {len(task_inputs)} task samples")
        selected_indices = self._select_coreset_samples_with_agent(task_inputs, task_labels, remaining)
        print(f"CSReL: Selected {len(selected_indices)} samples")
        
        if len(selected_indices) > 0:
            # Add selected samples to buffer
            sel_x = task_inputs[selected_indices].to(self.device)
            sel_y = task_labels[selected_indices].to(self.device)
            self.buffer.add_data(examples=sel_x, labels=sel_y)
            print(f"CSReL: Added {len(selected_indices)} samples to buffer")
        else:
            print("CSReL: No samples selected!")



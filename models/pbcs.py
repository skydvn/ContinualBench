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
import os
import pickle
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
        parser.add_argument('--pbcs_inner_optim', type=str, default='sgd',
                           choices=['sgd', 'adam'],
                           help='Inner optimizer: sgd or adam')
        parser.add_argument('--pbcs_inner_wd', type=float, default=0.0,
                           help='Weight decay for inner optimizer')
        parser.add_argument('--pbcs_div_tol', type=float, default=9.0,
                           help='Divergence tolerance threshold')
        parser.add_argument('--pbcs_buffer_path', type=str, default='./pbcs_buffer',
                           help='Path for PBCS buffer storage')
        parser.add_argument('--pbcs_chunk_size', type=int, default=1000,
                           help='Chunk size for file-based processing')
        
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
        self.inner_optim = getattr(args, 'pbcs_inner_optim', 'sgd')
        self.inner_wd = getattr(args, 'pbcs_inner_wd', 0.0)
        self.div_tol = getattr(args, 'pbcs_div_tol', 9.0)
        
        # Initialize coreset selector
        selector_kwargs = {
            'device': self.device,
            'K': self.K,
            'outer_lr': self.outer_lr,
            'inner_lr': self.inner_lr,
            'max_outer_iter': self.max_outer_iter,
            'epoch_converge': self.epoch_converge,
            'coreset_size': int(self.coreset_ratio * args.buffer_size),
            'use_variance_reduction': self.use_vr,
            'clip_grad': self.clip_grad,
            'clip_constant': self.clip_constant,
            'inner_optim': self.inner_optim,
            'inner_wd': self.inner_wd,
            'div_tol': self.div_tol
        }
        
        if self.selection_type == 'pixel':
            self.coreset_selector = PixelLevelCoresetSelector(**selector_kwargs)
        else:
            self.coreset_selector = ProbabilisticCoresetSelector(**selector_kwargs)
        
        # File-based storage for task data (memory-efficient)
        self.buffer_path = getattr(args, 'pbcs_buffer_path', './pbcs_buffer')
        self.chunk_size = getattr(args, 'pbcs_chunk_size', 1000)
        os.makedirs(self.buffer_path, exist_ok=True)
        
        # Store file paths instead of data in memory
        self.task_data_files = {}  # Store file paths for each task
        self.task_sample_hashes = set()  # Track unique samples to avoid duplicates
        
        print(f"PBCS initialized with {self.selection_type}-level selection")
        print(f"Coreset size: {self.coreset_selector.coreset_size}")
        print(f"Buffer size: {self.args.buffer_size}")
        print(f"Using file-based storage at: {self.buffer_path}")
    
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
        
        # Store current task data using file-based storage (memory-efficient)
        self._store_task_samples(not_aug_inputs, labels)
        
        # Get replay data from buffer
        if not self.buffer.is_empty():
            buf_inputs, buf_labels = self.buffer.get_data(
                self.args.minibatch_size, transform=self.transform, device=self.device
            )
            # Ensure buffer data is on the same device as current inputs
            buf_inputs = buf_inputs.to(inputs.device)
            buf_labels = buf_labels.to(labels.device)
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
    
    def _store_task_samples(self, inputs, labels):
        """Store samples from current task using memory-efficient file-based storage (avoid duplicates)."""
        if self.current_task not in self.task_data_files:
            # Create temporary file for this task
            temp_file = os.path.join(self.buffer_path, f'task_{self.current_task}_data.pkl')
            self.task_data_files[self.current_task] = temp_file
            
            # Initialize file and set for tracking unique samples
            with open(temp_file, 'wb') as f:
                pass  # Create empty file
            
            # Initialize set to track unique samples (using hash of data)
            self.task_sample_hashes = set()
        
        # Convert to numpy for efficient storage
        inputs_np = inputs.cpu().numpy()
        labels_np = labels.cpu().numpy()
        
        # Check for duplicates using hash of data
        new_samples = []
        for i in range(len(inputs_np)):
            # Create hash of the sample data
            sample_hash = hash((inputs_np[i].tobytes(), labels_np[i]))
            
            if sample_hash not in self.task_sample_hashes:
                self.task_sample_hashes.add(sample_hash)
                new_samples.append((inputs_np[i], labels_np[i]))
        
        # Only store new samples
        if new_samples:
            with open(self.task_data_files[self.current_task], 'ab') as f:
                # Store in chunks to avoid memory spikes
                for i in range(0, len(new_samples), self.chunk_size):
                    chunk_samples = new_samples[i:i+self.chunk_size]
                    chunk_inputs = np.array([s[0] for s in chunk_samples])
                    chunk_labels = np.array([s[1] for s in chunk_samples])
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
            inputs_tensor = torch.from_numpy(inputs).to(self.device)
            labels_tensor = torch.from_numpy(labels).to(self.device)
            return inputs_tensor, labels_tensor
        
        return None, None
    
    def _fair_reduce_buffer(self, target_size: int):
        """
        Fairly reduce existing buffer to target size.
        Uses class-balanced reduction similar to other methods.
        
        Args:
            target_size: Target number of samples to keep in buffer
        """
        if target_size <= 0 or len(self.buffer) == 0:
            return
        
        # Get all buffer data
        buf_data = self.buffer.get_all_data()
        buf_inputs = buf_data[0]  # First element is examples
        buf_labels = buf_data[1] if len(buf_data) > 1 else None  # Second element is labels
        
        if buf_labels is None:
            print("Warning: Buffer has no labels, cannot perform class-balanced reduction")
            return
        
        # Get actual buffer size (using len() which respects num_seen_examples)
        current_size = len(self.buffer)
        
        if current_size <= target_size:
            return  # No reduction needed
        
        # Preserve num_seen_examples for reservoir sampling
        original_num_seen = self.buffer.num_seen_examples
        
        # Empty buffer and refill with reduced samples
        self.buffer.empty()
        
        # Collect all samples to keep
        unique_classes = buf_labels.unique()
        samples_per_class = max(1, target_size // len(unique_classes))
        
        selected_inputs_list = []
        selected_labels_list = []
        
        for class_id in unique_classes:
            # Get indices for this class
            class_mask = (buf_labels == class_id)
            class_inputs = buf_inputs[class_mask]
            class_labels = buf_labels[class_mask]
            
            # Keep samples for this class (proportional to class frequency)
            keep_count = min(len(class_inputs), samples_per_class)
            
            if keep_count > 0:
                if len(class_inputs) > keep_count:
                    # Random selection within class (maintains diversity)
                    selected_indices = torch.randperm(len(class_inputs))[:keep_count]
                    selected_inputs = class_inputs[selected_indices]
                    selected_labels = class_labels[selected_indices]
                else:
                    selected_inputs = class_inputs
                    selected_labels = class_labels
                
                selected_inputs_list.append(selected_inputs)
                selected_labels_list.append(selected_labels)
        
        # Concatenate all selected samples
        if selected_inputs_list:
            all_selected_inputs = torch.cat(selected_inputs_list, dim=0).to(self.device)
            all_selected_labels = torch.cat(selected_labels_list, dim=0).to(self.device)
            
            initial_count = len(all_selected_labels)
            
            # Ensure we have exactly target_size samples (might be slightly off due to class distribution)
            if initial_count > target_size:
                # Too many samples, trim to target_size
                trim_indices = torch.randperm(initial_count)[:target_size]
                all_selected_inputs = all_selected_inputs[trim_indices]
                all_selected_labels = all_selected_labels[trim_indices]
                final_count = target_size
            elif initial_count < target_size:
                # Too few samples - pad by duplicating some (better than having fewer)
                deficit = target_size - initial_count
                pad_indices = torch.randint(0, initial_count, (deficit,))
                all_selected_inputs = torch.cat([all_selected_inputs, all_selected_inputs[pad_indices]], dim=0)
                all_selected_labels = torch.cat([all_selected_labels, all_selected_labels[pad_indices]], dim=0)
                final_count = target_size
                print(f"Info: Selected {initial_count} samples from {len(unique_classes)} classes, padded to {target_size} for buffer consistency.")
            else:
                final_count = target_size
            
            # After empty(), buffer tensors are deleted, so we need to reinitialize
            # Set the buffer contents directly (exactly target_size)
            samples_to_use = min(final_count, self.args.buffer_size)
            
            # Initialize buffer tensors with the correct size
            self.buffer.num_seen_examples = 0  # Reset before init
            self.buffer.init_tensors(
                all_selected_inputs[:samples_to_use],
                all_selected_labels[:samples_to_use],
                None,  # logits
                None,  # task_labels
                None   # true_labels
            )
            
            # Copy data directly into buffer tensors
            self.buffer.examples[:samples_to_use].copy_(all_selected_inputs[:samples_to_use])
            if hasattr(self.buffer, 'labels') and self.buffer.labels is not None:
                self.buffer.labels[:samples_to_use].copy_(all_selected_labels[:samples_to_use])
            
            # CRITICAL: Set num_seen_examples to samples_to_use so len() returns correct size
            # Buffer.__len__() returns min(num_seen_examples, buffer_size)
            self.buffer.num_seen_examples = samples_to_use
            
            # Verify the size is correct immediately after setting
            actual_buffer_len = len(self.buffer)
            if actual_buffer_len != samples_to_use:
                print(f"ERROR: Buffer size mismatch! Expected {samples_to_use}, got {actual_buffer_len}")
                print(f"  num_seen_examples={self.buffer.num_seen_examples}, buffer_size={self.args.buffer_size}")
                print(f"  Setting num_seen_examples to {samples_to_use} to fix...")
                self.buffer.num_seen_examples = samples_to_use
                actual_buffer_len = len(self.buffer)
                print(f"  After fix: buffer size = {actual_buffer_len}")
            else:
                print(f"Buffer size correctly set to {samples_to_use}")
        
        final_buffer_size = len(self.buffer)
        print(f"Buffer reduced: {current_size} -> {final_buffer_size} samples (target: {target_size})")
        
        # Verify reduction worked
        if final_buffer_size > target_size:
            print(f"Warning: Buffer reduction incomplete. Expected {target_size}, got {final_buffer_size}")
    
    def _cleanup_task_data(self):
        """Clean up task data files and memory."""
        # Clear the hash set for the current task
        if hasattr(self, 'task_sample_hashes'):
            self.task_sample_hashes.clear()
        
        # Optionally remove the task data file to save disk space
        # (comment out if you want to keep files for debugging)
        if self.current_task in self.task_data_files:
            task_file = self.task_data_files[self.current_task]
            if os.path.exists(task_file):
                try:
                    os.remove(task_file)
                    print(f"Cleaned up task data file: {task_file}")
                except OSError as e:
                    print(f"Warning: Could not remove task file {task_file}: {e}")
        
        # Clear GPU cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    def end_task(self, dataset):
        """
        Called at the end of each task to perform coreset selection.
        
        Args:
            dataset: Current dataset
        """
        print(f"End of task {self.current_task}: Performing coreset selection")
        
        # Load task data from file (memory-efficient)
        task_inputs, task_labels = self._load_task_samples_efficient(self.current_task)
        
        if task_inputs is None or task_labels is None:
            print("No task data available for coreset selection")
            return
        
        print(f"Task data shape: {task_inputs.shape}, Labels: {task_labels.shape}")
        
        # Calculate coreset size first
        num_tasks_seen = self.current_task + 1
        fair_allocation = self.args.buffer_size // num_tasks_seen  # Fair share for current task
        ratio_based = int(self.coreset_ratio * len(task_inputs))
        estimated_coreset_size = min(fair_allocation, ratio_based)
        
        # Fair buffer management: reduce existing buffer if needed to make room for new task
        current_buffer_size = len(self.buffer)
        if current_buffer_size > 0 and self.current_task > 0:
            # If buffer would exceed capacity with new samples, reduce existing buffer
            if current_buffer_size + estimated_coreset_size > self.args.buffer_size:
                # Reduce existing buffer: keep fair share for previous tasks
                target_buffer_size = self.args.buffer_size - estimated_coreset_size
                target_buffer_size = max(target_buffer_size, fair_allocation * self.current_task)  # At least fair share
                target_buffer_size = max(1, target_buffer_size)  # Ensure at least 1 slot
                
                if current_buffer_size > target_buffer_size:
                    reduction_needed = current_buffer_size - target_buffer_size
                    print(f"Reducing buffer from {current_buffer_size} to {target_buffer_size} samples (removing {reduction_needed})")
                    self._fair_reduce_buffer(target_buffer_size)
        
        # Recalculate coreset size after buffer reduction
        remaining_slots = self.args.buffer_size - len(self.buffer)
        
        # If buffer reduction didn't free enough space, force additional reduction
        if remaining_slots < estimated_coreset_size and len(self.buffer) > 0:
            # Need more space - reduce buffer further
            additional_reduction = estimated_coreset_size - remaining_slots
            new_target_size = len(self.buffer) - additional_reduction
            new_target_size = max(new_target_size, fair_allocation * self.current_task, 1)
            if new_target_size < len(self.buffer):
                print(f"Additional buffer reduction needed: {len(self.buffer)} -> {new_target_size}")
                self._fair_reduce_buffer(new_target_size)
                remaining_slots = self.args.buffer_size - len(self.buffer)
        
        coreset_size = min(remaining_slots, fair_allocation, ratio_based)
        coreset_size = max(coreset_size, 1)  # Ensure at least 1 sample
        
        self.coreset_selector.coreset_size = coreset_size
        
        print(f"Selecting {coreset_size} samples from {len(task_inputs)} task samples")
        print(f"Buffer capacity: {len(self.buffer)}/{self.args.buffer_size} (remaining: {remaining_slots})")
        
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
            # Handle both numpy array and torch tensor indices
            if isinstance(selected_indices, np.ndarray):
                selected_indices_tensor = torch.from_numpy(selected_indices).long()
            else:
                selected_indices_tensor = selected_indices.long() if isinstance(selected_indices, torch.Tensor) else torch.tensor(selected_indices).long()
            
            if len(selected_indices_tensor) > 0:
                selected_inputs = task_inputs[selected_indices_tensor]
                selected_labels = task_labels[selected_indices_tensor]
                
                # Add to buffer
                self.buffer.add_data(
                    examples=selected_inputs.to(self.device),
                    labels=selected_labels.to(self.device)
                )
                
                print(f"Added {len(selected_inputs)} samples to buffer")
                print(f"Buffer now contains {len(self.buffer)} samples")
                
              
            
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
        
        # Clean up task data files and memory
        self._cleanup_task_data()
        
        print(f"Task {self.current_task} completed")
    
    def begin_task(self, dataset):
        """
        Called at the beginning of each task.
        
        Args:
            dataset: Current dataset
        """
        print(f"Beginning task {self.current_task}")
        
        # Initialize file-based storage for new task
        if self.current_task not in self.task_data_files:
            temp_file = os.path.join(self.buffer_path, f'task_{self.current_task}_data.pkl')
            self.task_data_files[self.current_task] = temp_file
            
            # Initialize file
            with open(temp_file, 'wb') as f:
                pass  # Create empty file
            
            # Initialize set to track unique samples
            self.task_sample_hashes = set()
    
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
            'clip_grad': self.clip_grad,
            'memory_efficient': True,
            'buffer_path': self.buffer_path,
            'chunk_size': self.chunk_size,
            'task_files_count': len(self.task_data_files)
        }
        return stats
    
    def get_memory_usage(self):
        """Get current memory usage statistics."""
        import psutil
        import gc
        
        # Get system memory usage
        process = psutil.Process()
        memory_info = process.memory_info()
        
        # Get GPU memory usage if available
        gpu_memory = None
        if torch.cuda.is_available():
            gpu_memory = {
                'allocated': torch.cuda.memory_allocated() / 1024**3,  # GB
                'cached': torch.cuda.memory_reserved() / 1024**3,  # GB
            }
        
        stats = {
            'ram_usage_mb': memory_info.rss / 1024**2,  # MB
            'ram_usage_gb': memory_info.rss / 1024**3,  # GB
            'gpu_memory_gb': gpu_memory,
            'task_files_count': len(self.task_data_files),
            'buffer_samples': len(self.buffer),
            'chunk_size': self.chunk_size
        }
        
        return stats


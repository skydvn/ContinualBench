# -*-coding:utf8-*-

import torch
from torch.utils.data import DataLoader
import torchvision
import pickle
import os
import random
import copy
import numpy as np
from typing import Dict, List, Tuple, Optional, Union

from .csrel_coreset_functions import (
    select_by_loss_diff, get_class_dic, get_subset_by_id, make_class_sizes, add_new_data
)
from .csrel_loss_functions import CompliedLoss


class CSReLSelectionAgent:
    """CSReL Selection Agent for coreset selection."""
    
    def __init__(self, 
                 local_path: str,
                 transforms: Optional[torch.nn.Module],
                 init_size: int,
                 selection_steps: int,
                 cur_train_lr: float,
                 cur_train_steps: int,
                 use_cuda: bool,
                 eval_mode: str,
                 early_stop: int,
                 eval_steps: int,
                 model_params: Dict,
                 ref_train_params: Dict,
                 seed: int,
                 ref_model: Optional[torch.nn.Module] = None,
                 class_balance: bool = True,
                 only_new_data: bool = True,
                 loss_params: Optional[Dict] = None,
                 save_checkpoint: bool = False):
        """Initialize CSReL Selection Agent."""
        
        # All related settings
        self.local_path = local_path
        if not os.path.exists(self.local_path):
            os.makedirs(self.local_path)
        
        self.transforms = transforms
        self.init_size = init_size
        self.selection_steps = selection_steps
        self.cur_train_lr = cur_train_lr
        self.cur_train_steps = cur_train_steps
        self.eval_mode = eval_mode
        self.early_stop = early_stop
        self.eval_steps = eval_steps
        self.class_balance = class_balance
        self.only_new_data = only_new_data
        self.loss_params = loss_params
        self.ref_model = ref_model
        self.ref_train_params = ref_train_params
        self.model_params = model_params
        self.seed = seed
        self.save_checkpoint = save_checkpoint
        
        # Make train_params
        if loss_params is None:
            loss_params = {
                'ce_factor': 1.0,
                'mse_factor': 0.0
            }
        
        self.train_params = {
            'lr': self.cur_train_lr,
            'steps': self.cur_train_steps,
            'batch_size': ref_train_params.get('batch_size', 32),
            'use_cuda': use_cuda,
            'loss_params': loss_params
        }
        
        # File paths
        self.cur_train_file = os.path.join(self.local_path, 'cur_train.pkl')
        
        # Transforms
        self.to_pil = torchvision.transforms.ToPILImage()
        self.to_tensor = torchvision.transforms.ToTensor()
        
        # Set random seed
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if use_cuda:
            torch.cuda.manual_seed(seed)

    def make_data_loader(self, 
                        x: np.ndarray, 
                        y: np.ndarray, 
                        fname: str, 
                        batch_size: int = 32,
                        id_list: Optional[List] = None,
                        id2logit: Optional[Dict] = None,
                        extra_data: Optional[List] = None) -> Tuple[DataLoader, str]:
        """Make data loader for training."""
        
        # Create dataset
        class SimpleDataset:
            def __init__(self, data, labels, transforms=None):
                self.data = data
                self.labels = labels
                self.transforms = transforms
            
            def __len__(self):
                return len(self.data)
            
            def __getitem__(self, idx):
                # Convert numpy array to tensor
                if isinstance(self.data[idx], np.ndarray):
                    if self.data[idx].dtype == np.uint8:
                        # Convert to float and normalize
                        img = torch.from_numpy(self.data[idx]).float() / 255.0
                        if img.dim() == 3 and img.shape[0] == 3:
                            img = img.permute(1, 2, 0)  # CHW to HWC
                    else:
                        img = torch.from_numpy(self.data[idx]).float()
                else:
                    img = self.data[idx]
                
                # Ensure proper tensor format
                if img.dim() == 3 and img.shape[2] == 3:  # HWC format
                    img = img.permute(2, 0, 1)  # Convert to CHW
                elif img.dim() == 2:  # Grayscale
                    img = img.unsqueeze(0)  # Add channel dimension
                
                # Apply transforms if provided
                if self.transforms is not None:
                    try:
                        img = self.transforms(img)
                    except:
                        # If transforms fail, convert to PIL and back
                        if img.dim() == 3:
                            img_pil = self.to_pil(img)
                            img = self.transforms(img_pil)
                        else:
                            img = self.transforms(img)
                
                # Ensure final tensor is 3D (C, H, W) - no batch dimension
                if img.dim() == 4:
                    img = img.squeeze(0)  # Remove batch dimension
                elif img.dim() == 2:
                    img = img.unsqueeze(0)  # Add channel dimension
                
                return img, self.labels[idx]
        
        # Create dataset
        dataset = SimpleDataset(x, y, self.transforms)
        
        # Create data loader
        data_loader = DataLoader(
            dataset, 
            batch_size=batch_size, 
            shuffle=True, 
            drop_last=False,
            num_workers=0  # Avoid multiprocessing issues
        )
        
        # Save data to file for compatibility
        data_file = os.path.join(self.local_path, fname)
        with open(data_file, 'wb') as fw:
            for i in range(len(x)):
                d_id = i if id_list is None else id_list[i]
                lab = y[i]
                logit = id2logit.get(d_id) if id2logit is not None else None
                
                if logit is not None:
                    pickle.dump((d_id, x[i], lab, logit), fw)
                else:
                    pickle.dump((d_id, x[i], lab), fw)
        
        return data_loader, data_file

    def train_ref_model(self, 
                       x: np.ndarray, 
                       y: np.ndarray, 
                       id2logit: Optional[Dict] = None,
                       extra_data: Optional[List] = None,
                       log_file: Optional[str] = None,
                       verbose: bool = True) -> None:
        """Train reference model on given data."""
        
        if verbose:
            print("Training Reference Model ===")
        
        # Create data loader
        train_loader, data_file = self.make_data_loader(
            x=x,
            y=y,
            fname='ref_train.pkl',
            batch_size=self.ref_train_params['batch_size'],
            id2logit=id2logit,
            extra_data=extra_data
        )
        
        # Check if reference model is available
        if self.ref_model is None:
            print("Error: Reference model not provided, cannot train reference model")
            raise ValueError("Reference model must be set before training")
        
        # Train the reference model
        self.ref_model.train()
        
        # Ensure model is on the correct device
        if self.train_params['use_cuda'] and torch.cuda.is_available():
            self.ref_model = self.ref_model.cuda()
        
        optimizer = torch.optim.SGD(self.ref_model.parameters(), lr=self.ref_train_params['lr'])
        criterion = CompliedLoss(
            ce_factor=self.ref_train_params.get('ce_factor', 1.0),
            mse_factor=self.ref_train_params.get('mse_factor', 0.0),
            reduction='mean'
        )
        
        # Training loop
        for epoch in range(self.ref_train_params['epochs']):
            epoch_loss = 0.0
            for batch_idx, (sps, labs) in enumerate(train_loader):
                # Ensure all tensors are on the same device as the model
                device = next(self.ref_model.parameters()).device
                sps = sps.to(device)
                labs = labs.to(device)
                
                # Get logits if available
                logits = None
                if id2logit is not None:
                    # This would need to be implemented based on specific requirements
                    pass
                
                optimizer.zero_grad()
                outputs = self.ref_model(sps)
                loss = criterion(outputs, labs, logits)
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
            
            if verbose and epoch % 10 == 0:
                print(f"Epoch {epoch}, Loss: {epoch_loss / len(train_loader):.4f}")
        
        # Set model to eval mode
        self.ref_model.eval()
        
        if verbose:
            print("Reference model training completed")

    def compute_loss_dic(self, 
                        ref_model: torch.nn.Module, 
                        data_loader: DataLoader, 
                        aug_iters: int, 
                        use_cuda: bool, 
                        loss_params: Dict) -> Dict[int, float]:
        """Compute loss dictionary for reference model."""
        ref_model.eval()
        loss_fn = CompliedLoss(
            ce_factor=loss_params['ce_factor'],
            mse_factor=loss_params['mse_factor'],
            reduction='none'
        )
        
        if use_cuda:
            ref_model.cuda()
        
        loss_dic = {}
        
        with torch.no_grad():
            for i in range(aug_iters):
                for batch_idx, data in enumerate(data_loader):
                    print(f"Debug: Batch {batch_idx}, data type: {type(data)}, data length: {len(data) if hasattr(data, '__len__') else 'no length'}")
                    # Handle different data formats - be more flexible
                    try:
                        if len(data) == 4:
                            d_ids, sps, labs, logit = data
                        elif len(data) == 3:
                            d_ids, sps, labs = data
                            logit = None
                        elif len(data) == 2:
                            # Handle case where only (sps, labs) are returned
                            sps, labs = data
                            d_ids = torch.arange(len(sps))  # Create sequential IDs
                            logit = None
                        else:
                            raise ValueError(f"Unexpected data format with {len(data)} elements")
                    except Exception as e:
                        print(f"Error unpacking data: {e}, data type: {type(data)}, data length: {len(data) if hasattr(data, '__len__') else 'no length'}")
                        # Fallback: assume it's (sps, labs) format
                        sps, labs = data
                        d_ids = torch.arange(len(sps))
                        logit = None
                    
                    if use_cuda:
                        sps = sps.cuda()
                        labs = labs.cuda()
                        if logit is not None:
                            logit = logit.cuda()
                    # Fix tensor shape if needed (remove extra dimensions)
                    if len(sps.shape) == 5:  # [batch, 1, channels, height, width]
                        sps = sps.squeeze(1)  # Remove the extra dimension
                    
                    loss = loss_fn(ref_model(sps), labs, logit)
                    if use_cuda:
                        loss = loss.cpu()
                    loss = loss.clone().detach().numpy()
                    
                    batch_size = sps.shape[0]
                    for j in range(batch_size):
                        d_id = int(d_ids[j].numpy())
                        if d_id not in loss_dic:
                            loss_dic[d_id] = [loss[j]]
                        else:
                            loss_dic[d_id].append(loss[j])
        
        # Average losses across augmentations
        for d_id in loss_dic.keys():
            loss_dic[d_id] = float(np.mean(loss_dic[d_id]))
        
        if use_cuda:
            ref_model.cpu()
        
        return loss_dic

    def incremental_selection(self, 
                            x: np.ndarray, 
                            y: np.ndarray, 
                            select_size: int, 
                            id_list: Optional[List] = None,
                            loss_dic: Optional[Dict] = None, 
                            loss_dic_dump_file: Optional[str] = None,
                            verbose: bool = True, 
                            class_pool: Optional[List] = None, 
                            id2logit: Optional[Dict] = None, 
                            ideal_logit: bool = False, 
                            extra_data: Optional[List] = None) -> List:
        """Perform incremental coreset selection."""
        
        if select_size >= x.shape[0]:
            print('Warning: select size greater than data size', select_size, x.shape[0])
        
        # The id of each sample is assigned according to order
        if loss_dic is None:
            # Train reference model
            if self.ref_model is None:
                print("Error: Reference model not set, cannot perform incremental selection")
                return []
            
            # Compute loss dict
            temp_loader, temp_file = self.make_data_loader(
                x=x,
                y=y,
                fname='temp_data.pkl',
                batch_size=256,
                id_list=id_list,
                id2logit=id2logit
            )
            
            if ideal_logit:
                ref_loss_params = {
                    'ce_factor': 1.0,
                    'mse_factor': 0.0
                }
            else:
                ref_loss_params = self.train_params['loss_params']
            
            ref_loss_dic = self.compute_loss_dic(
                ref_model=self.ref_model,
                data_loader=temp_loader,
                aug_iters=1,
                use_cuda=self.train_params['use_cuda'],
                loss_params=ref_loss_params
            )
            
            if loss_dic_dump_file is not None:
                with open(loss_dic_dump_file, 'wb') as fw:
                    pickle.dump(ref_loss_dic, fw)
            
            os.remove(temp_file)
        else:
            ref_loss_dic = loss_dic
        
        # Init model and selection
        all_selected_ids = set()
        incremental_size = max(int(select_size / self.selection_steps), 1)
        
        # Create initial model (simplified for mammoth integration)
        if self.ref_model is None:
            print("Error: No reference model available for selection")
            return []
        
        init_model = copy.deepcopy(self.ref_model)
        all_class_ids = get_class_dic(y=y)
        class_ids = {}
        
        if class_pool is None:
            for i in range(self.model_params['num_class']):
                class_ids[i] = set()
        else:
            for i in class_pool:
                class_ids[i] = set()
        
        # Create training dataset
        class RandomDataset:
            def __init__(self, data, labels, transforms=None):
                self.data = data
                self.labels = labels
                self.transforms = transforms
            
            def __len__(self):
                return len(self.data)
            
            def __getitem__(self, idx):
                if isinstance(self.data[idx], np.ndarray):
                    if self.data[idx].dtype == np.uint8:
                        img = torch.from_numpy(self.data[idx]).float() / 255.0
                        if img.dim() == 3 and img.shape[0] == 3:
                            img = img.permute(1, 2, 0)
                    else:
                        img = torch.from_numpy(self.data[idx]).float()
                else:
                    img = self.data[idx]
                
                if img.dim() == 3 and img.shape[2] == 3:
                    img = img.permute(2, 0, 1)
                elif img.dim() == 2:
                    img = img.unsqueeze(0)
                
                if self.transforms is not None:
                    try:
                        img = self.transforms(img)
                    except:
                        if img.dim() == 3:
                            img_pil = self.to_pil(img)
                            img = self.transforms(img_pil)
                        else:
                            img = self.transforms(img)
                
                # Ensure final tensor is 3D (C, H, W) - no batch dimension
                if img.dim() == 4:
                    img = img.squeeze(0)  # Remove batch dimension
                elif img.dim() == 2:
                    img = img.unsqueeze(0)  # Add channel dimension
                
                return img, self.labels[idx]
        
        cur_train_dataset = RandomDataset(
            data=x,
            labels=y,
            transforms=self.transforms
        )
        
        train_loader = DataLoader(cur_train_dataset, batch_size=self.train_params['batch_size'], drop_last=False)
        
        if self.eval_mode in ['acc', 'avg_loss', 'loss_var']:
            full_train_loader, full_data_file = self.make_data_loader(
                x=x,
                y=y,
                batch_size=self.train_params['batch_size'],
                fname='full_data.pkl',
                id_list=id_list,
                id2logit=id2logit
            )
        else:
            full_train_loader = None
            full_data_file = ''
        
        if id_list is None:
            full_ids = list(range(x.shape[0]))
        else:
            full_ids = id_list
        
        # Make initial set
        if self.init_size > 0:
            if bool(self.class_balance):
                class_size = []
                base_size = int(self.init_size // self.model_params['num_class'])
                for i in range(self.model_params['num_class']):
                    class_size.append(base_size)
                res = self.init_size - self.model_params['num_class'] * base_size
                for i in range(self.model_params['num_class']):
                    if res == 0:
                        break
                    class_size[i] = class_size[i] + 1
                    res -= 1
                init_ids = set()
                for i in range(self.model_params['num_class']):
                    cids = all_class_ids[i]
                    init_cids = random.sample(list(cids), class_size[i])
                    for d_id in init_cids:
                        init_ids.add(d_id)
            else:
                init_ids = random.sample(full_ids, self.init_size)
                init_ids = set(init_ids)
            
            init_data = get_subset_by_id(
                x=x, y=y, ids=init_ids,
                transforms=self.to_pil if self.transforms is not None else None,
                id_list=id_list)
            
            for di in init_data:
                d_id = int(di[0])
                lab = int(di[2])
                class_ids[lab].add(d_id)
            
            with open(self.cur_train_file, 'wb') as fw:
                for di in init_data:
                    pickle.dump(di, fw)
            
            for d_id in init_ids:
                all_selected_ids.add(d_id)
            
            if self.save_checkpoint:
                self.dump_selected_ids(selected_ids=all_selected_ids)
            
            # Train initial model (simplified)
            init_model.train()
            if self.train_params['use_cuda'] and torch.cuda.is_available():
                init_model = init_model.cuda()
            
            optimizer = torch.optim.SGD(init_model.parameters(), lr=self.train_params['lr'])
            criterion = CompliedLoss(
                ce_factor=self.train_params['loss_params']['ce_factor'],
                mse_factor=self.train_params['loss_params']['mse_factor'],
                reduction='mean'
            )
            
            for epoch in range(self.train_params['steps']):
                for batch_idx, (sps, labs) in enumerate(train_loader):
                    if self.train_params['use_cuda'] and torch.cuda.is_available():
                        sps = sps.cuda()
                        labs = labs.cuda()
                    
                    optimizer.zero_grad()
                    outputs = init_model(sps)
                    loss = criterion(outputs, labs)
                    loss.backward()
                    optimizer.step()
            
            init_model.eval()
        
        # Incremental selection loop
        iteration = 0
        max_iterations = 10  # Prevent infinite loops
        
        while len(all_selected_ids) < select_size and iteration < max_iterations:
            iteration += 1
            
            id_pool = set()
            for d_id in full_ids:
                if bool(self.only_new_data):
                    if d_id not in all_selected_ids:
                        id_pool.add(d_id)
                else:
                    id_pool.add(d_id)
            
            if len(id_pool) == 0:
                break
            
            if bool(self.class_balance):
                class_sizes = make_class_sizes(
                    class_ids=class_ids,
                    incremental_size=min(incremental_size, select_size - len(all_selected_ids))
                )
            else:
                class_sizes = None
            
            rand_data = get_subset_by_id(
                x=x,
                y=y,
                ids=id_pool,
                transforms=self.to_pil if self.transforms is not None else None,
                id_list=id_list,
                id2logit=id2logit
            )
            
            selected_data, _ = select_by_loss_diff(
                ref_loss_dic=ref_loss_dic,
                rand_data=rand_data,
                model=init_model,
                incremental_size=min(incremental_size, select_size - len(all_selected_ids)),
                transforms=self.transforms,
                on_cuda=self.train_params['use_cuda'],
                loss_params=self.train_params['loss_params'],
                class_sizes=class_sizes
            )
            
            flg_add = False
            for di in selected_data:
                d_id = int(di[0])
                lab = int(di[2])
                if d_id not in all_selected_ids:
                    all_selected_ids.add(d_id)
                    class_ids[lab].add(d_id)
                    flg_add = True
            
            if not flg_add:
                break
            
            if self.save_checkpoint:
                self.dump_selected_ids(selected_ids=all_selected_ids)
        
        # Clean up
        if os.path.exists(self.cur_train_file):
            os.remove(self.cur_train_file)
        if full_data_file and os.path.exists(full_data_file):
            os.remove(full_data_file)
        
        # Return the actual selected data, not just IDs
        selected_data_list = []
        for d_id in all_selected_ids:
            # Find the corresponding data for this ID
            for di in rand_data:
                if di[0] == d_id:
                    selected_data_list.append(di)
                    break
        print(f"Selected {len(selected_data_list)} data out of {select_size}")
        return selected_data_list

    def dump_selected_ids(self, selected_ids: set):
        """Dump selected IDs to file."""
        dump_file = os.path.join(self.local_path, 'selected_ids.pkl')
        with open(dump_file, 'wb') as fw:
            pickle.dump(selected_ids, fw)

    def load_selected_ids(self) -> set:
        """Load selected IDs from file."""
        dump_file = os.path.join(self.local_path, 'selected_ids.pkl')
        if os.path.exists(dump_file):
            with open(dump_file, 'rb') as fr:
                return pickle.load(fr)
        return set()

    def clear_path(self):
        """Clear temporary files."""
        if os.path.exists(self.local_path):
            import shutil
            shutil.rmtree(self.local_path)
            os.makedirs(self.local_path)

    def reset_ref_model(self):
        """Reset reference model."""
        self.ref_model = None

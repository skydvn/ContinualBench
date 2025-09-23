# -*-coding:utf8-*-

import os
import torch
from torch.utils.data import DataLoader
import numpy as np
import random
import pickle
import copy
from typing import Dict, List, Tuple, Optional, Union

from .csrel_loss_functions import CompliedLoss


def random_select(id2cnt: Dict, select_size: int) -> Tuple[set, Dict]:
    """Random selection of samples."""
    selected_id2prob = {}
    all_ids = list(id2cnt.keys())
    selected_ids = set(random.sample(all_ids, select_size))
    for d_id in selected_ids:
        selected_id2prob[d_id] = 1.0
    return selected_ids, selected_id2prob


def add_new_data(data_file: str, new_data: List) -> None:
    """Add new data to existing data file."""
    ori_data = []
    ori_ids = set()
    if os.path.exists(data_file):
        with open(data_file, 'rb') as fr:
            while True:
                try:
                    di = pickle.load(fr)
                    d_id = di[0]
                    ori_ids.add(int(d_id))
                    ori_data.append(di)
                except EOFError:
                    break
    all_data = ori_data
    for di in new_data:
        d_id = int(di[0])
        if d_id not in ori_ids:
            all_data.append(di)
    random.shuffle(all_data)
    with open(data_file, 'wb') as fw:
        for di in all_data:
            pickle.dump(di, fw)


def select_by_loss_diff(ref_loss_dic: Dict[int, float], 
                       rand_data: List, 
                       model: torch.nn.Module, 
                       incremental_size: int, 
                       transforms: Optional[torch.nn.Module] = None, 
                       on_cuda: bool = True, 
                       loss_params: Optional[Dict] = None, 
                       class_sizes: Optional[Dict] = None) -> Tuple[List, Dict]:
    """Select samples based on loss difference."""
    
    if loss_params is None:
        loss_params = {
            'ce_factor': 1.0,
            'mse_factor': 0.0
        }
    
    status = model.training
    model.eval()
    
    if on_cuda and torch.cuda.is_available():
        model = model.cuda()
    
    loss_fn = CompliedLoss(
        ce_factor=loss_params['ce_factor'],
        mse_factor=loss_params['mse_factor'],
        reduction='none'
    )
    
    selected_data = []
    selected_id2prob = {}
    
    # Process data in batches to avoid memory issues
    batch_size = min(32, len(rand_data))
    
    with torch.no_grad():
        for i in range(0, len(rand_data), batch_size):
            batch_data = rand_data[i:i + batch_size]
            batch_sps = []
            batch_labs = []
            batch_ids = []
            
            for di in batch_data:
                d_id, sp, lab = di[0], di[1], di[2]
                
                # Convert to tensor if needed
                if isinstance(sp, torch.Tensor):
                    # Already a tensor, ensure proper format
                    sp_tensor = sp
                elif isinstance(sp, np.ndarray):
                    if sp.dtype == np.uint8:
                        sp_tensor = torch.from_numpy(sp).float() / 255.0
                        if sp_tensor.dim() == 3 and sp_tensor.shape[0] == 3:
                            sp_tensor = sp_tensor.permute(1, 2, 0)  # CHW to HWC
                    else:
                        sp_tensor = torch.from_numpy(sp).float()
                elif hasattr(sp, 'mode') or str(type(sp)).find('PIL') != -1:  # PIL Image
                    import torchvision.transforms as T
                    to_tensor = T.ToTensor()
                    sp_tensor = to_tensor(sp)
                else:  # Other non-tensor types
                    import torchvision.transforms as T
                    to_tensor = T.ToTensor()
                    sp_tensor = to_tensor(sp)
                
                # Ensure proper tensor format
                if sp_tensor.dim() == 3 and sp_tensor.shape[2] == 3:  # HWC format
                    sp_tensor = sp_tensor.permute(2, 0, 1)  # Convert to CHW
                elif sp_tensor.dim() == 2:  # Grayscale
                    sp_tensor = sp_tensor.unsqueeze(0)  # Add channel dimension
                
                # Apply transforms if provided
                if transforms is not None:
                    try:
                        sp_tensor = transforms(sp_tensor)
                    except:
                        # If transforms fail, convert to PIL and back
                        import torchvision.transforms as T
                        to_pil = T.ToPILImage()
                        to_tensor = T.ToTensor()
                        sp_tensor = to_tensor(transforms(to_pil(sp_tensor)))
                
                # Ensure tensor has correct dimensions (3D: C, H, W)
                if sp_tensor.dim() == 4:
                    sp_tensor = sp_tensor.squeeze(0)  # Remove extra batch dimension
                elif sp_tensor.dim() == 2:
                    sp_tensor = sp_tensor.unsqueeze(0)  # Add channel dimension
                
                #print(f"Debug: Processed tensor shape: {sp_tensor.shape}")
                batch_sps.append(sp_tensor)
                batch_labs.append(lab)
                batch_ids.append(d_id)
            
            if not batch_sps:
                continue
            
            # Stack tensors - ensure all tensors have the same shape
            if batch_sps:
                # Check dimensions of first tensor
                first_tensor = batch_sps[0]
               # print(f"Debug: First tensor shape: {first_tensor.shape}")
                
                # Ensure all tensors have the same shape as the first one
                normalized_tensors = []
                for i, tensor in enumerate(batch_sps):
                    if tensor.shape != first_tensor.shape:
                        #print(f"Debug: Tensor {i} shape mismatch: {tensor.shape} vs {first_tensor.shape}")
                        # Resize to match first tensor
                        if tensor.dim() == 3 and first_tensor.dim() == 3:
                            # Both are 3D, check if we need to permute
                            if tensor.shape[0] == 3 and first_tensor.shape[2] == 3:
                                # tensor is CHW, first_tensor is HWC
                                tensor = tensor.permute(1, 2, 0)
                            elif tensor.shape[2] == 3 and first_tensor.shape[0] == 3:
                                # tensor is HWC, first_tensor is CHW
                                tensor = tensor.permute(2, 0, 1)
                    normalized_tensors.append(tensor)
                
                sps = torch.stack(normalized_tensors, dim=0)
                #print(f"Debug: Final sps shape: {sps.shape}")
            else:
                continue
            
            labs = torch.tensor(batch_labs, dtype=torch.long)
            
            # Move to device
            if on_cuda and torch.cuda.is_available():
                sps = sps.cuda()
                labs = labs.cuda()
            
            # Compute losses
            outputs = model(sps)
            losses = loss_fn(outputs, labs)
            
            # Compute loss differences
            for j, (d_id, loss) in enumerate(zip(batch_ids, losses)):
                ref_loss = ref_loss_dic.get(d_id, 0.0)
                loss_diff = loss.item() - ref_loss
                
                # Store loss difference for sorting
                batch_data[j] = (d_id, batch_data[j][1], batch_data[j][2], loss_diff)
        
        # Sort by loss difference (descending)
        rand_data.sort(key=lambda x: x[3] if len(x) > 3 else 0, reverse=True)
        
        # Select samples based on class sizes if provided
        if class_sizes is not None:
            class_counts = {}
            for di in rand_data:
                d_id, sp, lab = di[0], di[1], di[2]
                lab = int(lab)
                
                if lab not in class_counts:
                    class_counts[lab] = 0
                
                if class_counts[lab] < class_sizes.get(lab, 0):
                    selected_data.append((d_id, sp, lab))
                    selected_id2prob[d_id] = 1.0
                    class_counts[lab] += 1
                    
                    if len(selected_data) >= incremental_size:
                        break
        else:
            # Select top samples by loss difference
            for di in rand_data[:incremental_size]:
                d_id, sp, lab = di[0], di[1], di[2]
                selected_data.append((d_id, sp, lab))
                selected_id2prob[d_id] = 1.0
    
    model.train(status)
    return selected_data, selected_id2prob


def get_class_dic(y: np.ndarray) -> Dict[int, set]:
    """Get class dictionary mapping class to sample indices."""
    class_dic = {}
    for i, label in enumerate(y):
        label = int(label)
        if label not in class_dic:
            class_dic[label] = set()
        class_dic[label].add(i)
    return class_dic


def get_subset_by_id(x: np.ndarray, 
                    y: np.ndarray, 
                    ids: Union[List, set], 
                    transforms: Optional[torch.nn.Module] = None, 
                    id_list: Optional[List] = None, 
                    id2logit: Optional[Dict] = None) -> List:
    """Get subset of data by sample IDs."""
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
        # Always create tensor first
        sp = torch.tensor(x[pi], dtype=torch.float32).clone().detach()
        
        # Apply transforms if provided (but keep as tensor)
        if transforms is not None:
            # If transforms expects PIL Image, convert temporarily
            if hasattr(transforms, '__call__'):
                try:
                    # Try to apply transforms directly to tensor
                    sp = transforms(sp)
                except:
                    # If that fails, convert to PIL and back
                    import torchvision.transforms as T
                    to_pil = T.ToPILImage()
                    to_tensor = T.ToTensor()
                    sp = to_tensor(transforms(to_pil(sp)))
        
        if id_list is None:
            d_id = pi
        else:
            d_id = id_list[pi]
        
        lab = y[pi]
        
        if id2logit is not None and d_id in id2logit:
            logit = id2logit[d_id]
            selected_data.append((d_id, sp, lab, logit))
        else:
            selected_data.append((d_id, sp, lab))
    
    return selected_data


def make_class_sizes(class_ids: Dict[int, set], 
                    incremental_size: int) -> Dict[int, int]:
    """Make class sizes for balanced selection."""
    class_sizes = {}
    num_classes = len(class_ids)
    
    if num_classes == 0:
        return class_sizes
    
    base_size = incremental_size // num_classes
    remainder = incremental_size % num_classes
    
    for i, class_id in enumerate(class_ids.keys()):
        class_sizes[class_id] = base_size
        if i < remainder:
            class_sizes[class_id] += 1
    
    return class_sizes

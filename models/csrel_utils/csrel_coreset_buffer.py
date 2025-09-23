# -*-coding:utf8-*-

import copy
import torch
import pickle
import os
import numpy as np
import torchvision
import random
from typing import Dict, List, Tuple, Optional, Union

from .csrel_selection_agent import CSReLSelectionAgent
from .csrel_coreset_functions import get_class_dic, get_subset_by_id


class CSReLCoresetBuffer:
    """CSReL Coreset Buffer for continual learning."""
    
    def __init__(self, 
                 local_path: str,
                 model_params: Dict,
                 transforms: Optional[torch.nn.Module],
                 selection_params: Dict,
                 buffer_size: int,
                 use_cuda: bool,
                 task_dic: Dict,
                 seed: int,
                 selection_transforms: Optional[torch.nn.Module] = None,
                 extra_data_mode: Optional[str] = None):
        """Initialize CSReL Coreset Buffer."""
        
        self.local_path = local_path
        if not os.path.exists(self.local_path):
            os.makedirs(self.local_path)
        
        self.model_params = model_params
        self.transforms = transforms
        self.selection_params = selection_params
        self.use_cuda = use_cuda
        self.seed = seed
        self.buffer_size = buffer_size
        self.task_dic = task_dic
        self.selection_transforms = selection_transforms
        self.extra_data_mode = extra_data_mode
        
        # Build coreset selector
        self.coreset_selector = CSReLSelectionAgent(
            local_path=self.local_path,
            transforms=self.transforms if self.selection_transforms is None else self.selection_transforms,
            init_size=0,
            selection_steps=self.selection_params['selection_steps'],
            cur_train_lr=self.selection_params['cur_train_lr'],
            cur_train_steps=self.selection_params['cur_train_steps'],
            use_cuda=self.use_cuda,
            eval_mode='none',
            early_stop=-1,
            eval_steps=100,
            model_params=self.model_params,
            ref_train_params=self.selection_params['ref_train_params'],
            seed=self.seed,
            ref_model=None,
            class_balance=self.selection_params['class_balance'],
            only_new_data=True,
            loss_params=None if 'loss_params' not in self.selection_params else self.selection_params['loss_params']
        )
        
        self.data = []
        self.id2task = {}
        self.to_tensor = torchvision.transforms.ToTensor()
        self.to_pil = torchvision.transforms.ToPILImage()
        self.id_bias = 0

    def update_buffer(self, 
                     task_cnts: List[int], 
                     task_id: int, 
                     cur_x: np.ndarray, 
                     cur_y: np.ndarray, 
                     full_cur_x: np.ndarray, 
                     full_cur_y: np.ndarray, 
                     cur_id2logit: Optional[Dict] = None,
                     next_x: Optional[np.ndarray] = None, 
                     next_y: Optional[np.ndarray] = None) -> None:
        """Update the coreset buffer with new task data."""
        
        # Distribute buffer size to each task
        task_sizes = []
        for i in range(task_id + 1):
            task_sizes.append(int(task_cnts[i] / sum(task_cnts) * self.buffer_size))
        
        rest_size = self.buffer_size - sum(task_sizes)
        for j in range(rest_size):
            task_sizes[j] += 1
        
        new_id2task = {}
        new_data = []
        for i in range(task_id + 1):
            new_data.append([])
        
        pre_select_size = 0
        
        # Make id list
        cur_id_list = []
        for i in range(cur_x.shape[0]):
            cur_id_list.append(i + self.id_bias)
        
        if cur_id2logit is not None:  # Support for adding knowledge distillation
            new_id2logit = {}
            for d_id in cur_id2logit.keys():
                new_did = d_id + self.id_bias
                new_id2logit[new_did] = cur_id2logit[d_id]
        else:
            new_id2logit = None
        
        # Re-select previous tasks
        for i in range(task_id):
            print(f'\tSelecting coreset for task {i}, size: {task_sizes[i]}')
            
            # Get data
            id_pool = []
            prv_x = []
            prv_y = []
            id2logit = {}
            
            for di in self.data[i]:
                if len(di) == 3:
                    d_id, sp, lab = di
                elif len(di) == 4:
                    d_id, sp, lab, logit = di
                    id2logit[d_id] = logit
                else:
                    raise ValueError('Invalid data length')
                
                if isinstance(sp, torch.Tensor):
                    prv_x.append(sp.numpy())
                else:
                    prv_x.append(self.to_tensor(sp).numpy())
                prv_y.append(lab)
                id_pool.append(d_id)
            
            prv_x = np.stack(prv_x, axis=0)
            prv_y = np.array(prv_y)
            
            if len(id2logit) == 0:
                id2logit = None
            
            # Load loss dic
            loss_dic_file = os.path.join(self.local_path, 'ref_loss_dic' + str(i) + '.pkl')
            if os.path.exists(loss_dic_file):
                with open(loss_dic_file, 'rb') as fr:
                    ref_loss_dic = pickle.load(fr)
            else:
                ref_loss_dic = None
            
            # Add support for extra data
            extra_data = self.make_extra_data(
                cur_x=cur_x,
                cur_y=cur_y,
                cur_id_list=cur_id_list,
                task_id=task_id,
                c_tid=i,
                task_sizes=task_sizes,
                next_x=next_x,
                next_y=next_y
            )
            
            # Coreset selection
            selected_data = self.coreset_selector.incremental_selection(
                x=prv_x,
                y=prv_y,
                select_size=task_sizes[i],
                loss_dic=ref_loss_dic,
                verbose=False,
                class_pool=self.task_dic[i],
                id_list=id_pool,
                id2logit=id2logit,
                extra_data=extra_data
            )
            self.coreset_selector.clear_path()
            
            # Update data and id2task
            for si in selected_data:
                new_data[i].append(si)
                d_id = int(si[0])
                new_id2task[d_id] = i
                pre_select_size += 1
            id_pool.clear()
        
        # Select current task data
        cur_select_size = self.buffer_size - pre_select_size
        
        # Train current ref model
        print(f'\tTraining ref model for task {task_id}, size: {cur_select_size}')
        
        if 'ref_sample_per_task' in self.selection_params['ref_train_params'] and \
                self.selection_params['ref_train_params']['ref_sample_per_task'] > 0:
            extra_data = self.make_extra_ref_samples(
                sample_per_task=self.selection_params['ref_train_params']['ref_sample_per_task'])
        else:
            extra_data = None
        
        self.coreset_selector.train_ref_model(
            x=full_cur_x,
            y=full_cur_y,
            verbose=False,
            extra_data=extra_data,
            log_file=os.path.join(self.local_path, 'holdout_model_loss' + str(task_id) + '.pkl')
        )
        
        loss_dic_dump_file = os.path.join(self.local_path, 'ref_loss_dic' + str(task_id) + '.pkl')
        
        # Coreset selection
        print(f'\tSelecting coreset for task {task_id}')
        extra_data = self.make_extra_data(
            cur_x=cur_x,
            cur_y=cur_y,
            cur_id_list=cur_id_list,
            task_id=task_id,
            c_tid=task_id,
            task_sizes=task_sizes,
            next_x=next_x,
            next_y=next_y
        )
        
        cur_selected_data = self.coreset_selector.incremental_selection(
            x=cur_x,
            y=cur_y,
            select_size=cur_select_size,
            loss_dic=None,
            loss_dic_dump_file=loss_dic_dump_file,
            verbose=False,
            class_pool=self.task_dic[task_id],
            id_list=cur_id_list,
            id2logit=new_id2logit,
            extra_data=extra_data
        )
        
        self.coreset_selector.clear_path()
        self.id_bias += cur_x.shape[0]
        self.coreset_selector.reset_ref_model()
        
        # Update data and id2task
        for si in cur_selected_data:
            new_data[task_id].append(si)
            d_id = int(si[0])
            new_id2task[d_id] = task_id
            pre_select_size += 1
        
        # Update data
        self.data = new_data
        self.id2task = new_id2task

    def make_extra_data(self, 
                       cur_x: np.ndarray, 
                       cur_y: np.ndarray, 
                       cur_id_list: List[int], 
                       task_id: int, 
                       c_tid: int, 
                       task_sizes: List[int], 
                       next_x: Optional[np.ndarray] = None, 
                       next_y: Optional[np.ndarray] = None) -> Optional[List]:
        """Make extra data for selection."""
        extra_data = []
        
        if self.extra_data_mode is not None and 'other_task' in self.extra_data_mode:
            for j in range(task_id):  # For previous data
                if j == c_tid:
                    continue
                for di in self.data[j]:
                    extra_data.append(di)
            
            # For current data
            if c_tid < task_id:
                for i in range(cur_x.shape[0]):
                    extra_data.append((cur_id_list[i], cur_x[i], cur_y[i]))
            
            # For next data
            if next_x is not None and next_y is not None:
                for i in range(next_x.shape[0]):
                    extra_data.append((self.id_bias + cur_x.shape[0] + i, next_x[i], next_y[i]))
        
        return extra_data

    def make_extra_ref_samples(self, sample_per_task: int) -> List:
        """Make extra reference samples."""
        extra_ref_samples = []
        for i in range(len(self.data)):
            if len(self.data[i]) > 0:
                sample_size = min(sample_per_task, len(self.data[i]))
                sampled_data = random.sample(self.data[i], sample_size)
                extra_ref_samples.extend(sampled_data)
        return extra_ref_samples

    def get_data(self):
        """Get data from buffer."""
        for i in range(len(self.data)):
            sps = []
            labs = []
            logits = []
            
            for di in self.data[i]:
                if len(di) == 3:
                    d_id, sp, lab = di
                elif len(di) == 4:
                    d_id, sp, lab, logit = di
                    logits.append(torch.tensor(logit, dtype=torch.float32))
                else:
                    raise ValueError('Invalid data length')
                
            if self.transforms is not None:
                aug_sp = self.transforms(sp)
            else:
                aug_sp = sp
            
            # Ensure we have a tensor, not PIL Image
            if isinstance(aug_sp, torch.Tensor):
                # Already a tensor, ensure proper format
                if aug_sp.dim() == 3 and aug_sp.shape[2] == 3:  # HWC format
                    aug_sp = aug_sp.permute(2, 0, 1)  # Convert to CHW
                elif aug_sp.dim() == 2:  # Grayscale
                    aug_sp = aug_sp.unsqueeze(0)  # Add channel dimension
            elif hasattr(aug_sp, 'mode') or str(type(aug_sp)).find('PIL') != -1: 
                import torchvision.transforms as T
                to_tensor = T.ToTensor()
                aug_sp = to_tensor(aug_sp)
            elif isinstance(aug_sp, np.ndarray):
                if aug_sp.dtype == np.uint8:
                    aug_sp = torch.from_numpy(aug_sp).float() / 255.0
                    if aug_sp.dim() == 3 and aug_sp.shape[0] == 3:
                        aug_sp = aug_sp.permute(1, 2, 0)  # CHW to HWC
                else:
                    aug_sp = torch.from_numpy(aug_sp).float()
                # Convert to CHW format
                if aug_sp.dim() == 3 and aug_sp.shape[2] == 3:
                    aug_sp = aug_sp.permute(2, 0, 1)
            
            sps.append(aug_sp)
            labs.append(lab)
            
            # Check if we have any samples to avoid empty tensor stack
            if len(sps) == 0:
                continue  # Skip empty tasks
            
            out_data = [torch.stack(sps, dim=0), torch.tensor(labs, dtype=torch.long)]
            if len(logits) > 0:
                out_data.append(torch.stack(logits, dim=0))
            
            yield out_data

    def get_sub_data(self, size: int) -> Union[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        """Get subset of data from buffer."""
        all_data = []
        for i in range(len(self.data)):
            all_data = all_data + self.data[i]
        
        if len(all_data) == 0:
            return torch.empty(0), torch.empty(0, dtype=torch.long)
        
        inds = random.sample(list(range(len(all_data))), min(size, len(all_data)))
        selected_sps = []
        selected_labs = []
        selected_logits = []
        
        for ind in inds:
            di = all_data[ind]
            if len(di) == 3:
                d_id, sp, lab = di
            elif len(di) == 4:
                d_id, sp, lab, logit = di
                selected_logits.append(torch.tensor(logit, dtype=torch.float32))
            else:
                raise ValueError('Invalid data length')
            
            if self.transforms is not None:
                aug_sp = self.transforms(sp)
            else:
                aug_sp = sp
            
            selected_sps.append(aug_sp)
            selected_labs.append(lab)
        
        out_data = [torch.stack(selected_sps, dim=0), torch.tensor(selected_labs, dtype=torch.long)]
        if len(selected_logits) > 0:
            out_data.append(torch.stack(selected_logits, dim=0))
        
        return tuple(out_data)

    def shuffle_data(self):
        """Shuffle data in buffer."""
        random.shuffle(self.data)

    def is_empty(self) -> bool:
        """Check if buffer is empty."""
        if len(self.data) == 0:
            return True
        
        # Check if all tasks are empty
        for task_data in self.data:
            if len(task_data) > 0:
                return False
        
        return True

    def dump_data(self, task_id: int):
        """Dump buffer data to file."""
        dump_file = os.path.join(self.local_path, 'buffer_data' + str(task_id) + '.pkl')
        with open(dump_file, 'wb') as fw:
            pickle.dump(self.data, fw)

    def load_data(self, task_id: int):
        """Load buffer data from file."""
        dump_file = os.path.join(self.local_path, 'buffer_data' + str(task_id) + '.pkl')
        if os.path.exists(dump_file):
            with open(dump_file, 'rb') as fr:
                self.data = pickle.load(fr)

    def set_current_model(self, model: torch.nn.Module):
        """Set current model for reference."""
        self.current_model = model
        # Also set the reference model in the selection agent
        self.coreset_selector.ref_model = model

    def clear_path(self):
        """Clear temporary files."""
        if os.path.exists(self.local_path):
            import shutil
            shutil.rmtree(self.local_path)
            os.makedirs(self.local_path, exist_ok=True)


class UniformBuffer:
    """Uniform random buffer for comparison."""
    
    def __init__(self, 
                 local_path: str, 
                 transforms: Optional[torch.nn.Module],
                 buffer_size: int, 
                 use_cuda: bool = True, 
                 task_dic: Optional[Dict] = None, 
                 seed: int = 42):
        """Initialize Uniform Buffer."""
        
        self.local_path = local_path
        if not os.path.exists(self.local_path):
            os.makedirs(self.local_path)
        
        self.transforms = transforms
        self.buffer_size = buffer_size
        self.use_cuda = use_cuda
        self.task_dic = task_dic
        self.seed = seed
        
        self.data = []
        self.id2task = {}
        self.to_tensor = torchvision.transforms.ToTensor()
        self.to_pil = torchvision.transforms.ToPILImage()
        self.id_bias = 0

    def update_buffer(self, task_cnts: List[int], task_id: int, cur_x: np.ndarray, cur_y: np.ndarray):
        """Update buffer with uniform random selection."""
        # Distribute buffer size to each task
        task_sizes = []
        for i in range(task_id + 1):
            task_sizes.append(int(task_cnts[i] / sum(task_cnts) * self.buffer_size))
        
        rest_size = self.buffer_size - sum(task_sizes)
        for j in range(rest_size):
            task_sizes[j] += 1
        
        new_id2task = {}
        new_data = []
        for i in range(task_id + 1):
            new_data.append([])
        
        # Resize previous task data
        for i in range(task_id):
            new_data = random.sample(self.data[i], task_sizes[i])
            self.data[i] = copy.deepcopy(new_data)
        
        # Select current task data
        selected_ids = random.sample(list(range(cur_x.shape[0])), task_sizes[task_id])
        to_pil = torchvision.transforms.ToPILImage()
        cur_data = []
        
        for i, idx in enumerate(selected_ids):
            sp = cur_x[idx]
            lab = cur_y[idx]
            d_id = i + self.id_bias
            
            if self.transforms is not None:
                aug_sp = self.transforms(sp)
            else:
                aug_sp = sp
            
            cur_data.append((d_id, aug_sp, lab))
            new_id2task[d_id] = task_id
        
        self.data.append(cur_data)
        self.id2task.update(new_id2task)
        self.id_bias += cur_x.shape[0]

    def get_data(self):
        """Get data from buffer."""
        for i in range(len(self.data)):
            sps = []
            labs = []
            for di in self.data[i]:
                sp, lab = di
                if self.transforms is not None:
                    aug_sp = self.transforms(sp)
                else:
                    aug_sp = sp
                sps.append(aug_sp)
                labs.append(lab)
            out_data = [torch.stack(sps, dim=0), torch.tensor(labs, dtype=torch.long)]
            yield out_data

    def get_sub_data(self, size: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get subset of data from buffer."""
        all_data = []
        for i in range(len(self.data)):
            all_data = all_data + self.data[i]
        
        if len(all_data) == 0:
            return torch.empty(0), torch.empty(0, dtype=torch.long)
        
        inds = random.sample(list(range(len(all_data))), min(size, len(all_data)))
        selected_sps = []
        selected_labs = []
        
        for ind in inds:
            di = all_data[ind]
            sp, lab = di
            if self.transforms is not None:
                aug_sp = self.transforms(sp)
            else:
                aug_sp = sp
            selected_sps.append(aug_sp)
            selected_labs.append(lab)
        
        return torch.stack(selected_sps, dim=0), torch.tensor(selected_labs, dtype=torch.long)

    def shuffle_data(self):
        """Shuffle data in buffer."""
        random.shuffle(self.data)

    def is_empty(self) -> bool:
        """Check if buffer is empty."""
        if len(self.data) == 0:
            return True
        
        # Check if all tasks are empty
        for task_data in self.data:
            if len(task_data) > 0:
                return False
        
        return True

    def dump_data(self, task_id: int):
        """Dump buffer data to file."""
        dump_file = os.path.join(self.local_path, 'buffer_data' + str(task_id) + '.pkl')
        with open(dump_file, 'wb') as fw:
            pickle.dump(self.data, fw)

    def load_data(self, task_id: int):
        """Load buffer data from file."""
        dump_file = os.path.join(self.local_path, 'buffer_data' + str(task_id) + '.pkl')
        if os.path.exists(dump_file):
            with open(dump_file, 'rb') as fr:
                self.data = pickle.load(fr)

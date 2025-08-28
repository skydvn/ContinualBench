import torch
from argparse import ArgumentParser
from models.utils.continual_model import ContinualModel
from utils.buffer import Buffer
from utils.args import add_rehearsal_args
from models.bcsr_utils.bcsr_coreset import BCSR_Coreset
from models.bcsr_utils.selection import classwise_fair_selection
import copy
import numpy as np
from tqdm import tqdm


class BCSRWrapper(torch.nn.Module):
    """Wrapper to make the backbone compatible with BCSR's expected interface."""
    
    def __init__(self, backbone):
        super().__init__()
        self.backbone = backbone
        # Ensure all parameters require gradients
        for p in self.backbone.parameters():
            p.requires_grad = True
    
    def forward(self, x, task_id=None):
        # Forward without detaching or no_grad
        out = self.backbone(x)
        # Ensure output requires grad (in case some layers break grad)
        if not out.requires_grad:
            out.requires_grad_(True)
        return out
    
    def parameters(self):
        return self.backbone.parameters()
    
    def state_dict(self):
        return self.backbone.state_dict()
    
    def load_state_dict(self, state_dict):
        return self.backbone.load_state_dict(state_dict)
    
    def named_parameters(self):
        return self.backbone.named_parameters()
    
    def modules(self):
        return self.backbone.modules()

class Bcsr(ContinualModel):
    NAME = 'bcsr'
    COMPATIBILITY = ['class-il', 'task-il', 'domain-il']

    @staticmethod
    def get_parser(parser: ArgumentParser) -> ArgumentParser:
        # standard rehearsal args (e.g., --buffer_size, --minibatch_size)
        add_rehearsal_args(parser)
        # BCSR-specific args
        parser.add_argument('--bcsr_lr_proxy', type=float, default=0.01,
                            help='LR for proxy model in BCSR')
        parser.add_argument('--bcsr_beta', type=float, default=1.0,
                            help='Regularization trade-off beta')
        parser.add_argument('--bcsr_outer_it', type=int, default=50,
                            help='Outer iterations (weight updates)')
        parser.add_argument('--bcsr_inner_it', type=int, default=1,
                            help='Inner iterations (proxy steps)')
        parser.add_argument('--bcsr_weight_lr', type=float, default=1e-1,
                            help='Step size for sample weights')
        parser.add_argument('--bcsr_candidate_bs', type=int, default=600,
                            help='Number of candidates per selection round')
        return parser

    def __init__(self, backbone, loss, args, transform, dataset=None):
        super().__init__(backbone, loss, args, transform, dataset=dataset)
        self.buffer = Buffer(self.args.buffer_size, device=self.device)
        
        # Create a wrapper for the backbone to handle BCSR's interface
        self.bcsr_backbone = BCSRWrapper(copy.deepcopy(self.net))
        
        
        # Initialize BCSR coreset selector with wrapped backbone as proxy
        self.coreset_selector = BCSR_Coreset(
            proxy_model=self.bcsr_backbone,
            lr_proxy_model=args.bcsr_lr_proxy,
            beta=args.bcsr_beta,
            out_dim=self.N_CLASSES,                 # logits dimension
            max_outer_it=args.bcsr_outer_it,
            max_inner_it=args.bcsr_inner_it,
            weight_lr=args.bcsr_weight_lr,
            candidate_batch_size=args.bcsr_candidate_bs,
            device=str(self.device)
        )

    def observe(self, inputs, labels, not_aug_inputs, epoch=None):
        self.opt.zero_grad()
        # standard replay
        if not self.buffer.is_empty():
            buf_inputs, buf_labels = self.buffer.get_data(
                self.args.minibatch_size, transform=self.transform, device=self.device
            )
            inputs = torch.cat([inputs, buf_inputs], dim=0)
            labels = torch.cat([labels, buf_labels], dim=0)

        outputs = self.net(inputs)
        loss = self.loss(outputs, labels)
        loss.backward()
        self.opt.step()
        return loss.item()

    def end_task(self, dataset):
        # 1) Reduce existing buffer fairly across seen classes
        if self.current_task > 0:
            examples_per_class = self.args.buffer_size // (self.cpt * self.current_task)
            buf_x, buf_y = self.buffer.get_all_data()
            self.buffer.empty()
            for cid in buf_y.unique():
                idx = (buf_y == cid)
                keep = min(idx.sum().item(), examples_per_class)
                if keep > 0:
                    self.buffer.add_data(examples=buf_x[idx][:keep], labels=buf_y[idx][:keep])

        # 2) Determine remaining buffer slots
        remaining = self.buffer.buffer_size - len(self.buffer)
        if remaining <= 0:
            return

        # 3) Update BCSR backbone with current model weights
        self.bcsr_backbone.load_state_dict(self.net.state_dict())
        self.bcsr_backbone.to(self.device)
    
        # 4) Collect candidates batch-wise
        batch_candidates_x = []
        batch_candidates_y = []
    
        for batch in tqdm(dataset.train_loader, desc="Processing batches"):
            _, y, not_aug = batch[0], batch[1], batch[2]
            data = not_aug.to(self.device)
            labels = y.to(self.device)

            # Run BCSR coreset selection for this batch
            try:
                pick, _ = self.coreset_selector.coreset_select(
                    self.bcsr_backbone,
                    data.cpu().numpy(),
                    labels.cpu().numpy(),
                    task_id=self.current_task + 1,
                    topk=len(data)
                )
                pick = pick.to(torch.int)
            except Exception as e:
                print(f"Batch BCSR failed: {e}, fallback to random selection")
                pick = np.random.permutation(len(data))

            batch_candidates_x.append(data[pick].cpu())
            batch_candidates_y.append(labels[pick].cpu())
    
        # 5) Concatenate selected candidates across batches
        X_sel = torch.cat(batch_candidates_x, dim=0)
        Y_sel = torch.cat(batch_candidates_y, dim=0)

        # 6) Class-fair selection across accumulated candidates
        num_per_label = [
            len((Y_sel == (jj + self.cpt * self.current_task)).nonzero()) 
            for jj in range(self.cpt)
        ]

        selected = classwise_fair_selection(
            task=self.current_task + 1,
            cand_target=Y_sel.numpy(),
            sorted_index=np.arange(len(Y_sel)),  # already selected per batch
            num_per_label=num_per_label,
            n_classes=self.cpt,
            memory_size=remaining,
            is_shuffle=True
        )

        # 7) Add to buffer
        sel_x = X_sel[selected].to(self.device)
        sel_y = Y_sel[selected].to(self.device)
        if sel_x.shape[0] > 0:
            self.buffer.add_data(examples=sel_x, labels=sel_y)




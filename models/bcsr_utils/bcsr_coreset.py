import numpy as np
import torch
from .bcsr_training import Training


class BCSR_Coreset:
    """"
    Coreset selection based on bilevel optimzation

    Args:
        proxy_model: model for coreset selection
        lr_proxy_model: learning rare for proxy_model
        beta: balance the loss and regularizer
        out_dim: input dimension
        max_outer_it: outer loops for bilevel optimization
        max_inner_it: inner loops for bilevel optimization
        weight_lr: step size for updating sample weights
        candidate_batch_size: number of coreset candidates
    """
    def __init__(self, proxy_model, lr_proxy_model,  beta, out_dim=10, max_outer_it=50, max_inner_it=1,
                 weight_lr=1e-1, candidate_batch_size=600, logging_period=1000, device='cuda'):
        self.out_dim = out_dim
        self.device = device
        self.max_outer_it = max_outer_it
        self.max_inner_it = max_inner_it
        self.weight_lr = weight_lr
        self.candidate_batch_size = candidate_batch_size
        self.logging_period = logging_period
        self.nystrom_batch = None
        self.nystrom_normalization = None
        self.param_size=  []
        self.seed = 0
        self.lr_proxy_model = lr_proxy_model
        self.training_model_op = Training(proxy_model, beta, device, lr_proxy_model, lr_weights=self.weight_lr)
        for p in self.training_model_op.proxy_model.parameters():
            self.param_size.append(p.size())


    def outer_loss(self, X, y, task_id, topk, ref_x=None, ref_y=None):
        if isinstance(y, np.ndarray):
            y = torch.from_numpy(y).float()
        if isinstance(X, np.ndarray):
            X = torch.from_numpy(X).float()
        n = X.shape[0]
        coreset_weights = 1.0 / n * torch.ones([n], dtype=torch.float, requires_grad=True)

        _, _, outer_loss = self.training_model_op.train_outer(X, y, task_id, coreset_weights, topk, ref_x,
                                                                            ref_y)
        return outer_loss

    def projection_onto_simplex(self, v, b=1):
        v = v.cpu().detach().numpy()
        n_features = v.shape[0]
        u = np.sort(v)[::-1]
        cssv = np.cumsum(u) - b
        ind = np.arange(n_features) + 1
        cond = u - cssv / ind > 0
        rho = ind[cond][-1]
        theta = cssv[cond][-1] / float(rho)
        w = np.maximum(v - theta, 0)
        w = torch.from_numpy(w).cuda()
        w.requires_grad = True
        return w

    def coreset_select(self, model, X, y, task_id,  topk, out_loss=None, ref_x=None, ref_y=None):
        np.random.seed(self.seed)
        if isinstance(y, np.ndarray):
            y = torch.from_numpy(y).float()
        if isinstance(X, np.ndarray):
            X = torch.from_numpy(X).float()
        n = X.shape[0]
      
        self.training_model_op.proxy_model.load_state_dict(model.state_dict())
        # initialize sample weights
        coreset_weights = 1.0/n*torch.ones([n], dtype=torch.float, requires_grad=True)
       
        coreset_weights = self.projection_onto_simplex(coreset_weights)

        self.training_model_op.lr_p = self.lr_proxy_model
        self.training_model_op.lr_w = self.weight_lr


        try:
            #print(f"Starting bilevel optimization with {self.max_outer_it} outer iterations...")
            for i in range(self.max_outer_it):
                #print(f"Outer iteration {i+1}/{self.max_outer_it}")
                inner_loss = self.training_model_op.train_inner(X, y, task_id, coreset_weights, self.max_inner_it)
                coreset_weights, _, outer_loss = self.training_model_op.train_outer(X, y, task_id, coreset_weights,
                                                                                    topk, ref_x, ref_y)
                coreset_weights = self.projection_onto_simplex(coreset_weights)
                total_loss = torch.mean(outer_loss).item()
                if i%10 == 0:
                    print(f'Inner loss: {inner_loss:.3f}, outer loss: {total_loss:.3f}')
            
            if out_loss is not None and n == 50:
                out_loss.append(total_loss)

            # Ensure coreset_weights is on the right device and has proper shape
            if coreset_weights.device != torch.device(self.device):
                coreset_weights = coreset_weights.to(self.device)
            
            # Use multinomial sampling for selection
            result = torch.multinomial(coreset_weights, topk, replacement=False)
            return result, out_loss
            
        except Exception as e:
            print(f"BCSR coreset selection failed: {e}")
            import traceback
            traceback.print_exc()
            print("Falling back to random selection")
            # Fallback: return random indices
            indices = torch.randperm(n)[:min(topk, n)]
            return indices, out_loss


        #return torch.multinomial(coreset_weights, topk, replacement=False), out_loss


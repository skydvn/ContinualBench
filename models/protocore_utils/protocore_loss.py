import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import SGD, Adam
from argparse import ArgumentParser
from models.protocore_utils.proto_visualize import Visualizer


class PrototypicalLoss(nn.Module):
    def __init__(self):
        super(PrototypicalLoss, self).__init__()
        pass

    def forward(self, support_embeddings, support_labels, query_embeddings, query_labels, n_way, n_support, n_query):
        """
        Computes the prototypical loss for few-shot learning.

        Args:
            support_embeddings: Tensor of shape (n_way * n_support, embedding_dim)
            support_labels: Tensor of shape (n_way * n_support,)
            query_embeddings: Tensor of shape (n_way * n_query, embedding_dim)
            query_labels: Tensor of shape (n_way * n_query,)
            n_way: Number of classes in the episode
            n_support: Number of support examples per class
            n_query: Number of query examples per class

        Returns:
            loss: Scalar tensor representing the prototypical loss
        """
        # Compute prototypes by averaging support embeddings for each class
        prototypes = self._compute_prototypes(support_embeddings, support_labels, n_way, n_support)

        # Compute distances from query embeddings to prototypes
        distances = self._compute_distances(query_embeddings, prototypes)

        # Convert distances to log probabilities (negative distances with softmax)
        log_probs = F.log_softmax(-distances, dim=1)

        # Compute cross-entropy loss
        loss = F.nll_loss(log_probs, query_labels)

        # Compute accuracy
        predictions = torch.argmax(-distances, dim=1)
        accuracy = (predictions == query_labels).float().mean()

        return loss, accuracy

    def _compute_prototypes(self, support_embeddings, support_labels, n_way, n_support):
        """
        Compute class prototypes by averaging support embeddings.

        Args:
            support_embeddings: Tensor of shape (n_way * n_support, embedding_dim)
            support_labels: Tensor of shape (n_way * n_support,)
            n_way: Number of classes
            n_support: Number of support examples per class

        Returns:
            prototypes: Tensor of shape (n_way, embedding_dim)
        """
        embedding_dim = support_embeddings.size(1)
        prototypes = torch.zeros(n_way, embedding_dim, device=support_embeddings.device)

        for i in range(n_way):
            class_mask = (support_labels == i)
            class_embeddings = support_embeddings[class_mask]
            prototypes[i] = class_embeddings.mean(dim=0)

        return prototypes

    def _compute_distances(self, query_embeddings, prototypes):
        """
        Compute Euclidean distances between query embeddings and prototypes.

        Args:
            query_embeddings: Tensor of shape (n_query, embedding_dim)
            prototypes: Tensor of shape (n_way, embedding_dim)

        Returns:
            distances: Tensor of shape (n_query, n_way)
        """
        n_query = query_embeddings.size(0)
        n_way = prototypes.size(0)

        # Expand dimensions for broadcasting
        query_expanded = query_embeddings.unsqueeze(1).expand(n_query, n_way, -1)
        prototype_expanded = prototypes.unsqueeze(0).expand(n_query, n_way, -1)

        # Compute squared Euclidean distances
        distances = torch.pow(query_expanded - prototype_expanded, 2).sum(dim=2)

        return distances


def test_prototypical_loss():
    pass


def prototypical_loss(input, target, n_query):
    """
    Compute prototypical loss and accuracy using fixed n_query per class.

    Args:
        input (Tensor): Model output of shape [N, D]
        target (Tensor): Ground truth labels of shape [N]
        n_query (int): Number of query samples per class

    Returns:
        Tuple[Tensor, Tensor]: loss and accuracy
    """
    classes = torch.unique(target)
    n_classes = len(classes)

    support_idxs = []
    query_idxs = []

    for cls in classes:
        cls_idxs = (target == cls).nonzero(as_tuple=True)[0]
        if len(cls_idxs) < n_query + 1:
            raise ValueError(f"Not enough samples for class {cls.item()}: need > {n_query}, got {len(cls_idxs)}")
        query_idxs.append(cls_idxs[-n_query:])
        support_idxs.append(cls_idxs[:-n_query])

    support_idxs = torch.cat(support_idxs)
    query_idxs = torch.cat(query_idxs)

    support = input[support_idxs]
    query = input[query_idxs]

    prototypes = []
    for cls in classes:
        cls_support = support[(target[support_idxs] == cls)]
        prototypes.append(cls_support.mean(0))
    prototypes = torch.stack(prototypes)

    dists = euclidean_dist(query, prototypes)
    log_p_y = F.log_softmax(-dists, dim=1)

    target_inds = torch.arange(n_classes, device=target.device).repeat_interleave(n_query)
    loss = -log_p_y[range(len(query_idxs)), target_inds].mean()

    pred = log_p_y.argmax(dim=1)
    acc = (pred == target_inds).float().mean()

    return loss, acc


"""
    Supplementary Functions:
"""
def euclidean_dist(x, y):
    '''
    Compute euclidean distance between two tensors
    '''
    # x: N x D
    # y: M x D

    n = x.size(0)
    m = y.size(0)
    d = x.size(1)
    if d != y.size(1):
        raise Exception

    x = x.unsqueeze(1).expand(n, m, d)
    y = y.unsqueeze(0).expand(n, m, d)

    return torch.pow(x - y, 2).sum(2)


def hyperbolic_distance(u, v, eps=1e-5):
    """
    Compute hyperbolic distance between points u and v in Poincaré ball.

    Args:
        u, v (Tensor): [N, D] embeddings inside the unit ball
        eps (float): numerical stability
    Returns:
        Tensor: [N, M] pairwise hyperbolic distances
    """
    u_norm = torch.clamp(torch.norm(u, dim=-1, keepdim=True), max=1 - eps)
    v_norm = torch.clamp(torch.norm(v, dim=-1, keepdim=True), max=1 - eps)

    diff = torch.norm(u.unsqueeze(1) - v.unsqueeze(0), dim=-1) ** 2 + 1e-10

    print(f"diff: {diff}")

    denominator = (1 - u_norm ** 2) * (1 - v_norm.transpose(0, 1) ** 2)
    dist = torch.acosh(1 + 2 * diff / denominator.clamp_min(eps))
    return dist


import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class HyperbolicOps:
    """Hyperbolic operations in the Poincaré ball model"""

    @staticmethod
    def project(x, c=1.0, eps=1e-5):
        """Project points to the Poincaré ball"""
        c = torch.as_tensor(c, device=x.device, dtype=x.dtype)
        norm = torch.clamp(torch.norm(x, dim=-1, keepdim=True), min=eps)
        max_norm = (1 - eps) / torch.sqrt(c)
        cond = norm > max_norm
        projected = x / norm * max_norm
        return torch.where(cond, projected, x)

    @staticmethod
    def exp_map(v, x, c=1.0):
        """Exponential map at point x"""
        c = torch.as_tensor(c, device=x.device, dtype=x.dtype)
        sqrt_c = torch.sqrt(c)
        v_norm = torch.clamp(torch.norm(v, dim=-1, keepdim=True), min=1e-10)
        lambda_x = 2 / (1 - c * torch.sum(x * x, dim=-1, keepdim=True))

        result = x + (2 / (sqrt_c * lambda_x)) * torch.tanh(sqrt_c * lambda_x * v_norm / 2) * v / v_norm
        return HyperbolicOps.project(result, c)

    @staticmethod
    def log_map(y, x, c=1.0):
        """Logarithmic map at point x"""
        c = torch.as_tensor(c, device=x.device, dtype=x.dtype)
        sqrt_c = torch.sqrt(c)
        diff = y - x
        lambda_x = 2 / (1 - c * torch.sum(x * x, dim=-1, keepdim=True))

        diff_norm = torch.clamp(torch.norm(diff, dim=-1, keepdim=True), min=1e-10)
        result = (2 / (sqrt_c * lambda_x)) * torch.atanh(sqrt_c * diff_norm) * diff / diff_norm
        return result

    @staticmethod
    def distance(x, y, c=1.0):
        """Hyperbolic distance in Poincaré ball"""
        c = torch.as_tensor(c, device=x.device, dtype=x.dtype)
        sqrt_c = torch.sqrt(c)
        diff = x - y
        diff_norm_sq = torch.sum(diff * diff, dim=-1)

        x_norm_sq = torch.sum(x * x, dim=-1)
        y_norm_sq = torch.sum(y * y, dim=-1)

        num = 2 * diff_norm_sq
        denom = (1 - c * x_norm_sq) * (1 - c * y_norm_sq)

        return (2 / sqrt_c) * torch.atanh(sqrt_c * torch.sqrt(torch.clamp(num / denom, max=1.0 - 1e-5)))


class HyperbolicContrastiveLoss(nn.Module):
    """Hyperbolic Contrastive Learning Loss"""

    def __init__(self, temperature=0.1, curvature=1.0):
        super().__init__()
        self.temperature = temperature
        self.curvature = curvature
        self.hyperbolic_ops = HyperbolicOps()

    def forward(self, embeddings, labels):
        """
        Args:
            embeddings: Tensor of shape [N, D] in Poincaré ball
            labels: Tensor of shape [N] with class labels
        """
        N = embeddings.size(0)

        # Ensure embeddings are in Poincaré ball
        embeddings = self.hyperbolic_ops.project(embeddings, self.curvature)

        # Compute pairwise hyperbolic distances
        distances = torch.zeros(N, N, device=embeddings.device)
        for i in range(N):
            for j in range(N):
                if i != j:
                    distances[i, j] = self.hyperbolic_ops.distance(
                        embeddings[i:i + 1], embeddings[j:j + 1], self.curvature
                    )

        # Convert distances to similarities (negative distances)
        similarities = -distances / self.temperature

        # Create masks
        labels = labels.contiguous().view(-1, 1)
        mask = torch.eq(labels, labels.T).float().to(embeddings.device)

        # Remove diagonal (self-similarity)
        mask = mask * (1 - torch.eye(N, device=embeddings.device))

        # Compute InfoNCE loss in hyperbolic space
        exp_sim = torch.exp(similarities)

        # Denominator: sum over all negative samples
        denominator = torch.sum(exp_sim, dim=1, keepdim=True) - torch.diag(exp_sim).unsqueeze(1)

        # Numerator: sum over positive samples
        numerator = torch.sum(mask * exp_sim, dim=1, keepdim=True)

        # Avoid log(0)
        numerator = torch.clamp(numerator, min=1e-8)
        denominator = torch.clamp(denominator, min=1e-8)

        # InfoNCE loss
        loss = -torch.log(numerator / (numerator + denominator))

        # Only consider samples that have positive pairs
        pos_counts = torch.sum(mask, dim=1)
        valid_samples = pos_counts > 0

        if valid_samples.sum() > 0:
            loss = loss[valid_samples].mean()
        else:
            loss = torch.tensor(0.0, device=embeddings.device, requires_grad=True)

        return loss


class HyperbolicEncoder(nn.Module):
    """Encoder that maps to hyperbolic space"""

    def __init__(self, input_dim, hidden_dim, output_dim, curvature=1.0):
        super().__init__()
        self.curvature = curvature
        self.hyperbolic_ops = HyperbolicOps()

        # Euclidean layers
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

        # Hyperbolic layer
        self.to_hyperbolic = nn.Linear(output_dim, output_dim)

    def forward(self, x):
        # Euclidean encoding
        h = self.encoder(x)

        # Map to hyperbolic space
        # Use small initialization to stay near origin
        h_hyp = torch.tanh(self.to_hyperbolic(h)) * 0.1

        # Project to Poincaré ball
        return self.hyperbolic_ops.project(h_hyp, self.curvature)


class HierarchicalHyperbolicCL(nn.Module):
    """Hierarchical Hyperbolic Contrastive Learning"""

    def __init__(self, input_dim, embed_dim, num_levels=3, curvature=1.0, temperature=0.1):
        super().__init__()
        self.num_levels = num_levels
        self.curvature = curvature
        self.temperature = temperature

        # Multi-level encoders
        self.encoders = nn.ModuleList([
            HyperbolicEncoder(input_dim, embed_dim, embed_dim, curvature)
            for _ in range(num_levels)
        ])

        # Hyperbolic contrastive losses
        self.cl_losses = nn.ModuleList([
            HyperbolicContrastiveLoss(temperature, curvature)
            for _ in range(num_levels)
        ])

        # Level-specific curvatures (increasing for finer levels)
        self.level_curvatures = [curvature * (i + 1) for i in range(num_levels)]

    def forward(self, x, labels, hierarchy_labels=None):
        """
        Args:
            x: Input features
            labels: Ground truth labels
            hierarchy_labels: List of labels for each hierarchical level
        """
        total_loss = 0
        embeddings_by_level = []

        if hierarchy_labels is None:
            hierarchy_labels = [labels] * self.num_levels

        for level in range(self.num_levels):
            # Encode at current level
            embeddings = self.encoders[level](x)
            embeddings_by_level.append(embeddings)

            # Compute contrastive loss at current level
            level_labels = hierarchy_labels[min(level, len(hierarchy_labels) - 1)]
            loss = self.cl_losses[level](embeddings, level_labels)
            total_loss += loss

        return total_loss, embeddings_by_level


# Example usage and training loop
def train_hyperbolic_cl(model, dataloader, optimizer, device, epochs=100):
    """Training loop for hyperbolic contrastive learning"""
    model.train()

    for epoch in range(epochs):
        total_loss = 0

        for batch_idx, (data, labels) in enumerate(dataloader):
            data, labels = data.to(device), labels.to(device)

            optimizer.zero_grad()

            if isinstance(model, HierarchicalHyperbolicCL):
                loss, embeddings = model(data, labels)
            else:
                embeddings = model(data)
                loss = HyperbolicContrastiveLoss()(embeddings, labels)

            loss.backward()

            # Gradient clipping for stability in hyperbolic space
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()
            total_loss += loss.item()

            if batch_idx % 100 == 0:
                print(f'Epoch {epoch}, Batch {batch_idx}, Loss: {loss.item():.4f}')

        print(f'Epoch {epoch}, Average Loss: {total_loss / len(dataloader):.4f}')


# Utility functions for hyperbolic visualization
def hyperbolic_to_euclidean_viz(embeddings, curvature=1.0):
    """Convert hyperbolic embeddings to Euclidean for visualization"""
    # Project to unit disk for 2D visualization
    if embeddings.size(-1) > 2:
        # Use PCA or take first 2 dimensions
        embeddings = embeddings[..., :2]

    # Ensure they're in the Poincaré disk
    norms = torch.norm(embeddings, dim=-1, keepdim=True)
    max_norm = 1.0 - 1e-5
    embeddings = torch.where(norms > max_norm, embeddings / norms * max_norm, embeddings)

    return embeddings.detach().cpu().numpy()


# Example of creating hierarchical labels
def create_hierarchy_labels(labels, num_levels=3):
    """Create hierarchical labels from flat labels"""
    hierarchy = []
    unique_labels = torch.unique(labels)

    for level in range(num_levels):
        if level == 0:
            # Coarsest level: group labels
            level_labels = labels // (len(unique_labels) // 2)
        elif level == num_levels - 1:
            # Finest level: original labels
            level_labels = labels
        else:
            # Intermediate levels
            group_size = len(unique_labels) // (2 ** level)
            level_labels = labels // max(1, group_size)

        hierarchy.append(level_labels)

    return hierarchy


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import numpy as np
    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE
    import seaborn as sns
    from matplotlib.patches import Circle
    import warnings

    test_prototypical_loss()
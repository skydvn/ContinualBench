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


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import numpy as np
    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE
    import seaborn as sns
    from matplotlib.patches import Circle
    import warnings

    test_prototypical_loss()
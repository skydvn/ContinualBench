import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import SGD, Adam
from argparse import ArgumentParser
from proto_visualize import PrototypicalVisualizer


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
    """Test function to demonstrate usage."""
    # Parameters
    n_way = 5
    n_support = 5
    n_query = 10
    embedding_dim = 128

    # Create sample data
    support_embeddings = torch.randn(n_way * n_support, embedding_dim)
    support_labels = torch.repeat_interleave(torch.arange(n_way), n_support)
    query_embeddings = torch.randn(n_way * n_query, embedding_dim)
    query_labels = torch.repeat_interleave(torch.arange(n_way), n_query)

    # Test standard implementation
    # Initialize the visualizer
    visualizer = PrototypicalVisualizer(figsize=(16, 12))

    # Run prototypical loss
    criterion = PrototypicalLoss()
    loss, accuracy = criterion(support_embeddings, support_labels,
                               query_embeddings, query_labels,
                               n_way=5, n_support=5, n_query=10)
    print(f"Standard - Loss: {loss:.4f}, Accuracy: {accuracy:.4f}")

    # Get prototypes and predictions
    prototypes = criterion._compute_prototypes(support_embeddings, support_labels, 5, 5)
    distances = criterion._compute_distances(query_embeddings, prototypes)
    predictions = torch.argmin(distances, dim=1)

    # Generate comprehensive visualization
    fig = visualizer.visualize_episode(
        support_embeddings, support_labels,
        query_embeddings, query_labels,
        prototypes, predictions,
        method='pca',  # or 'tsne'
        title="5-way 5-shot Classification Episode"
    )
    plt.savefig("plot.png")  # saves to file
    plt.close()


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import numpy as np
    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE
    import seaborn as sns
    from matplotlib.patches import Circle
    import warnings

    test_prototypical_loss()
import torch
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from matplotlib.patches import Circle

class PrototypicalVisualizer:
    def __init__(self, figsize=(15, 10)):
        self.figsize = figsize
        self.colors = plt.cm.tab10(np.linspace(0, 1, 10))

    def visualize_episode(self, support_embeddings, support_labels, query_embeddings,
                          query_labels, prototypes, predictions=None, method='pca',
                          title="Prototypical Network Visualization"):
        """
        Visualize a few-shot learning episode with support, query, and prototypes.

        Args:
            support_embeddings: Support set embeddings
            support_labels: Support set labels
            query_embeddings: Query set embeddings
            query_labels: Query set labels
            prototypes: Class prototypes
            predictions: Query predictions (optional)
            method: Dimensionality reduction method ('pca' or 'tsne')
            title: Plot title
        """
        # Combine all embeddings for consistent dimensionality reduction
        all_embeddings = torch.cat([support_embeddings, query_embeddings, prototypes], dim=0)
        all_embeddings_np = all_embeddings.detach().cpu().numpy()

        # Apply dimensionality reduction
        if method == 'pca':
            reducer = PCA(n_components=2, random_state=42)
            reduced = reducer.fit_transform(all_embeddings_np)
            explained_var = reducer.explained_variance_ratio_.sum()
        else:  # tsne
            reducer = TSNE(n_components=2, random_state=42, perplexity=min(30, len(all_embeddings_np) // 4))
            reduced = reducer.fit_transform(all_embeddings_np)
            explained_var = None

        # Split back into components
        n_support = len(support_embeddings)
        n_query = len(query_embeddings)
        n_prototypes = len(prototypes)

        support_2d = reduced[:n_support]
        query_2d = reduced[n_support:n_support + n_query]
        prototypes_2d = reduced[n_support + n_query:]

        # Create the plot
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=self.figsize)
        fig.suptitle(title, fontsize=16, fontweight='bold')

        # Plot 1: Overview with all points
        self._plot_overview(ax1, support_2d, support_labels, query_2d, query_labels,
                            prototypes_2d, method, explained_var)

        # Plot 2: Classification results
        if predictions is not None:
            self._plot_predictions(ax2, query_2d, query_labels, predictions)
        else:
            ax2.text(0.5, 0.5, 'No predictions provided', ha='center', va='center',
                     transform=ax2.transAxes, fontsize=12)
            ax2.set_title('Predictions')

        # Plot 3: Distance heatmap
        self._plot_distance_heatmap(ax3, query_embeddings, prototypes)

        # Plot 4: Class statistics
        self._plot_class_stats(ax4, support_labels, query_labels, predictions)

        plt.tight_layout()
        return fig

    def _plot_overview(self, ax, support_2d, support_labels, query_2d, query_labels,
                       prototypes_2d, method, explained_var):
        """Plot overview with support, query, and prototypes."""
        n_classes = len(torch.unique(support_labels))

        # Plot support points
        for i in range(n_classes):
            mask = support_labels.cpu().numpy() == i
            ax.scatter(support_2d[mask, 0], support_2d[mask, 1],
                       c=[self.colors[i]], s=60, alpha=0.7, marker='o',
                       label=f'Support Class {i}', edgecolors='black', linewidth=0.5)

        # Plot query points
        for i in range(n_classes):
            mask = query_labels.cpu().numpy() == i
            ax.scatter(query_2d[mask, 0], query_2d[mask, 1],
                       c=[self.colors[i]], s=80, alpha=0.9, marker='^',
                       edgecolors='black', linewidth=1)

        # Plot prototypes
        for i in range(n_classes):
            ax.scatter(prototypes_2d[i, 0], prototypes_2d[i, 1],
                       c=[self.colors[i]], s=200, marker='*',
                       edgecolors='black', linewidth=2, alpha=1.0)

            # Add circles around prototypes
            circle = Circle((prototypes_2d[i, 0], prototypes_2d[i, 1]),
                            radius=0.3, fill=False, color=self.colors[i],
                            linewidth=2, linestyle='--', alpha=0.5)
            ax.add_patch(circle)

        # Create custom legend
        legend_elements = []
        legend_elements.append(plt.scatter([], [], c='gray', s=60, marker='o',
                                           edgecolors='black', label='Support'))
        legend_elements.append(plt.scatter([], [], c='gray', s=80, marker='^',
                                           edgecolors='black', label='Query'))
        legend_elements.append(plt.scatter([], [], c='gray', s=200, marker='*',
                                           edgecolors='black', label='Prototype'))
        ax.legend(handles=legend_elements, loc='upper right')

        title = f'Episode Overview ({method.upper()})'
        if explained_var is not None:
            title += f' - Variance: {explained_var:.2%}'
        ax.set_title(title)
        ax.grid(True, alpha=0.3)

    def _plot_predictions(self, ax, query_2d, query_labels, predictions):
        """Plot query predictions vs ground truth."""
        correct = (predictions.cpu().numpy() == query_labels.cpu().numpy())

        # Plot correct predictions
        ax.scatter(query_2d[correct, 0], query_2d[correct, 1],
                   c='green', s=100, marker='o', alpha=0.8,
                   label=f'Correct ({correct.sum()})', edgecolors='black')

        # Plot incorrect predictions
        if not correct.all():
            ax.scatter(query_2d[~correct, 0], query_2d[~correct, 1],
                       c='red', s=100, marker='x', alpha=0.8,
                       label=f'Incorrect ({(~correct).sum()})', linewidth=3)

        accuracy = correct.mean()
        ax.set_title(f'Query Predictions (Acc: {accuracy:.2%})')
        ax.legend()
        ax.grid(True, alpha=0.3)

    def _plot_distance_heatmap(self, ax, query_embeddings, prototypes):
        """Plot heatmap of distances between queries and prototypes."""
        distances = torch.cdist(query_embeddings, prototypes, p=2)
        distances_np = distances.detach().cpu().numpy()

        im = ax.imshow(distances_np, cmap='viridis', aspect='auto')
        ax.set_xlabel('Prototype Class')
        ax.set_ylabel('Query Sample')
        ax.set_title('Query-Prototype Distances')

        # Add colorbar
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        # Add text annotations
        if distances_np.shape[0] <= 20 and distances_np.shape[1] <= 10:
            for i in range(distances_np.shape[0]):
                for j in range(distances_np.shape[1]):
                    ax.text(j, i, f'{distances_np[i, j]:.1f}',
                            ha='center', va='center', color='white', fontsize=8)

    def _plot_class_stats(self, ax, support_labels, query_labels, predictions):
        """Plot class-wise statistics."""
        n_classes = len(torch.unique(support_labels))

        if predictions is not None:
            # Calculate per-class accuracy
            accuracies = []
            for i in range(n_classes):
                mask = query_labels.cpu().numpy() == i
                if mask.sum() > 0:
                    acc = (predictions.cpu().numpy()[mask] == i).mean()
                    accuracies.append(acc)
                else:
                    accuracies.append(0)

            bars = ax.bar(range(n_classes), accuracies, color=self.colors[:n_classes], alpha=0.7)
            ax.set_xlabel('Class')
            ax.set_ylabel('Accuracy')
            ax.set_title('Per-Class Accuracy')
            ax.set_ylim(0, 1)
            ax.grid(True, alpha=0.3, axis='y')

            # Add value labels on bars
            for bar, acc in zip(bars, accuracies):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width() / 2., height + 0.01,
                        f'{acc:.2%}', ha='center', va='bottom')
        else:
            # Just show class distribution
            support_counts = [(support_labels == i).sum().item() for i in range(n_classes)]
            query_counts = [(query_labels == i).sum().item() for i in range(n_classes)]

            x = np.arange(n_classes)
            width = 0.35

            ax.bar(x - width / 2, support_counts, width, label='Support',
                   color=self.colors[:n_classes], alpha=0.7)
            ax.bar(x + width / 2, query_counts, width, label='Query',
                   color=self.colors[:n_classes], alpha=0.5)

            ax.set_xlabel('Class')
            ax.set_ylabel('Count')
            ax.set_title('Class Distribution')
            ax.legend()
            ax.grid(True, alpha=0.3, axis='y')

    def plot_training_progress(self, losses, accuracies, title="Training Progress"):
        """Plot training loss and accuracy over episodes."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

        episodes = range(1, len(losses) + 1)

        # Plot loss
        ax1.plot(episodes, losses, 'b-', linewidth=2, alpha=0.7)
        ax1.fill_between(episodes, losses, alpha=0.3)
        ax1.set_xlabel('Episode')
        ax1.set_ylabel('Loss')
        ax1.set_title('Training Loss')
        ax1.grid(True, alpha=0.3)

        # Plot accuracy
        ax2.plot(episodes, accuracies, 'g-', linewidth=2, alpha=0.7)
        ax2.fill_between(episodes, accuracies, alpha=0.3)
        ax2.set_xlabel('Episode')
        ax2.set_ylabel('Accuracy')
        ax2.set_title('Training Accuracy')
        ax2.grid(True, alpha=0.3)
        ax2.set_ylim(0, 1)

        fig.suptitle(title, fontsize=14, fontweight='bold')
        plt.tight_layout()
        return fig
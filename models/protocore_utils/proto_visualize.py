import os

import torch
import torchvision
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import umap
from typing import Tuple, Optional
import warnings
warnings.filterwarnings('ignore')


class Visualizer:
    def __init__(self, save_dir=None):
        self.save_dir = save_dir

    def visualize_episode(self, embeddings, labels, task, epoch,
                          prototypes=None, predictions=None, syn_proto = None,
                          method="tsne", title="Episode Visualization"):
        """
        Visualize embeddings + prototypes in 2D, coloring prototypes by predicted class.

        Args:
            embeddings (Tensor): [N, D] all features
            labels (Tensor): [N] labels
            prototypes (Tensor): [C, D] prototypes per class (optional)
            predictions (Tensor): [C, #classes] logits or probabilities per prototype (optional)
            method (str): "tsne" or "pca"
            title (str): plot title
        """
        # Convert tensors → numpy
        X = embeddings.detach().cpu().numpy()
        y = labels.detach().cpu().numpy()

        P, P_syn, P_pred = None, None, None
        if syn_proto is not None:
            P_syn = syn_proto.detach().cpu().numpy()
        if prototypes is not None:
            P = prototypes.detach().cpu().numpy()
            if predictions is not None:
                P_pred = predictions.argmax(dim=1).cpu().numpy()  # predicted class

        # Dimensionality reduction
        if method.lower() == "tsne":
            reducer = TSNE(n_components=2, init="pca", random_state=42)
        else:
            reducer = PCA(n_components=2)

        Z = reducer.fit_transform(X)

        # Scatter plot for embeddings
        fig, ax = plt.subplots(figsize=(8, 6))
        scatter = ax.scatter(Z[:, 0], Z[:, 1], c=y, cmap="tab20", alpha=0.6, s=10)

        # Prototypes: map them with same reducer
        # print(f"X: {np.shape(X)}")
        # print(f"P: {np.shape(P)}")
        # print(f"P_syn: {np.shape(P_syn)}")
        if P is not None:
            all_concat = np.vstack([X, P])
            all_Z = reducer.fit_transform(all_concat)
            PZ = all_Z[-P.shape[0]:]

            if P_pred is not None:
                ax.scatter(
                    PZ[:, 0], PZ[:, 1],
                    c=P_pred, cmap="tab20",
                    marker="X", s=200, edgecolor="k", linewidth=1.2,
                    label="Prototypes"
                )
            else:
                ax.scatter(
                    PZ[:, 0], PZ[:, 1],
                    c="black", marker="X", s=200, edgecolor="k", linewidth=1.2,
                    label="Prototypes"
                )

        # Synthetic coreset: map them with same reducer
        if P_syn is not None:
            P_syn_all = P_syn.reshape(-1, 512)
            all_concat = np.vstack([X, P_syn_all])
            all_Z = reducer.fit_transform(all_concat)
            PZ = all_Z[-P_syn_all.shape[0]:]

            repeat_count = P_syn.shape[1]  # e.g., if shape is (10, 3, 512) -> 3
            P_pred_expanded = np.repeat(P_pred, repeat_count)

            if P_pred is not None:
                ax.scatter(
                    PZ[:, 0], PZ[:, 1],
                    c=P_pred_expanded, cmap="tab20",
                    marker= '*', s=200, edgecolor="k", linewidth=1.2,
                    label="Prototypes"
                )
            else:
                ax.scatter(
                    PZ[:, 0], PZ[:, 1],
                    c="black", marker= '*', s=200, edgecolor="k", linewidth=1.2,
                    label="Prototypes"
                )

        # Synthetic prototypes: map them with same reducer
        if P_syn is not None:
            P_syn_one =P_syn.mean(axis=1)
            all_concat = np.vstack([X, P_syn_one])
            all_Z = reducer.fit_transform(all_concat)
            PZ = all_Z[-P_syn_one.shape[0]:]

            if P_pred is not None:
                ax.scatter(
                    PZ[:, 0], PZ[:, 1],
                    c=P_pred, cmap="tab20",
                    marker= '^', s=200, edgecolor="k", linewidth=1.2,
                    label="Prototypes"
                )
            else:
                ax.scatter(
                    PZ[:, 0], PZ[:, 1],
                    c="black", marker= '^', s=200, edgecolor="k", linewidth=1.2,
                    label="Prototypes"
                )


        ax.set_title(title)
        ax.legend(*scatter.legend_elements(num=None), title="Classes",
                  bbox_to_anchor=(1.05, 1), loc='upper left')

        plt.tight_layout()

        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir, exist_ok=True)
        # Save if dir is set
        if self.save_dir is not None:
            path = f"{self.save_dir}/{task}{epoch}{title.replace(' ', '_')}.png"
            plt.savefig(path)
            print(f"[Visualizer] Saved figure to {path}")

        return fig


class HyperbolicVisualizer:
    def __init__(self, save_dir=None):
        self.save_dir = save_dir
        if self.save_dir and not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir, exist_ok=True)

    def _euclidean_to_poincare(self, X: np.ndarray) -> np.ndarray:
        """Convert Euclidean coordinates to Poincaré ball coordinates"""
        # Normalize to unit ball first
        X_norm = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-8)

        # Apply stereographic projection to Poincaré ball
        # Scale down to ensure points are inside unit ball
        scale_factor = 0.95
        return X_norm * scale_factor

    def visualize_episode(self, embeddings, labels, task, epoch,
                          prototypes=None, predictions=None, syn_proto=None,
                          method="humap", title="Episode Visualization"):
        """
        Visualize embeddings + prototypes in 2D using hyperbolic UMAP, coloring prototypes by predicted class.

        Args:
            embeddings (Tensor): [N, D] all features
            labels (Tensor): [N] labels
            task: task identifier
            epoch: epoch number
            prototypes (Tensor): [C, D] prototypes per class (optional)
            predictions (Tensor): [C, #classes] logits or probabilities per prototype (optional)
            syn_proto (Tensor): [C, K, D] synthetic prototypes (optional)
            method (str): "humap" (hyperbolic umap), "tsne", or "pca"
            title (str): plot title
        """
        import os

        # Convert tensors → numpy
        X = embeddings.detach().cpu().numpy()
        y = labels.detach().cpu().numpy()

        P, P_syn, P_pred = None, None, None
        if syn_proto is not None:
            P_syn = syn_proto.detach().cpu().numpy()
        if prototypes is not None:
            P = prototypes.detach().cpu().numpy()
            if predictions is not None:
                P_pred = predictions.argmax(dim=1).cpu().numpy()  # predicted class

        # Dimensionality reduction
        if method.lower() == "humap":
            # Apply PCA preprocessing for better results
            pca = PCA(n_components=min(50, X.shape[1]), random_state=42)
            X_reduced = pca.fit_transform(X)

            # Apply hyperbolic UMAP
            hyp_umap = HyperbolicUMAP(n_components=2, n_neighbors=15, min_dist=0.1, random_state=42)
            Z = hyp_umap.fit_transform(X_reduced)
            is_hyperbolic = True

        elif method.lower() == "tsne":
            reducer = TSNE(n_components=2, init="pca", random_state=42)
            Z = reducer.fit_transform(X)
            is_hyperbolic = False

        else:  # PCA
            reducer = PCA(n_components=2, random_state=42)
            Z = reducer.fit_transform(X)
            is_hyperbolic = False

        # Create figure
        fig, ax = plt.subplots(figsize=(10, 8))

        # Scatter plot for embeddings
        scatter = ax.scatter(Z[:, 0], Z[:, 1], c=y, cmap="tab20", alpha=0.6, s=15)

        # Handle prototypes
        if P is not None:
            if method.lower() == "humap":
                # Transform prototypes using same PCA and hyperbolic UMAP
                P_reduced = pca.transform(P)
                all_data = np.vstack([X_reduced, P_reduced])
                hyp_umap_all = HyperbolicUMAP(n_components=2, n_neighbors=15, min_dist=0.1, random_state=42)
                all_Z = hyp_umap_all.fit_transform(all_data)
                PZ = all_Z[-P.shape[0]:]
            else:
                # Use standard approach for other methods
                all_concat = np.vstack([X, P])
                if method.lower() == "tsne":
                    reducer = TSNE(n_components=2, init="pca", random_state=42)
                else:
                    reducer = PCA(n_components=2, random_state=42)
                all_Z = reducer.fit_transform(all_concat)
                PZ = all_Z[-P.shape[0]:]

            if P_pred is not None:
                ax.scatter(
                    PZ[:, 0], PZ[:, 1],
                    c=P_pred, cmap="tab20",
                    marker="X", s=300, edgecolor="k", linewidth=1.5,
                    label="Prototypes", alpha=0.9
                )
            else:
                ax.scatter(
                    PZ[:, 0], PZ[:, 1],
                    c="black", marker="X", s=300, edgecolor="k", linewidth=1.5,
                    label="Prototypes", alpha=0.9
                )

        # Handle synthetic prototypes (individual points)
        if P_syn is not None:
            P_syn_flat = P_syn.reshape(-1, P_syn.shape[-1])  # [C*K, D]

            if method.lower() == "humap":
                # Transform synthetic prototypes
                P_syn_reduced = pca.transform(P_syn_flat)
                all_data = np.vstack([X_reduced, P_syn_reduced])
                hyp_umap_all = HyperbolicUMAP(n_components=2, n_neighbors=15, min_dist=0.1, random_state=42)
                all_Z = hyp_umap_all.fit_transform(all_data)
                PZ_syn = all_Z[-P_syn_flat.shape[0]:]
            else:
                all_concat = np.vstack([X, P_syn_flat])
                if method.lower() == "tsne":
                    reducer = TSNE(n_components=2, init="pca", random_state=42)
                else:
                    reducer = PCA(n_components=2, random_state=42)
                all_Z = reducer.fit_transform(all_concat)
                PZ_syn = all_Z[-P_syn_flat.shape[0]:]

            repeat_count = P_syn.shape[1]  # K
            if P_pred is not None:
                P_pred_expanded = np.repeat(P_pred, repeat_count)
                ax.scatter(
                    PZ_syn[:, 0], PZ_syn[:, 1],
                    c=P_pred_expanded, cmap="tab20",
                    marker='*', s=150, edgecolor="k", linewidth=1.0,
                    label="Synthetic Prototypes", alpha=0.8
                )
            else:
                ax.scatter(
                    PZ_syn[:, 0], PZ_syn[:, 1],
                    c="red", marker='*', s=150, edgecolor="k", linewidth=1.0,
                    label="Synthetic Prototypes", alpha=0.8
                )

            # Handle synthetic prototypes (mean points)
            P_syn_mean = P_syn.mean(axis=1)  # [C, D]

            if method.lower() == "humap":
                P_syn_mean_reduced = pca.transform(P_syn_mean)
                all_data = np.vstack([X_reduced, P_syn_mean_reduced])
                hyp_umap_all = HyperbolicUMAP(n_components=2, n_neighbors=15, min_dist=0.1, random_state=42)
                all_Z = hyp_umap_all.fit_transform(all_data)
                PZ_mean = all_Z[-P_syn_mean.shape[0]:]
            else:
                all_concat = np.vstack([X, P_syn_mean])
                if method.lower() == "tsne":
                    reducer = TSNE(n_components=2, init="pca", random_state=42)
                else:
                    reducer = PCA(n_components=2, random_state=42)
                all_Z = reducer.fit_transform(all_concat)
                PZ_mean = all_Z[-P_syn_mean.shape[0]:]

            if P_pred is not None:
                ax.scatter(
                    PZ_mean[:, 0], PZ_mean[:, 1],
                    c=P_pred, cmap="tab20",
                    marker='^', s=250, edgecolor="k", linewidth=1.5,
                    label="Mean Synthetic Prototypes", alpha=0.9
                )
            else:
                ax.scatter(
                    PZ_mean[:, 0], PZ_mean[:, 1],
                    c="blue", marker='^', s=250, edgecolor="k", linewidth=1.5,
                    label="Mean Synthetic Prototypes", alpha=0.9
                )

        # Set up the plot
        if is_hyperbolic:
            # Draw unit circle for Poincaré ball
            circle = plt.Circle((0, 0), 1, fill=False, color='black',
                                linestyle='--', alpha=0.5, linewidth=2)
            ax.add_patch(circle)
            ax.set_xlim(-1.1, 1.1)
            ax.set_ylim(-1.1, 1.1)
            ax.set_aspect('equal')
            title += " (Hyperbolic - Poincaré Ball)"

        ax.set_title(f"{title} - Task {task}, Epoch {epoch}", fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)

        # Create legend for classes
        handles, class_labels = scatter.legend_elements(num=None)
        class_legend = ax.legend(handles, [f"Class {i}" for i in range(len(handles))],
                                 title="Classes", bbox_to_anchor=(1.05, 1), loc='upper left')

        # Add legend for prototypes if they exist
        if P is not None or P_syn is not None:
            ax.add_artist(class_legend)  # Keep the class legend
            proto_handles = []
            proto_labels = []

            if P is not None:
                proto_handles.append(plt.scatter([], [], marker="X", s=200, c="gray", edgecolor="k"))
                proto_labels.append("Prototypes")

            if P_syn is not None:
                proto_handles.append(plt.scatter([], [], marker="*", s=100, c="red", edgecolor="k"))
                proto_labels.append("Synthetic Prototypes")
                proto_handles.append(plt.scatter([], [], marker="^", s=150, c="blue", edgecolor="k"))
                proto_labels.append("Mean Synthetic")

            ax.legend(proto_handles, proto_labels, title="Prototypes",
                      bbox_to_anchor=(1.05, 0.7), loc='upper left')

        plt.tight_layout()

        # Save if directory is set
        if self.save_dir is not None:
            path = f"{self.save_dir}/task_{task}_epoch_{epoch}_{title.replace(' ', '_')}.png"
            plt.savefig(path, dpi=300, bbox_inches='tight')
            print(f"[HyperbolicVisualizer] Saved figure to {path}")

        return fig


class HyperbolicUMAP:
    """Hyperbolic UMAP implementation using Poincaré ball model"""

    def __init__(self, n_components: int = 2, n_neighbors: int = 15,
                 min_dist: float = 0.1, metric: str = 'euclidean',
                 random_state: int = 42):
        self.n_components = n_components
        self.n_neighbors = n_neighbors
        self.min_dist = min_dist
        self.metric = metric
        self.random_state = random_state
        self.embedding_ = None

    def _euclidean_to_poincare(self, X: np.ndarray) -> np.ndarray:
        """Convert Euclidean coordinates to Poincaré ball coordinates"""
        # Normalize to unit ball first
        X_norm = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-8)

        # Apply stereographic projection to Poincaré ball
        # Scale down to ensure points are inside unit ball
        scale_factor = 0.95
        return X_norm * scale_factor

    def _poincare_distance(self, x: np.ndarray, y: np.ndarray) -> float:
        """Compute Poincaré distance between two points"""
        diff = x - y
        diff_norm_sq = np.sum(diff ** 2)
        x_norm_sq = np.sum(x ** 2)
        y_norm_sq = np.sum(y ** 2)

        numerator = 2 * diff_norm_sq
        denominator = (1 - x_norm_sq) * (1 - y_norm_sq)

        # Avoid division by zero and numerical issues
        denominator = np.maximum(denominator, 1e-8)

        return np.arccosh(1 + numerator / denominator)

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """Fit hyperbolic UMAP and return transformed data"""
        print("Applying standard UMAP...")
        # First apply standard UMAP
        standard_umap = umap.UMAP(
            n_components=self.n_components,
            n_neighbors=self.n_neighbors,
            min_dist=self.min_dist,
            metric=self.metric,
            random_state=self.random_state
        )

        euclidean_embedding = standard_umap.fit_transform(X)

        print("Converting to hyperbolic space...")
        # Convert to hyperbolic (Poincaré ball) coordinates
        self.embedding_ = self._euclidean_to_poincare(euclidean_embedding)

        return self.embedding_
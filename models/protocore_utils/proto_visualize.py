import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import torch
import numpy as np


class Visualizer:
    def __init__(self, save_dir=None):
        self.save_dir = save_dir

    def visualize_episode(self, embeddings, labels, task, epoch,
                          prototypes=None, predictions=None,
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

        P, P_pred = None, None
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

        ax.set_title(title)
        ax.legend(*scatter.legend_elements(num=None), title="Classes",
                  bbox_to_anchor=(1.05, 1), loc='upper left')

        plt.tight_layout()

        # Save if dir is set
        if self.save_dir is not None:
            path = f"{self.save_dir}/{task}{epoch}{title.replace(' ', '_')}.png"
            plt.savefig(path)
            print(f"[Visualizer] Saved figure to {path}")

        return fig

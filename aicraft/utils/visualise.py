import matplotlib.pyplot as plt
import torch
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE


class Visualisations:
    @staticmethod
    def show_images(data, labels_map: dict = None) -> None:
        if labels_map is None:
            if hasattr(data, "class_to_idx"):
                labels_map = {v: k for k, v in data.class_to_idx.items()}
            elif hasattr(data.dataset, "class_to_idx"):
                labels_map = {v: k for k, v in data.dataset.class_to_idx.items()}
            else:
                raise AttributeError(
                    "The dataset doesn't have a class_to_idx attribute"
                )
        figure = plt.figure(figsize=(8, 8))
        cols, rows = 3, 3
        for i in range(1, cols * rows + 1):
            sample_idx = torch.randint(len(data), size=(1,)).item()
            img, label = data[sample_idx]
            figure.add_subplot(rows, cols, i)
            plt.title(labels_map[label])
            plt.axis("off")
            plt.imshow(img.squeeze(), cmap="gray")

    @staticmethod
    def show_image(img: torch.Tensor, label: str = None):
        img = img.cpu().squeeze().detach().numpy()
        if label is not None:
            plt.title(label)
        plt.axis("off")
        plt.imshow(img.squeeze(), cmap="gray")

    @staticmethod
    def show_3_channel_image(img: torch.Tensor, label: str = None):
        img = img.permute(1, 2, 0)
        img = torch.clamp(img, 0, 1)
        Visualisations.show_image(img, label)

    @staticmethod
    def plot_3_channel_images_to_compare(
        a: torch.Tensor, b: torch.Tensor, la: str = None, lb: str = None
    ):
        a = a.permute(1, 2, 0)
        a = torch.clamp(a, 0, 1)
        a = a.cpu().squeeze().detach().numpy()

        b = b.permute(1, 2, 0)
        b = torch.clamp(b, 0, 1)
        b = b.cpu().squeeze().detach().numpy()

        plt.figure(figsize=(10, 5))

        plt.subplot(1, 2, 1)
        plt.imshow(a)
        if la is not None:
            plt.title(la)
        plt.axis("off")

        plt.subplot(1, 2, 2)
        plt.imshow(b)
        if lb is not None:
            plt.title(lb)
        plt.axis("off")

        plt.show()

    @staticmethod
    def compare_images(
        original: torch.Tensor, reconstructed: torch.Tensor, label: str = None
    ):
        original = original.cpu().squeeze().detach().numpy()
        reconstructed = reconstructed.cpu().squeeze().detach().numpy()
        # Create a new figure with a specified size
        figure = plt.figure(figsize=(6, 3))

        # Add the first subplot for the original image
        ax1 = figure.add_subplot(1, 2, 1)
        ax1.set_title(label + " (original)" if label else "Original")
        ax1.axis("off")
        ax1.imshow(original, cmap="gray")

        # Add the second subplot for the reconstructed image
        ax2 = figure.add_subplot(1, 2, 2)
        ax2.set_title(label + " (reconstructed)" if label else "Reconstructed")
        ax2.axis("off")
        ax2.imshow(reconstructed, cmap="gray")

        figure.tight_layout()
        plt.show()

    @staticmethod
    def plot_2d_embeddings(embeddings: torch.Tensor, labels: torch.Tensor):
        """
        Plot 2D embeddings with colors based on labels.

        Parameters:
        embeddings (np.ndarray): 2D numpy array of shape (n_samples, 2)
        labels (np.ndarray): 1D numpy array of shape (n_samples,)
        """
        embeddings = embeddings.cpu().detach().numpy()
        labels = labels.cpu().detach().numpy()
        # Create a new figure
        plt.figure(figsize=(10, 8))

        # Create a scatter plot
        scatter = plt.scatter(
            embeddings[:, 0],
            embeddings[:, 1],
            c=labels,
            cmap="viridis",
            s=50,
            alpha=0.7,
        )

        # Add a color bar
        cbar = plt.colorbar(scatter)
        cbar.set_label("Labels")

        # Add title and labels to the plot
        plt.title("2D Embeddings")
        plt.xlabel("Embedding Dimension 1")
        plt.ylabel("Embedding Dimension 2")

        # Show the plot
        plt.show()

    @staticmethod
    def plot_embeddings_pca(embeddings: torch.Tensor, labels: torch.Tensor):
        """
        Plot high-dimensional embeddings reduced to 2D using PCA.

        Parameters:
        embeddings (np.ndarray): High-dimensional numpy array of shape (n_samples, n_features)
        labels (np.ndarray): 1D numpy array of shape (n_samples,)
        """
        # Apply PCA to reduce dimensions to 2
        pca = PCA(n_components=2)
        reduced_embeddings = pca.fit_transform(embeddings.cpu().detach().numpy())

        # Create a new figure
        plt.figure(figsize=(10, 8))

        # Create a scatter plot
        scatter = plt.scatter(
            reduced_embeddings[:, 0],
            reduced_embeddings[:, 1],
            c=labels.cpu().detach().numpy(),
            cmap="viridis",
            s=50,
            alpha=0.7,
        )

        # Add a color bar
        cbar = plt.colorbar(scatter)
        cbar.set_label("Labels")

        # Add title and labels to the plot
        plt.title("2D PCA of High-Dimensional Embeddings")
        plt.xlabel("Principal Component 1")
        plt.ylabel("Principal Component 2")

        # Show the plot
        plt.show()

    @staticmethod
    def plot_embeddings_tsne(embeddings: torch.Tensor, labels: torch.Tensor):
        """
        Plot high-dimensional embeddings reduced to 2D using t-SNE.

        Parameters:
        embeddings (np.ndarray): High-dimensional numpy array of shape (n_samples, n_features)
        labels (np.ndarray): 1D numpy array of shape (n_samples,)
        """
        # Apply t-SNE to reduce dimensions to 2
        tsne = TSNE(n_components=2, random_state=42)
        reduced_embeddings = tsne.fit_transform(embeddings.cpu().detach().numpy())

        # Create a new figure
        plt.figure(figsize=(10, 8))

        # Create a scatter plot
        scatter = plt.scatter(
            reduced_embeddings[:, 0],
            reduced_embeddings[:, 1],
            c=labels.cpu().detach().numpy(),
            cmap="viridis",
            s=50,
            alpha=0.7,
        )

        # Add a color bar
        cbar = plt.colorbar(scatter)
        cbar.set_label("Labels")

        # Add title and labels to the plot
        plt.title("2D t-SNE of High-Dimensional Embeddings")
        plt.xlabel("t-SNE Component 1")
        plt.ylabel("t-SNE Component 2")

        # Show the plot
        plt.show()

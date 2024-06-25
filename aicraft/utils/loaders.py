import torchvision
import torch
import os
import logging

from typing import Tuple
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

if "TERMINAL_MODE" in os.environ:
    if os.environ["TERMINAL_MODE"] == "1":
        from tqdm import tqdm
    else:
        from tqdm.notebook import tqdm
else:
    from tqdm.notebook import tqdm

torch.manual_seed(42)


class FashionMNISTLoader:
    def __init__(self, val_split: float = 0.2):
        transforms = torchvision.transforms.Compose(
            [
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Pad(2),
                # torchvision.transforms.Normalize((0.5,), (0.5,))  # formula (x - mean / std)
            ]
        )
        self.full_train_dataset = torchvision.datasets.FashionMNIST(
            root="./data",  # Specify the path where the data should be saved
            train=True,  # Load the training set
            download=True,  # Download the data if not already present
            transform=transforms,  # Apply transformations
        )

        train_size = int((1 - val_split) * len(self.full_train_dataset))
        val_size = len(self.full_train_dataset) - train_size

        self.train, self.validation = random_split(
            self.full_train_dataset, [train_size, val_size]
        )

        self.test = torchvision.datasets.FashionMNIST(
            root="./data", train=False, download=True, transform=transforms
        )

    def get_loaders(
        self, batch_size: int = 46
    ) -> Tuple[DataLoader, DataLoader, DataLoader]:
        return (
            DataLoader(self.train, batch_size=batch_size, shuffle=True),
            DataLoader(self.validation, batch_size=batch_size, shuffle=True),
            DataLoader(self.test, batch_size=batch_size, shuffle=True),
        )


class CartoonFacesLoader:
    def __init__(
        self,
        path: str = "../data/cartoon-faces",
        val_split: float = 0.1,
        test_split: float = 0.1,
        means: list[float] = [0.924, 0.884, 0.854],
        stds: list[float] = [0.181, 0.230, 0.276],
    ):
        if means is not None and stds is not None:
            logger.info(
                "Using provided means and stds for the cartoon faces dataset..."
            )
            logger.info(f"\nmeans: {means}\nstds: {stds}")
            self.means = means
            self.stds = stds
        else:
            logger.info(
                "Computing mean and std of dataset for the cartoon faces dataset..."
            )
            ds = datasets.ImageFolder(
                root=path, transform=transforms.Compose([transforms.ToTensor()])
            )
            self.means, self.stds = self.compute_mean_std(ds)
            del ds

        self.dataset = datasets.ImageFolder(
            root=path,
            transform=transforms.Compose(
                [
                    transforms.Resize((512, 512)),
                    transforms.ToTensor(),
                    # transforms.Normalize(
                    #     mean=self.means,
                    #     std=self.stds,
                    # ),
                ]
            ),
        )

        train_size = int((1 - val_split - test_split) * len(self.dataset))
        val_size = int(val_split * len(self.dataset))
        test_size = len(self.dataset) - (train_size + val_size)

        self.train, self.validation, self.test = random_split(
            self.dataset, [train_size, val_size, test_size]
        )

    @staticmethod
    def compute_mean_std(
        dataset: datasets.ImageFolder,
    ) -> Tuple[list[float], list[float]]:
        loader = DataLoader(dataset, batch_size=512, shuffle=False, num_workers=2)

        # Initialize variables to store sum and squared sum of pixel values
        mean = torch.zeros(3)
        std = torch.zeros(3)
        total_samples = 0

        tk0 = tqdm(loader, total=int(len(loader)), desc="Computing Mean and STD")
        for batch_idx, (data, _) in enumerate(tk0):
            # Reshape images to (N, 3, H, W) -> (N, 3, H * W)
            data = data.view(data.size(0), data.size(1), -1)
            # Update mean and std accumulators
            mean += data.mean(2).sum(0)
            std += data.std(2).sum(0)
            total_samples += data.size(0)

        # Divide by the number of samples to get the average
        mean /= total_samples
        std /= total_samples

        return mean.tolist(), std.tolist()

    def get_loaders(
        self, batch_size: int = 64
    ) -> Tuple[DataLoader, DataLoader, DataLoader]:
        return (
            DataLoader(self.train, batch_size=batch_size, shuffle=True),
            DataLoader(self.validation, batch_size=batch_size, shuffle=True),
            DataLoader(self.test, batch_size=batch_size, shuffle=True),
        )

    def denormalize(self, tensor: torch.Tensor) -> torch.Tensor:
        if tensor.is_mps or tensor.is_cuda:
            tensor = tensor.cpu()

        result = torch.empty_like(tensor)
        for i, (t, m, s) in enumerate(zip(tensor, self.means, self.stds)):
            result[i] = t * s + m

        return result

    def normalize(self, tensor: torch.Tensor) -> torch.Tensor:
        if tensor.is_mps or tensor.is_cuda:
            tensor = tensor.cpu()
        for t, m, s in zip(tensor, self.means, self.stds):
            t.sub_(m).div_(s)
        return tensor

    def __len__(self) -> int:
        return len(self.dataset)

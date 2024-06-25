from torch import nn
import torch
from pthflops import count_ops
from safetensors.torch import save_model, load_model
from typing import Tuple

from .encoder import Encoder
from .decoder import Decoder
from .sampling import Sampling


class VariationalAutoEncoder(nn.Module):
    def __init__(self, embedding_size: int = 2):
        super().__init__()
        self.encoder = Encoder(embedding_size=embedding_size)
        self.decoder = Decoder(embedding_size=embedding_size)
        self.sampling = Sampling()

    def forward(self, x: torch.Tensor):
        mu, sigma = self.encoder(x)
        z = self.sampling(mu, sigma)
        x = self.decoder(z)
        return mu, sigma, x

    def trainable_parameters(self):
        """
        Return the total number of trainable parameters in the model.
        """
        return sum(p.numel() for p in self.model.parameters() if p.requires_grad)

    def save(self, path: str) -> None:
        """
        Save the model to the specified path.

        Parameters:
            path (str): The path where the model will be saved.

        Returns:
            None
        """
        save_model(self, path)

    def load(self, path: str) -> None:
        """
        Load the model from the specified path.

        Parameters:
            path (str): The path to load the model from.

        Returns:
            None
        """
        load_model(self, path)

    def no_of_params(self) -> int:
        """
        A function that calculates the total number of parameters in a model.
        Returns:
            int: The total number of parameters.
        """
        _no_params = 0
        for p in self.model.parameters():
            _no_params += p.nelement()
        return _no_params

    def size_in_memory(self) -> int:
        """
        Calculate the total size in memory taken by the model parameters and buffers.
        This function does not take any parameters and returns an integer representing the total size in bytes.
        """
        _param_size = 0
        for p in self.model.parameters():
            _param_size += p.nelement() * p.element_size()
        buffer_size = sum(
            [buf.nelement() * buf.element_size() for buf in self.model.buffers()]
        )
        return _param_size + buffer_size

    def no_of_flops(self, dtype: torch.dtype = torch.float32) -> int:
        """
        Calculate the number of floating-point operations for the given model and input data.

        Parameters:
            dtype (torch.dtype): The data type for the input data. Default is torch.float32.

        Returns:
            int: The total number of floating-point operations.
        """
        return count_ops(self.model, torch.randn(1, 3, 224, 224).to(dtype=dtype))

    def generate_embedding(
        self, x: torch.utils.data.DataLoader, device: torch.device = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        z_means = torch.Tensor()
        z_stds = torch.Tensor()
        labels = torch.Tensor()
        if device is not None:
            z_means = z_means.to(device)
            z_stds = z_stds.to(device)
            labels = labels.to(device)
            self.to(device)
        for batch, l in x:
            if device is not None:
                batch = batch.to(device)
                l = l.to(device)
            mu, log_var = self.encoder(batch)
            z_means = torch.cat((z_means, mu), 0)
            z_stds = torch.cat((z_stds, log_var), 0)
            labels = torch.cat((labels, l), 0)
        embeddings = self.sampling(z_means, z_stds)
        return embeddings, labels, z_means, z_stds

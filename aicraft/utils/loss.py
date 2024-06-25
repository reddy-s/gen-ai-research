from torch import nn
import torch


class CustomLosses:
    @staticmethod
    def kld_loss(mu, log_var):
        return torch.mean(
            -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp(), dim=1)
        )

    @staticmethod
    def kld_loss_tweaked(mu, log_var):
        return torch.mean(
            torch.sum(
                -0.5 * (1 + log_var - mu.pow(2) - log_var.exp()),
                dim=1
            )
        )

    @staticmethod
    def vae_loss(recon_x, x, mu, log_var, beta: int = 500):
        reconstruction_loss = beta * nn.functional.binary_cross_entropy(
            recon_x, x, reduction="mean"
        )
        kld_loss = CustomLosses.kld_loss(mu, log_var)
        total_loss = reconstruction_loss + kld_loss
        return total_loss, reconstruction_loss, kld_loss

    @staticmethod
    def vae_loss_with_rmse(recon_x, x, mu, log_var, beta: int = 2000):
        reconstruction_loss = beta * nn.functional.mse_loss(recon_x, x)
        kld_loss = CustomLosses.kld_loss_tweaked(mu, log_var)
        total_loss = reconstruction_loss + kld_loss
        return total_loss, reconstruction_loss, kld_loss

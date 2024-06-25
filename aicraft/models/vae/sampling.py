import torch
import torch.nn as nn


class Sampling(nn.Module):
    def forward(self, z_mean, z_log_var):
        epsilon = torch.randn_like(z_mean).to(z_mean.device)
        z_sigma = torch.exp(0.5 * z_log_var)
        sample = z_mean + (z_sigma * epsilon)
        return sample

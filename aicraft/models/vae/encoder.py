from torch import nn
from torch import Tensor


class Encoder(nn.Module):
    def __init__(self, embedding_size: int = 2):
        super().__init__()
        self.block1 = nn.Sequential(
            nn.Conv2d(
                in_channels=1, out_channels=32, kernel_size=(3, 3), stride=2, padding=1
            ),
            nn.BatchNorm2d(32),
            nn.ReLU(),
        )

        self.block2 = nn.Sequential(
            nn.Conv2d(
                in_channels=32, out_channels=64, kernel_size=(3, 3), stride=2, padding=1
            ),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )

        self.block3 = nn.Sequential(
            nn.Conv2d(
                in_channels=64,
                out_channels=128,
                kernel_size=(3, 3),
                stride=2,
                padding=1,
            ),
            nn.BatchNorm2d(128),
            nn.ReLU(),
        )

        self.z_mean = nn.Linear(128 * 4 * 4, embedding_size)
        self.z_log_var = nn.Linear(128 * 4 * 4, embedding_size)

    def forward(self, x: Tensor):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = x.view(-1, 128 * 4 * 4)
        mu = self.z_mean(x)
        sigma = self.z_log_var(x)
        return mu, sigma

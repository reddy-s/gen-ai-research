from torch import nn
from torch import Tensor


class Decoder(nn.Module):
    def __init__(self, embedding_size: int = 16):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(embedding_size, 256 * 16 * 16))

        self.block4 = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels=256,
                out_channels=128,
                kernel_size=(3, 3),
                stride=2,
                padding=1,
                output_padding=1,
            ),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(),
        )

        self.block3 = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels=128,
                out_channels=64,
                kernel_size=(3, 3),
                stride=2,
                padding=1,
                output_padding=1,
            ),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(),
        )

        self.block2 = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels=64,
                out_channels=32,
                kernel_size=(3, 3),
                stride=2,
                padding=1,
                output_padding=1,
            ),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(),
        )

        self.block1 = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels=32,
                out_channels=16,
                kernel_size=(3, 3),
                stride=2,
                padding=1,
                output_padding=1,
            ),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(),
        )

        self.reconstructor = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels=16,
                out_channels=3,
                kernel_size=(3, 3),
                stride=2,
                padding=1,
                output_padding=1,
            ),
            nn.Sigmoid(),
        )

    def forward(self, x: Tensor):
        x = self.mlp(x)
        x = x.view(-1, 256, 16, 16)
        x = self.block4(x)
        x = self.block3(x)
        x = self.block2(x)
        x = self.block1(x)
        x = self.reconstructor(x)
        return x

from torch import nn
from torch import Tensor


class Decoder(nn.Module):
    def __init__(self, embedding_size: int = 2):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(embedding_size, 128 * 4 * 4))

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
            nn.ReLU(),
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
            nn.ReLU(),
        )

        self.block1 = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels=32,
                out_channels=1,
                kernel_size=(3, 3),
                stride=2,
                padding=1,
                output_padding=1,
            ),
            nn.BatchNorm2d(1),
            nn.ReLU(),
        )

        self.reconstructor = nn.Sequential(
            nn.Conv2d(
                in_channels=1, out_channels=1, kernel_size=(3, 3), stride=1, padding=1
            ),
            nn.Sigmoid(),
        )

        self.model = nn.Sequential(
            self.mlp,
            nn.Unflatten(1, (128, 4, 4)),
            self.block3,
            self.block2,
            self.block1,
            self.reconstructor,
        )

    def forward(self, x: Tensor):
        # return self.model(x)
        x = self.mlp(x)
        x = x.view(-1, 128, 4, 4)
        x = self.block3(x)
        x = self.block2(x)
        x = self.block1(x)
        x = self.reconstructor(x)
        return x

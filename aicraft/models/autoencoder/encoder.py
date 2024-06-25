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

        self.mlp = nn.Sequential(nn.Linear(128 * 4 * 4, embedding_size))

        self.model = nn.Sequential(
            self.block1, self.block2, self.block3, nn.Flatten(), self.mlp
        )

    def forward(self, x: Tensor):
        # return self.model(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = x.view(-1, 128 * 4 * 4)
        x = self.mlp(x)
        return x

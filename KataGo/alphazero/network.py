import torch.nn as nn
import torch.nn.functional as F


class ResBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.norm1 = nn.GroupNorm(1, channels)
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.norm2 = nn.GroupNorm(1, channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)

    def forward(self, x):
        out = self.conv1(F.silu(self.norm1(x)))
        out = self.conv2(F.silu(self.norm2(out)))
        return out + x


class ResNet(nn.Module):
    def __init__(self, board_size, num_planes, num_blocks=1, num_channels=32):
        super().__init__()
        self.board_size = board_size
        self.action_size = board_size * board_size

        self.start_layer = nn.Sequential(
            nn.Conv2d(num_planes, num_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(1, num_channels),
            nn.SiLU(inplace=True),
        )

        self.trunk = nn.ModuleList([ResBlock(num_channels) for _ in range(num_blocks)])

        self.policy_head = nn.Sequential(
            nn.Conv2d(num_channels, num_channels, kernel_size=1, bias=False),
            nn.GroupNorm(1, num_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(num_channels, 1, kernel_size=1, bias=True),
        )

        self.value_head = nn.Sequential(
            nn.Conv2d(num_channels, num_channels, kernel_size=1, bias=False),
            nn.GroupNorm(1, num_channels),
            nn.SiLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(num_channels, num_channels // 2),
            nn.SiLU(inplace=True),
            nn.Linear(num_channels // 2, 3),
        )

    def forward(self, x):
        x = self.start_layer(x)
        for block in self.trunk:
            x = block(x)
        policy_logits = self.policy_head(x).flatten(1)
        value_logits = self.value_head(x)
        return policy_logits, value_logits

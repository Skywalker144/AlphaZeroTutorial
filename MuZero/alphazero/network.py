import torch
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


class RepresentationNet(nn.Module):
    def __init__(self, num_planes, num_channels=32, num_blocks=1):
        super().__init__()
        self.start_layer = nn.Sequential(
            nn.Conv2d(num_planes, num_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(1, num_channels),
            nn.SiLU(inplace=True),
        )
        self.trunk = nn.ModuleList([ResBlock(num_channels) for _ in range(num_blocks)])

    def forward(self, x):
        x = self.start_layer(x)
        for block in self.trunk:
            x = block(x)
        return x


class DynamicsNet(nn.Module):
    def __init__(self, board_size, num_channels=32, num_blocks=1):
        super().__init__()
        self.board_size = board_size
        self.start_layer = nn.Sequential(
            nn.Conv2d(num_channels + 2, num_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(1, num_channels),
            nn.SiLU(inplace=True),
        )
        self.trunk = nn.ModuleList([ResBlock(num_channels) for _ in range(num_blocks)])

    def forward(self, hidden_state, action, to_play):
        action_plane = torch.zeros(
            hidden_state.shape[0],
            1,
            self.board_size,
            self.board_size,
            dtype=hidden_state.dtype,
            device=hidden_state.device,
        )
        row = action // self.board_size
        col = action % self.board_size
        batch_index = torch.arange(hidden_state.shape[0], device=hidden_state.device)
        action_plane[batch_index, 0, row, col] = 1.0
        to_play_plane = (
            (to_play > 0)
            .to(hidden_state.dtype)
            .view(-1, 1, 1, 1)
            .expand(-1, 1, self.board_size, self.board_size)
        )
        x = torch.cat([hidden_state, action_plane, to_play_plane], dim=1)
        x = self.start_layer(x)
        for block in self.trunk:
            x = block(x)
        return x


class PredictionNet(nn.Module):
    def __init__(self, num_channels=32):
        super().__init__()
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
            nn.Linear(num_channels // 2, 1),
        )

    def forward(self, hidden_state):
        policy_logits = self.policy_head(hidden_state).flatten(1)
        value = torch.tanh(self.value_head(hidden_state)).squeeze(1)
        return policy_logits, value


class MuZeroNet(nn.Module):
    def __init__(self, board_size, num_planes, num_blocks=1, num_channels=32):
        super().__init__()
        self.board_size = board_size
        self.action_size = board_size * board_size
        self.representation = RepresentationNet(num_planes, num_channels, num_blocks)
        self.dynamics = DynamicsNet(board_size, num_channels, num_blocks)
        self.prediction = PredictionNet(num_channels)

    def initial_inference(self, observation):
        return self.prediction(self.representation(observation))

    def recurrent_inference(self, hidden_state, action, to_play):
        return self.prediction(self.dynamics(hidden_state, action, to_play))

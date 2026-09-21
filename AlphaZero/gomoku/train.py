import torch.optim as optim

from alphazero import AlphaZero, ResNet, auto_device
from envs.gomoku import Gomoku

board_size = 9

train_args = {
    "board_size": board_size,
    "num_planes": 3,
    "num_blocks": 4,
    "num_channels": 64,
    "lr": 0.0001,
    "weight_decay": 1e-4,
    "num_simulations": 100,
    "c_puct": 1.5,
    "dirichlet_total_concentration": 0.03 * board_size ** 2,
    "dirichlet_noise_weight": 0.25,
    "train_steps": 200,
    "batch_size": 128,
    "replay_ratio": 8,
    "bootstrap_games": 100,
    "min_rows": 40000,
    "taper_window_exponent": 0.8,
    "expand_window_per_row": 0.3,
    "keep_target_rows": 10000000,
    "data_dir": "data/gomoku",
    "device": auto_device(),
}


def main():
    game = Gomoku(board_size=train_args["board_size"])
    model = ResNet(
        game.board_size,
        game.num_planes,
        num_blocks=train_args["num_blocks"],
        num_channels=train_args["num_channels"],
    )
    optimizer = optim.AdamW(
        model.parameters(), lr=train_args["lr"], weight_decay=train_args["weight_decay"]
    )

    az = AlphaZero(game, model, optimizer, train_args)
    az.load_checkpoint()
    az.learn()


if __name__ == "__main__":
    main()

import torch.optim as optim

from alphazero import AlphaZero, ResNet, auto_device
from envs.tictactoe import TicTacToe

board_size = 3

train_args = {
    "board_size": board_size,
    "num_planes": 3,
    "num_blocks": 3,
    "num_channels": 32,
    "lr": 0.0001,
    "weight_decay": 1e-4,
    "num_simulations": 50,
    "c_puct": 1.5,
    "dirichlet_total_concentration": 0.03 * board_size ** 2,
    "dirichlet_noise_weight": 0.25,
    "train_steps": 50,
    "batch_size": 128,
    "replay_ratio": 8,
    "bootstrap_games": 20,
    "min_rows": 6400,
    "taper_window_exponent": 0.8,
    "expand_window_per_row": 0.3,
    "keep_target_rows": 100000,
    "save_interval": 5,
    "winrate_sample_every": 5,
    "data_dir": "data/tictactoe",
    "device": auto_device(),
}


def main():
    game = TicTacToe()
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

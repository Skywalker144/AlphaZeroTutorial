import os

from alphazero import MuZero
from envs.tictactoe import TicTacToe

board_size = 3

train_args = {
    "mode": "train",
    "num_blocks": 1,
    "num_channels": 16,
    "num_simulations": 20,
    "data_dir": os.path.join(os.path.dirname(__file__), "data"),
}


def main():
    game = TicTacToe()
    az = MuZero(game, train_args)
    az.learn()


if __name__ == "__main__":
    main()

import os

from alphazero import MuZero
from envs.gomoku import Gomoku

board_size = 9

train_args = {
    "mode": "train",
    "num_blocks": 1,
    "num_channels": 32,
    "num_simulations": 100,
    "data_dir": os.path.join(os.path.dirname(__file__), "data"),
}


def main():
    game = Gomoku(board_size=board_size)
    az = MuZero(game, train_args)
    az.learn()


if __name__ == "__main__":
    main()

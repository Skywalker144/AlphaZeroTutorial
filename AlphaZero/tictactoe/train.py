import os

from alphazero import AlphaZero
from envs.tictactoe import TicTacToe

board_size = 3

train_args = {
    "num_blocks": 1,
    "num_channels": 16,
    "num_simulations": 50,
    "data_dir": os.path.join(os.path.dirname(__file__), "data"),
}


def main():
    game = TicTacToe()
    az = AlphaZero(game, train_args)
    az.learn()


if __name__ == "__main__":
    main()

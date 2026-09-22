from alphazero import AlphaZero
from envs.gomoku import Gomoku

board_size = 9

train_args = {
    "num_blocks": 1,
    "num_channels": 32,
    "num_simulations": 100,
    "data_dir": "data/gomoku",
}


def main():
    game = Gomoku(board_size=board_size)
    az = AlphaZero(game, train_args)
    az.load_checkpoint()
    az.learn()


if __name__ == "__main__":
    main()

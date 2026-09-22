from alphazero import AlphaZero
from envs.tictactoe import TicTacToe

board_size = 3

train_args = {
    "num_blocks": 3,
    "num_channels": 32,
    "num_simulations": 50,
    "data_dir": "data/tictactoe",
}


def main():
    game = TicTacToe()
    az = AlphaZero(game, train_args)
    az.load_checkpoint()
    az.learn()


if __name__ == "__main__":
    main()

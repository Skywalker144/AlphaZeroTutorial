from alphazero.play import play
from envs.gomoku import Gomoku
from gomoku.train import board_size, train_args


def main():
    play(Gomoku(board_size=board_size), train_args, side_names=("Black", "White"))


if __name__ == "__main__":
    main()

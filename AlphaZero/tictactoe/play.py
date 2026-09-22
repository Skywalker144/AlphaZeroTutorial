from alphazero.play import play
from envs.tictactoe import TicTacToe
from tictactoe.train import train_args


def main():
    play(TicTacToe(), train_args, side_names=("X", "O"))


if __name__ == "__main__":
    main()

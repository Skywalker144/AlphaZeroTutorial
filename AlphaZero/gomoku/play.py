import numpy as np

from alphazero import MCTS, ResNet, auto_device
from alphazero.utils import print_board
from envs.gomoku import Gomoku
from gomoku.train import board_size, train_args


def main():
    device = auto_device()
    game = Gomoku(board_size=board_size)
    model = ResNet(
        game.board_size,
        game.num_planes,
        num_blocks=train_args["num_blocks"],
        num_channels=train_args["num_channels"],
    )
    model.to(device)

    checkpoint_path = input("Checkpoint path (Enter for latest): ").strip()
    import glob
    import os

    if not checkpoint_path:
        candidates = glob.glob(os.path.join(train_args["data_dir"], "checkpoints", "*.pth"))
        checkpoint_path = max(candidates, key=os.path.getmtime) if candidates else ""
    if checkpoint_path:
        import torch

        ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        print(f"Loaded {checkpoint_path}")
    else:
        print("No checkpoint found, using random weights.")

    mcts = MCTS(
        game,
        train_args,
        model,
        device,
    )
    human_side = int(input("Play as Black (1) or White (-1): ").strip())

    state = game.get_initial_state()
    to_play = 1
    print_board(state)

    while not game.is_terminal(state, to_play):
        if to_play == human_side:
            while True:
                text = input("Your move (row col): ").strip()
                try:
                    row, col = map(int, text.split())
                except ValueError:
                    print("Invalid input, use 'row col'.")
                    continue
                action = row * game.board_size + col
                if game.get_legal_action_mask(state, to_play)[action]:
                    break
                print("Illegal move.")
        else:
            mcts_policy = mcts.search(state, to_play, train_args["num_simulations"])
            action = int(np.argmax(mcts_policy))
            row, col = divmod(action, game.board_size)
            print(f"AlphaZero plays: {row} {col}")

        state = game.get_next_state(state, action, to_play)
        to_play = -to_play
        print_board(state)

    winner = game.get_winner(state, to_play)
    if winner == 1:
        print("Black wins!")
    elif winner == -1:
        print("White wins!")
    else:
        print("Draw!")


if __name__ == "__main__":
    main()

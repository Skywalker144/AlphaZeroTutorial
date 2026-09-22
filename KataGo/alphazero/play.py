import argparse

import numpy as np

from .mcts import MCTS
from .network import ResNet
from .utils import auto_device, find_latest_model, print_board


def play(game, train_args, side_names):
    parser = argparse.ArgumentParser(description="AlphaZero 人机对弈")
    parser.add_argument(
        "-n",
        "--num-simulations",
        type=int,
        default=None,
        help="MCTS 模拟次数；0 表示纯网络。默认取 train_args 中的配置。",
    )
    cli_args = parser.parse_args()
    if cli_args.num_simulations is not None:
        train_args = {**train_args, "num_simulations": cli_args.num_simulations}

    positive_name, negative_name = side_names
    
    train_args = {**train_args, "mode": "eval"}
    num_simulations = train_args["num_simulations"]
    device = auto_device()
    model = ResNet(
        game.board_size,
        game.num_planes,
        num_blocks=train_args["num_blocks"],
        num_channels=train_args["num_channels"],
    )
    model.to(device)

    checkpoint_path, state_dict = find_latest_model(
        train_args["data_dir"], map_location=device
    )
    if state_dict is not None:
        model.load_state_dict(state_dict)
        print(f"Loaded {checkpoint_path}")
    else:
        print("No checkpoint found, using random weights.")

    mcts = MCTS(
        game,
        train_args,
        model,
        device,
    )
    human_side = int(
        input(f"Play as {positive_name} (1) or {negative_name} (-1): ").strip()
    )

    state = game.get_initial_state()
    to_play = 1
    turn_number = 0
    print_board(state)

    while not game.is_terminal(state, to_play):
        policy = None
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
            policy, root_value, _ = mcts.search(state, to_play, num_simulations, turn_number, False)
            action = int(np.argmax(policy))
            row, col = divmod(action, game.board_size)
            print(f"AlphaZero plays: {row} {col}  root_value={root_value:+.3f}")

        state = game.get_next_state(state, action, to_play)
        to_play = -to_play
        turn_number += 1
        print_board(state, policy)

    winner = game.get_winner(state, to_play)
    if winner == 1:
        print(f"{positive_name} wins!")
    elif winner == -1:
        print(f"{negative_name} wins!")
    else:
        print("Draw!")

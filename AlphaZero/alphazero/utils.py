import numpy as np
import torch


def auto_device():
    if torch.cuda.is_available():
        return "cuda"
    mps_backend = getattr(torch.backends, "mps", None)
    if mps_backend is not None and mps_backend.is_available():
        return "mps"
    return "cpu"


def softmax(x):
    x = np.asarray(x, dtype=np.float64)
    x = x - np.max(x)
    exp_x = np.exp(x)
    return exp_x / np.sum(exp_x)


def add_dirichlet_noise(policy, total_concentration, noise_weight=0.25):
    """Mix uniform Dirichlet noise into a root policy.

    ``total_concentration`` is the sum of the Dirichlet parameters.  It is
    deliberately independent of the number of currently legal actions; the
    per-action concentration is derived from it below.
    """
    nonzero_mask = policy > 0
    nonzero_count = np.sum(nonzero_mask)
    if nonzero_count <= 1:
        return policy
    per_action_concentration = total_concentration / nonzero_count
    noise = np.random.dirichlet([per_action_concentration] * nonzero_count)
    new_policy = policy.copy()
    new_policy[nonzero_mask] = (
        (1 - noise_weight) * policy[nonzero_mask] + noise_weight * noise
    )
    return new_policy


def random_augment_batch(batch, board_size):
    augmented_batch = []
    for sample in batch:
        k = np.random.randint(0, 4)
        flip = np.random.choice([True, False])

        state = sample["encoded_state"]
        policy = sample["policy_target"].reshape(board_size, board_size)

        aug_state = np.rot90(state, k, axes=(1, 2))
        aug_policy = np.rot90(policy, k)

        if flip:
            aug_state = np.flip(aug_state, axis=2)
            aug_policy = np.flip(aug_policy, axis=1)

        new_sample = sample.copy()
        new_sample["encoded_state"] = aug_state.copy()
        new_sample["policy_target"] = aug_policy.flatten().copy()
        augmented_batch.append(new_sample)
    return augmented_batch


def print_board(board):
    rows, cols = board.shape
    print("   ", end="")
    for col in range(cols):
        print(f"{col:2d} ", end="")
    print()
    for row in range(rows):
        print(f"{row:2d} ", end="")
        for col in range(cols):
            if board[row, col] == 1:
                print("  X", end="")
            elif board[row, col] == -1:
                print("  O", end="")
            else:
                print("  .", end="")
        print()

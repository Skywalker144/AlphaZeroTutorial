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


def add_dirichlet_noise(policy, total_concentration, legal_actions_mask, noise_weight=0.25):
    """
    训练时 在根节点策略中混入 Dirichlet Noise 以鼓励探索：

        noisy_policy = (1 - noise_weight) * policy + noise_weight * noise
    
    其中 total_concentration 一般可以设置为 0.03 * board_size^2
    noise_weight 一般是 0.25
    
    只给合法动作加噪声，非法位置保持为 0。每个合法动作的浓度为 total_concentration / 合法动作数，
    total_concentration 越小，噪声越尖锐，即越集中在少数动作上。
    """
    legal_actions_count = np.sum(legal_actions_mask)
    if legal_actions_count <= 1:
        return policy
    per_action_concentration = total_concentration / legal_actions_count
    noise = np.random.dirichlet([per_action_concentration] * legal_actions_count)
    noisy_policy = policy.copy()
    noisy_policy[legal_actions_mask] = (
        (1 - noise_weight) * policy[legal_actions_mask] + noise_weight * noise
    )
    return noisy_policy


def random_augment_batch(batch, board_size):
    """
    对整个 batch 应用同一个随机对称变换 让网络学会对称性
    """
    states = np.stack([sample["encoded_state"] for sample in batch])
    policies = np.stack(
        [sample["policy_target"].reshape(board_size, board_size) for sample in batch]
    )

    k = np.random.randint(0, 4)
    flip = np.random.choice([True, False])

    states = np.rot90(states, k, axes=(2, 3))
    policies = np.rot90(policies, k, axes=(1, 2))
    if flip:
        states = np.flip(states, axis=3)
        policies = np.flip(policies, axis=2)

    augmented_batch = []
    for sample, state, policy in zip(batch, states, policies):
        new_sample = sample.copy()
        new_sample["encoded_state"] = np.ascontiguousarray(state)
        new_sample["policy_target"] = policy.flatten()
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

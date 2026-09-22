import glob
import os

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


def interpolate_early(turn_number, halflife, early_value, late_value, board_size):
    halflives = (turn_number / halflife) * (19.0 / board_size)
    return late_value + (early_value - late_value) * 0.5 ** halflives


def root_policy_temperature(args, turn_number, board_size):
    return interpolate_early(
        turn_number,
        args.get("chosen_move_temperature_halflife", 19.0),
        args.get("root_policy_temperature_early", 1.3),
        args.get("root_policy_temperature", 1.1),
        board_size,
    )


def chosen_move_temperature(args, turn_number, board_size):
    return interpolate_early(
        turn_number,
        args.get("chosen_move_temperature_halflife", 19.0),
        args.get("chosen_move_temperature_early", 0.75),
        args.get("chosen_move_temperature", 0.15),
        board_size,
    )


def value_target(winner, to_play):
    outcome = winner * to_play
    target = np.zeros(3, dtype=np.float32)
    if outcome > 0:
        target[0] = 1.0
    elif outcome < 0:
        target[2] = 1.0
    else:
        target[1] = 1.0
    return target


def search_visit_counts(args, board_size):
    full = max(
        args.get("full_search_visits_floor", 50),
        round(args.get("num_simulations", 1.66 * board_size ** 2)),
    )
    cheap = min(
        full,
        max(
            args.get("cheap_search_visits_floor", 20),
            round(args.get("cheap_search_visits", 0.28 * board_size ** 2)),
        ),
    )
    return full, cheap


def apply_temperature(probs, temperature):
    if temperature <= 1e-4:
        result = np.zeros_like(probs)
        result[np.argmax(probs)] = 1.0
        return result
    positive = probs > 0
    scaled = np.full(probs.shape, -np.inf)
    scaled[positive] = np.log(probs[positive]) / temperature
    scaled -= np.max(scaled)
    result = np.exp(scaled)
    return result / np.sum(result)


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


def load_state_dict(path, map_location="cpu"):
    """从 .pth 文件读出模型权重。

    同时支持两种格式：
    - 完整的训练 checkpoint（含 model_state_dict / optimizer_state_dict ...）
    - 单纯由 ``torch.save(model.state_dict(), ...)`` 保存的 ``model_*.pth``
    """
    obj = torch.load(path, map_location=map_location, weights_only=False)
    if isinstance(obj, dict) and "model_state_dict" in obj:
        return obj["model_state_dict"]
    return obj


def find_latest_model(data_dir, map_location="cpu"):
    """找出 data_dir 下最新的、能真正读出来的模型文件。

    会同时搜索 ``checkpoints/`` 和 ``models/``，按修改时间从新到旧尝试。
    读到损坏文件（例如保存时被 Ctrl-C 打断而截断的 checkpoint）就跳过，
    返回 ``(path, state_dict)``；全部失败则返回 ``(None, None)``。
    """
    candidates = []
    for sub in ("checkpoints", "models"):
        candidates.extend(glob.glob(os.path.join(data_dir, sub, "*.pth")))
    candidates.sort(key=os.path.getmtime, reverse=True)
    for path in candidates:
        try:
            return path, load_state_dict(path, map_location=map_location)
        except Exception:
            print(f"Skipping unreadable file, trying the previous one: {path}")
            continue
    return None, None


BOARD_GAP = "   "
POLICY_WIDTH = 5
POLICY_SHOW_THRESHOLD = 0.01
EMPTY_DOT = "·"


def _board_line(board, row, cols):
    if row == -1:
        return "   " + "".join(f"{col:2d} " for col in range(cols))
    parts = [f"{row:2d} "]
    for col in range(cols):
        value = board[row, col]
        if value == 1:
            marker = "X"
        elif value == -1:
            marker = "O"
        else:
            marker = EMPTY_DOT
        parts.append(f" {marker} ")
    return "".join(parts)


def _policy_line(policy, row, cols):
    if row == -1:
        return "   " + "".join(f"{col:^{POLICY_WIDTH}}" for col in range(cols))
    parts = [f"{row:2d} "]
    for col in range(cols):
        probability = policy[row * cols + col]
        if probability < POLICY_SHOW_THRESHOLD:
            text = EMPTY_DOT
        else:
            # 显示千分比，取三位整数（小数点后三位），如 0.25 -> 250
            text = f"{round(probability * 1000):03d}"
        parts.append(f"{text:^{POLICY_WIDTH}}")
    return "".join(parts)


def print_board(board, policy=None):
    """打印棋盘；若给定 ``policy``（展平的长度 board_size**2 概率分布），
    在棋盘右侧并排打印该分布，低于 1% 的位置显示为点。
    """
    rows, cols = board.shape
    for row in range(-1, rows):
        line = _board_line(board, row, cols)
        if policy is not None:
            line += BOARD_GAP + _policy_line(policy, row, cols)
        print(line)

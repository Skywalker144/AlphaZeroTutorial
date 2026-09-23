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


def _transform_action(action, k, flip, board_size):
    plane = np.zeros((board_size, board_size), dtype=np.int8)
    plane[action // board_size, action % board_size] = 1
    plane = np.rot90(plane, k)
    if flip:
        plane = np.flip(plane, axis=1)
    row, col = np.argwhere(plane == 1)[0]
    return int(row) * board_size + int(col)


def random_augment_batch(batch, board_size):
    """
    对整个 batch 应用同一个随机对称变换 让网络学会对称性

    MuZero 的样本里有起点观测、K 个动作和 K+1 组策略目标，三者必须用同一个变换，
    动作编号也要一并重映射。
    """
    k = int(np.random.randint(0, 4))
    flip = bool(np.random.choice([True, False]))

    augmented_batch = []
    for sample in batch:
        observation = np.rot90(sample["observation"], k, axes=(1, 2))
        policy_targets = np.rot90(
            sample["policy_targets"].reshape(-1, board_size, board_size), k, axes=(1, 2)
        )
        if flip:
            observation = np.flip(observation, axis=2)
            policy_targets = np.flip(policy_targets, axis=2)
        actions = np.array(
            [_transform_action(action, k, flip, board_size) for action in sample["actions"]],
            dtype=np.int64,
        )
        new_sample = sample.copy()
        new_sample["observation"] = np.ascontiguousarray(observation)
        new_sample["actions"] = actions
        new_sample["policy_targets"] = np.ascontiguousarray(
            policy_targets.reshape(-1, board_size * board_size)
        )
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

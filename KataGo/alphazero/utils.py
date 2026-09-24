import glob
import math
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


def reduced_search_limit(args, win_loss_history, full_visits, cheap_visits):
    """
    soft resignation：局势已经一边倒时，给终局前的搜索降低访问数、降低样本权重，而不是直接认输。

    - 取最近 reduce_visits_threshold_lookback 步的 WinLoss（绝对视角），算
      extreme = max(min, -max) 并 clamp 到 1；
    - 若 extreme 超过 reduce_visits_threshold，则按 (extreme - threshold)/(1 - threshold) 的平方
      在 full_visits 与 reduced_visits_min 之间插值，并把样本权重从 1 向 reduced_visits_weight 插值。
    """
    if not args.get("reduce_visits", True):
        return full_visits, 1.0
    lookback = args.get("reduce_visits_threshold_lookback", 3)
    if len(win_loss_history) < lookback:
        return full_visits, 1.0
    threshold = args.get("reduce_visits_threshold", 0.9)
    recent = win_loss_history[-lookback:]
    extreme = max(min(recent), -max(recent))
    if extreme > 1.0:
        extreme = 1.0
    amount = extreme - threshold
    if amount <= 0.0:
        return full_visits, 1.0
    prop = (amount / (1.0 - threshold)) ** 2
    min_visits = args.get("reduced_visits_min", cheap_visits)
    weight = args.get("reduced_visits_weight", 0.1)
    visits = round(full_visits + prop * (min_visits - full_visits))
    visits = max(visits, min_visits)
    return visits, 1.0 + prop * (weight - 1.0)


def policy_surprise(prior_policy, mcts_policy):
    support = mcts_policy > 0
    prior = np.maximum(prior_policy[support], 1e-100)
    target = mcts_policy[support]
    surprise = np.sum(target * (np.log(target) - np.log(prior)))
    return max(0.0, float(surprise))


def value_surprise(search_wdl, nn_wdl):
    search = np.maximum(search_wdl, 0.0)
    total = search.sum()
    if total <= 0.0:
        return 0.0
    search = search / total
    surprise = np.sum(
        search * (np.log(np.maximum(search, 1e-100)) - np.log(np.maximum(nn_wdl, 1e-100)))
    )
    return min(max(0.0, float(surprise)), 1.0)


def redistribute_surprise_weights(samples, policy_data_weight, value_data_weight):
    sum_weights = sum(sample["weight"] for sample in samples)
    if sum_weights < 1.0:
        return
    sum_policy = sum(sample["policy_surprise"] * sample["weight"] for sample in samples)
    sum_value = sum(sample["value_surprise"] * sample["weight"] for sample in samples)
    average_policy = sum_policy / sum_weights
    average_value = sum_value / sum_weights
    if average_value < 0.010:
        value_data_weight *= average_value / 0.010
    threshold = average_policy * 1.5
    policy_props = [
        sample["weight"] * sample["policy_surprise"]
        + (1.0 - sample["weight"]) * max(0.0, sample["policy_surprise"] - threshold)
        for sample in samples
    ]
    value_props = [sample["weight"] * sample["value_surprise"] for sample in samples]
    sum_policy_prop = max(sum(policy_props), 1e-10)
    sum_value_prop = max(sum(value_props), 1e-10)
    for i, sample in enumerate(samples):
        sample["weight"] = (
            (1.0 - policy_data_weight - value_data_weight) * sample["weight"]
            + policy_data_weight * policy_props[i] * sum_weights / sum_policy_prop
            + value_data_weight * value_props[i] * sum_weights / sum_value_prop
        )


def soft_policy_target(policy, temperature):
    soft = np.power(policy, 1.0 / temperature)
    total = soft.sum()
    if total > 0:
        soft = soft / total
    return soft


def finish_game_samples(memory, winner, game, args):
    future_wdl = value_target(winner, 1).astype(np.float64)
    now_factor = 1.0 / (1.0 + game.board_size ** 2 * 0.016)
    for sample in reversed(memory):
        search_wdl = sample["search_wdl"]
        nn_wdl = sample["nn_wdl"]
        if sample["to_play"] == -1:
            search_wdl = search_wdl[::-1]
            nn_wdl = nn_wdl[::-1]
        future_wdl = future_wdl + now_factor * (search_wdl - future_wdl)
        sample["value_surprise"] = value_surprise(future_wdl, nn_wdl)

    redistribute_surprise_weights(
        memory,
        args.get("policy_surprise_data_weight", 0.5),
        args.get("value_surprise_data_weight", 0.1),
    )

    temperature = args.get("soft_policy_temperature", 4.0)
    action_size = game.board_size ** 2
    samples = []
    for index, sample in enumerate(memory):
        weight = max(0.0, sample["weight"])
        count = int(weight)
        if np.random.random() < weight - count:
            count += 1
        if count == 0:
            continue
        policy_target = sample["mcts_policy"]
        if index + 1 < len(memory):
            opponent_policy = memory[index + 1]["mcts_policy"]
            opponent_weight = 1.0
        else:
            opponent_policy = np.zeros(action_size)
            opponent_weight = 0.0
        row = {
            "encoded_state": game.encode_state(sample["state"], sample["to_play"]),
            "policy_target": policy_target,
            "opponent_policy": opponent_policy,
            "policy_target_soft": soft_policy_target(policy_target, temperature),
            "opponent_policy_soft": soft_policy_target(opponent_policy, temperature),
            "opponent_weight": opponent_weight,
            "value_target": value_target(winner, sample["to_play"]),
        }
        samples.extend(row.copy() for _ in range(count))
    return samples


def lcb_play_selection(root, action_size, args):
    weights = np.zeros(action_size)
    for child in root.children:
        weights[child.action_taken] = child.visits
    if not args.get("use_lcb_for_selection", True):
        total = weights.sum()
        if total > 0:
            weights /= total
        return weights

    children = root.children
    num = len(children)
    if num == 0:
        return weights

    utility_radius = 1.0
    lcb_stdevs = args.get("lcb_stdevs", 5.0)
    min_visit_prop = args.get("min_visit_prop_for_lcb", 0.15)
    use_non_buggy = args.get("use_non_buggy_lcb", True)

    zero_radius = 2.0 * utility_radius * lcb_stdevs
    radius = [zero_radius] * num
    lcb = [-zero_radius] * num
    for i, child in enumerate(children):
        visits = child.visits
        if visits <= 0:
            continue
        avg = float((child.wdl_sum[0] - child.wdl_sum[2]) / visits)
        sq = float(child.utility_sq_sum / visits)
        weight_sum = float(visits)
        weight_sq_sum = float(visits)
        ess = weight_sum * weight_sum / weight_sq_sum
        prior_weight = weight_sum / (ess * ess * ess)
        sq = max(sq, avg * avg)
        sq = (sq * weight_sum + (sq + utility_radius * utility_radius) * prior_weight) / (weight_sum + prior_weight)
        weight_sum += prior_weight
        weight_sq_sum += prior_weight * prior_weight
        ess = weight_sum * weight_sum / weight_sq_sum
        variance = max(0.0, sq - avg * avg)
        radius[i] = lcb_stdevs * math.sqrt(variance / ess)
        lcb[i] = -avg - radius[i]

    best_goodness = -1e30
    non_lcb_best_weight = -1e30
    for child in children:
        weight = float(child.visits)
        goodness = (
            weight * max(0.0, weight - 1.0) / max(1.0, weight)
            + 2.0 * float(child.prior)
        )
        if goodness > best_goodness:
            best_goodness = goodness
            non_lcb_best_weight = weight

    best_lcb = -1e10
    best_lcb_idx = -1
    for i, child in enumerate(children):
        weight = float(child.visits)
        if weight > 0 and weight >= min_visit_prop * non_lcb_best_weight:
            if lcb[i] > best_lcb:
                best_lcb = lcb[i]
                best_lcb_idx = i

    eligible = best_lcb_idx >= 0 if use_non_buggy else best_lcb_idx > 0
    if eligible:
        adjusted = float(children[best_lcb_idx].visits)
        for i, child in enumerate(children):
            if i == best_lcb_idx:
                continue
            excess = best_lcb - lcb[i]
            if excess < 0:
                continue
            r = radius[i]
            factor = (r + excess) / (r + 0.20 * excess)
            lbound = factor * factor * float(child.visits)
            if lbound > adjusted:
                adjusted = lbound
        weights[children[best_lcb_idx].action_taken] = adjusted

    total = weights.sum()
    if total > 0:
        weights /= total
    return weights


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


def add_dirichlet_noise(policy, total_concentration, legal_actions_mask, board_size, noise_weight=0.25):
    """
    训练时 在根节点策略中混入 Dirichlet Noise 以鼓励探索：

        noisy_policy = (1 - noise_weight) * policy + noise_weight * noise
    
    其中 total_concentration 一般可以设置为 0.03 * board_size^2
    noise_weight 一般是 0.25
    
    只给合法动作加噪声，非法位置保持为 0。alpha 总量为 total_concentration：一半平均分给合法动作，
    另一半按先验取对数后的形状分配（只保留高于均值的那部分）。对数先验封顶为
    0.01 * (19 / board_size)^2，使 19x19 上等价于 KataGo 的 0.01。
    """
    legal_actions_count = int(np.sum(legal_actions_mask))
    if legal_actions_count <= 1:
        return policy
    legal_policy = policy[legal_actions_mask]
    prior_cap = 0.01 * (19.0 / board_size) ** 2
    log_policy = np.log(np.minimum(prior_cap, legal_policy) + 1e-20)
    shaped = np.maximum(0.0, log_policy - log_policy.mean())
    shaped_sum = shaped.sum()
    uniform = 1.0 / legal_actions_count
    if shaped_sum <= 0.0:
        proportions = np.full(legal_actions_count, uniform)
    else:
        proportions = 0.5 * (shaped / shaped_sum + uniform)
    noise = np.random.dirichlet(proportions * total_concentration)
    noisy_policy = policy.copy()
    noisy_policy[legal_actions_mask] = (
        (1 - noise_weight) * legal_policy + noise_weight * noise
    )
    return noisy_policy


def random_augment_batch(batch, board_size):
    """
    对整个 batch 应用同一个随机对称变换 让网络学会对称性
    """
    states = np.stack([sample["encoded_state"] for sample in batch])
    policy_planes = np.stack(
        [
            np.stack(
                [
                    sample["policy_target"],
                    sample["opponent_policy"],
                    sample["policy_target_soft"],
                    sample["opponent_policy_soft"],
                ]
            ).reshape(4, board_size, board_size)
            for sample in batch
        ]
    )

    k = np.random.randint(0, 4)
    flip = np.random.choice([True, False])

    states = np.rot90(states, k, axes=(2, 3))
    policy_planes = np.rot90(policy_planes, k, axes=(2, 3))
    if flip:
        states = np.flip(states, axis=3)
        policy_planes = np.flip(policy_planes, axis=3)

    augmented_batch = []
    for sample, state, planes in zip(batch, states, policy_planes):
        new_sample = sample.copy()
        new_sample["encoded_state"] = np.ascontiguousarray(state)
        new_sample["policy_target"] = planes[0].flatten()
        new_sample["opponent_policy"] = planes[1].flatten()
        new_sample["policy_target_soft"] = planes[2].flatten()
        new_sample["opponent_policy_soft"] = planes[3].flatten()
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

import math

import numpy as np
import torch

from .utils import (
    add_dirichlet_noise,
    apply_temperature,
    root_policy_temperature,
    softmax,
    value_target,
)


class Node:
    def __init__(self, state, to_play, prior=0.0, parent=None, action_taken=None):
        self.state = state
        self.to_play = to_play
        self.prior = prior
        self.parent = parent
        self.action_taken = action_taken
        self.children = []
        self.wdl_sum = np.zeros(3)
        self.utility_sq_sum = 0.0
        self.visits = 0
        self.prior_policy = None
        self.nn_wdl = None

    def update(self, wdl):
        self.wdl_sum += wdl
        self.utility_sq_sum += (wdl[0] - wdl[2]) ** 2
        self.visits += 1

    def q_value(self):
        if self.visits == 0:
            return 0.0
        return float((self.wdl_sum[0] - self.wdl_sum[2]) / self.visits)


class MCTS:
    def __init__(self, game, args, model, device):
        self.game = game
        self.args = args
        self.model = model.to(device).eval()
        self.device = device

    @torch.inference_mode()
    def nn_inference(self, state, to_play):
        # state, to_play -> encoded_state ---NeuralNetwork---> policy, value
        encoded = self.game.encode_state(state, to_play)
        tensor = torch.tensor(encoded, dtype=torch.float32, device=self.device).unsqueeze(0)
        policy_logits, value_logits = self.model(tensor)
        policy_logits = policy_logits[0, 0].cpu().numpy()
        value_probs = softmax(value_logits.flatten().float().cpu().numpy())

        policy_logits = self.game.mask_illegal_actions(state, to_play, policy_logits)
        return softmax(policy_logits), value_probs

    def select(self, node, is_root, cheap):
        # 选择 PUCT值 最大的节点
        c_puct = self.args.get("c_puct", 1.5)
        if cheap or not is_root:
            reduction_max = self.args.get("fpu_reduction_max", 0.2)
        else:
            reduction_max = self.args.get("root_fpu_reduction_max", 0.0)
        policy_mass_visited = 0.0
        for child in node.children:
            if child.visits > 0:
                policy_mass_visited += child.prior
        mix = min(1.0, policy_mass_visited ** 2)
        nn_value = float(node.nn_wdl[0] - node.nn_wdl[2])
        base_value = mix * node.q_value() + (1.0 - mix) * nn_value
        fpu_value = base_value - reduction_max * math.sqrt(policy_mass_visited)
        best_score = -float("inf")
        best_child = None
        for child in node.children:
            if child.visits == 0:
                value = fpu_value
            else:
                value = -child.q_value()
            score = value + c_puct * child.prior * math.sqrt(node.visits) / (1 + child.visits)
            if score > best_score:
                best_score = score
                best_child = child
        return best_child

    def expand(self, node, policy):
        # 按照 NN Policy 展开该叶节点的所有合法子节点
        legal_actions_mask = self.game.get_legal_action_mask(node.state, node.to_play)
        for action in np.flatnonzero(legal_actions_mask):
            next_state = self.game.get_next_state(node.state, action, node.to_play)
            node.children.append(
                Node(
                    next_state,
                    -node.to_play,
                    prior=policy[action],
                    parent=node,
                    action_taken=action
                )
            )

    def backpropagate(self, node, wdl):
        # 沿路径回传 Value 并更新统计信息
        while node is not None:
            node.update(wdl)
            wdl = wdl[::-1]
            node = node.parent

    def advance(self, root, action):
        if root is None:
            return None
        for i, child in enumerate(root.children):
            if child.action_taken == action:
                child.parent = None
                root.children = []
                return child
        return None

    @torch.inference_mode()
    def search(self, state, to_play, num_simulations, turn_number, cheap, root=None):

        if root is None:
            policy, value = self.nn_inference(state, to_play)

            if self.args.get("mode", "train") == "eval" and num_simulations == 0:
                return policy, float(value[0] - value[2]), None

            root = Node(state, to_play)

            if self.args.get("mode", "train") == "train" and not cheap:
                policy = apply_temperature(
                    policy,
                    root_policy_temperature(self.args, turn_number, self.game.board_size),
                )
                policy = add_dirichlet_noise(
                    policy,
                    self.args.get("dirichlet_total_concentration", 0.03 * self.game.board_size ** 2),
                    legal_actions_mask=self.game.get_legal_action_mask(state, to_play),
                    board_size=self.game.board_size,
                    noise_weight=self.args.get("dirichlet_noise_weight", 0.25),
                )

            root.prior_policy = policy
            root.nn_wdl = value
            self.expand(root, policy)
            self.backpropagate(root, value)
            remaining = num_simulations
        else:
            remaining = max(0, num_simulations + 1 - root.visits)

        for _ in range(remaining):
            node = root
            while node.children:
                node = self.select(node, node is root, cheap)

            if self.game.is_terminal(node.state, node.to_play):
                value = value_target(self.game.get_winner(node.state, node.to_play), node.to_play)
            else:
                policy, value = self.nn_inference(node.state, node.to_play)
                node.prior_policy = policy
                node.nn_wdl = value
                self.expand(node, policy)

            self.backpropagate(node, value)

        mcts_policy = np.zeros(self.game.board_size ** 2)
        for child in root.children:
            mcts_policy[child.action_taken] = child.visits
        mcts_policy /= np.sum(mcts_policy)
        return mcts_policy, root.q_value(), root

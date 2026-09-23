import math

import numpy as np
import torch

from .utils import add_dirichlet_noise, softmax


class Node:
    def __init__(self, to_play, prior=0.0, parent=None, action_taken=None):
        self.hidden_state = None
        self.to_play = to_play
        self.prior = prior
        self.parent = parent
        self.action_taken = action_taken
        self.children = []
        self.value_sum = 0.0
        self.visits = 0

    def update(self, value):
        self.value_sum += value
        self.visits += 1

    def q_value(self):
        if self.visits == 0:
            return 0.0
        return self.value_sum / self.visits


class MCTS:
    def __init__(self, game, args, model, device):
        self.game = game
        self.args = args
        self.model = model.to(device).eval()
        self.device = device

    @torch.inference_mode()
    def representation(self, observation):
        tensor = torch.tensor(observation, dtype=torch.float32, device=self.device).unsqueeze(0)
        return self.model.representation(tensor)

    @torch.inference_mode()
    def prediction(self, hidden_state):
        policy_logits, value = self.model.prediction(hidden_state)
        policy_logits = policy_logits.flatten().float().cpu().numpy()
        return policy_logits, float(value.item())

    @torch.inference_mode()
    def dynamics(self, hidden_state, action, to_play):
        action_tensor = torch.tensor([action], dtype=torch.long, device=self.device)
        to_play_tensor = torch.tensor([to_play], dtype=torch.float32, device=self.device)
        return self.model.dynamics(hidden_state, action_tensor, to_play_tensor)

    def select(self, node):
        # 选择 PUCT值 最大的节点
        c_puct = self.args.get("c_puct", 1.5)
        best_score = -float("inf")
        best_child = None
        for child in node.children:
            score = -child.q_value() + c_puct * child.prior * math.sqrt(node.visits) / (1 + child.visits)
            if score > best_score:
                best_score = score
                best_child = child
        return best_child

    def expand(self, node, policy):
        # 隐状态节点按完整动作空间展开，不查询真实棋盘的合法动作
        for action in range(self.game.board_size ** 2):
            node.children.append(
                Node(
                    -node.to_play,
                    prior=policy[action],
                    parent=node,
                    action_taken=action,
                )
            )

    def backpropagate(self, node, value):
        # 沿路径回传 Value 并更新统计信息
        while node is not None:
            node.update(value)
            value = -value
            node = node.parent

    @torch.inference_mode()
    def search(self, state, to_play, num_simulations):

        observation = self.game.encode_state(state, to_play)
        root = Node(to_play)
        root.hidden_state = self.representation(observation)
        policy_logits, value = self.prediction(root.hidden_state)

        if self.args.get("mode", "train") == "eval" and num_simulations == 0:
            policy_logits = self.game.mask_illegal_actions(state, to_play, policy_logits)
            return softmax(policy_logits), value

        # 根节点面对真实棋盘，用环境给出的合法动作屏蔽先验
        legal_actions_mask = self.game.get_legal_action_mask(state, to_play)
        policy_logits = self.game.mask_illegal_actions(state, to_play, policy_logits)
        policy = softmax(policy_logits)

        if self.args.get("mode", "train") == "train":
            policy = add_dirichlet_noise(
                policy,
                self.args.get("dirichlet_total_concentration", 0.03 * self.game.board_size ** 2),
                legal_actions_mask=legal_actions_mask,
                noise_weight=self.args.get("dirichlet_noise_weight", 0.25),
            )

        for action in np.flatnonzero(legal_actions_mask):
            action = int(action)
            root.children.append(
                Node(-to_play, prior=policy[action], parent=root, action_taken=action)
            )
        self.backpropagate(root, value)

        for _ in range(num_simulations):
            node = root
            while node.children:
                node = self.select(node)

            node.hidden_state = self.dynamics(node.parent.hidden_state, node.action_taken, node.to_play)
            policy_logits, value = self.prediction(node.hidden_state)
            self.expand(node, softmax(policy_logits))
            self.backpropagate(node, value)

        mcts_policy = np.zeros(self.game.board_size ** 2)
        for child in root.children:
            mcts_policy[child.action_taken] = child.visits
        mcts_policy /= np.sum(mcts_policy)
        return mcts_policy, root.q_value()

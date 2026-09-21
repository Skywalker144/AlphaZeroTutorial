import math

import numpy as np
import torch

from .utils import add_dirichlet_noise, softmax


class Node:
    def __init__(self, state, to_play, prior=0.0, parent=None, action_taken=None):
        self.state = state
        self.to_play = to_play
        self.prior = prior
        self.parent = parent
        self.action_taken = action_taken
        self.children = []
        self.value = 0.0
        self.visits = 0

    def update(self, value):
        self.value += value
        self.visits += 1

    def q_value(self):
        if self.n == 0:
            return 0
        return -self.value / self.visits


class MCTS:
    def __init__(self, game, args, model, device):
        self.game = game
        self.args = args
        self.model = model.to(device).eval()
        self.device = device

    @torch.inference_mode()
    def _inference(self, state, to_play):
        encoded = self.game.encode_state(state, to_play)
        tensor = torch.tensor(encoded, dtype=torch.float32, device=self.device).unsqueeze(0)
        policy_logits, value = self.model(tensor)
        policy_logits = policy_logits.flatten().cpu().numpy()

        legal_mask = self.game.get_is_legal_actions(state, to_play)
        policy_logits = np.where(legal_mask, policy_logits, -1e10)
        return softmax(policy_logits), float(value.item())

    def select(self, node):
        c_puct = self.args.get("c_puct", 1.5)
        best_score = -float("inf")
        best_child = None
        for child in node.children:
            score = child.q_value() + c_puct * child.prior * math.sqrt(node.n) / (1 + child.n)
            if score > best_score:
                best_score = score
                best_child = child
        return best_child

    def expand(self, node, policy):
        for action, prob in enumerate(policy):
            if prob > 0:
                next_state = self.game.get_next_state(node.state, action, node.to_play)
                node.children.append(
                    Node(next_state, -node.to_play, prior=prob, parent=node, action_taken=action)
                )

    def backpropagate(self, node, value):
        while node is not None:
            node.update(value)
            value = -value
            node = node.parent

    @torch.inference_mode()
    def search(self, state, to_play, num_simulations):
        root = Node(state, to_play)

        policy, value = self._inference(state, to_play)
        policy = add_dirichlet_noise(
            policy,
            self.args.get("dirichlet_total_concentration", 0.03 * self.game.board_size ** 2),
            self.args.get("dirichlet_noise_weight", 0.25),
        )
        self.expand(root, policy)
        self.backpropagate(root, value)

        for _ in range(num_simulations - 1):
            node = root
            while node.is_expanded():
                node = self.select(node)

            if self.game.is_terminal(node.state):
                value = self.game.get_winner(node.state) * node.to_play
            else:
                policy, value = self._inference(node.state, node.to_play)
                self.expand(node, policy)

            self.backpropagate(node, value)

        mcts_policy = np.zeros(self.game.board_size ** 2)
        for child in root.children:
            mcts_policy[child.action_taken] = child.n
        mcts_policy /= np.sum(mcts_policy)
        return mcts_policy

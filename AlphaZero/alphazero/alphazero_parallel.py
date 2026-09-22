import math

import numpy as np
import torch

from .mcts import Node
from .utils import add_dirichlet_noise, softmax


# -- 与 MCTS 等价的树操作（自由函数版，供多棵树复用） ----------------------


def _select(node, args):
    """选择 PUCT 值最大的子节点（与 MCTS.select 一致）。"""
    c_puct = args.get("c_puct", 1.5)
    best_score = -float("inf")
    best_child = None
    for child in node.children:
        score = -child.q_value() + c_puct * child.prior * math.sqrt(node.visits) / (1 + child.visits)
        if score > best_score:
            best_score = score
            best_child = child
    return best_child


def _expand(node, policy, game):
    """按照 NN Policy 展开叶节点的所有合法子节点（与 MCTS.expand 一致）。"""
    legal_actions_mask = game.get_legal_action_mask(node.state, node.to_play)
    for action in np.flatnonzero(legal_actions_mask):
        next_state = game.get_next_state(node.state, action, node.to_play)
        node.children.append(
            Node(
                next_state,
                -node.to_play,
                prior=policy[action],
                parent=node,
                action_taken=action,
            )
        )


def _backpropagate(node, value):
    """沿路径回传 value 并翻转视角（与 MCTS.backpropagate 一致）。"""
    while node is not None:
        node.update(value)
        value = -value
        node = node.parent


# -- 单局的搜索 / 对局状态机 ------------------------------------------------


class _Search:
    """一局中"当前这一步棋"的 MCTS，可暂停在一个等待评估的叶子上。

    对应串行版 ``MCTS.search`` 的一次调用：先评估根节点（不计入模拟数），
    再做 ``num_simulations`` 次模拟，每次模拟走到叶子——终局叶子直接回传，
    非终局叶子挂起等待 NN 评估。
    """

    def __init__(self, state, to_play, num_simulations):
        self.root = Node(state, to_play)
        self.num_simulations = num_simulations
        self.simulations = 0
        self.pending = None  # 等待 NN 评估的节点（根节点评估也走这里）


class _GameSession:
    """一局完整对局的状态机：搜索 -> 落子 -> 搜索 -> ... -> 终局。

    对外只暴露两个入口：

    - ``advance(requests)``：推进到本局恰好需要一个 NN 评估（把待评估节点
      追加进 requests），或者本局下完（返回 True）。
    - ``deliver(node, policy, value)``：把批量推理的结果交给挂起的叶子，
      展开 + 回传后继续。
    """

    def __init__(self, game, args):
        self.game = game
        self.args = args
        self.state = game.get_initial_state()
        self.to_play = 1
        self.memory = []
        self.search = None
        self.result = None  # (samples, winner, game_len)

    # -- 搜索生命周期 -----------------------------------------------------

    def _num_simulations(self):
        return int(self.args.get("num_simulations", 1.7 * self.game.board_size ** 2))

    def _start_search(self):
        search = _Search(self.state, self.to_play, self._num_simulations())
        search.pending = search.root  # 根节点评估最先入队
        self.search = search
        return search.root

    def _finish_move(self):
        """搜索结束：记录样本、按温度选动作落子、判断终局。"""
        search = self.search
        mcts_policy = np.zeros(self.game.board_size ** 2)
        for child in search.root.children:
            mcts_policy[child.action_taken] = child.visits
        mcts_policy /= np.sum(mcts_policy)

        self.memory.append({
            "state": self.state,
            "to_play": self.to_play,
            "mcts_policy": mcts_policy,
        })

        half_life = self.args.get("half_life", self.game.board_size)
        if len(self.memory) < half_life:
            action = np.random.choice(len(mcts_policy), p=mcts_policy)
        else:
            action = int(np.argmax(mcts_policy))

        self.state = self.game.get_next_state(self.state, action, self.to_play)
        self.to_play = -self.to_play
        self.search = None

        if self.game.is_terminal(self.state, self.to_play):
            winner = self.game.get_winner(self.state, self.to_play)
            samples = [
                {
                    "encoded_state": self.game.encode_state(sample["state"], sample["to_play"]),
                    "policy_target": sample["mcts_policy"],
                    "value_target": float(winner) * sample["to_play"],
                }
                for sample in self.memory
            ]
            self.result = (samples, winner, len(self.memory))

    # -- 状态机入口 -------------------------------------------------------

    def advance(self, requests):
        """推进直到本局需要一个 NN 评估，或本局结束。

        需要评估时把 ``(self, node)`` 追加进 requests 并返回 False；
        本局结束时返回 True（结果在 ``self.result``）。
        """
        while self.result is None:
            if self.search is None:
                requests.append((self, self._start_search()))
                return False

            search = self.search
            if search.simulations >= search.num_simulations:
                # 本轮搜索已完成，落子；若仍有棋可下就开下一次搜索
                self._finish_move()
                if self.result is not None:
                    return True
                requests.append((self, self._start_search()))
                return False

            node = search.root
            while node.children:
                node = _select(node, self.args)

            # 一次模拟 = 走到一个叶子；终局叶子直接回传，非终局叶子挂起等
            # 批量评估（评估由 deliver 完成），两种情况都计一次模拟。
            search.simulations += 1
            if self.game.is_terminal(node.state, node.to_play):
                value = self.game.get_winner(node.state, node.to_play) * node.to_play
                _backpropagate(node, value)
                continue

            search.pending = node
            requests.append((self, node))
            return False
        return True

    def deliver(self, node, policy, value):
        """把批量推理结果交给挂起的节点：根节点加噪声，然后展开 + 回传。"""
        search = self.search
        if node is search.root:
            policy = add_dirichlet_noise(
                policy,
                self.args.get("dirichlet_total_concentration", 0.03 * self.game.board_size ** 2),
                legal_actions_mask=self.game.get_legal_action_mask(node.state, node.to_play),
                noise_weight=self.args.get("dirichlet_noise_weight", 0.25),
            )
        _expand(node, policy, self.game)
        _backpropagate(node, value)
        search.pending = None


# -- 协调器 -----------------------------------------------------------------


class ParallelSelfPlayer:
    """并行 selfplay 后端：跨对局合并 batch 推理。

    用法::

        player = ParallelSelfPlayer(game, args, model, device)
        for samples, winner, game_len in player.run(num_games):
            replay_buffer.add_game(samples)

    yield 的三元组与 ``AlphaZero.selfplay()`` 的返回值格式完全一致，按
    对局完成（而非开始）的顺序产出。

    参数（均从 ``args`` 读取）：

    parallel
        选择后端的开关，默认 True（使用本并行后端）；
        设为 False 退回串行 selfplay（见 trainer.AlphaZero）。
    num_parallel_games
        同时保持活跃的对局数，也约等于每轮批量推理的 batch 大小。默认 32。
    其余搜索相关参数（num_simulations / c_puct / half_life /
    dirichlet_total_concentration）含义与串行版完全相同。
    """

    def __init__(self, game, args, model, device):
        self.game = game
        self.args = args
        self.model = model.to(device).eval()
        self.device = device
        self.num_parallel_games = max(1, int(args.get("num_parallel_games", 32)))

    @torch.inference_mode()
    def _batch_inference(self, nodes):
        """把多个待评估节点合并成一次前向，返回 [(policy, value), ...]。

        与 ``MCTS.nn_inference`` 语义一致：先 mask 非法动作再 softmax。
        """
        encoded = np.stack([self.game.encode_state(node.state, node.to_play) for node in nodes])
        tensor = torch.from_numpy(encoded).to(dtype=torch.float32, device=self.device)
        policy_logits, values = self.model(tensor)

        logits_batch = policy_logits.reshape(len(nodes), -1).cpu().numpy()
        values_batch = values.reshape(-1).cpu().numpy()
        results = []
        for node, logits, value in zip(nodes, logits_batch, values_batch):
            logits = self.game.mask_illegal_actions(node.state, node.to_play, logits)
            results.append((softmax(logits), float(value)))
        return results

    def run(self, total_games):
        """收集 ``total_games`` 局，按完成顺序 yield (samples, winner, game_len)。"""
        active = []
        started = 0
        while started < total_games or active:
            # 池子持续补充：谁下完了立刻开新局，直到攒够 total_games
            while len(active) < self.num_parallel_games and started < total_games:
                active.append(_GameSession(self.game, self.args))
                started += 1

            # 每个活跃对局推进到恰好需要一个评估（或下完）
            requests = []
            for session in active:
                session.advance(requests)

            finished = [s for s in active if s.result is not None]
            active = [s for s in active if s.result is None]

            # 所有挂起请求合并成一次前向，再分发回去
            if requests:
                results = self._batch_inference([node for _, node in requests])
                for (session, node), (policy, value) in zip(requests, results):
                    session.deliver(node, policy, value)

            for session in finished:
                yield session.result

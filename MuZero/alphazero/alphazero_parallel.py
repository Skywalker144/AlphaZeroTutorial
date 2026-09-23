import gc
import math

import numpy as np
import torch

from .utils import add_dirichlet_noise, softmax


class _Node:
    """与 ``mcts.Node`` 接口一致的轻量节点。

    - 用 ``__slots__`` 去掉实例 ``__dict__``，MCTS 展开会产生海量节点。
    - ``state`` 只有根节点持有（真实棋盘）；树节点是隐状态内部节点，没有
      对应的真实棋盘，因此为 ``None``。这也意味着树内不做终局判定，与串行
      ``MCTS`` 一致。

    树操作依赖的字段逐字段语义与 ``mcts.Node`` 完全相同。
    """

    __slots__ = (
        "state",
        "hidden_state",
        "to_play",
        "prior",
        "parent",
        "action_taken",
        "children",
        "value_sum",
        "visits",
    )

    def __init__(self, state, to_play, prior=0.0, parent=None, action_taken=None):
        self.state = state
        self.hidden_state = None
        self.to_play = to_play
        self.prior = prior
        self.parent = parent
        self.action_taken = action_taken
        self.children = []
        self.value_sum = 0.0
        self.visits = 0


# -- 与 MCTS 等价的树操作（自由函数版，供多棵树复用） ----------------------


def _select(node, pb_c_base, pb_c_init):
    """选择 PUCT 值最大的子节点（与 MCTS.select 一致）。"""
    pb_c = math.log((node.visits + pb_c_base + 1) / pb_c_base) + pb_c_init
    pb_c *= math.sqrt(node.visits)
    best_score = -float("inf")
    best_child = None
    for child in node.children:
        visits = child.visits
        value = (1.0 - child.value_sum / visits) / 2.0 if visits else 0.0
        score = value + pb_c * child.prior / (1 + visits)
        if score > best_score:
            best_score = score
            best_child = child
    return best_child


def _expand(node, policy, actions):
    """按给定动作集合展开子节点。

    根节点传入合法动作，树节点传入完整动作空间（与 MCTS.expand / search
    中根的展开逻辑一致）。
    """
    to_play = -node.to_play
    children = node.children
    for action in actions:
        action = int(action)
        children.append(_Node(None, to_play, policy[action], node, action))


def _backpropagate(node, value):
    """沿路径回传 value 并翻转视角（与 MCTS.backpropagate 一致）。"""
    while node is not None:
        node.value_sum += value
        node.visits += 1
        value = -value
        node = node.parent


def _discard_tree(root):
    """丢弃一棵搜索树：断开 ``parent`` 反向引用，消除父子引用环。"""
    stack = [root]
    while stack:
        node = stack.pop()
        stack.extend(node.children)
        node.parent = None


# -- 单局的搜索 / 对局状态机 ------------------------------------------------


class _Search:
    """一局中"当前这一步棋"的 MCTS，可暂停在一个等待评估的叶子上。

    对应串行版 ``MCTS.search`` 的一次调用：先评估根节点（不计入模拟数），
    再做 ``num_simulations`` 次模拟，每次模拟走到一个叶子挂起等待 NN 评估。
    """

    __slots__ = ("root", "num_simulations", "simulations", "pending")

    def __init__(self, root, num_simulations):
        self.root = root
        self.num_simulations = num_simulations
        self.simulations = 0
        self.pending = None


class _GameSession:
    """一局完整对局的状态机：搜索 -> 落子 -> 搜索 -> ... -> 终局。

    对外只暴露两个入口：

    - ``advance(requests)``：推进到本局恰好需要一个 NN 评估（把待评估节点
      追加进 requests），或者本局下完（返回 True）。
    - ``deliver(node, policy_logits, value)``：把批量推理的结果交给挂起的
      叶子，展开 + 回传后继续。
    """

    __slots__ = (
        "game",
        "args",
        "action_size",
        "pb_c_base",
        "pb_c_init",
        "num_simulations",
        "mode",
        "half_life",
        "dirichlet_concentration",
        "dirichlet_noise_weight",
        "state",
        "to_play",
        "memory",
        "search",
        "result",
    )

    def __init__(self, game, args):
        self.game = game
        self.args = args
        # 每局固定不变的超参数在构造时解析一次，避免每次模拟都查 args。
        self.action_size = game.board_size ** 2
        self.pb_c_base = args.get("pb_c_base", 19652)
        self.pb_c_init = args.get("pb_c_init", 1.25)
        self.num_simulations = int(
            args.get("num_simulations", 1.7 * game.board_size ** 2)
        )
        self.mode = args.get("mode", "train")
        self.half_life = args.get("half_life", game.board_size)
        self.dirichlet_concentration = args.get(
            "dirichlet_total_concentration", 0.03 * game.board_size ** 2
        )
        self.dirichlet_noise_weight = args.get("dirichlet_noise_weight", 0.25)

        self.state = game.get_initial_state()
        self.to_play = 1
        self.memory = []
        self.search = None
        self.result = None  # (memory, winner, game_len)

    # -- 搜索生命周期 -----------------------------------------------------

    def _start_search(self):
        root = _Node(self.state, self.to_play)
        search = _Search(root, self.num_simulations)
        search.pending = root  # 根节点评估最先入队
        self.search = search
        return root

    def _finish_move(self):
        """搜索结束：按温度选动作落子、记录样本、判断终局。"""
        game = self.game
        search = self.search
        root = search.root
        raw_policy = np.zeros(self.action_size)
        for child in root.children:
            raw_policy[child.action_taken] = child.visits
        raw_policy /= np.sum(raw_policy)

        if len(self.memory) + 1 < self.half_life:
            action = int(np.random.choice(self.action_size, p=raw_policy))
        else:
            action = int(np.argmax(raw_policy))

        self.memory.append({
            "observation": game.encode_state(self.state, self.to_play),
            "player": self.to_play,
            "action": action,
            "mcts_policy": raw_policy,
        })

        self.state = game.get_next_state(self.state, action, self.to_play)
        self.to_play = -self.to_play
        _discard_tree(root)
        self.search = None

        if game.is_terminal(self.state, self.to_play):
            winner = game.get_winner(self.state, self.to_play)
            for step in self.memory:
                step["value_target"] = float(winner) * step["player"]
            self.result = (self.memory, winner, len(self.memory))

    # -- 状态机入口 -------------------------------------------------------

    def advance(self, requests):
        """推进直到本局需要一个 NN 评估，或本局结束。"""
        while self.result is None:
            if self.search is None:
                root = self._start_search()
                requests.append((self, root))
                return False

            search = self.search
            if search.simulations >= search.num_simulations:
                self._finish_move()
                if self.result is not None:
                    return True
                root = self._start_search()
                requests.append((self, root))
                return False

            node = search.root
            while node.children:
                node = _select(node, self.pb_c_base, self.pb_c_init)
            # 一次模拟 = 走到一个叶子挂起等待批量评估，并计一次模拟。
            search.simulations += 1
            search.pending = node
            requests.append((self, node))
            return False
        return True

    def deliver(self, node, policy_logits, value):
        """把批量推理结果交给挂起的节点：根节点 mask + 噪声，然后展开 + 回传。"""
        game = self.game
        if node.parent is None:
            legal_actions_mask = game.get_legal_action_mask(node.state, node.to_play)
            policy_logits = np.where(legal_actions_mask, policy_logits, -np.inf)
            policy = softmax(policy_logits)
            if self.mode == "train":
                policy = add_dirichlet_noise(
                    policy,
                    self.dirichlet_concentration,
                    legal_actions_mask=legal_actions_mask,
                    noise_weight=self.dirichlet_noise_weight,
                )
            actions = np.flatnonzero(legal_actions_mask).tolist()
        else:
            policy = softmax(policy_logits)
            actions = range(self.action_size)
        _expand(node, policy, actions)
        _backpropagate(node, value)
        self.search.pending = None


# -- 协调器 -----------------------------------------------------------------


class ParallelSelfPlayer:
    """并行 selfplay 后端：跨对局合并 batch 推理。

    用法::

        player = ParallelSelfPlayer(game, args, model, device)
        for samples, winner, game_len in player.run(num_games):
            replay_buffer.add_game(samples)

    yield 的三元组与 ``MuZero.selfplay()`` 的返回值格式完全一致，按对局完成
    （而非开始）的顺序产出。

    与 KataGo 后端最大的不同在 ``_batch_inference``：MuZero 的每个待评估节点
    需要先经过 ``representation``（根）或 ``dynamics``（树节点）得到隐状态，再
    统一过 ``prediction``，因此推理分两段完成。

    参数（均从 ``args`` 读取）：

    parallel
        选择后端的开关，默认 True（使用本并行后端）；
        设为 False 退回串行 selfplay（见 trainer.MuZero）。
    num_parallel_games
        同时保持活跃的对局数，也约等于每轮批量推理的 batch 大小。默认 32。
    其余搜索相关参数（num_simulations / pb_c_base / pb_c_init / half_life /
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
        """把多个待评估节点合并成一次前向，返回 [(policy_logits, value), ...]。

        根节点用 ``representation`` 得到隐状态，树节点用 ``dynamics`` 从父节点
        隐状态 + 动作推进；两类隐状态拼成一个 batch 后统一过 ``prediction``。
        根节点的 mask / softmax / dirichlet 与树节点的 softmax 交给 ``deliver``，
        与串行 ``MCTS.search`` 的分工一致。
        """
        game = self.game
        n = len(nodes)
        hidden = [None] * n
        root_idx = [i for i, node in enumerate(nodes) if node.parent is None]
        tree_idx = [i for i, node in enumerate(nodes) if node.parent is not None]

        if root_idx:
            observations = np.stack(
                [game.encode_state(nodes[i].state, nodes[i].to_play) for i in root_idx]
            ).astype(np.float32)
            root_hidden = self.model.representation(
                torch.from_numpy(observations).to(self.device)
            )
            for k, i in enumerate(root_idx):
                hidden[i] = root_hidden[k:k + 1]

        if tree_idx:
            parent_hidden = torch.cat(
                [nodes[i].parent.hidden_state for i in tree_idx], dim=0
            )
            actions = torch.tensor(
                [nodes[i].action_taken for i in tree_idx],
                dtype=torch.long,
                device=self.device,
            )
            tree_hidden = self.model.dynamics(parent_hidden, actions)
            for k, i in enumerate(tree_idx):
                hidden[i] = tree_hidden[k:k + 1]

        hidden_batch = torch.cat(hidden, dim=0)
        policy_logits, value = self.model.prediction(hidden_batch)

        logits = policy_logits.reshape(n, -1).float().cpu().numpy().astype(np.float64)
        values = value.reshape(n).float().cpu().numpy()
        for i, node in enumerate(nodes):
            node.hidden_state = hidden[i]
        return [(logits[i], float(values[i])) for i in range(n)]

    def run(self, total_games):
        """收集 ``total_games`` 局，按完成顺序 yield (samples, winner, game_len)。"""
        game = self.game
        args = self.args
        num_parallel_games = self.num_parallel_games
        batch_inference = self._batch_inference

        # selfplay 阶段会产生海量短生命周期的树节点。树在丢弃前会用
        # ``_discard_tree`` 主动断环，可被引用计数立即释放，因此这里关掉循环
        # GC 不会泄漏，却能省掉反复扫描/晋升这些节点的巨大开销；退出时恢复。
        gc_was_enabled = gc.isenabled()
        if gc_was_enabled:
            gc.disable()
        try:
            active = []
            started = 0
            while started < total_games or active:
                # 池子持续补充：谁下完了立刻开新局，直到攒够 total_games
                while len(active) < num_parallel_games and started < total_games:
                    active.append(_GameSession(game, args))
                    started += 1

                # 每个活跃对局推进到恰好需要一个评估（或下完）
                requests = []
                for session in active:
                    session.advance(requests)

                finished = []
                remaining = []
                for session in active:
                    if session.result is None:
                        remaining.append(session)
                    else:
                        finished.append(session.result)
                active = remaining

                # 所有挂起请求合并成一次前向，再分发回去
                if requests:
                    results = batch_inference([node for _, node in requests])
                    for (session, node), (logits, value) in zip(requests, results):
                        session.deliver(node, logits, value)

                for result in finished:
                    yield result
        finally:
            if gc_was_enabled:
                gc.enable()

import gc
import math

import numpy as np
import torch

from .utils import (
    add_dirichlet_noise,
    apply_temperature,
    chosen_move_temperature,
    root_policy_temperature,
    search_visit_counts,
)


class _Node:
    """与 ``mcts.Node`` 接口一致的轻量节点。

    两处纯性能上的区别：

    - 用 ``__slots__`` 去掉实例 ``__dict__``：MCTS 展开会产生海量节点，
      构造与属性访问的开销非常可观。
    - ``state`` 可以是 ``None``，表示惰性棋局状态：子节点刚展开时并不立刻
      拷贝棋盘，等真正被访问（选择到 / 送去推理）时才由 ``_materialize``
      从父节点算出来。终局判定、展开逻辑使用的状态值完全一致，且
      ``get_next_state`` 是纯函数，因此搜索行为逐位不变。

    树操作依赖的字段逐字段语义与 ``mcts.Node`` 完全相同。
    """

    __slots__ = (
        "state",
        "to_play",
        "prior",
        "parent",
        "action_taken",
        "children",
        "value_sum",
        "visits",
        "legal_actions_mask",
    )

    def __init__(self, state, to_play, prior=0.0, parent=None, action_taken=None):
        self.state = state
        self.to_play = to_play
        self.prior = prior
        self.parent = parent
        self.action_taken = action_taken
        self.children = []
        self.value_sum = 0.0
        self.visits = 0
        self.legal_actions_mask = None


def _materialize(node, game):
    """返回 ``node`` 的棋局状态；惰性节点在首次访问时才计算并缓存。

    父节点一定已经物化（它先被选中并评估过才会展开），且 ``get_next_state``
    是纯函数，所以惰性求值的结果与展开时立即求值完全相同。
    """
    state = node.state
    if state is None:
        parent = node.parent
        state = game.get_next_state(parent.state, node.action_taken, parent.to_play)
        node.state = state
    return state


# -- 与 MCTS 等价的树操作（自由函数版，供多棵树复用） ----------------------


def _select(node, c_puct):
    """选择 PUCT 值最大的子节点（与 MCTS.select 一致）。"""
    sqrt_visits = math.sqrt(node.visits)
    best_score = -float("inf")
    best_child = None
    # 内联 q_value() 并只做一次 sqrt，避免每个子节点的方法调用与重复开方。
    # 逐项运算顺序与 MCTS.select 完全一致，保证浮点结果逐位相同。
    for child in node.children:
        visits = child.visits
        if visits:
            q = child.value_sum / visits
        else:
            q = 0
        score = -q + c_puct * child.prior * sqrt_visits / (1 + visits)
        if score > best_score:
            best_score = score
            best_child = child
    return best_child


def _expand(node, policy, game, legal_actions_mask=None):
    """按照 NN Policy 展开叶节点的所有合法子节点（与 MCTS.expand 一致）。

    子节点的 ``state`` 先留空，真正访问时再由 ``_materialize`` 计算，避免为
    那些永远不会被访问到的分支拷贝棋盘。
    """
    if legal_actions_mask is None:
        legal_actions_mask = game.get_legal_action_mask(_materialize(node, game), node.to_play)
    next_to_play = -node.to_play
    children = node.children
    # tolist() 把 numpy 标量转成 Python int，后续索引/divmod 更快。
    for action in np.flatnonzero(legal_actions_mask).tolist():
        children.append(_Node(None, next_to_play, policy[action], node, action))


def _backpropagate(node, value):
    """沿路径回传 value 并翻转视角（与 MCTS.backpropagate 一致）。"""
    while node is not None:
        node.value_sum += value
        node.visits += 1
        value = -value
        node = node.parent


def _discard_tree(root):
    """丢弃一棵搜索树：断开 ``parent`` 反向引用，消除父子引用环。

    ``children`` 与 ``parent`` 互相引用会形成环，Python 的循环 GC 必须反复
    扫描这些海量节点。搜索结束时主动断环后，整棵树可被引用计数立即释放，
    后续在 selfplay 热路径里关闭循环 GC 也不会造成内存泄漏。
    """
    stack = [root]
    while stack:
        node = stack.pop()
        stack.extend(node.children)
        node.parent = None


def _advance(root, action, game):
    children = root.children
    promoted = None
    for i, child in enumerate(children):
        if child.action_taken == action:
            promoted = child
            del children[i]
            break
    if promoted is None:
        _discard_tree(root)
        return None
    _materialize(promoted, game)
    promoted.parent = None
    root.children = []
    for child in children:
        _discard_tree(child)
    return promoted


# -- 单局的搜索 / 对局状态机 ------------------------------------------------


class _Search:
    """一局中"当前这一步棋"的 MCTS，可暂停在一个等待评估的叶子上。

    对应串行版 ``MCTS.search`` 的一次调用：先评估根节点（不计入模拟数），
    再做 ``num_simulations`` 次模拟，每次模拟走到叶子——终局叶子直接回传，
    非终局叶子挂起等待 NN 评估。
    """

    __slots__ = ("root", "num_simulations", "simulations", "pending")

    def __init__(self, root, num_simulations):
        self.root = root
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

    __slots__ = (
        "game",
        "args",
        "action_size",
        "c_puct",
        "num_simulations",
        "cheap_search_prob",
        "cheap_search_visits",
        "is_cheap",
        "carried_root",
        "turn_number",
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
        self.c_puct = args.get("c_puct", 1.5)
        self.num_simulations, self.cheap_search_visits = search_visit_counts(
            args, game.board_size
        )
        self.cheap_search_prob = args.get("cheap_search_prob", 0.75)
        self.is_cheap = False
        self.carried_root = None
        self.turn_number = 0
        self.dirichlet_concentration = args.get(
            "dirichlet_total_concentration", 0.03 * game.board_size ** 2
        )
        self.dirichlet_noise_weight = args.get("dirichlet_noise_weight", 0.25)

        self.state = game.get_initial_state()
        self.to_play = 1
        self.memory = []
        self.search = None
        self.result = None  # (samples, winner, game_len)

    # -- 搜索生命周期 -----------------------------------------------------

    def _start_search(self):
        self.is_cheap = np.random.random() < self.cheap_search_prob
        if self.is_cheap and self.carried_root is not None:
            root = self.carried_root
            self.carried_root = None
            num_simulations = max(0, self.cheap_search_visits + 1 - root.visits)
        else:
            if self.carried_root is not None:
                _discard_tree(self.carried_root)
                self.carried_root = None
            root = _Node(self.state, self.to_play)
            num_simulations = (
                self.cheap_search_visits if self.is_cheap else self.num_simulations
            )
        search = _Search(root, num_simulations)
        if not root.children:
            search.pending = root  # 根节点评估最先入队
        self.search = search
        return root

    def _finish_move(self):
        """搜索结束：记录样本、按温度选动作落子、判断终局。"""
        search = self.search
        game = self.game
        mcts_policy = np.zeros(self.action_size)
        for child in search.root.children:
            mcts_policy[child.action_taken] = child.visits
        mcts_policy /= np.sum(mcts_policy)

        if not self.is_cheap:
            self.memory.append({
                "state": self.state,
                "to_play": self.to_play,
                "mcts_policy": mcts_policy,
            })

        temperature = chosen_move_temperature(
            self.args, self.turn_number, self.game.board_size
        )
        action = np.random.choice(
            self.action_size, p=apply_temperature(mcts_policy, temperature)
        )

        self.state = game.get_next_state(self.state, action, self.to_play)
        self.to_play = -self.to_play
        self.turn_number += 1
        self.search = None

        if game.is_terminal(self.state, self.to_play):
            _discard_tree(search.root)
            winner = game.get_winner(self.state, self.to_play)
            encode_state = game.encode_state
            memory = self.memory
            samples = [
                {
                    "encoded_state": encode_state(sample["state"], sample["to_play"]),
                    "policy_target": sample["mcts_policy"],
                    "value_target": float(winner) * sample["to_play"],
                }
                for sample in memory
            ]
            self.result = (samples, winner, self.turn_number)
        else:
            # 提升落子对应的子节点供下一步复用，其余断环立即释放。
            self.carried_root = _advance(search.root, action, game)

    # -- 状态机入口 -------------------------------------------------------

    def advance(self, requests):
        """推进直到本局需要一个 NN 评估，或本局结束。

        需要评估时把 ``(self, node)`` 追加进 requests 并返回 False；
        本局结束时返回 True（结果在 ``self.result``）。
        """
        while self.result is None:
            if self.search is None:
                self._start_search()
                if self.search.pending is None:
                    continue
                requests.append((self, self.search.root))
                return False

            search = self.search
            if search.simulations >= search.num_simulations:
                # 本轮搜索已完成，落子；若仍有棋可下就开下一次搜索
                self._finish_move()
                if self.result is not None:
                    return True
                self._start_search()
                if self.search.pending is None:
                    continue
                requests.append((self, self.search.root))
                return False

            game = self.game
            c_puct = self.c_puct
            node = search.root
            while node.children:
                node = _select(node, c_puct)

            # 一次模拟 = 走到一个叶子；终局叶子直接回传，非终局叶子挂起等
            # 批量评估（评估由 deliver 完成），两种情况都计一次模拟。
            search.simulations += 1
            # 选中叶子时才真正算出它的棋局状态（惰性展开）。
            state = _materialize(node, game)
            if game.is_terminal(state, node.to_play):
                value = game.get_winner(state, node.to_play) * node.to_play
                _backpropagate(node, value)
                continue

            search.pending = node
            requests.append((self, node))
            return False
        return True

    def deliver(self, node, policy, value):
        """把批量推理结果交给挂起的节点：根节点加噪声，然后展开 + 回传。"""
        search = self.search
        # 合法动作 mask 在批量推理时已经算过并挂在 node 上，这里直接复用。
        legal_actions_mask = getattr(node, "legal_actions_mask", None)
        if legal_actions_mask is None:
            game = self.game
            legal_actions_mask = game.get_legal_action_mask(
                _materialize(node, game), node.to_play
            )
        if (
            node is search.root
            and self.args.get("mode", "train") == "train"
            and not self.is_cheap
        ):
            policy = apply_temperature(
                policy,
                root_policy_temperature(self.args, self.turn_number, self.game.board_size),
            )
            policy = add_dirichlet_noise(
                policy,
                self.dirichlet_concentration,
                legal_actions_mask=legal_actions_mask,
                noise_weight=self.dirichlet_noise_weight,
            )
        _expand(node, policy, self.game, legal_actions_mask)
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
    其余搜索相关参数（num_simulations / c_puct / 根温度 /
    落子温度 / dirichlet_total_concentration）含义与串行版完全相同。
    """

    def __init__(self, game, args, model, device):
        self.game = game
        self.args = args
        self.model = model.to(device).eval()
        self.device = device
        self.num_parallel_games = max(1, int(args.get("num_parallel_games", 32)))
        self.action_size = game.board_size ** 2
        size = game.board_size
        self.input_shape = (game.num_planes, size, size)

    @torch.inference_mode()
    def _batch_inference(self, nodes):
        """把多个待评估节点合并成一次前向，返回 [(policy, value), ...]。

        与 ``MCTS.nn_inference`` 语义一致：先 mask 非法动作再 softmax。
        """
        game = self.game
        n = len(nodes)
        action_size = self.action_size

        # 直接构造 float32 的 batch（省掉先堆 int8 再转 float32 的一遍拷贝）。
        encoded = np.empty((n,) + self.input_shape, dtype=np.float32)
        masks = np.empty((n, action_size), dtype=bool)
        encode_state = game.encode_state
        get_legal_action_mask = game.get_legal_action_mask
        for i, node in enumerate(nodes):
            state = _materialize(node, game)
            to_play = node.to_play
            encoded[i] = encode_state(state, to_play)
            mask = get_legal_action_mask(state, to_play)
            masks[i] = mask
            node.legal_actions_mask = mask  # 供 deliver/_expand 复用

        policy_logits, values = self.model(torch.from_numpy(encoded).to(self.device))

        # 一次性完成 mask + softmax，逐行结果与 utils.softmax 逐位一致。
        logits = policy_logits.reshape(n, -1).float().cpu().numpy().astype(np.float64)
        logits[~masks] = -np.inf
        logits -= logits.max(axis=1, keepdims=True)
        np.exp(logits, out=logits)
        logits /= logits.sum(axis=1, keepdims=True)

        values = values.reshape(-1).float().cpu().numpy()
        return [(logits[i], float(values[i])) for i in range(n)]

    def run(self, total_games):
        """收集 ``total_games`` 局，按完成顺序 yield (samples, winner, game_len)。"""
        game = self.game
        args = self.args
        num_parallel_games = self.num_parallel_games
        batch_inference = self._batch_inference

        # selfplay 阶段会产生海量短生命周期的树节点。树在丢弃前会用
        # ``_discard_tree`` 主动断环，可被引用计数立即释放，因此这里关掉循环
        # GC 不会泄漏，却能省掉反复扫描/晋升这些节点的巨大开销；退出时恢复
        # （嵌套调用也安全，gc 内部用计数管理）。
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
                    for (session, node), (policy, value) in zip(requests, results):
                        session.deliver(node, policy, value)

                for result in finished:
                    yield result
        finally:
            if gc_was_enabled:
                gc.enable()

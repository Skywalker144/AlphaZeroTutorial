# MuZero（适用于棋类）

**MuZero** 和 AlphaZero 一样，通过**自对弈 + MCTS + 神经网络训练**学习下棋。理解它时，可以先从 AlphaZero 中替换掉一行代码：

```python
# AlphaZero：按照真实规则，算出走完这一步后的棋盘
next_state = game.get_next_state(state, action)

# MuZero：用神经网络，算出走完这一步后的内部特征
next_hidden_state = dynamics_network(hidden_state, action)
```

AlphaZero 搜索树里的节点保存**真实棋盘**，MuZero 搜索树里的节点保存**隐状态（hidden state）**。这个隐状态就是一个特征张量，后面所有模拟都在这些特征上继续计算。

本文先讲双人轮流落子、零和、没有中间得分的棋类：`to_play = 1` 表示黑方，`-1` 表示白方，value 始终从**当前节点轮到走棋的一方**来看，赢为 `+1`，输为 `-1`，和棋为 `0`。

这里先省略 reward 分支，把最终胜负交给 value 学习。原论文的无中间奖励棋类实验也省略了 reward prediction loss；通用 MuZero 的 reward 分支在后文单独说明。这样可以先看清楚：**即使没有 reward loss，dynamics 也能训练。** [原论文，附录 G](https://arxiv.org/html/1911.08265v2#A7)

## MuZero 的三个神经网络模块

### 先看 hidden state 的形状

**在原论文的棋类实现中，hidden state 的长宽确实与棋盘相同，通道数为 256。** 例如，忽略 batch 维度后，围棋是 `[256, 19, 19]`，国际象棋是 `[256, 8, 8]`，将棋是 `[256, 9, 9]`。这是网络架构的选择，MuZero 算法本身并不要求隐状态一定保留棋盘长宽。[原论文，网络输入与架构](https://arxiv.org/html/1911.08265v2#A5)

为了把尺寸写得具体，下面统一用一个**15×15、每步选择一个位置落子、没有 pass 动作的五子棋示例**：

| 符号 | 含义 | 本文示例 |
|---|---|---|
| `B` | 一批棋局的数量 | 搜索时可取 `1` |
| `H, W` | 棋盘长宽 | `15, 15` |
| `C_in` | 输入棋局编码的通道数 | `3` |
| `C` | hidden state 的通道数 | `128` |
| `A` | 固定动作空间的大小 | `225` |

本文张量维度按 PyTorch 的 `[B, C, H, W]` 顺序书写。`3` 个输入通道、`128` 个隐状态通道是**教学示例的配置**，不是原论文的固定参数。

```text
真实棋盘                 编码后的棋盘                    hidden state
[15, 15]      ──────>    [B, 3, 15, 15]      ── h ──>   [B, 128, 15, 15]
```

可以把 `[128, 15, 15]` 理解成 **128 张叠在一起的 15×15 特征图**。但这些通道不是“第 1 层黑棋、第 2 层白棋”这样预先规定的棋盘层，而是网络自己学出的浮点数特征。

`hidden_state[:, :, row, col]` 取出该空间位置上的 128 维特征。经过多层卷积，这些特征可以包含周围甚至整个棋盘的信息，不能把它直接当成“这个格子上放了什么棋子”。**空间尺寸相同，不代表每个数都有真实棋盘上的明确含义。**

### 1、Representation：真实棋局 → 隐状态

Representation 记作 $h$，负责把当前真实棋局转换成隐状态：

$$
s^0 = h(o_t)
$$

这里 $o_t$ 是时刻 $t$ 的棋局输入，可以包含当前棋盘、历史棋盘和轮到谁走棋等信息；上标 $0$ 表示“从这个真实棋局出发，还没有在模型里模拟任何一步”。

示例中先把真实棋盘编码为三个通道：

```python
observation[:, 0] = 当前玩家的棋子位置    # 有棋子为 1，否则为 0
observation[:, 1] = 对手的棋子位置        # 有棋子为 1，否则为 0
observation[:, 2] = 当前玩家身份          # 黑方时整层为 1，白方时整层为 0

# observation.shape == [B, 3, 15, 15]
hidden_state = representation_network(observation)
# hidden_state.shape == [B, 128, 15, 15]
```

一种实现方式是：

```text
[B, 3, 15, 15]
      │
      ▼  3×3 卷积：3 个输入通道 → 128 个输出通道
      │  stride=1, padding=1，长宽保持不变
[B, 128, 15, 15]
      │
      ▼  若干残差块：保持通道数和长宽
[B, 128, 15, 15] = s⁰
```

注意，这里的“表示”不是一定要把张量变小。这个例子反而增加了通道数，它要做的是**把棋盘编码转换成后续网络适合使用的特征**。

### 2、Dynamics：隐状态 + 动作 → 下一隐状态

Dynamics 记作 $g$，负责在隐状态中“走一步”：

$$
s^{k+1} = g(s^k, a_{t+k})
$$

这里 $a_{t+k}$ 表示在真实时刻 $t+k$ 执行的动作，$s^{k+1}$ 表示从起点累计模拟了 $k+1$ 步之后的隐状态。训练时用真实执行过的动作；搜索时则可以尝试候选动作。

动作也要先变成张量。比如五子棋中的动作是“在第 4 行、第 7 列落子”（下标从 0 开始），可以编码成一张只有落子位置为 `1`、其余位置为 `0` 的特征图：

```python
action_plane = zeros(B, 1, 15, 15)
action_plane[:, 0, 4, 7] = 1        # 此处假设这一批都演示相同动作

# 沿通道维拼接，不是把两个张量相加
x = cat([hidden_state, action_plane], dim=1)
# [B, 128, 15, 15] 与 [B, 1, 15, 15] 拼接 → [B, 129, 15, 15]

next_hidden_state = dynamics_body(x)
# [B, 128, 15, 15]
```

对应的形状变化是：

```text
hidden state [B, 128, 15, 15] ──┐
                               ├─ 沿通道拼接 → [B, 129, 15, 15]
动作编码      [B,   1, 15, 15] ──┘                       │
                                                        ▼
                                      3×3 卷积：129 → 128 个通道
                                      stride=1, padding=1
                                                        │
                                                        ▼
                                                   若干残差块
                                                        │
                                                        ▼
                                          [B, 128, 15, 15]
                                           next hidden state
```

输入和输出的隐状态形状相同，因此可以连续调用**同一个 dynamics 网络**：

```python
s0 = representation_network(observation_t)
s1 = dynamics_network(s0, action_t)
s2 = dynamics_network(s1, action_t_plus_1)
s3 = dynamics_network(s2, action_t_plus_2)
```

`s1`、`s2`、`s3` 都是 `[B, 128, 15, 15]`，但内容不同，分别表示模拟 1、2、3 步后的局面特征。**这里没有先还原成棋盘，也没有重新调用 representation。**

动作通道数取决于游戏。落子游戏可以用一个位置通道；移动棋子的游戏还要表达起点、终点、升变等信息。一般写成：`[B, C + C_action, H, W] → [B, C, H, W]`。

### 3、Prediction：隐状态 → 策略和价值

Prediction 记作 $f$，作用与 AlphaZero 的策略、价值预测相似，只是输入换成了 hidden state：

$$
p^k, v^k = f(s^k)
$$

```python
policy_logits, value = prediction_network(hidden_state)

# hidden_state.shape  == [B, 128, 15, 15]
# policy_logits.shape == [B, 225]
# value.shape         == [B, 1]
```

一种便于理解的双头结构如下；中间层宽度只是示例，图中省略中间激活函数：

```text
                         hidden state [B, 128, 15, 15]
                                      │
                   ┌──────────────────┴──────────────────┐
                   ▼                                     ▼
             Policy head                            Value head
        1×1 卷积：128 → 2                       1×1 卷积：128 → 1
          [B, 2, 15, 15]                          [B, 1, 15, 15]
                   │                                     │
                flatten                               flatten
               [B, 450]                              [B, 225]
                   │                                     │
           全连接：450 → 225                  全连接：225 → 128 → 1
                   │                                     │
        policy_logits [B, 225]                     tanh → [B, 1]
```

- **Policy**：225 个动作各有一个 logit，经过 softmax 后得到先验概率。动作编号 `row * 15 + col` 对应一个落子位置。网络始终输出完整的 225 项，不在 hidden state 上判断哪个位置可以落子。
- **Value**：每个局面一个标量，表示当前玩家的预期胜负，范围为 `[-1, 1]`。

这三个模块的输入输出可以放在一起看：

| 模块 | 输入 | 输出 |
|---|---|---|
| Representation $h$ | 棋局编码 `[B, 3, 15, 15]` | 隐状态 `[B, 128, 15, 15]` |
| Dynamics $g$ | 隐状态与动作编码拼接后 `[B, 129, 15, 15]` | 下一隐状态 `[B, 128, 15, 15]` |
| Prediction $f$ | 隐状态 `[B, 128, 15, 15]` | 策略 logits `[B, 225]`、价值 `[B, 1]` |

## MuZero 搜索算法流程

和 AlphaZero 一样，走一步棋之前，先进行多次 MCTS 模拟，每次包括**选择、扩展、反向传播**。

但根节点和后续节点的隐状态来源不同：

```python
# 根节点：手里有真实棋局，所以调用 h
root.hidden_state = representation_network(observation)

# 后续节点：搜索假设动作的结果，所以调用 g
child.hidden_state = dynamics_network(parent.hidden_state, action)

# 两种节点都用 f 预测策略和价值
policy_logits, value = prediction_network(node.hidden_state)
```

下面是突出主流程的伪代码，沿用 AlphaZero 的固定 `c_puct` 写法，省略批量推理等工程细节。搜索时将网络的 batch 维取出，policy 按动作编号索引，value 作为标量使用。`Node` 默认 `n=0`、`v=0`、`children={}`，其中 `v` 存储累计价值，`is_expanded()` 表示已经创建子节点。

### 1、选择

从根节点开始，选择 PUCT 最大的子节点，直到到达一个未展开的节点：

```python
while node.is_expanded():
    node = select(node)


def select(node):
    return max(node.children.values(), key=get_puct)


def get_puct(node, c_puct=1.25):
    # node.v / node.n 是子节点玩家的价值；对父节点玩家要取反
    q = -node.v / node.n if node.n > 0 else 0
    u = c_puct * node.prior * sqrt(node.parent.n) / (1 + node.n)
    return q + u
```

这部分和 AlphaZero 基本一致。区别发生在下一步：**未展开的节点没有真实棋盘，需要用 dynamics 生成隐状态。**

### 2、扩展

这里讲的是 **dynamics 生成的树内节点**：prediction 对完整动作空间输出概率，直接据此创建子节点。整个过程没有合法动作查询，也没有 `mask_logits`。

```python
# 先得到当前节点的隐状态
node.hidden_state = dynamics_network(
    node.parent.hidden_state,
    node.action_taken
)

# 再预测策略、创建子节点，并返回当前节点的价值
value = expand(node)


def expand(node):
    policy_logits, value = prediction_network(node.hidden_state)
    policy = softmax(policy_logits)

    # 树内节点按完整动作空间展开，不使用真实棋盘的合法动作集合
    for action in range(action_space_size):
        node.children[action] = Node(
            hidden_state=None,          # 真正搜索到这里时，才调用 dynamics
            to_play=-node.to_play,
            prior=policy[action],
            parent=node,
            action_taken=action
        )

    return value
```

**隐状态搜索与真实落子的区别**

在隐状态搜索中，每个动作仍对应固定的动作编号。例如 `action = 67` 仍表示尝试在第 4 行、第 7 列落子。`g(hidden_state, 67)` 只进行网络计算，不检查这个位置在真实棋盘上是否已经有棋子。

所以准确地说，**树内没有显式的合法性判断和屏蔽步骤**。真实游戏的规则仍然存在，只是没有被拿来筛选树内节点的动作。内部 policy 通过训练学习动作的先验概率，搜索再结合预测价值进行选择。

**根节点是特殊情况**：此时手里有当前真实棋盘，可以向环境查询合法动作，让最终实际执行的动作符合规则。虽然根节点也有 hidden state，但合法动作集合来自外部真实环境，不是从 hidden state 中计算出来的。原版 MuZero 只在根节点进行这种屏蔽。[原论文，附录 A](https://arxiv.org/html/1911.08265v2#A1)

终局同理：真实自对弈由环境判断何时结束；树内不调用真实规则判断终局，继续使用网络预测。后文会说明怎样训练终局之后的预测。

### 3、反向传播

这里的“反向传播”指 **MCTS 的价值回传**，只是更新节点统计量，**不会更新神经网络参数**。训练网络时的 `loss.backward()` 是另一件事。

在本文省略中间奖励、折扣为 `1` 的棋类设定下，和 AlphaZero 一样，每向上一层就切换一次玩家视角：

```python
def backpropagate(node, value):
    while node is not None:
        node.v += value
        node.n += 1
        value = -value
        node = node.parent
```

### 完整搜索与最终策略

把根节点初始化单独写成 `expand_root()`：只有这个函数从真实棋盘获取合法动作，并屏蔽对应 logits；模拟循环中的 `expand()` 始终使用上面的完整动作空间。

```python
def expand_root(root, real_state, add_dirichlet=False):
    policy_logits, value = prediction_network(root.hidden_state)

    # 这是环境对当前真实棋盘的判断，不是隐状态上的规则推演
    legal_actions = game.get_legal_actions(real_state)
    policy_logits = mask_logits(policy_logits, legal_actions)
    # 将非法位置设为 -inf，再做 softmax
    policy = softmax(policy_logits)

    if add_dirichlet:
        policy = mix_dirichlet(policy, legal_actions)

    for action in legal_actions:
        root.children[action] = Node(
            hidden_state=None,
            to_play=-root.to_play,
            prior=policy[action],
            parent=root,
            action_taken=action
        )

    return value


@no_grad()  # 搜索收集统计量，不构建训练用的梯度图
def search(state, to_play, self_play=True):
    observation = game.encode_state(state, to_play)

    root = Node(
        hidden_state=representation_network(observation),
        to_play=to_play,
        parent=None
    )

    expand_root(
        root,
        real_state=state,
        add_dirichlet=self_play
    )

    for _ in range(num_simulations):
        node = root

        # 1. 选择
        while node.is_expanded():
            node = select(node)

        # 2. 扩展：这里只计算 hidden state，不推演真实棋盘
        node.hidden_state = dynamics_network(
            node.parent.hidden_state,
            node.action_taken
        )
        value = expand(node)

        # 3. 回传搜索价值
        backpropagate(node, value)

    # 访问次数分布作为 policy 训练目标
    mcts_policy = zeros(action_space_size)
    for action, child in root.children.items():
        mcts_policy[action] = child.n
    mcts_policy /= sum(mcts_policy)

    # 评估时选择访问次数最多的动作
    best_action = max(root.children, key=lambda a: root.children[a].n)
    return mcts_policy, best_action
```

上面假设输入棋局尚未结束、至少有一个合法动作，且 `num_simulations > 0`。自对弈时可按访问次数分布采样动作，增加探索；评估时选择访问次数最多的动作。

## 自对弈数据：训练目标从哪里来？

每次完成搜索后，**在真实环境中实际走一步**，保存这一时刻的数据：

```python
mcts_policy, best_action = search(state, to_play)
action = sample(mcts_policy)

history.append({
    "observation": game.encode_state(state, to_play),
    "player": to_play,
    "action": action,
    "mcts_policy": mcts_policy
})

# 真正落子时，仍然需要真实游戏规则
state = game.get_next_state(state, action, to_play)
to_play = -to_play
```

一局结束后，把最终结果转换成各个时刻玩家的相对胜负：

```python
winner = game.get_winner(state)     # 黑胜 +1，白胜 -1，和棋 0

for item in history:
    item["value_target"] = winner * item["player"]

replay_buffer.add(history)
```

因此每个真实时刻 $j$ 都有两个监督目标：

- **策略目标 $\pi_j$**：当时从真实棋局出发，运行 MCTS 得到的根节点访问次数分布。
- **价值目标 $z_j$**：整局最终胜负，转换成该时刻轮到走棋的一方的视角。

这些目标在训练时作为固定数据使用。实际落子的 `action` 是 dynamics 的输入，**不是**拿来替代整个 MCTS 分布的 policy 标签。

## Dynamics 到底怎么训练？

### 1、先把网络沿真实动作展开

假设一局棋中有下面这段经历，且黑方最终获胜：

```text
真实棋局：   o_t  ── a_t ──>  o_{t+1}  ── a_{t+1} ──>  o_{t+2}
轮到谁下：   黑方              白方                     黑方
策略目标：   π_t               π_{t+1}                  π_{t+2}
价值目标：   +1                -1                       +1
```

训练时只把起点 `o_t` 交给 representation，然后把这局棋**实际走过的动作**依次交给 dynamics：

```python
s0 = representation_network(observation_t)
p0, v0 = prediction_network(s0)

s1 = dynamics_network(s0, action_t)
p1, v1 = prediction_network(s1)

s2 = dynamics_network(s1, action_t_plus_1)
p2, v2 = prediction_network(s2)
```

对应关系是：

```text
o_t ── h ──> s0 ── g(·, a_t) ──> s1 ── g(·, a_{t+1}) ──> s2
              │                    │                       │
              f                    f                       f
              │                    │                       │
              ▼                    ▼                       ▼
           p0, v0               p1, v1                  p2, v2
              │                    │                       │
           对照目标             对照目标                对照目标
              │                    │                       │
              ▼                    ▼                       ▼
          π_t, +1             π_{t+1}, -1             π_{t+2}, +1
```

特别注意：`p1` 对应的是 **下一真实棋局重新进行 MCTS 后得到的 `π_{t+1}`**，不是起点 `π_t`，也不是起点搜索树中某个内部节点的访问分布。

如果每走一步都改成 `s1 = h(observation_t_plus_1)`，再让 prediction 预测，那么这一步的损失就绕过了 dynamics，不能教会它怎样生成下一隐状态。所以这段训练必须让预测沿着 `h → g → g` 连起来。

### 2、hidden state 没有直接标签，后续预测有标签

最容易困惑的是：**真实数据里只有棋盘，哪里来的“正确 hidden state”让 dynamics 拟合？**

答案是：基础 MuZero **不给 hidden state 本身设置直接的监督标签**。它没有要求 `s1` 的每个元素必须等于某个标准张量，而是要求从 `s1` 读出的策略和价值预测准确。[原论文，第 3 节](https://arxiv.org/html/1911.08265v2#S3)

```python
# 用来训练的是下一步的预测误差
loss1 = policy_loss(p1, target_policy_t_plus_1)
loss1 += value_loss(v1, target_value_t_plus_1)

# 基础 MuZero 不要求添加这样的 hidden state 对齐损失：
# mse(s1, representation_network(observation_t_plus_1))
```

因此，`g(h(o_t), a_t)` 与 `h(o_{t+1})` **形状相同，但数值不必逐元素相等**。前者由模型模拟一步得到，后者由真实下一棋局重新编码得到；训练要求它们支持正确的预测，没有要求它们内部表示完全一样。

这也解释了为什么 hidden state 不必能还原出真实棋盘：需要监督的是**接下来怎么走、最终能不能赢**，而不是每个格子的重建结果。

### 3、policy/value 的损失会经过 prediction，传回 dynamics

看 `loss1` 的计算依赖：

```text
正向计算：o_t → h → s0 → g(·, a_t) → s1 → f → p1, v1 → loss1

梯度回传：loss1 → f 的参数
               → s1 → g 的参数
                    → s0 → h 的参数
```

虽然损失写在 `p1`、`v1` 上，但它们是由 `s1` 算出来的，`s1` 又是由 dynamics 算出来的。因此执行 `loss.backward()` 时，梯度会沿这条计算链回传，**同时修改 prediction、dynamics 和 representation 的参数**。

例如，上面的 `o_{t+1}` 轮到白方走，而这局最后黑方获胜，所以 `v1` 的目标是 `-1`。如果网络预测成 `+0.8`，误差会推动 prediction 改变对 `s1` 的评估，也会推动 dynamics 改变生成 `s1` 的方式。它们共同学习让这条预测链的输出更接近目标。

继续看第二步的损失：

```text
loss2 → f → s2 → 第二次调用 g → s1 → 第一次调用 g → s0 → h
```

**第二步的误差也会影响第一步的 dynamics 计算。** 所以 `s1` 既要能让 prediction 看懂当前局面，也要保留足够的信息，让 dynamics 接着往后推演。

这里两次调用的 `g` 使用**同一套参数**，所有步的梯度会汇总到这套参数上。多个 `f` 的调用同样共享参数。这就是沿展开序列进行反向传播，也称为 BPTT。

| 损失来自哪里 | 会更新哪些模块 |
|---|---|
| 第 0 步的 policy/value | $f$、$h$；这一步尚未经过 $g$ |
| 第 1 步的 policy/value | $f$、$g$、$h$ |
| 第 2 步及更后面的 policy/value | $f$、前面各次调用的 $g$、$h$ |

因此，**dynamics 学到的是：输入当前特征和动作后，产生一个能支持后续策略、价值预测以及继续推演的特征张量。** 它不是只靠一个 reward loss 学会“走棋”。

### 4、完整的训练更新

策略损失使用 MCTS 分布与网络分布之间的交叉熵；价值损失使用最终胜负与预测价值之间的均方误差：

$$
L_{policy}^{k} = -\sum_a \pi_{t+k}(a)\log p^k(a)
$$

$$
L_{value}^{k} = (v^k-z_{t+k})^2
$$

从一个起点展开 $K$ 个动作，会产生 **$K+1$ 组 policy/value 预测**：

$$
L = \sum_{k=0}^{K}\left(L_{policy}^{k}+L_{value}^{k}\right)
    + \lambda\|\theta\|^2
$$

这里 $\theta$ 包含 $h$、$g$、$f$ 的所有参数。下面用 PyTorch 风格伪代码写出一次更新，先演示没有跨过终局的片段；实际 batch 的有效步数和终局处理见后文。

```python
# optimizer 必须包含 h、g、f 三个模块的参数
# 正则化可在 optimizer 中配置；下方只写 policy/value 损失
optimizer.zero_grad()

# sample.observation:   [B, C_in, H, W]，只有起点的棋局输入
# sample.actions:       [B, K]，自对弈中实际执行的 K 个动作
# sample.target_policy: [B, K+1, A]，各真实时刻的 MCTS 分布
# sample.target_value:  [B, K+1, 1]，各时刻玩家视角下的最终胜负

hidden_state = representation_network(sample.observation)
loss = 0.0

for k in range(K + 1):
    policy_logits, value = prediction_network(hidden_state)

    # 策略目标是完整概率分布，而不是某个动作编号
    policy_loss = -(
        sample.target_policy[:, k] * log_softmax(policy_logits, dim=-1)
    ).sum(dim=-1).mean()

    value_loss = (
        value - sample.target_value[:, k]
    ).square().mean()

    loss = loss + policy_loss + value_loss

    if k < K:
        # 动作编码、通道拼接在 dynamics 内部完成
        hidden_state = dynamics_network(
            hidden_state,
            sample.actions[:, k]
        )

# 所有步一起反向传播，梯度经过整个展开链
loss = loss / (K + 1)
loss.backward()
optimizer.step()
```

这里按预测步数取平均只是教学代码的损失缩放方式。理解梯度路径时，最需要注意的是：

- **不能在展开过程中随意对 hidden state 调用 `detach()`**，否则后续损失无法穿过被截断的位置回传。
- **不能只把 prediction 的参数交给 optimizer**，三个模块需要联合训练。
- **训练不用对 MCTS 本身求导**。MCTS 先产生保存在数据中的目标，训练时再重新计算这条可微分的网络链。

原论文使用多步展开，并包含额外的梯度缩放、隐状态缩放等训练细节；这些细节不改变上述梯度路径。[原论文，附录 G](https://arxiv.org/html/1911.08265v2#A7)

### 5、展开到终局怎么办？

前面的训练代码只展示了尚未结束的片段。完整实现还要处理“剩余实际步数不足 $K$”以及“模型继续推演到终局之后”的情况，不能读取不存在的下一步 MCTS 策略。

原版 MuZero 使用**吸收状态**的思路：真实棋局结束后，结果不再改变，让模型在继续展开时也保持与这个结果一致的预测。[原论文，附录 A](https://arxiv.org/html/1911.08265v2#A1)

对本文的“无 reward 分支、value 表示最终胜负”的约定，一种具体处理方式是：

- 终局前使用实际动作、MCTS 策略和最终胜负。
- 终局以及补齐的后续步骤不再有真实 MCTS 策略，屏蔽这些步骤的 policy loss；补齐动作从固定动作空间取值。
- value 仍以固定的最终结果为目标，并转换到该步的玩家视角。如果沿用代码中每步切换玩家的约定，黑胜对应的目标仍按 `+1, -1, +1, ...` 交替；改变的是视角，不是比赛结果。

这里要求的是**预测结果一致**，并不要求终局后的 hidden state 张量每一步都完全相同。

## 通用 MuZero 中的 reward 分支

如果环境存在中间奖励，dynamics 还需要预测**执行这个动作立刻得到的 reward**：

$$
r^{k+1}, s^{k+1} = g(s^k, a_{t+k})
$$

```python
reward, next_hidden_state = dynamics_network(hidden_state, action)

# reward.shape            == [B, 1]，此处用标量回归演示
# next_hidden_state.shape == [B, C, H, W]
```

例如可以在 dynamics 的特征上分出一个 reward head：

```text
hidden state + action → dynamics 特征 → next hidden state [B, C, H, W]
                              │
                              └─ reward head ──> reward [B, 1]
```

此时自对弈还需要保存环境反馈的真实奖励 `u_{t+1}`。训练每次调用 dynamics 后，多加一项：

```python
reward, hidden_state = dynamics_network(hidden_state, sample.actions[:, k])
loss += reward_loss(reward, sample.target_reward[:, k])
# target_reward[:, k] 保存执行 a_{t+k} 后得到的 u_{t+k+1}
```

总损失变为：

$$
L = \sum_{k=0}^{K}\left(L_{policy}^{k}+L_{value}^{k}\right)
    + \sum_{k=1}^{K}L_{reward}^{k}
    + \lambda\|\theta\|^2
$$

注意 reward loss 从 `k=1` 开始：起点还没经过动作，不存在这一展开链中的第 0 步奖励预测。

reward loss 给 dynamics 增加了直接监督，但**后续 policy/value loss 经过隐状态回传的训练路径仍然存在**。两种信号可以同时训练 dynamics；仅拟合即时 reward，并不足以要求隐状态保留后续决策需要的信息。

搜索回传也要与奖励定义一致。对双方轮流行动的零和游戏，如果 reward 定义为**刚刚落子的父节点玩家**所得，value 定义为**下一节点玩家**的未来回报，则：

```python
parent_value = child.reward - discount * child_value
```

若把终局胜负计入最后一步 reward，终局之后的未来回报 value 就应为 `0`，避免同一胜负被 reward 和 value 重复计算。本文前面的棋类版本采用另一套统一约定：不累计 reward，value 直接学习最终胜负。

## AlphaZero 和 MuZero 的学习区别

| 问题 | AlphaZero | 本文的棋类 MuZero |
|---|---|---|
| 搜索节点保存什么？ | 真实棋盘 | hidden state 特征张量 |
| 怎样得到下一节点？ | 规则引擎计算下一棋盘 | dynamics 计算下一隐状态 |
| 网络怎样评估节点？ | 从真实棋局预测 policy/value | 从隐状态预测 policy/value |
| policy 跟谁学？ | MCTS 的访问次数分布 | 各真实时刻的 MCTS 分布 |
| value 跟谁学？ | 最终胜负 | 各展开步玩家视角下的最终胜负 |
| dynamics 跟谁学？ | 没有这个模块 | 后续预测的误差沿展开链回传 |

把训练过程连起来看：

```text
真实自对弈提供：起点棋局 + 实际动作序列 + 各时刻 MCTS 策略 + 最终胜负
                                  │
                                  ▼
                h 编码起点，g 沿动作连续展开，f 在每步预测
                                  │
                                  ▼
                     各步预测与真实轨迹中的目标比较
                                  │
                                  ▼
                    梯度沿整条计算链更新 h、g、f
```

**Representation 学会怎样表示棋局，dynamics 学会怎样根据动作更新这些特征，prediction 学会从特征中判断怎么走、能不能赢。三者通过同一条预测链联合训练。**

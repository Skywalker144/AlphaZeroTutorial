# KataGo 技巧

本路线在 [AlphaZero](AlphaZero.md) 的完整流程之上，逐项加入 [KataGo](https://github.com/lightvector/KataGo) 的搜索与训练技巧。每条技巧都按「解决什么问题 → 做法 → 本仓库实现 → 实验与结论」展开，并明确标注与 KataGo 的差异。

本仓库是纯 Python 的教学子集：不编译原生引擎，不包含围棋专用的目数、归属等目标。并行自我对弈与批量推理只是为了让实验跑得更快，教学以串行实现为准，两条路径的算法保持一致（有等价性测试守护）。

使用与运行方式见 [KataGo README](../KataGo/README.md)。

## 目录

- [与 AlphaZero 的差异总览](#与-alphazero-的差异总览)
- [PlayoutCapRandomization](#playoutcaprandomization)
- [TreeReuse](#treereuse)
- [WDLValueHead](#wdlvaluehead)
- [FPU](#fpu)
- [LCB](#lcb)
- [RootTemperature / ChosenMoveTemperature](#roottemperature--chosenmovetemperature)
- [ShapedDirichletNoise](#shapeddirichletnoise)
- [SoftResign](#softresign)
- [Policy(Value)SurpriseWeighting](#policyvaluesurpriseweighting)
- [未实现 / 后续](#未实现--后续)

## 与 AlphaZero 的差异总览

| 技巧 | 一句话 | 与 KataGo 的差异 |
| --- | --- | --- |
| PlayoutCapRandomization | 按概率混用 cheap / full search，化解 policy 需要深搜索与 value 需要更多独立对局的矛盾 | 一致 |
| TreeReuse | 落子后保留被选分支的子树，供下一步 reuse | 一致 |
| WDLValueHead | value 输出胜/平/负三分类，交叉熵训练，搜索用 `P(胜) − P(负)` | 一致 |
| FPU | 未访问子节点价值 = 父价值与网络价值的混合 − 缩减量 | 未实现 `fpuLossProp` |
| LCB | 用「均值 − k·标准误」修正根节点的 policy target，避免选到虚高的着法 | 未实现 forced-playout pruning |
| RootTemperature / ChosenMoveTemperature | 根策略温度 + 落子温度，随对局进度衰减 | 一致 |
| ShapedDirichletNoise | alpha 一半均匀、一半按 `log(先验)` 形状 | 先验封顶按棋盘面积缩放 |
| SoftResign | 一边倒时降低 full search 的访问数与样本权重，仍下到真实终局 | `reduced_visits_min` 取 cheap 访问数 |
| Policy(Value)SurpriseWeighting | 按 policy/value 意外度重分配每局权重，按权重决定写入份数 | value surprise 用平滑前瞻版（KataGo 默认） |

下表之外的功能（并行自我对弈、批量推理、动态回放窗口、checkpoint、棋盘对称增强）两套实现都有，不属于本路线新增。

---

## PlayoutCapRandomization

### 解决什么问题

AlphaZero 每一步都用相同次数做全量搜索，并且每一步都记录训练样本。问题在于两个训练目标对搜索预算的需求是**矛盾**的：

- **policy target** 是 MCTS 的访问分布，它需要**足够深的搜索**才准确；
- **value target** 只是这局棋的最终胜负，它**与这一步搜得多深无关**，真正想要的是**更多、更相互独立的完整对局**（更多样的终局结果来训练价值网络）。

在固定算力下，把每一步都搜得很深，就必然减少能下完的局数，反之亦然。AlphaZero 固定一个较高的访问数，等于把预算几乎全押在 policy 上，value 拿到的是数量偏少、且来自「强而相似」搜索的对局。

### 做法

PCR 把预算拆开：每一步以概率 $p$ 做 full search，其余做访问数很少的 cheap search。cheap 搜索**只负责推进对局、选择落子，不产生（或以极低权重产生）policy target**；full 搜索产生正常的 policy target。于是：

- 每一局仍然走到真实终局，因此局内所有局面都有正确的 value target；
- 大多数步很便宜，同样的算力能下完**更多、更多样**的对局，给 value 提供更多独立样本；
- full 步数变少，但每步都有足够深的搜索，policy 目标质量不降。

换句话说，PCR 用「大多数步只要够用来把棋走完」换取「更多完整对局」，同时把深搜索集中在一部分步上以保证 policy 目标。cheap 分支不施加根噪声与根温度。

### 本仓库实现

- `cheap_search_prob`，默认 `0.75`。
- 访问次数由 `search_visit_counts` 决定：

  $$
  \text{full} = \max(\text{full\_floor},\ \text{num\_simulations}), \qquad
  \text{cheap} = \min(\text{full},\ \max(\text{cheap\_floor},\ \text{cheap\_visits}))
  $$

  默认 `num_simulations = 1.66·board_size²`、`cheap_search_visits = 0.28·board_size²`，并有安全下限 `full_search_visits_floor = 50`、`cheap_search_visits_floor = 20`；`tictactoe/train.py` 把 `num_simulations` 设为 50，`gomoku/train.py` 设为 100。
- cheap 与 full 是 `if/else` 两个分支：cheap 不做根噪声/温度，也不做 SoftResign 的降访问。
- cheap 位置的样本会以 `weight = 0` 记录在 `memory` 里（见 SoftResign 与 Policy(Value)SurpriseWeighting），对局结束后仍可能因为「意外度高」重新获得权重。

### 实验与结论

本仓库暂无受控的棋力对照实验；`KataGo/tmp/` 下保留过一些临时的预算—棋力测量脚本与曲线（未纳入版本管理）。

---

## TreeReuse

### 解决什么问题

每落一步都从零重建搜索树，会浪费前一步已经算出的子树。落子后，被选中着法对应的子树对当前局面依然有效。

### 做法

落子时把被选中的子节点提升为新根，保留其子树供下一次搜索复用；下一次搜索只需要补足到目标访问数。

### 本仓库实现

- `MCTS.advance(root, action)`（并行版 `_advance`）：找到 `action_taken == action` 的子节点，断开它与父节点的引用，其余子树立即释放；返回提升后的新根。
- **是否复用由「本步」的搜索模式决定**：
  - **cheap search 复用** `carried_root`，并把访问数补足到 cheap 预算：

    $$
    \text{remaining} = \max(0,\ \text{num\_simulations} + 1 - \text{root.visits})
    $$

    其中 `+1` 是因为本次搜索的根节点评估不计入模拟次数。
  - **full search 必须先清树、从新根开始搜索**。

### 为什么 full search 必须清树

full search 的产物是这一步的 **policy target（访问分布）**，它必须来自一次干净、完整的搜索：

1. **根先验必须重新计算，噪声/温度只能施加一次。** full search 要求用**当前局面**的网络输出来构造根先验，并把根温度与 Dirichlet 噪声恰好施加一次。复用旧树时，根节点的先验是**上一步**已经加过温度/噪声的旧策略，本次要么重复叠加、要么直接沿用上一步采样的噪声，得到的 target 有偏。
2. **访问预算必须是本次的完整预算。** 复用会把旧树（可能来自一次 cheap search，访问很少）里已有的 visits 算进根节点；`num\_simulations + 1 - root.visits` 的补足语义只会跑更少的新模拟，于是 policy target 里混入了不同搜索模式、不同噪声下产生的访问，进一步偏。
3. **树复用本就是为 cheap search 设计的。** cheap search 只用来选落子、不产生训练目标，继续沿用同一棵树、把访问数补到 cheap 预算既省算力也不会污染任何 target；full search 则相反，它的全部意义就在于产出一个可信的 target。

对应 KataGo：full search 前 `clearBotBeforeSearch`，cheap（不记录）时把 `clearBotBeforeSearchThisMove` 关掉以复用树。

### 实验与结论

复用与否不改变单步结果，只影响访问数是否被重复计算，因此不单独做棋力实验。

---

## WDLValueHead

### 解决什么问题

单个 `tanh` 标量只能给出一个期望值，无法表达「胜 / 和 / 负」的结构，和棋尤其容易被压缩掉。KataGo 让网络直接输出胜、平、负三个概率。

### 做法

value head 输出 3 个 logits 并 softmax 得到 $(P(\text{胜}), P(\text{和}), P(\text{负}))$。搜索时把标量价值取为期望：

$$
v = P(\text{胜}) - P(\text{负}) \in [-1, 1]
$$

训练时用交叉熵拟合最终胜负的 one-hot 目标。

### 本仓库实现

- `network.py`：value head 最后一层输出 3 维，`forward` 返回 `(policy_logits, value_logits)`，不再有 `tanh`。
- `value_target(winner, to_play)` 生成 one-hot 目标：胜 `[1,0,0]`、和 `[0,1,0]`、负 `[0,0,1]`（以该局面行棋方视角）。
- 节点存 3 维累加 `wdl_sum`，$q\text{-value} = (wdl\_sum[0] - wdl\_sum[2]) / \text{visits}$；回传时逐层翻转视角等价于把 `[w, d, l]` 反转为 `[l, d, w]`。
- 损失：$\text{value\_loss} = \text{value\_loss\_scale} \cdot \text{CE}$，默认 `value_loss_scale = 1.2`。

### 实验与结论

未训练网络的 value 交叉熵约为 $\log 3 \approx 1.099$；随训练下降。与 AlphaZero 标量价值 + MSE 的棋力对比尚未系统进行。

---

## FPU

（First Play Urgency）

### 解决什么问题

子节点还没被访问时价值未定义。AlphaZero 直接把未访问子节点的价值当作 0（和棋）。当父节点明显处于劣势（已访问着法价值都为负）时，未访问着法的 0 比它们都好，于是搜索会给每个合法着法都发一次「免费」访问，浪费预算；反之在明显优势时又会抑制一些其实不差的着法。

### 做法

未访问子节点的价值取「父节点价值」减去一个缩减量，缩减量随已覆盖的 policy 质量增大。KataGo 进一步让父节点价值在网络原始价值与搜索平均值之间按已覆盖 policy 质量混合：

$$
\text{mass} = \sum_{\text{已访问子节点}} \text{prior}, \qquad
\text{mix} = \min(1,\ \text{mass}^{pow})
$$

$$
\text{base} = \text{mix} \cdot \text{parentSearchValue} + (1 - \text{mix}) \cdot \text{parentNNValue}
$$

$$
\text{fpu} = \text{base} - \text{fpuReductionMax} \cdot \sqrt{\text{mass}}
$$

（式中均为父节点行棋方视角。）`pow` 对应 KataGo 的 `fpuParentWeightByVisitedPolicyPow`。

### 本仓库实现

- `mix = min(1, mass**2)`，`base = mix·node.q_value() + (1−mix)·(nn_wdl[0] − nn_wdl[2])`。
- `node.nn_wdl` 与 `node.prior_policy` 在每个被扩展的节点上保存（不再只有根节点）。
- 缩减量的上限 `reduction_max`：
  - full search 的**根节点**用 `root_fpu_reduction_max`，默认 `0.0`；
  - 其余情况（非根节点，以及 cheap search 的根节点）用 `fpu_reduction_max`，默认 `0.2`。

  即 cheap search 时根节点按普通节点处理——与 KataGo 在 `removeRootNoise`（cheap 不记录）时把 `rootFpuReductionMax` 设回 `fpuReductionMax` 一致。
- 未实现 KataGo 的 `fpuLossProp` / `rootFpuLossProp`。

### 实验与结论

本仓库暂无受控实验。可预期的效果是早期搜索更集中、更少「每个着法各访问一次」。

---

## LCB

（Lower Confidence Bound）

### 解决什么问题

每个子节点的平均价值都是从有限次访问估出来的，访问少时估计噪声大。直接挑「平均价值最高」的着法，容易选中一个靠少数几次走运刷出来的虚高着法。LCB 用「均值 − k·标准误」排序，偏好被稳定验证过的好着法。

### 做法

KataGo 里 LCB 同时服务两件事，由 `useLcbForSelection` 控制：

- **对弈 / 评估**：`getChosenMoveLoc` 用带 LCB 修正的权重选着法；
- **自我对弈训练**：选落子时**临时关掉** LCB（用原始访问权重），选完再恢复，让 LCB 只作用在 **policy target** 上（`play.cpp` 的 HACK）。

每个子节点的 LCB（`getSelfUtilityLCBAndRadius`）：设 $u$ 为平均 utility（win−loss），另外存 utility 平方均值来估方差：

$$
\text{ess} = \frac{(\sum w)^2}{\sum w^2}, \qquad
\text{priorWeight} = \frac{\sum w}{\text{ess}^3}
$$

$$
\overline{u^2} \leftarrow \frac{\overline{u^2}\,\sum w + (\overline{u^2} + R^2)\,\text{priorWeight}}{\sum w + \text{priorWeight}}
$$

$$
\text{stderr} = \sqrt{\frac{\overline{u^2} - \bar u^2}{\text{ess}}}, \qquad
\text{radius} = k\,\text{stderr}, \qquad
\text{lcb} = u_{\text{父节点视角}} - \text{radius}
$$

其中 $R$ 是 utility 的最大半径（本仓库取 1）。

生成 policy target 权重时（`getPlaySelectionValues`）：

1. 先找最「稳定被探索」的子节点作 `nonLCBBest`；
2. 在 $\text{weight} \ge \text{minVisitPropForLCB}\cdot\text{nonLCBBestWeight}$ 的子节点里取 `lcb` 最大者；
3. 把它加权到足以压过其它人：

$$
\text{radiusFactor} = \frac{\text{radius}_i + \text{excess}}{\text{radius}_i + 0.20\,\text{excess}}, \qquad
\text{excess} = \text{bestLcb} - \text{lcb}_i
$$

取 $\max(\text{weight}_{\text{best}},\ \max_i \text{radiusFactor}^2\cdot\text{weight}_i)$ 作为它的新权重。

那个 $\text{minVisitPropForLCB}$ 门槛是为了防止「只访问一两次但 LCB 很高」的着法劫持 target。`useNonBuggyLcb` 修的是早年 $\text{bestLcbIndex} > 0$ 把下标 0 排除掉的 off-by-one。

### 本仓库实现

- `utils.py::lcb_play_selection(root, action_size, args)`：返回 LCB 修正并归一化后的分布。`weight = child.visits`（本仓库没有 forced playouts / retrospective pruning，跳过那一步）。
- 节点新增 `utility_sq_sum`（精确平方和）用于算方差；串行 `Node.update` 与并行 `_backpropagate` 都更新。
- 训练（`trainer.selfplay` / 并行 `_finish_move`）：**落子用原始访问分布**，**policy target（含 policy surprise）用 LCB 修正后的分布**。
- 评估（`play.py`）：**落子用 LCB 修正后的分布**。
- 配置默认对齐 KataGo selfplay：`use_lcb_for_selection = true`、`lcb_stdevs = 5.0`、`min_visit_prop_for_lcb = 0.15`、`use_non_buggy_lcb = true`。

### 实验与结论

本仓库暂无受控实验。

---

## RootTemperature / ChosenMoveTemperature

### 解决什么问题

- AlphaZero 的 PUCT 有一个倾向：即使某个着法并不比其它着法好，访问分布也会比先验更尖锐（公式里 `1 + n_i` 带来的离散化效应）。加上网络在开局可能过早地对近似等价的着法形成意见，探索会不足。
- 直接按访问分布采样落子，前期需要随机性探索，后期则希望更接近贪心。

### 做法

- **RootTemperature**：搜索前对根策略做温度软化 $p_i \propto p_i^{1/T}$（$T > 1$ 更均匀）。只在根节点、只在训练、只在 full search 施加。它在 logit 空间等价于把所有 logits 朝 0 收缩，搜索必须主动「对抗」这个力才能保持策略尖锐，从而让近似等价的着法不被先验过早压制。
- **ChosenMoveTemperature**：落子时再对访问分布施加温度，按对局进度从高到低衰减，后期趋近「直接选访问数最多」。
- 两者共用一个半衰期，并按棋盘尺寸缩放。

### 本仓库实现

- `interpolate_early`：

  $$
  \text{halflives} = \frac{\text{turn}}{\text{halflife}} \cdot \frac{19}{\text{board\_size}}, \qquad
  \text{value} = \text{late} + (\text{early} - \text{late}) \cdot 0.5^{\text{halflives}}
  $$

  默认 `chosen_move_temperature_halflife = 19`，于是实际半衰期约为棋盘边长（19×19 为 19 手）。
- `root_policy_temperature`：early `1.3` → late `1.1`；`chosen_move_temperature`：early `0.75` → late `0.15`。
- 与 AlphaZero 路线的差异：AlphaZero 没有任何温度调度。

### 实验与结论

本仓库暂无受控实验。

---

## ShapedDirichletNoise

### 解决什么问题

AlphaZero 的根节点用 Dirichlet 噪声替换 25% 的先验，且给每个合法着法分配相同的 alpha。但真正的「盲点好手」即使先验很低（例如 0.008%），它的先验仍然高于棋盘上绝大多数随手棋、废棋。均匀噪声让好手和废棋被抽中的概率完全相同，浪费了探索预算。

KataGo 让 a 的一半均匀撒，另一半集中到「先验明显高于大多数着法」的子集上。

### 做法

对每个合法着法：

$$
p_i = \log(\min(\text{cap},\ \text{prior}_i) + 10^{-20}), \qquad
\text{shaped}_i = \max(0,\ p_i - \overline{p})
$$

$$
\text{proportion}_i = \frac{1}{2}\cdot\frac{\text{shaped}_i}{\sum \text{shaped}} + \frac{1}{2}\cdot\frac{1}{\text{legalCount}}
$$

然后 $\alpha_i = \text{proportion}_i \cdot \text{totalConcentration}$，采样 Dirichlet 噪声，再与先验混合：

$$
\text{policy} = (1-w)\cdot\text{prior} + w\cdot\text{noise}
$$

### 本仓库实现

- `add_dirichlet_noise(policy, total_concentration, legal_actions_mask, board_size, noise_weight)`。
- 先验封顶 `cap = 0.01 · (19 / board_size)²`：19×19 上等于 KataGo 的 `0.01`，小棋盘上相应放大，避免所有先验都被压平而退化为均匀噪声。
- `total_concentration` 默认 `0.03 · board_size²`（KataGo 默认固定 `10.83 = 0.03·361`；SkyZero 的 baseline 配置用 `6.75`），`dirichlet_noise_weight` 默认 `0.25`。
- `shaped` 之和为 0（先验近似相等）时退化为纯均匀。

### 实验与结论

在 3×3 上，若未做面积缩放，所有先验都会大于 0.01 而被压平、退化为均匀噪声；缩放后高先验着法能拿到更大的 alpha。受控棋力实验尚未进行（KataGo 官方也标注该技巧缺乏严格验证）。

---

## SoftResign

（ReduceVisits）

### 解决什么问题

自对弈后期一方已经必胜/必败时：继续全量搜索是浪费；直接认输（hard resign）又会引入训练偏差——败方从没学过在劣势下抵抗，胜方也没学过如何转化优势，且价值网络在「已定」区间缺样本。KataGo 的折中是：**不认输**，把棋下到真实终局（保证 value target 是真的），但逐步降低搜索访问数与样本权重。

### 做法

维护每一步根节点的 WinLoss 值（固定视角）。每一步开始前看最近 `lookback` 步：

$$
\text{extreme} = \max\big(\min(\text{recent}),\ -\max(\text{recent})\big),\ \text{clamp 到 } 1
$$

若超过阈值：

$$
\text{prop} = \left(\frac{\text{extreme} - \text{threshold}}{1 - \text{threshold}}\right)^2
$$

$$
\text{visits} = \text{round}(\text{full} + \text{prop}\cdot(\text{minVisits} - \text{full})), \quad \text{至少为 minVisits}
$$

$$
\text{weight} = 1 + \text{prop}\cdot(\text{reducedWeight} - 1)
$$

### 本仓库实现

- `reduce_visits`（默认 `True`）、`reduce_visits_threshold = 0.9`、`reduce_visits_threshold_lookback = 3`、`reduced_visits_min`（默认取 cheap 访问数）、`reduced_visits_weight = 0.1`。
- 历史用**固定（player 1）视角**存入：`root_value * to_play`。这一步是必须的：节点 `q` 是行棋方视角、会随行棋方交替符号，若不转成固定视角，`max(min, −max)` 恒为负、永不触发。
- **只作用于 full search**：cheap 分支的访问数本来就很低，不受影响。历史不足 `lookback` 步时不生效。

### 实验与结论

本仓库暂无受控实验。

---

## Policy(Value)SurpriseWeighting

### 解决什么问题

一局里最有训练价值的是「让网络意外」的局面——搜索找到了网络没看到的着法，或搜索价值与网络价值相差很大。但它们和其他局面权重相同。

### 做法

两种「意外度」：

- **PolicySurprise**：搜索访问分布相对加噪根先验的 KL。

  $$
  \text{surprise} = \sum_{\text{target}} \text{target}\cdot(\log \text{target} - \log \text{prior}) \ge 0
  $$

- **ValueSurprise**：平滑前瞻终局价值相对原始 NN WDL 的 KL。KataGo 默认版从真实终局 one-hot 出发，沿各步搜索价值反向做 TD 平滑：

  $$
  \text{future} \leftarrow \text{future} + \text{nowFactor}\cdot(\text{searchValue}_i - \text{future}), \qquad
  \text{nowFactor} = \frac{1}{1 + \text{area}\cdot 0.016}
  $$

  再算 $\text{KL}(\text{future} \,\|\, \text{nnValue})$，clamp 到 $[0,1]$。

对局结束后，把每步的权重 `weight` 重分配：

$$
\text{policyProp}_i = w_i\cdot\text{ps}_i + (1 - w_i)\cdot\max(0,\ \text{ps}_i - 1.5\cdot\overline{\text{ps}})
$$

$$
\text{valueProp}_i = w_i\cdot\text{vs}_i
$$

$$
w_i \leftarrow (1 - p_w - v_w)\cdot w_i + p_w\cdot\frac{\text{policyProp}_i}{\sum\text{policyProp}}\cdot\sum w + v_w\cdot\frac{\text{valueProp}_i}{\sum\text{valueProp}}\cdot\sum w
$$

其中第一项里的 $(1 - w_i)$ 项给「低权重行」（cheap 行 $w=0$、或被 SoftResign 降权的行）一个额外机会：即使搜索很浅，只要意外度超过阈值 $1.5\overline{\text{ps}}$，也能重新获得权重。最后按权重决定写入训练数据的份数：$\lfloor w_i \rfloor$ 份，再以 $w_i - \lfloor w_i \rfloor$ 的概率多写一份。

### 本仓库实现

- `policy_surprise(prior_policy, mcts_policy)`、`value_surprise(...)`、`redistribute_surprise_weights(...)`、`finish_game_samples(...)` 都在 `utils.py`。
- `policy_surprise_data_weight = 0.5`、`value_surprise_data_weight = 0.1`；当 value surprise 的加权均值低于 `0.010` 时按比例衰减 value 权重，避免除以接近 0 的量。
- `finish_game_samples` 先按 `nowFactor = 1/(1 + board_size²·0.016)` 反向平滑算 value surprise，再重分配权重，再按权重重复写入样本。
- PolicySurprise 的先验用的是搜索里实际施加的**温度 + 噪声**后的根先验：KataGo 的 `maybeAddPolicyNoiseAndTemp` 也是先温度、后噪声地写进 `noisedPolicyProbs`，两者一致。
- 权重用「写入份数」表达，因此 `train_step` 用普通均值（不再需要加权 loss）。cheap 行会以 `weight = 0` 记录，只有通过上面的 $(1-w_i)$ 项才可能被写入。

### 实验与结论

本仓库暂无受控实验。KataGo 官方认为 PolicySurprise 是较大的提升之一，而 ValueSurprise 更偏实验性、缺少严格验证。

---

## 未实现 / 后续

**搜索 / 训练**

- 强制探索、策略目标剪枝（policy target pruning）、cheap 位置的重新搜索（reanalyze）。
- Subtree value bias、dynamic variance cPUCT、optimistic policy、辅助短程价值目标。
- FPU 的 `fpuLossProp` / `rootFpuLossProp`。

**自对弈数据生成**

- **PDA**（PlayoutDoublingAdvantage）：以很低概率让一方获得非对称的更多 playouts（KataGo 配置 `normalAsymmetricPlayoutProb`、`maxAsymmetricRatio`），并把该优势作为网络输入特征，使网络对不同搜索预算稳健，也用于让子评估。本仓库未实现。
- **SidePosition**：落子后以一定概率额外叉出一个「旁支」局面（用温度采样或随机合法着法走出），对它单独做一次完整搜索并生成额外训练行（KataGo 配置 `sidePositionProb`），增加真实轨迹之外的数据覆盖。本仓库未实现。

**围棋专用 / 其它**

- 目数、归属、全局池化；graph search。

**评估**

- 系统化的棋力评估入口与固定对手。

参考：[KataGo 论文](https://arxiv.org/abs/1902.10565)、[KataGo 官方方法说明](https://github.com/lightvector/KataGo/blob/master/docs/KataGoMethods.md)、[原始 KataGo 仓库](https://github.com/lightvector/KataGo)。

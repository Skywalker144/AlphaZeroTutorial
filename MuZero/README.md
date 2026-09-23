# MuZero

用 Python、NumPy 和 PyTorch 实现 MuZero：搜索树里的节点保存**学习的隐状态**，节点之间的转移由**动力学网络**计算，而不是用真实规则算出下一棋盘。支持井字棋和自由规则五子棋，提供串行与批量推理并行两种自我对弈后端。

三个网络的分工、隐状态 MCTS 的流程、以及 K 步展开的训练目标见 [MuZero 技巧](../docs/MuZero.md)。

## 环境与入口

使用 Python 3.10 或更高版本。从仓库根目录进入 `MuZero/` 后执行安装、训练、对弈和测试命令：

```bash
cd MuZero
python -m pip install -r requirements.txt

# 井字棋
python -m tictactoe.train
python -m tictactoe.play
python -m tictactoe.play -n 0      # 纯网络，不做 MCTS 搜索
python -m tictactoe.play -n 1000   # 搜索 1000 次

# 五子棋
python -m gomoku.train
python -m gomoku.play

# 测试
python -m pip install pytest
python -m pytest tests/ -q
```

本机使用 Conda 环境时先执行 `conda activate pytorch`；非交互命令也可使用 `conda run -n pytorch python ...`。

设备选择见 [auto_device](alphazero/utils.py)：优先 CUDA，其次 MPS，最后 CPU。对弈入口额外支持命令行参数 `-n/--num-simulations`，`0` 表示纯网络。

## 代码阅读顺序

| 模块 | 内容 |
| --- | --- |
| [井字棋环境](envs/tictactoe.py)、[五子棋环境](envs/gomoku.py) | 状态、合法动作、落子、胜负与网络输入编码 |
| [network.py](alphazero/network.py) | 三个模块：Representation `h`、Dynamics `g`（输入隐状态 + 动作编码）、Prediction `f` |
| [mcts.py](alphazero/mcts.py) | 隐状态 MCTS：根用 `h`+`f` 并对真实棋盘屏蔽合法动作，树内用 `g`+`f` 按完整动作空间展开 |
| [alphazero_parallel.py](alphazero/alphazero_parallel.py) | 并行自我对弈：跨对局合并 batch，两段式推理（先 `h`/`g` 再 `f`） |
| [replay_buffer.py](alphazero/replay_buffer.py) | 整局历史、窗口裁剪、按起点构造 K 步展开样本 |
| [trainer.py](alphazero/trainer.py) | 自我对弈、K 步展开训练、损失与 checkpoint |
| [utils.py](alphazero/utils.py) | Dirichlet 噪声、含动作重映射的棋盘对称增强、设备选择与棋盘显示 |
| [tests/](tests/) | 游戏规则、搜索、三网络训练、checkpoint 与回放采样测试 |

两个游戏均为交替行动的双人零和棋盘游戏。五子棋默认使用 9×9 棋盘，连续五子及以上获胜，不含禁手。

## 配置与训练产物

训练配置分别位于 [tictactoe/train.py](tictactoe/train.py) 和 [gomoku/train.py](gomoku/train.py) 的 `train_args` 中（含 `mode`：训练为 `train`，对弈时自动切为 `eval` 以关闭 Dirichlet 噪声）。MuZero 特有的关键参数是 `unroll_steps`（展开步数 K，默认 5）。

训练产物默认生成在对应游戏目录下的 `data/`，例如运行 `python -m tictactoe.train` 会写入 `tictactoe/data/`，内含 `models/`、`checkpoints/` 和统计图片。训练流程、checkpoint、动态回放窗口与绘图行为与 AlphaZero 路线一致，详见 [AlphaZero README](../AlphaZero/README.md#配置与训练产物)。

## 当前行为与边界

- 自我对弈默认走**并行**后端（跨对局把待评估节点合并成 batch 推理，`parallel=False` 退回串行）；`num_parallel_games` 控制同时活跃的对局数，默认 32。`num_parallel_games=1` 时并行后端与串行逐位一致，见 [tests/test_parallel.py](tests/test_parallel.py) 的等价性测试。
- 隐状态搜索中，树内节点按**完整动作空间**展开，不查询真实棋盘的合法动作，也不在树内判断终局；只有根节点用环境给出的合法动作屏蔽先验。真实规则仍用于推进实际对局和判定胜负。
- PUCT 探索系数采用 `pb_c_init + log((N + pb_c_base + 1) / pb_c_base)`，其中 `N` 是父节点访问次数，默认 `pb_c_init=1.25`、`pb_c_base=19652`；串行与并行使用相同公式。选择阶段先把已访问子节点的 Q 转为父节点玩家视角，再按 `(Q + 1) / 2` 缩放到 `[0,1]`，未访问动作的价值项为 `0`，与官方棋类配置一致。网络输出、价值回传和节点统计仍使用 `[-1,1]`。
- 按 [MuZero.md](../docs/MuZero.md) 的棋类约定，**省略 reward 分支**：value 直接学习最终胜负（标量 `tanh`，不是胜/平/负三分类），训练时终局之后的步屏蔽 policy loss、value 目标按交替视角沿用最终胜负。根观测编码玩家身份，`g` 仅接收隐状态和动作，隐状态在终局后仍会继续更新。
- 训练沿 `h → g → g → …` 展开 K 步，K+1 组 policy/value 损失的梯度同时更新 `h`、`g`、`f`。
- `h`、`g` 按样本将隐状态归一化到 `[0,1]`；训练采用官方伪代码的预测损失梯度缩放和循环隐状态梯度缩放，具体位置与日志损失含义见 [训练更新](../docs/MuZero.md#4完整的训练更新)。
- 当前没有系统化的棋力评估入口；自我对弈胜率与训练损失只用于观察流程。

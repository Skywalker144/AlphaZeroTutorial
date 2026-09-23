# KataGo

在 [AlphaZero 实现](../AlphaZero/README.md) 的基础上，逐项加入 KataGo 的搜索与训练技巧。使用 Python、NumPy 和 PyTorch，支持井字棋与自由规则五子棋，无需编译 C++，也不依赖原生 KataGo 引擎。

各技巧的原理、公式、与 KataGo 的差异见 [KataGo 技巧](../docs/KataGo.md)。

## 环境与入口

使用 Python 3.10 或更高版本。从仓库根目录进入 `KataGo/` 后执行安装、训练、对弈和测试命令：

```bash
cd KataGo
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

设备选择见 [auto_device](alphazero/utils.py)：优先 CUDA，其次 MPS，最后 CPU。训练配置中的 `device` 可显式指定设备。对弈入口额外支持命令行参数 `-n/--num-simulations`，`0` 表示纯网络。

## 代码阅读顺序

| 模块 | 内容 |
| --- | --- |
| [井字棋环境](envs/tictactoe.py)、[五子棋环境](envs/gomoku.py) | 状态、合法动作、落子、胜负与网络输入编码 |
| [network.py](alphazero/network.py) | ResNet 主干、策略 logits 和胜/平/负三分类 value head |
| [mcts.py](alphazero/mcts.py) | 串行搜索：PUCT + FPU 选择、扩展、WDL 回传、树复用、根温度与 shaped 噪声 |
| [alphazero_parallel.py](alphazero/alphazero_parallel.py) | 并行自我对弈与批量推理，算法与串行版保持一致 |
| [replay_buffer.py](alphazero/replay_buffer.py) | 动态回放窗口、样本保留与随机采样 |
| [trainer.py](alphazero/trainer.py) | 自我对弈、PlayoutCapRandomization、训练调度、损失与 checkpoint |
| [utils.py](alphazero/utils.py) | 访问次数调度、温度、Dirichlet 噪声、SoftResign、Surprise 权重、棋盘对称增强与设备选择 |
| [tests/](tests/) | 游戏规则、搜索、训练、恢复与串行/并行等价性测试 |

两个游戏均为交替行动的双人零和棋盘游戏。五子棋默认使用 9×9 棋盘，连续五子及以上获胜，不含禁手。环境接口和动作编码以对应实现为准。

## 配置与训练产物

训练配置分别位于 [tictactoe/train.py](tictactoe/train.py) 和 [gomoku/train.py](gomoku/train.py) 的 `train_args` 中（含 `mode`：训练为 `train`，对弈时自动切为 `eval` 以关闭根噪声与温度）。所有 KataGo 技巧的参数都从 `args` 读取并带默认值，直接写在配置里即可覆盖；各参数的默认值见 [KataGo 技巧](../docs/KataGo.md) 对应小节。

训练产物默认生成在对应游戏目录下的 `data/`，例如运行 `python -m tictactoe.train` 会写入 `tictactoe/data/`，内含 `models/`、`checkpoints/` 和统计图片。训练流程、checkpoint、回放窗口与绘图行为与 AlphaZero 路线一致，详见 [AlphaZero README](../AlphaZero/README.md#配置与训练产物)。

## 已实现的技巧

- [PlayoutCapRandomization](../docs/KataGo.md#playoutcaprandomization)
- [TreeReuse](../docs/KataGo.md#treereuse)
- [WDLValueHead](../docs/KataGo.md#wdlvaluehead)
- [FPU](../docs/KataGo.md#fpu)
- [RootTemperature / ChosenMoveTemperature](../docs/KataGo.md#roottemperature--chosenmovetemperature)
- [ShapedDirichletNoise](../docs/KataGo.md#shapeddirichletnoise)
- [SoftResign](../docs/KataGo.md#softresign)
- [Policy(Value)SurpriseWeighting](../docs/KataGo.md#policyvaluesurpriseweighting)

## 当前行为与边界

- 自我对弈默认使用并行对局与批量网络推理；设置 `parallel=False` 时使用逐局、逐节点的串行实现。两条路径有等价性测试，教学以串行版为准。
- 训练数据按对局结束后重分配的权重决定写入份数（`floor(weight)` 份 + 概率性一份），cheap 位置以 `weight = 0` 记录。
- 对弈入口自动加载最新可用的模型权重，并在 AlphaZero 落子时打印 MCTS 访问分布与根节点价值估计。
- checkpoint 尚未保存完整配置与随机数状态，因此不保证逐步精确复现；全局迭代编号会随 checkpoint 保存并在续训时恢复。
- 当前没有系统化的棋力评估入口；自我对弈中的黑白胜率与训练损失只用于观察流程。
- 这里是可运行的教学子集，不是完整 KataGo：强制探索、策略目标剪枝、cheap 位置重新搜索、围棋专用辅助目标等见 [未实现 / 后续](../docs/KataGo.md#未实现--后续)。

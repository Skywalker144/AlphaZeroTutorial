# AlphaZero

用 Python、NumPy 和 PyTorch 实现策略价值网络、MCTS、自我对弈与训练，支持井字棋和自由规则五子棋，无需编译 C++。

当前提供可运行的训练与命令行对弈代码，不附带预训练模型；系统化教程和固定对手评估尚待完善。

## 环境与入口

使用 Python 3.10 或更高版本。从仓库根目录进入 `AlphaZero/` 后执行安装、训练、对弈和测试命令：

```bash
cd AlphaZero
python -m pip install -r requirements.txt

# 井字棋
python -m tictactoe.train
python -m tictactoe.play

# 五子棋
python -m gomoku.train
python -m gomoku.play

# 测试
python -m pip install pytest
python -m pytest tests/ -q
```

本机使用 Conda 环境时先执行 `conda activate pytorch`；非交互命令也可使用 `conda run -n pytorch python ...`。

设备选择见 [auto_device](alphazero/utils.py)：优先 CUDA，其次 MPS，最后 CPU。训练配置中的 `device` 可显式指定设备。

## 代码阅读顺序

| 模块 | 内容 |
| --- | --- |
| [井字棋环境](envs/tictactoe.py)、[五子棋环境](envs/gomoku.py) | 状态、合法动作、落子、胜负与网络输入编码 |
| [network.py](alphazero/network.py) | ResNet 主干、策略 logits 和价值输出 |
| [mcts.py](alphazero/mcts.py) | PUCT 选择、扩展、视角转换与价值回传 |
| [replay_buffer.py](alphazero/replay_buffer.py) | 动态回放窗口、样本保留与随机采样 |
| [trainer.py](alphazero/trainer.py) | 自我对弈、训练调度、损失、checkpoint 与绘图 |
| [utils.py](alphazero/utils.py) | 根节点噪声、棋盘对称增强、设备选择与棋盘显示 |
| [tests/](tests/) | 游戏规则、搜索、训练和恢复测试 |

两个游戏均为交替行动的双人零和棋盘游戏。五子棋默认使用 9×9 棋盘，连续五子及以上获胜，不包含禁手。环境接口和动作编码以对应实现为准。

## 配置与训练产物

训练配置分别位于 [tictactoe/train.py](tictactoe/train.py) 和 [gomoku/train.py](gomoku/train.py) 的 `train_args` 中；当前没有单独的 `config.py` 或命令行参数解析。训练产物默认生成在对应游戏目录下的 `data/`，例如运行 `python -m tictactoe.train` 会写入 `tictactoe/data/`，内含 `models/`、`checkpoints/` 和统计图片。

- 迭代从 0 开始：第 0 轮收集固定的 `bootstrap_games` 局用于估计平均行长，第 1 轮据此补跑到 `min_rows`（乘以 1.05 的余量因子），之后按目标回放比自适应。
- 默认持续训练，`Ctrl+C` 或正常结束都会保存完整 checkpoint 和曲线。`num_iterations` 表示目标总迭代数，达到后停止；不设置则持续训练。
- 训练开始时自动恢复 `data/checkpoints/checkpoint.pth`（若存在），迭代编号、回放数据与训练统计都接着继续。开启独立实验时使用不同的 `data_dir`。
- 每 `save_interval` 轮迭代：模型权重另存为 `data/models/model_<iter>.pth`，完整状态覆盖写入 `data/checkpoints/checkpoint.pth`（只保留最新一份，用于续训）。
- 每个迭代结束后都会更新 `data/` 根目录下的训练面板图 `training.png`（损失、胜率、行长度的深色 2×2 dashboard），并导出可复现绘图的 `losses.csv` 与 `games.csv`。这些产物不纳入 Git。
- 对弈入口会询问 checkpoint 路径；回车选择最新文件，没有文件时使用随机权重。落子输入为从零开始的行、列坐标。

默认配置面向持续训练。仅检查流程时，应使用较小网络、较少搜索次数、较低的 `min_rows`、较小批次和有限的训练轮数；同时确保回放缓冲区能提供完整批次。

## 当前行为与边界

- 自我对弈逐局执行，搜索逐节点推理；没有并行自我对弈或批量搜索推理。
- 自我对弈按搜索访问次数分布采样落子，没有随对局进度变化的温度调度；对弈入口按最大访问次数选择动作，但目前沿用训练配置中的根节点噪声。
- 回放窗口随累计数据量增长，采样局数按目标回放比调度，属于基础 AlphaZero 流程之外的训练策略。
- checkpoint 尚未保存完整配置与随机数状态，因此不保证逐步精确复现；全局迭代编号会随 checkpoint 保存并在续训时恢复。
- 自我对弈中的黑白胜率与训练损失用于观察流程，不代表相对于固定对手的棋力；当前没有自动化棋力评估入口。

原始算法参考：[AlphaZero 论文](https://arxiv.org/abs/1712.01815)。当前实现的配置、训练策略和运行规模不等同于论文实验。

# AlphaZeroTutorial

用 Python 理解和实现 AlphaZero、KataGo 与 MuZero。重点是算法原理、可阅读的实现和可复现的小规模实验。

AlphaZero 路线提供井字棋和五子棋的 Python 实现，包含自我对弈、训练、模型保存与恢复、命令行对弈和测试。KataGo 路线在此基础上实现了部分搜索与训练技巧。MuZero 仍为路线规划；目前没有系统化的棋力评估结论。

## 三条路线

每条路线有两份文档：`docs/` 下的**教学文档**讲算法原理，路线目录内的 `README.md` 讲怎么运行。

| 路线 | 学习重点 | 教学文档 | 使用文档 |
| --- | --- | --- | --- |
| AlphaZero | 游戏环境、策略与价值网络、MCTS、自我对弈、训练与评估 | [AlphaZero.md](docs/AlphaZero.md) | [README](AlphaZero/README.md) |
| KataGo | 搜索与训练效率改进，以及逐项对照实验 | [KataGo.md](docs/KataGo.md) | [README](KataGo/README.md) |
| MuZero | 学习环境模型、隐状态搜索与序列训练 | [MuZero.md](docs/MuZero.md) | [README](MuZero/README.md) |

建议先完成 AlphaZero，再按兴趣进入 KataGo 或 MuZero；研究 MuZero 无需先完成 KataGo。

## 实现方式

- 游戏逻辑、搜索和训练流程使用 Python，网络使用 PyTorch，数组处理使用 NumPy。
- 不需要编写或编译 C++/CUDA 扩展，不依赖原生 KataGo 引擎；可使用 PyTorch 提供的 GPU 加速。
- 算法教学文档集中在 `docs/`；每条路线的代码、配置、测试和实验输出放在自己的目录内。
- 初期优先构建易于阅读的小规模闭环；批量推理等性能优化在基础行为正确后逐项引入。
- 入门建议从已提供的井字棋开始，再扩展到默认 9×9 的自由规则五子棋；具体章节安排尚待完善。

KataGo 路线以明确标注的算法子集开展教学和实验，围棋专用目标是否实现取决于所选游戏。MuZero 路线保留模型学习的核心区别：搜索使用学习模型预测后继隐状态与奖励，环境仍用于真实交互。

# AlphaZeroTutorial

用 Python 理解和实现 AlphaZero、KataGo 与 MuZero。重点是算法原理、可阅读的实现和可复现的小规模实验。

AlphaZero 路线提供井字棋和五子棋的 Python 实现，包含自我对弈、训练、模型保存与恢复、命令行对弈和测试；运行方式见 [AlphaZero README](AlphaZero/README.md)。KataGo 和 MuZero 目前仍为路线规划，尚无训练棋力结论。

## 三条路线

| 目录 | 学习重点 | 与基础路线的关系 |
| --- | --- | --- |
| [AlphaZero](AlphaZero/README.md) | 游戏环境、策略与价值网络、MCTS、自我对弈、训练与评估 | 建立完整基础流程 |
| [KataGo](KataGo/README.md) | 搜索与训练效率改进，以及逐项对照实验 | 在理解 AlphaZero 后研究改进方法 |
| [MuZero](MuZero/README.md) | 学习环境模型、隐状态搜索与序列训练 | 在理解 AlphaZero 后研究模型学习与规划 |

建议先完成 AlphaZero，再按兴趣进入 KataGo 或 MuZero；研究 MuZero 无需先完成 KataGo。

## 实现方式

- 游戏逻辑、搜索和训练流程使用 Python，网络使用 PyTorch，数组处理使用 NumPy。
- 不需要编写或编译 C++/CUDA 扩展，不依赖原生 KataGo 引擎；可使用 PyTorch 提供的 GPU 加速。
- 每条路线的教程、代码、配置、测试和实验输出都放在自己的目录内。
- 初期优先构建易于阅读的小规模闭环；批量推理等性能优化在基础行为正确后逐项引入。
- 入门建议从已提供的井字棋开始，再扩展到默认 9×9 的自由规则五子棋；具体章节安排尚待完善。

KataGo 路线以明确标注的算法子集开展教学和实验，围棋专用目标是否实现取决于所选游戏。MuZero 路线保留模型学习的核心区别：搜索使用学习模型预测后继隐状态与奖励，环境仍用于真实交互。

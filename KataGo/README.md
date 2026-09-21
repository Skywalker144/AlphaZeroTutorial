# KataGo

状态：路线规划，尚未实现。

目标是用 Python 研究 KataGo 对 AlphaZero 搜索与训练流程的改进，采用逐项加入、逐项对照的教学方式。具体纳入的方法与实验顺序尚待确定。

## 候选内容

- 搜索预算随机化（playout cap randomization）。
- 强制探索（forced playouts）与策略目标剪枝（policy target pruning）。
- 全局池化、辅助预测目标及其对所选游戏的适用性。
- 回放窗口、数据采样与训练量之间的关系。

每项实验说明方法来源、适用条件、实现差异及比较预算。搜索或网络结构变化不能只凭训练损失判断棋力提升。

本路线实现的是明确标注范围的教学子集，不依赖原生 KataGo 引擎。围棋目数、归属等专用目标仅在游戏规则和训练标签支持时引入；用于五子棋等其他游戏时须说明取舍。

参考：[KataGo 论文](https://arxiv.org/abs/1902.10565)、[官方方法说明](https://github.com/lightvector/KataGo/blob/master/docs/KataGoMethods.md)。

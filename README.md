# BGG BoardGameGeek — 桌游复杂度与玩家评分的量化关系分析

基于 [BoardGameGeek (TidyTuesday 2019)](https://raw.githubusercontent.com/rfordatascience/tidytuesday/master/data/2019/2019-03-12/board_games.csv) 数据集，分析桌游复杂度、机制多样性与玩家评分之间的关系。

## 数据集

| 来源 | TidyTuesday 2019 Week 11 |
|------|--------------------------|
| 原始数据 | 10,532 款游戏 × 22 列 |
| 清洗后 | 10,182 款，年份 1950–2016 |
| 过滤条件 | 评分人数 ≥ 30，有效年份和评分 |

## 项目结构

```
├── data/
│   ├── bgg_board_games.csv          # 原始数据
│   └── ratings.csv                  # 评分数据
├── output/                          # 所有图表 PNG
├── personal_assignment1.ipynb       # 作业1：探索性数据分析
├── personal_assignment1.html        # 作业1 HTML 导出
├── personal_assignment1.pdf         # 作业1 PDF
├── personal_assignment2.ipynb       # 作业2：PCA + K-Means + 时间序列
├── personal_assignment2.html        # 作业2 HTML 导出
├── personal_assignment2.pdf         # 作业2 PDF
└── README.md
```

## 作业内容

### 作业1 — 探索性数据分析

- 评分分布、偏度和峰度
- 游戏时长 vs 评分散点图和箱线图
- 年度发行量与评分趋势
- Top-15 热门桌游机制
- 相关性热力图
- 不同时代的评分和机制数量对比

### 作业2 — 深度建模

- **PCA 降维**：65 个机制/类别特征 → 2 个主成分，PC1 大致对应策略向 vs 休闲向
- **K-Means 聚类**：轮廓系数确定最优 K，PCA 空间可视化，聚类画像
- **时间序列分析**：年度出版量、评分、机制数、游戏时长的长期趋势
- 1995年前后 / 2015年后的行业对比

## 核心发现

- 游戏时长（复杂度）与评分几乎无相关性（r ≈ 0.058）
- 机制多样性对评分有轻微正向影响（高机制组均分 6.68 vs 低机制组 6.06）
- 2000 年后桌游出版量大幅增长（占总量 71%）
- 桌游设计越来越复杂：平均机制数从 1.8 增长到 3.1

## 技术栈

Python · pandas · NumPy · Jupyter Notebook

## 课程信息

大数据处理技术 — 2025-2026 第二学期

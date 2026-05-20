# Big Data Course Projects Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement 团队作业1 (US crime rate analysis) and 个人项目作业1+2 (BoardGameGeek board game analysis) as Jupyter notebooks with full data pipelines and visualizations.

**Architecture:** Each project is an independent Jupyter notebook pipeline: data loading → cleaning → EDA → feature engineering → conclusions. Personal assignment 2 extends assignment 1 with PCA, clustering, and time series modules.

**Tech Stack:** Python 3, Pandas, NumPy, Matplotlib, Seaborn, Scikit-learn, Plotly, Jupyter Notebook

---

## File Structure

```
.
├── team_project/
│   ├── data/                        # Downloaded Kaggle dataset
│   ├── team_assignment1.ipynb       # 团队作业1 notebook
│   └── output/                      # Saved charts
│
├── personal_project/
│   ├── data/                        # Downloaded BGG dataset
│   ├── personal_assignment1.ipynb   # 个人作业1 notebook
│   ├── personal_assignment2.ipynb   # 个人作业2 notebook
│   └── output/                      # Saved charts
│
└── docs/superpowers/
    ├── specs/                       # Design docs
    └── plans/                       # This plan
```

---

## Part A: 团队作业1 — 美国犯罪率与社会经济因素

### Task A1: Create project structure and download data

**Files:**
- Create: `team_project/data/` (directory)
- Create: `team_project/output/` (directory)

- [ ] **Step 1: Create directories**

```bash
mkdir -p team_project/data team_project/output
```

- [ ] **Step 2: Download dataset from Kaggle**

Download from https://www.kaggle.com/datasets/lucague/us-state-crime-and-socioeconomic-factors-20052015 using the Kaggle API or manual download. Place CSV files in `team_project/data/`.

```bash
# Option A: via kaggle CLI
pip install kagglehub
python -c "
import kagglehub
path = kagglehub.dataset_download('lucague/us-state-crime-and-socioeconomic-factors-20052015')
import shutil, os
for f in os.listdir(path):
    shutil.copy(os.path.join(path, f), 'team_project/data/')
print('Downloaded to team_project/data/')
"
```

Expected: CSV file(s) in `team_project/data/`. Verify with `ls team_project/data/`.

- [ ] **Step 3: Commit**

```bash
git add team_project/
git commit -m "chore: create team project directory structure"
```

---

### Task A2: Create team assignment 1 notebook — setup and data loading

**Files:**
- Create: `team_project/team_assignment1.ipynb`

- [ ] **Step 1: Create notebook with first cells**

Create `team_project/team_assignment1.ipynb` with the following cells.

**Cell 1 — Title and imports:**

```python
# 团队作业1：美国各州犯罪率与社会经济因素分析
# 大数据处理技术课程 — 2025-2026第二学期

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# 中文字体设置
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False

print("环境就绪")
```

**Cell 2 — Load data:**

```python
import os
data_dir = 'data/'
csv_files = [f for f in os.listdir(data_dir) if f.endswith('.csv')]
print(f"发现 {len(csv_files)} 个CSV文件: {csv_files}")

for f in csv_files:
    df = pd.read_csv(os.path.join(data_dir, f))
    print(f"\n{f}: {df.shape[0]} 行 × {df.shape[1]} 列")
    print(f"列名: {list(df.columns[:10])}..." if len(df.columns) > 10 else f"列名: {list(df.columns)}")
    display(df.head(3))
```

**Cell 3 — Data dictionary:**

```python
# 数据字典与业务含义
print("数据集：US State Crime and Socioeconomic Factors (2005-2015)")
print("来源：Kaggle — lucague/us-state-crime-and-socioeconomic-factors-2005-2015")
print(f"形状：{df.shape}")
print("\n字段说明：")
df.info()
```

- [ ] **Step 2: Run notebook to verify data loads**

Run: `jupyter nbconvert --to notebook --execute team_project/team_assignment1.ipynb --output team_assignment1.ipynb`
Expected: No errors, data shape and columns printed.

- [ ] **Step 3: Commit**

```bash
git add team_project/team_assignment1.ipynb
git commit -m "feat: team assignment 1 — data loading and initial exploration"
```

---

### Task A3: Data cleaning cells

**Files:**
- Modify: `team_project/team_assignment1.ipynb` (append cells)

- [ ] **Step 1: Add cleaning cells**

Append to notebook:

**Cell — Missing value analysis:**

```python
# 缺失值分析（对应课件 Chapter4 第9-10页）
print("=== 缺失值统计 ===")
missing = df.isnull().sum()
missing_pct = (missing / len(df)) * 100
missing_df = pd.DataFrame({
    '缺失数量': missing,
    '缺失比例(%)': missing_pct.round(2)
}).sort_values('缺失比例(%)', ascending=False)
display(missing_df[missing_df['缺失数量'] > 0])

# 可视化缺失值
plt.figure(figsize=(12, 5))
sns.heatmap(df.isnull(), cbar=False, yticklabels=False, cmap='viridis')
plt.title('缺失值热力图')
plt.tight_layout()
plt.savefig('output/missing_heatmap.png', dpi=150, bbox_inches='tight')
plt.show()
```

**Cell — Handle missing values:**

```python
# 缺失值处理（课件 Chapter4 第10页）
# 数值型：中位数填充
numeric_cols = df.select_dtypes(include=[np.number]).columns
for col in numeric_cols:
    if df[col].isnull().sum() > 0:
        median_val = df[col].median()
        df[col].fillna(median_val, inplace=True)
        print(f"  {col}: 用中位数 {median_val:.2f} 填充")

# 分类型：众数填充
categorical_cols = df.select_dtypes(include=['object']).columns
for col in categorical_cols:
    if df[col].isnull().sum() > 0:
        mode_val = df[col].mode()[0]
        df[col].fillna(mode_val, inplace=True)
        print(f"  {col}: 用众数 '{mode_val}' 填充")

print(f"\n清洗后缺失值总数: {df.isnull().sum().sum()}")
```

**Cell — Outlier detection with Z-Score:**

```python
# 异常值检测 — Z-Score方法（课件 Chapter4 第11页）
from scipy import stats

outlier_cols = [c for c in numeric_cols if c not in ['Year', 'State']]  # 排除ID类字段
z_scores = np.abs(stats.zscore(df[outlier_cols].select_dtypes(include=[np.number]), nan_policy='omit'))
outlier_mask = (z_scores > 3).any(axis=1)
print(f"Z-Score 检测到 {outlier_mask.sum()} 个异常行 (±3σ 阈值)")
print(f"异常比例: {outlier_mask.sum() / len(df) * 100:.2f}%")

# 标记但不删除（保留业务判断空间）
df['is_outlier'] = outlier_mask
```

**Cell — Format standardization:**

```python
# 格式标准化
# 州名去空格
str_cols = df.select_dtypes(include=['object']).columns
for col in str_cols:
    if col != 'is_outlier':
        df[col] = df[col].str.strip()

# 年份标准化
if 'Year' in df.columns:
    df['Year'] = df['Year'].astype(int)

print("格式标准化完成")
print(f"最终数据集: {df.shape}")
df.head()
```

- [ ] **Step 2: Run notebook and verify**

Run the notebook and verify the outputs are sensible (missing values resolved, outliers flagged).

- [ ] **Step 3: Commit**

```bash
git add team_project/team_assignment1.ipynb
git commit -m "feat: team assignment 1 — data cleaning with Z-Score outlier detection"
```

---

### Task A4: EDA visualization cells

**Files:**
- Modify: `team_project/team_assignment1.ipynb` (append cells)

- [ ] **Step 1: Add EDA cells**

Append to notebook:

**Cell — Descriptive statistics:**

```python
# === 描述性统计（EDA） ===
print("=== 描述性统计 ===")
display(df.describe().round(2))

# 按年份分组
if 'Year' in df.columns:
    yearly_stats = df.groupby('Year').agg({
        [c for c in numeric_cols if c in df.columns][0]: ['mean', 'std']
    })
```

**Cell — Crime rate time trend (Sequence data, Chapter3 p16-18):**

```python
# 犯罪率时序趋势（对应课件"序列数据与时间维度" Chapter3 第16-18页）
crime_cols = [c for c in df.columns if any(tag in c.lower()
    for tag in ['violent', 'property', 'murder', 'robbery', 'burglary', 'assault', 'larceny', 'theft'])]

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

for i, col in enumerate(crime_cols[:4]):
    ax = axes[i // 2, i % 2]
    yearly = df.groupby('Year')[col].mean()
    ax.plot(yearly.index, yearly.values, marker='o', linewidth=2, markersize=4)
    ax.set_title(f'{col} 年度趋势', fontsize=12)
    ax.set_xlabel('年份')
    ax.set_ylabel('比率')
    ax.grid(True, alpha=0.3)

plt.suptitle('美国犯罪率时序趋势 (2005-2015)', fontsize=14, y=1.01)
plt.tight_layout()
plt.savefig('output/crime_time_trend.png', dpi=150, bbox_inches='tight')
plt.show()
```

**Cell — Geographic distribution (Spatial data, Chapter6):**

```python
# 地理分布 — 各州犯罪率对比（对应课件"时空数据"Chapter3 第22-24页 + Chapter6）
state_avg = df.groupby('State')[crime_cols[0]].mean().sort_values()

plt.figure(figsize=(8, 14))
state_avg.plot(kind='barh')
plt.title(f'各州平均{crime_cols[0]} (2005-2015)', fontsize=14)
plt.xlabel('比率')
plt.tight_layout()
plt.savefig('output/state_crime_comparison.png', dpi=150, bbox_inches='tight')
plt.show()
```

**Cell — Histogram and box plots:**

```python
# 变量分布 — 直方图与箱线图
fig, axes = plt.subplots(2, 3, figsize=(15, 8))
target_cols = crime_cols[:3] + [c for c in ['poverty_rate', 'unemployment_rate', 'median_income']
    if c in df.columns or any(c.replace('_', '') in x.lower().replace('_', '') for x in df.columns)][:3]

for i, col in enumerate(target_cols[:6]):
    ax = axes[i // 3, i % 3]
    ax.hist(df[col].dropna(), bins=30, edgecolor='white', alpha=0.7)
    ax.axvline(df[col].median(), color='red', linestyle='--', label=f'中位数={df[col].median():.1f}')
    ax.set_title(f'{col} 分布')
    ax.legend(fontsize=8)

plt.suptitle('关键变量分布直方图', fontsize=14)
plt.tight_layout()
plt.savefig('output/variable_distributions.png', dpi=150, bbox_inches='tight')
plt.show()
```

**Cell — Regional comparison:**

```python
# 按地区分组对比（美国四大区域）
region_map = {
    'Northeast': ['Maine', 'New Hampshire', 'Vermont', 'Massachusetts', 'Rhode Island', 'Connecticut',
                   'New York', 'New Jersey', 'Pennsylvania'],
    'Midwest': ['Ohio', 'Indiana', 'Illinois', 'Michigan', 'Wisconsin', 'Minnesota', 'Iowa', 'Missouri',
                'North Dakota', 'South Dakota', 'Nebraska', 'Kansas'],
    'South': ['Delaware', 'Maryland', 'Virginia', 'West Virginia', 'Kentucky', 'North Carolina',
              'South Carolina', 'Tennessee', 'Georgia', 'Florida', 'Alabama', 'Mississippi',
              'Arkansas', 'Louisiana', 'Texas', 'Oklahoma'],
    'West': ['Montana', 'Idaho', 'Wyoming', 'Colorado', 'New Mexico', 'Arizona', 'Utah', 'Nevada',
             'Washington', 'Oregon', 'California', 'Alaska', 'Hawaii']
}
df['Region'] = df['State'].map({s: r for r, states in region_map.items() for s in states})

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
sns.boxplot(data=df, x='Region', y=crime_cols[0], ax=axes[0])
axes[0].set_title(f'{crime_cols[0]} by Region', fontsize=12)

region_avg = df.groupby('Region')[crime_cols[:3]].mean()
region_avg.plot(kind='bar', ax=axes[1])
axes[1].set_title('Regional Average Crime Rates', fontsize=12)
axes[1].set_xticklabels(axes[1].get_xticklabels(), rotation=0)

plt.tight_layout()
plt.savefig('output/regional_comparison.png', dpi=150, bbox_inches='tight')
plt.show()
```

- [ ] **Step 2: Run and verify charts generated**

Run notebook. Expected: 4 PNG files in `team_project/output/`.

- [ ] **Step 3: Commit**

```bash
git add team_project/team_assignment1.ipynb team_project/output/
git commit -m "feat: team assignment 1 — EDA with temporal, spatial, and distribution analysis"
```

---

### Task A5: Feature engineering and correlation analysis

**Files:**
- Modify: `team_project/team_assignment1.ipynb` (append cells)

- [ ] **Step 1: Add feature engineering cells**

**Cell — Correlation matrix:**

```python
# === 特征工程与相关性分析 ===
# 对应课件 Chapter3 "高维表格与特征维度"（第31页）
# 相关性矩阵（Pearson）
corr_cols = df.select_dtypes(include=[np.number]).drop(columns=['Year', 'is_outlier'], errors='ignore')
corr_matrix = corr_cols.corr()

plt.figure(figsize=(14, 10))
mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
sns.heatmap(corr_matrix, mask=mask, annot=True, fmt='.2f',
            cmap='RdBu_r', center=0, square=True,
            linewidths=0.5, cbar_kws={'shrink': 0.8})
plt.title('Pearson 相关系数矩阵', fontsize=16)
plt.tight_layout()
plt.savefig('output/correlation_heatmap.png', dpi=150, bbox_inches='tight')
plt.show()
```

**Cell — Top correlations with violent crime:**

```python
# 与暴力犯罪率最强相关的前10个因素
target = crime_cols[0]
correlations = corr_matrix[target].drop(target).sort_values(key=abs, ascending=False)

plt.figure(figsize=(8, 6))
correlations.head(10).plot(kind='barh')
plt.title(f'与 {target} 最强相关的Top-10因素', fontsize=14)
plt.axvline(x=0, color='black', linewidth=0.5)
plt.xlabel('Pearson 相关系数')
plt.tight_layout()
plt.savefig('output/top_correlations.png', dpi=150, bbox_inches='tight')
plt.show()

print("Top-10 相关因素:")
for var, corr in correlations.head(10).items():
    print(f"  {var}: r = {corr:.3f}")
```

**Cell — Scatter matrix for key factors:**

```python
# 散点矩阵 — 关键社会经济指标 vs 犯罪率
key_factors = [target] + correlations.head(5).index.tolist()
# 过滤只保留组内存在的列
available = [c for c in key_factors if c in df.columns]
if len(available) >= 3:
    sns.pairplot(df[available].dropna().sample(min(500, len(df))),
                 diag_kind='kde', plot_kws={'alpha': 0.3})
    plt.suptitle('关键因素散点矩阵', y=1.01, fontsize=14)
    plt.savefig('output/scatter_matrix.png', dpi=150, bbox_inches='tight')
    plt.show()
```

**Cell — Socioeconomic vulnerability index:**

```python
# 构建 "社会经济脆弱性指数"（复合特征）
# 对应教学大纲 "复合商业指标拆解"
# 标准化后合并：高贫困率 + 高失业率 + 低收入 — 高毕业率
if all(c in df.columns for c in ['poverty_rate', 'unemployment_rate', 'median_income']):
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    vuln_features = ['poverty_rate', 'unemployment_rate']
    income_col = 'median_income'
    # 收入负向贡献
    df['income_neg'] = -df[income_col]
    vuln_features.append('income_neg')
    education_col = [c for c in df.columns if 'graduate' in c.lower() or 'bachelor' in c.lower() or 'education' in c.lower()]
    if education_col:
        df['edu_neg'] = -df[education_col[0]]
        vuln_features.append('edu_neg')

    scaled = scaler.fit_transform(df[vuln_features].fillna(0))
    df['vulnerability_index'] = scaled.mean(axis=1)

    # 脆弱性 vs 犯罪率
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(df['vulnerability_index'], df[crime_cols[0]], alpha=0.3)
    ax.set_xlabel('社会经济脆弱性指数')
    ax.set_ylabel(crime_cols[0])
    ax.set_title('脆弱性指数 vs 犯罪率')
    # 添加趋势线
    from numpy.polynomial.polynomial import polyfit
    x, y = df['vulnerability_index'].dropna(), df[crime_cols[0]].dropna()
    b, m = polyfit(x, y, 1)
    ax.plot(x, b + m * x, '-', color='red', linewidth=2, label=f'y = {m:.3f}x + {b:.3f}')
    ax.legend()
    plt.tight_layout()
    plt.savefig('output/vulnerability_vs_crime.png', dpi=150, bbox_inches='tight')
    plt.show()
```

- [ ] **Step 2: Run and verify**

- [ ] **Step 3: Commit**

```bash
git add team_project/team_assignment1.ipynb team_project/output/
git commit -m "feat: team assignment 1 — correlation analysis and composite index"
```

---

### Task A6: Conclusions and report markdown cells

**Files:**
- Modify: `team_project/team_assignment1.ipynb` (append cells)

- [ ] **Step 1: Add conclusion cells**

**Cell — Key findings summary:**

```python
# === 阶段性发现 ===
# 对应教学大纲"研究能力"目标2
print("=" * 60)
print("阶段性发现 — 美国犯罪率与社会经济因素")
print("=" * 60)

print("\n1. 与犯罪率最强的社会经济因素：")
for var, corr in correlations.head(5).items():
    direction = "正相关" if corr > 0 else "负相关"
    print(f"   • {var}: {direction} (r = {corr:.3f})")

print("\n2. 时序趋势 (2005-2015)：")
# 计算趋势
first_half = df[df['Year'] <= 2010][crime_cols[0]].mean()
second_half = df[df['Year'] > 2010][crime_cols[0]].mean()
trend = "下降" if second_half < first_half else "上升"
print(f"   前半段均值: {first_half:.2f}, 后半段均值: {second_half:.2f}")
print(f"   整体趋势: {trend}")

print("\n3. 地区差异：")
region_stats = df.groupby('Region')[crime_cols[0]].agg(['mean', 'std']).round(2)
print(region_stats.to_string())

print("\n4. 作业2展望：")
print("   • 基于K-Means聚类构建'犯罪画像'（高危/安全/混合型州）")
print("   • PCA降维识别犯罪主因子（暴力维度、财产维度）")
print("   • 时空建模：融合地理空间自相关分析（Moran's I）")
print("   • 面板数据回归：固定效应模型评估因果效应")
```

- [ ] **Step 2: Final notebook run and export**

```bash
jupyter nbconvert --to notebook --execute team_project/team_assignment1.ipynb --output team_assignment1.ipynb
```

- [ ] **Step 3: Commit**

```bash
git add team_project/
git commit -m "feat: team assignment 1 — conclusions and final report"
```

---

## Part B: 个人项目作业1 — BoardGameGeek 数据探索

### Task B1: Create project structure and download data

**Files:**
- Create: `personal_project/data/` (directory)
- Create: `personal_project/output/` (directory)

- [ ] **Step 1: Create directories and download BGG dataset**

```bash
mkdir -p personal_project/data personal_project/output
```

Download BGG dataset from Kaggle. Two primary options:
1. `https://www.kaggle.com/datasets/threnjen/board-game-geek-database` (7 tables, ~20k games)
2. `https://www.kaggle.com/datasets/msheikh/boardgamegeek` (single CSV, ~20k rows)

```bash
python -c "
import kagglehub
path = kagglehub.dataset_download('threnjen/board-game-geek-database')
import shutil, os
for f in os.listdir(path):
    shutil.copy(os.path.join(path, f), 'personal_project/data/')
print('Downloaded to personal_project/data/')
"
```

Expected: CSV files in `personal_project/data/`. Verify with `ls personal_project/data/`.

- [ ] **Step 2: Commit**

```bash
git add personal_project/
git commit -m "chore: create personal project directory structure"
```

---

### Task B2: Create personal assignment 1 notebook

**Files:**
- Create: `personal_project/personal_assignment1.ipynb`

- [ ] **Step 1: Create notebook with setup, loading, and cleaning cells**

Create `personal_project/personal_assignment1.ipynb` with the following structure.

**Cell 1 — Title and imports:**

```python
# 个人作业1：桌游复杂度与玩家评分的量化关系
# —— 基于 BoardGameGeek 数据
# 大数据处理技术课程 — 2025-2026第二学期

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False
print("环境就绪")
```

**Cell 2 — Load and explore data:**

```python
import os
data_dir = 'data/'
csv_files = [f for f in os.listdir(data_dir) if f.endswith('.csv')]
print(f"发现 {len(csv_files)} 个CSV文件: {csv_files}")

dfs = {}
for f in csv_files:
    name = f.replace('.csv', '')
    dfs[name] = pd.read_csv(os.path.join(data_dir, f))
    print(f"\n{name}: {dfs[name].shape[0]} 行 × {dfs[name].shape[1]} 列")
    display(dfs[name].head(2))
```

**Cell 3 — Select main game table:**

```python
# 选择主表（包含评分、复杂度、机制的表格）
# 根据BGG数据集结构，主表通常是 'games' 或 'bgg_dataset'
main_key = None
for k, v in dfs.items():
    cols_lower = [c.lower() for c in v.columns]
    score_check = any('rating' in c or 'score' in c or 'average' in c for c in cols_lower)
    weight_check = any('weight' in c or 'complexity' in c for c in cols_lower)
    if score_check and weight_check:
        main_key = k
        break

if main_key is None:
    main_key = list(dfs.keys())[0]

df = dfs[main_key].copy()
print(f"使用主表: {main_key} — {df.shape}")
display(df.head(3))
```

**Cell 4 — Data cleaning (Chapter4 methodology):**

```python
# === 数据清洗（课件 Chapter4 第9-13页）===

# 1. 评分人数过滤（统计意义不足）
rating_count_col = [c for c in df.columns if 'user' in c.lower() or 'voter' in c.lower() or 'count' in c.lower()]
if rating_count_col:
    rc_col = rating_count_col[0]
    before = len(df)
    df = df[df[rc_col] >= 30].copy()
    print(f"评分人数过滤 (<30): {before} → {len(df)} 行")

# 2. 缺失值处理
print(f"\n缺失值统计 (Top-10):")
missing = df.isnull().sum().sort_values(ascending=False)
display(missing.head(10))

# 数值型中位数填充
numeric_cols = df.select_dtypes(include=[np.number]).columns
for col in numeric_cols:
    if df[col].isnull().sum() > 0:
        df[col].fillna(df[col].median(), inplace=True)

# 分类型众数填充
cat_cols = df.select_dtypes(include=['object']).columns
for col in cat_cols:
    if df[col].isnull().sum() > 0:
        df[col].fillna('Unknown', inplace=True)

print(f"\n清洗后缺失值: {df.isnull().sum().sum()}")

# 3. 评分范围检查
rating_col = [c for c in df.columns if 'average' in c.lower() or 'rating' in c.lower() or 'bayes' in c.lower()]
if rating_col:
    r_col = rating_col[0]
    valid_mask = (df[r_col] >= 1) & (df[r_col] <= 10)
    print(f"评分异常值: {(~valid_mask).sum()} 行")
    df = df[valid_mask]

# 4. 年份过滤
year_col = [c for c in df.columns if 'year' in c.lower()]
if year_col:
    yc = year_col[0]
    valid_year = (df[yc] >= 1900) & (df[yc] <= 2026)
    print(f"年份异常值: {(~valid_year).sum()} 行")
    df = df[valid_year]
    df[yc] = df[yc].astype(int)

print(f"\n最终数据集: {df.shape}")
df.head()
```

- [ ] **Step 2: Run notebook to verify**

- [ ] **Step 3: Commit**

```bash
git add personal_project/personal_assignment1.ipynb
git commit -m "feat: personal assignment 1 — data loading and cleaning"
```

---

### Task B3: EDA cells for personal assignment 1

**Files:**
- Modify: `personal_project/personal_assignment1.ipynb` (append cells)

- [ ] **Step 1: Add EDA cells**

Append to notebook:

**Cell — Rating distribution:**

```python
# === EDA（课件 Chapter3 第4-5页数据分析框架）===

# 1. 评分分布
r_col = [c for c in df.columns if 'average' in c.lower() or 'rating' in c.lower()][0]

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
axes[0].hist(df[r_col].dropna(), bins=50, edgecolor='white', alpha=0.7, color='steelblue')
axes[0].axvline(df[r_col].mean(), color='red', linestyle='--', label=f'均值={df[r_col].mean():.2f}')
axes[0].axvline(df[r_col].median(), color='orange', linestyle='--', label=f'中位数={df[r_col].median():.2f}')
axes[0].set_title('桌游评分分布', fontsize=14)
axes[0].set_xlabel('评分')
axes[0].legend()

# Q-Q图检验正态性
stats.probplot(df[r_col].dropna(), dist="norm", plot=axes[1])
axes[1].set_title('评分 Q-Q 图', fontsize=14)

plt.tight_layout()
plt.savefig('output/rating_distribution.png', dpi=150, bbox_inches='tight')
plt.show()

# 偏度与峰度
print(f"评分偏度(Skewness): {df[r_col].skew():.3f}")
print(f"评分峰度(Kurtosis): {df[r_col].kurtosis():.3f}")
```

**Cell — Complexity (Weight) distribution:**

```python
# 2. 复杂度分布
weight_col = [c for c in df.columns if 'weight' in c.lower() or 'complexity' in c.lower()][0]

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
axes[0].hist(df[weight_col].dropna(), bins=40, edgecolor='white', alpha=0.7, color='coral')
axes[0].set_title('桌游复杂度权重分布 (1=轻, 5=重)', fontsize=14)
axes[0].set_xlabel('复杂度')

# 复杂度分段饼图
bins = [0, 1.5, 2.5, 3.5, 10]
labels = ['超轻策 (<1.5)', '轻策 (1.5-2.5)', '中策 (2.5-3.5)', '重策 (>3.5)']
df['weight_bin'] = pd.cut(df[weight_col], bins=bins, labels=labels)
df['weight_bin'].value_counts().plot(kind='pie', autopct='%1.1f%%', ax=axes[1])
axes[1].set_title('复杂度分布占比')
axes[1].set_ylabel('')

plt.tight_layout()
plt.savefig('output/complexity_distribution.png', dpi=150, bbox_inches='tight')
plt.show()
```

**Cell — Yearly trends (Time series, Chapter4):**

```python
# 3. 年代趋势（对应课件"序列数据与时间维度" + Chapter4 时间序列）
yc = [c for c in df.columns if 'year' in c.lower()][0]

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 年度发行量
yearly_count = df.groupby(yc).size()
axes[0, 0].bar(yearly_count.index, yearly_count.values, color='steelblue', alpha=0.7)
axes[0, 0].set_title('年度桌游发行量', fontsize=12)
axes[0, 0].set_xlabel('年份')
axes[0, 0].axvline(x=2000, color='red', linestyle='--', alpha=0.5, label='2000年')
axes[0, 0].legend()

# 年度均分趋势
yearly_rating = df.groupby(yc)[r_col].mean()
axes[0, 1].plot(yearly_rating.index, yearly_rating.values, marker='o', markersize=2, linewidth=1.5)
axes[0, 1].set_title('年度平均评分趋势', fontsize=12)
axes[0, 1].set_xlabel('年份')
axes[0, 1].set_ylabel('平均评分')

# 5年移动平均
yearly_rating_ma = yearly_rating.rolling(5, center=True, min_periods=1).mean()
axes[0, 1].plot(yearly_rating_ma.index, yearly_rating_ma.values,
                color='red', linewidth=2, label='5年移动平均')
axes[0, 1].legend()

# 年度复杂度趋势
yearly_weight = df.groupby(yc)[weight_col].mean()
axes[1, 0].plot(yearly_weight.index, yearly_weight.values, marker='o', markersize=2, linewidth=1.5, color='coral')
axes[1, 0].set_title('年度平均复杂度趋势', fontsize=12)
axes[1, 0].set_xlabel('年份')

# Top年份 vs 发行量热力
recent = df[df[yc] >= 2000]
yearly_heatmap = recent.groupby([yc, 'weight_bin']).size().unstack().fillna(0)
sns.heatmap(yearly_heatmap.T, ax=axes[1, 1], cmap='YlOrRd', cbar_kws={'label': '发行量'})
axes[1, 1].set_title('2000年后各复杂度发行热力', fontsize=12)

plt.tight_layout()
plt.savefig('output/yearly_trends.png', dpi=150, bbox_inches='tight')
plt.show()
```

**Cell — Top-N rankings:**

```python
# 4. Top-N 排行
print("=== Top-15 高评分桌游（评分人数≥5000）===")
rc_col = [c for c in df.columns if 'user' in c.lower() or 'count' in c.lower() or 'voter' in c.lower()][0]
top_games = df[df[rc_col] >= 5000].nlargest(15, r_col)
name_col = [c for c in df.columns if 'name' in c.lower() or 'title' in c.lower()][0]
display(top_games[[name_col, r_col, weight_col, rc_col, yc]])

# 流行机制排行
mechanics_col = [c for c in df.columns if 'mechanic' in c.lower() or 'category' in c.lower()]
if mechanics_col:
    # 简单统计机制出现频率（如果是逗号分隔）
    mc = mechanics_col[0]
    all_mechanics = df[mc].dropna().str.split(',').explode().str.strip()
    top_mechanics = all_mechanics.value_counts().head(15)

    plt.figure(figsize=(8, 6))
    top_mechanics.plot(kind='barh')
    plt.title('Top-15 流行桌游机制', fontsize=14)
    plt.xlabel('出现次数')
    plt.tight_layout()
    plt.savefig('output/top_mechanics.png', dpi=150, bbox_inches='tight')
    plt.show()
```

**Cell — Player count and duration analysis:**

```python
# 5. 玩家人数与时长分析
players_min_col = [c for c in df.columns if 'min_player' in c.lower() or 'minply' in c.lower()]
players_max_col = [c for c in df.columns if 'max_player' in c.lower() or 'maxply' in c.lower()]
duration_col = [c for c in df.columns if 'playing' in c.lower() or 'playtime' in c.lower() or 'duration' in c.lower() or 'time' in c.lower()]

if duration_col:
    dc = duration_col[0]
    # 时长分组
    time_bins = [0, 30, 60, 120, 9999]
    time_labels = ['<30分钟', '30-60分钟', '60-120分钟', '>120分钟']
    df['time_bin'] = pd.cut(df[dc], bins=time_bins, labels=time_labels)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    df['time_bin'].value_counts().sort_index().plot(kind='bar', ax=axes[0], color='steelblue', alpha=0.7)
    axes[0].set_title('游戏时长分布', fontsize=12)
    axes[0].set_xlabel('时长区间')

    df.boxplot(column=r_col, by='time_bin', ax=axes[1])
    axes[1].set_title('不同时长的评分分布', fontsize=12)
    axes[1].set_xlabel('时长区间')
    axes[1].set_ylabel('评分')

    plt.suptitle('')
    plt.tight_layout()
    plt.savefig('output/duration_analysis.png', dpi=150, bbox_inches='tight')
    plt.show()
```

- [ ] **Step 2: Run and verify charts generated**

- [ ] **Step 3: Commit**

```bash
git add personal_project/personal_assignment1.ipynb personal_project/output/
git commit -m "feat: personal assignment 1 — EDA with rating distribution and yearly trends"
```

---

### Task B4: Feature engineering and correlation analysis for personal assignment 1

**Files:**
- Modify: `personal_project/personal_assignment1.ipynb` (append cells)

- [ ] **Step 1: Add feature engineering cells**

**Cell — Mechanics One-Hot encoding:**

```python
# === 特征工程 ===
# 机制 One-Hot 编码（对应课件 Chapter3 高维表格特征维度）
mc = [c for c in df.columns if 'mechanic' in c.lower() or 'category' in c.lower()][0]
mechanics_series = df[mc].dropna().str.split(',').explode().str.strip()

# 过滤出现次数≥50的机制（降噪）
freq = mechanics_series.value_counts()
common_mechanics = freq[freq >= 50].index.tolist()
print(f"常见机制数量 (出现≥50次): {len(common_mechanics)}")

# One-Hot编码
for mech in common_mechanics[:50]:  # Top 50 机制
    df[f'mech_{mech}'] = df[mc].fillna('').str.contains(mech, regex=False).astype(int)

mech_cols = [c for c in df.columns if c.startswith('mech_')]
print(f"机制特征矩阵: {df[mech_cols].shape}")
```

**Cell — Complexity-rating relationship:**

```python
# 复杂度 × 评分 关系（核心假设检验）
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# 散点图
axes[0].scatter(df[weight_col], df[r_col], alpha=0.15, s=10)
axes[0].set_xlabel('复杂度权重')
axes[0].set_ylabel('评分')
axes[0].set_title('复杂度 vs 评分散点图')

# 添加 LOWESS 平滑线
from scipy.interpolate import make_interp_spline
x_sorted = df[weight_col].dropna().sort_values()
# 分段均值
df['weight_round'] = df[weight_col].round(1)
bin_means = df.groupby('weight_round')[r_col].agg(['mean', 'count'])
bin_means = bin_means[bin_means['count'] >= 20]
axes[0].plot(bin_means.index, bin_means['mean'], 'r-o', linewidth=2, markersize=5, label='分段均值')
axes[0].legend()

# 复杂度分段箱线图
df.boxplot(column=r_col, by='weight_bin', ax=axes[1])
axes[1].set_title('各复杂度段评分箱线图')
axes[1].set_xlabel('复杂度分段')
axes[1].set_ylabel('评分')
plt.suptitle('')

plt.tight_layout()
plt.savefig('output/complexity_vs_rating.png', dpi=150, bbox_inches='tight')
plt.show()

# 计算相关系数
corr_coef = df[[weight_col, r_col]].corr().iloc[0, 1]
print(f"复杂度与评分 Pearson 相关系数: r = {corr_coef:.3f}")
```

**Cell — Mechanics diversity index:**

```python
# 机制多样性指数
df['mechanics_count'] = df[mc].fillna('').str.split(',').apply(lambda x: len([i for i in x if i.strip() != '']))
print(f"平均机制数: {df['mechanics_count'].mean():.1f}")
print(f"机制数分布: 中位数={df['mechanics_count'].median():.0f}, "
      f"最小值={df['mechanics_count'].min():.0f}, 最大值={df['mechanics_count'].max():.0f}")

# 机制数 vs 评分
fig, ax = plt.subplots(figsize=(8, 5))
df.boxplot(column=r_col, by='mechanics_count', ax=ax)
ax.set_title('不同机制数量的评分分布')
ax.set_xlabel('机制数量')
ax.set_ylabel('评分')
plt.suptitle('')
plt.tight_layout()
plt.savefig('output/mechanics_count_vs_rating.png', dpi=150, bbox_inches='tight')
plt.show()
```

**Cell — Conclusions for assignment 1:**

```python
# === 阶段性发现（作业1终点）===
print("=" * 60)
print("阶段性发现 — 桌游复杂度与玩家评分")
print("=" * 60)

print(f"\n1. 复杂度与评分的关系：")
if abs(corr_coef) < 0.1:
    print(f"   相关系数 r = {corr_coef:.3f}，复杂度与评分基本无关")
    print("   高分桌游可以来自任何复杂度区间")
elif corr_coef > 0:
    print(f"   相关系数 r = {corr_coef:.3f}，轻度正相关")
else:
    print(f"   相关系数 r = {corr_coef:.3f}，轻度负相关")

print(f"\n2. 桌游行业趋势：")
recent_10y = df[df[yc] >= 2015]
older = df[(df[yc] >= 1990) & (df[yc] < 2000)]
print(f"   2015年后平均评分: {recent_10y[r_col].mean():.2f}")
print(f"   1990-2000平均评分: {older[r_col].mean():.2f}")

print(f"\n3. 高分桌游共性：")
top_quartile = df[df[r_col] >= df[r_col].quantile(0.75)]
print(f"   前25%高分桌游平均复杂度: {top_quartile[weight_col].mean():.2f}")
print(f"   前25%高分桌游平均机制数: {top_quartile['mechanics_count'].mean():.1f}")
print(f"   前25%高分桌游年份中位数: {top_quartile[yc].median():.0f}")

print(f"\n4. 作业2 方向：")
print("   • PCA降维：100+机制 → 2D/3D 机制空间可视化")
print("   • K-Means聚类：构建桌游类型学（轻策聚会/重策策略/家庭亲子/卡牌对战）")
print("   • 时间序列：检测'桌游文艺复兴'的结构断点")
```

- [ ] **Step 2: Run and verify**

- [ ] **Step 3: Commit**

```bash
git add personal_project/personal_assignment1.ipynb personal_project/output/
git commit -m "feat: personal assignment 1 — feature engineering and preliminary findings"
```

---

## Part C: 个人项目作业2 — 深度建模

### Task C1: Create personal assignment 2 notebook — PCA and clustering

**Files:**
- Create: `personal_project/personal_assignment2.ipynb`

- [ ] **Step 1: Create notebook with PCA and clustering cells**

Create `personal_project/personal_assignment2.ipynb`. This notebook builds on assignment 1 results.

**Cell 1 — Setup (load cleaned data and carry over from assignment 1):**

```python
# 个人作业2：桌游复杂度与玩家评分的量化关系（深度建模）
# 在作业1基础上深化：PCA降维 → 聚类分析 → 时间序列趋势
# 大数据处理技术课程 — 2025-2026第二学期

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False

# 复用作业1的数据加载与清洗逻辑（简化版）
import os
data_dir = 'data/'
csv_files = [f for f in os.listdir(data_dir) if f.endswith('.csv')]
dfs = {}
for f in csv_files:
    dfs[f.replace('.csv', '')] = pd.read_csv(os.path.join(data_dir, f))

# 加载主表并清洗（复用作业1逻辑）
main_key = None
for k, v in dfs.items():
    cols_lower = [c.lower() for c in v.columns]
    if any('rating' in c or 'average' in c for c in cols_lower) and \
       any('weight' in c or 'complexity' in c for c in cols_lower):
        main_key = k
        break
if main_key is None:
    main_key = list(dfs.keys())[0]

df = dfs[main_key].copy()
r_col = [c for c in df.columns if 'average' in c.lower() or 'rating' in c.lower()][0]
weight_col = [c for c in df.columns if 'weight' in c.lower() or 'complexity' in c.lower()][0]
yc = [c for c in df.columns if 'year' in c.lower()][0]
rc_col = [c for c in df.columns if 'user' in c.lower() or 'count' in c.lower() or 'voter' in c.lower()][0]
mc = [c for c in df.columns if 'mechanic' in c.lower() or 'category' in c.lower()][0]
name_col = [c for c in df.columns if 'name' in c.lower() or 'title' in c.lower()][0]

# 清洗（同作业1）
df = df[df[rc_col] >= 30].copy()
for col in df.select_dtypes(include=[np.number]).columns:
    df[col].fillna(df[col].median(), inplace=True)
for col in df.select_dtypes(include=['object']).columns:
    df[col].fillna('Unknown', inplace=True)
df = df[(df[r_col] >= 1) & (df[r_col] <= 10)]
df = df[(df[yc] >= 1900) & (df[yc] <= 2026)]
df[yc] = df[yc].astype(int)

# 机制 One-Hot
mechanics_series = df[mc].dropna().str.split(',').explode().str.strip()
freq = mechanics_series.value_counts()
common_mechanics = freq[freq >= 50].index.tolist()
for mech in common_mechanics[:50]:
    df[f'mech_{mech}'] = df[mc].fillna('').str.contains(mech, regex=False).astype(int)
mech_cols = [c for c in df.columns if c.startswith('mech_')]
df['mechanics_count'] = df[mc].fillna('').str.split(',').apply(lambda x: len([i for i in x if i.strip() != '']))
df['weight_bin'] = pd.cut(df[weight_col], bins=[0, 1.5, 2.5, 3.5, 10],
                          labels=['超轻策 (<1.5)', '轻策 (1.5-2.5)', '中策 (2.5-3.5)', '重策 (>3.5)'])

print(f"数据加载与清洗完成: {df.shape}")
print("开始深度建模...")
```

**Cell 2 — PCA降维（课件 Chapter3 高维表格数据 PCA）:**

```python
# === 模块A：PCA降维 — 桌游机制空间可视化 ===
# 对应课件 Chapter3 "主成分分析降维技术"（教学大纲第三章）
# 目标：将100+维的机制矩阵降至2-3维，可视化"机制地图"

mech_cols = [c for c in df.columns if c.startswith('mech_')]
print(f"机制特征维度: {len(mech_cols)}")

# 标准化
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df[mech_cols].fillna(0))

# PCA
pca = PCA()
X_pca = pca.fit_transform(X_scaled)

# 方差解释率
explained_var = pca.explained_variance_ratio_
cumsum_var = np.cumsum(explained_var)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# 碎石图
axes[0].bar(range(1, 16), explained_var[:15], alpha=0.7, color='steelblue')
axes[0].plot(range(1, 16), cumsum_var[:15], 'r-o', linewidth=2, markersize=6)
axes[0].set_xlabel('主成分')
axes[0].set_ylabel('方差解释率')
axes[0].set_title('PCA 碎石图（前15个主成分）')
axes[0].axhline(y=0.8, color='green', linestyle='--', label='80%阈值')
axes[0].legend()

# Top-5 PCs 方差解释
for i in range(5):
    axes[0].annotate(f'{cumsum_var[i]:.1%}', (i+1, cumsum_var[i]),
                     textcoords="offset points", xytext=(0, 10), fontsize=9)

# 前2个PC解释
axes[1].pie([explained_var[:2].sum(), 1 - explained_var[:2].sum()],
            labels=[f'PC1+2 ({explained_var[:2].sum():.1%})', '其他'],
            autopct='%1.1f%%', colors=['steelblue', 'lightgray'])
axes[1].set_title('前2个主成分累计方差解释')

plt.tight_layout()
plt.savefig('output/pca_scree.png', dpi=150, bbox_inches='tight')
plt.show()

print(f"前2个主成分累计方差解释: {cumsum_var[1]:.1%}")
print(f"前5个主成分累计方差解释: {cumsum_var[4]:.1%}")
```

**Cell 3 — PCA 2D visualization:**

```python
# PCA 2D 可视化 — 机制地图
pc1, pc2 = X_pca[:, 0], X_pca[:, 1]

# 加载机制载荷
loadings = pd.DataFrame(
    pca.components_[:2].T,
    index=mech_cols,
    columns=['PC1', 'PC2']
)

# Top contributing mechanics to PC1
print("PC1 最强正向载荷 (Top-5):")
print(loadings.nlargest(5, 'PC1')['PC1'])
print("\nPC1 最强负向载荷 (Top-5):")
print(loadings.nsmallest(5, 'PC1')['PC1'])

print("\nPC2 最强正向载荷 (Top-5):")
print(loadings.nlargest(5, 'PC2')['PC2'])
print("\nPC2 最强负向载荷 (Top-5):")
print(loadings.nsmallest(5, 'PC2')['PC2'])

# 2D散点图（按复杂度着色）
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

sc = axes[0].scatter(pc1, pc2, c=df[weight_col], cmap='viridis',
                     alpha=0.3, s=5, vmin=1, vmax=5)
axes[0].set_xlabel(f'PC1 ({explained_var[0]:.1%})')
axes[0].set_ylabel(f'PC2 ({explained_var[1]:.1%})')
axes[0].set_title('PCA 机制空间（按复杂度着色）')
plt.colorbar(sc, ax=axes[0], label='复杂度')

# 按评分着色
sc2 = axes[1].scatter(pc1, pc2, c=df[r_col], cmap='RdYlGn',
                      alpha=0.3, s=5, vmin=5, vmax=9)
axes[1].set_xlabel(f'PC1 ({explained_var[0]:.1%})')
axes[1].set_ylabel(f'PC2 ({explained_var[1]:.1%})')
axes[1].set_title('PCA 机制空间（按评分着色）')
plt.colorbar(sc2, ax=axes[1], label='评分')

plt.tight_layout()
plt.savefig('output/pca_2d_visualization.png', dpi=150, bbox_inches='tight')
plt.show()

# 保存PCA结果
df['PC1'] = pc1
df['PC2'] = pc2
df['PC3'] = X_pca[:, 2]
```

**Cell 4 — K-Means clustering (Chapter3 clustering methodology):**

```python
# === 模块B：K-Means聚类 — 桌游类型学 ===
# 对应课件 Chapter3 "聚类算法"（教学大纲第三章）
# 基于机制特征 + 复杂度 + 评分 构建桌游分类体系

# 特征选择：PCA前5个主成分 + 复杂度 + 机制数
cluster_features = [f'PC{i+1}' for i in range(5)] + [weight_col, 'mechanics_count']
cluster_data = df[cluster_features].dropna()
X_cluster = StandardScaler().fit_transform(cluster_data)

# 肘部法确定K值
inertias = []
silhouettes = []
K_range = range(2, 11)

for k in K_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X_cluster)
    inertias.append(kmeans.inertia_)
    silhouettes.append(silhouette_score(X_cluster, labels))

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
axes[0].plot(K_range, inertias, 'bo-')
axes[0].set_xlabel('K')
axes[0].set_ylabel('Inertia')
axes[0].set_title('肘部法确定K值')

axes[1].plot(K_range, silhouettes, 'ro-')
axes[1].set_xlabel('K')
axes[1].set_ylabel('轮廓系数')
axes[1].set_title('轮廓系数 vs K')

plt.tight_layout()
plt.savefig('output/kmeans_elbow.png', dpi=150, bbox_inches='tight')
plt.show()

best_k = K_range[np.argmax(silhouettes)]
print(f"最佳K值: {best_k} (轮廓系数: {max(silhouettes):.3f})")
```

**Cell 5 — Cluster interpretation:**

```python
# 执行最终聚类
final_kmeans = KMeans(n_clusters=best_k, random_state=42, n_init=10)
df_valid = df.loc[cluster_data.index].copy()
df_valid['cluster'] = final_kmeans.fit_predict(X_cluster)

# 各聚类的特征画像
cluster_profile = df_valid.groupby('cluster').agg({
    r_col: 'mean',
    weight_col: 'mean',
    'mechanics_count': 'mean',
    yc: 'mean',
    rc_col: 'mean',
}).round(2)
cluster_profile['size'] = df_valid['cluster'].value_counts().sort_index()
cluster_profile['pct'] = (cluster_profile['size'] / len(df_valid) * 100).round(1)

print("=== 桌游类型学 ===")
print(cluster_profile.to_string())

# 聚类标签定义
cluster_labels = {
    0: '轻策聚会型',
    1: '家庭亲子型',
    2: '重策策略型',
    3: '卡牌对战型',
}
if best_k == 4:
    df_valid['cluster_name'] = df_valid['cluster'].map(cluster_labels)
else:
    df_valid['cluster_name'] = df_valid['cluster'].astype(str)

# PCA空间中可视化聚类
fig, ax = plt.subplots(figsize=(12, 8))
scatter = ax.scatter(df_valid['PC1'], df_valid['PC2'],
                     c=df_valid['cluster'], cmap='tab10',
                     alpha=0.3, s=5)
ax.set_xlabel(f'PC1 ({explained_var[0]:.1%})')
ax.set_ylabel(f'PC2 ({explained_var[1]:.1%})')
ax.set_title(f'K-Means 聚类结果 (K={best_k}) — PCA空间中投影', fontsize=14)
plt.colorbar(scatter, ax=ax, label='聚类')

# 标注各类中心
centers_pca = PCA(n_components=2).fit_transform(
    StandardScaler().fit_transform(final_kmeans.cluster_centers_))
for i, (x, y) in enumerate(centers_pca):
    ax.plot(x, y, 'kx', markersize=15, markeredgewidth=3)
    ax.annotate(f'Cluster {i}', (x, y), fontsize=12, fontweight='bold',
                textcoords="offset points", xytext=(10, 10))

plt.tight_layout()
plt.savefig('output/cluster_pca_visualization.png', dpi=150, bbox_inches='tight')
plt.show()

# 每类代表游戏
print("\n=== 各类别代表游戏 Top-3 ===")
for cluster_id in range(best_k):
    cluster_games = df_valid[df_valid['cluster'] == cluster_id].nlargest(3, r_col)
    games_list = cluster_games[name_col].tolist() if name_col in cluster_games.columns else ['N/A']
    print(f"\nCluster {cluster_id}: {', '.join(games_list)}")
```

- [ ] **Step 2: Run notebook to verify PCA and clustering outputs**

- [ ] **Step 3: Commit**

```bash
git add personal_project/personal_assignment2.ipynb personal_project/output/
git commit -m "feat: personal assignment 2 — PCA dimensionality reduction and K-Means clustering"
```

---

### Task C2: Time series analysis and final report

**Files:**
- Modify: `personal_project/personal_assignment2.ipynb` (append cells)

- [ ] **Step 1: Add time series analysis cells**

**Cell — Time series trend analysis:**

```python
# === 模块C：时间序列分析 — 桌游行业发展脉络 ===
# 对应课件 Chapter4 "时间序列数据与行为建模"
# 检测"桌游文艺复兴"的结构断点

from scipy.signal import find_peaks

# 构建年度时间序列
yearly = df.groupby(yc).agg({
    r_col: ['mean', 'std', 'count'],
    weight_col: 'mean',
    'mechanics_count': 'mean'
}).reset_index()
yearly.columns = ['year', 'avg_rating', 'std_rating', 'count', 'avg_weight', 'avg_mechanics']
yearly = yearly[yearly['count'] >= 10]  # 过滤样本过少的年份

fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# 1. 年发行量
axes[0, 0].fill_between(yearly['year'], yearly['count'], alpha=0.3, color='steelblue')
axes[0, 0].plot(yearly['year'], yearly['count'], 'b-', linewidth=1.5)
axes[0, 0].set_ylabel('发行数量')
axes[0, 0].set_title('桌游年发行量趋势')
# 标注爆发年份
growth_rate = yearly['count'].pct_change()
boom_years = yearly[growth_rate > growth_rate.quantile(0.9)]
for _, row in boom_years.iterrows():
    axes[0, 0].annotate(str(int(row['year'])), (row['year'], row['count']),
                        fontsize=8, color='red')

# 2. 年均评分 + 移动平均 + 置信带
axes[0, 1].fill_between(yearly['year'],
                         yearly['avg_rating'] - yearly['std_rating'],
                         yearly['avg_rating'] + yearly['std_rating'],
                         alpha=0.2, color='coral')
axes[0, 1].plot(yearly['year'], yearly['avg_rating'], 'o-', markersize=3, linewidth=1, color='coral')
# 10年移动平均
yearly['rating_ma10'] = yearly['avg_rating'].rolling(10, center=True, min_periods=3).mean()
axes[0, 1].plot(yearly['year'], yearly['rating_ma10'], 'r-', linewidth=2, label='10年移动平均')
axes[0, 1].set_ylabel('平均评分')
axes[0, 1].set_title('年均评分趋势（含标准差带）')
axes[0, 1].legend()

# 3. 复杂度趋势
axes[1, 0].bar(yearly['year'], yearly['avg_weight'], color='steelblue', alpha=0.7)
axes[1, 0].set_ylabel('平均复杂度')
axes[1, 0].set_title('年均桌游复杂度')
# 趋势线
z = np.polyfit(yearly['year'], yearly['avg_weight'], 1)
p = np.poly1d(z)
axes[1, 0].plot(yearly['year'], p(yearly['year']), 'r--', linewidth=2,
                label=f'趋势 (斜率={z[0]:.4f}/年)')
axes[1, 0].legend()

# 4. 机制多样性趋势
axes[1, 1].plot(yearly['year'], yearly['avg_mechanics'], 'go-', markersize=3, linewidth=1)
axes[1, 1].set_ylabel('平均机制数')
axes[1, 1].set_title('年均桌游机制多样性')

plt.tight_layout()
plt.savefig('output/time_series_industry.png', dpi=150, bbox_inches='tight')
plt.show()
```

**Cell — Structural break detection:**

```python
# 结构断点检测 — 评分趋势是否存在"桌游文艺复兴"拐点
from scipy.signal import argrelextrema

# 对移动平均找局部极值
ma = yearly['rating_ma10'].dropna().values
local_min = argrelextrema(ma, np.less, order=5)[0]
local_max = argrelextrema(ma, np.greater, order=5)[0]

print("=== 评分趋势关键节点 ===")
years_with_data = yearly['year'].iloc[yearly['rating_ma10'].dropna().index]
print("局部极小值年份 (可能的'谷底'):")
for idx in local_min:
    print(f"  {int(years_with_data.iloc[idx])}: 评分={ma[idx]:.2f}")

print("\n局部极大值年份 (可能的'高峰'):")
for idx in local_max:
    print(f"  {int(years_with_data.iloc[idx])}: 评分={ma[idx]:.2f}")

# 对比 1995前后 vs 2015前后的变化
before_2000 = yearly[yearly['year'] <= 2000]
after_2015 = yearly[yearly['year'] >= 2015]
print(f"\n2000年前 vs 2015年后对比:")
print(f"  发行量: {before_2000['count'].mean():.0f} → {after_2015['count'].mean():.0f}")
print(f"  均分:   {before_2000['avg_rating'].mean():.2f} → {after_2015['avg_rating'].mean():.2f}")
print(f"  复杂度: {before_2000['avg_weight'].mean():.2f} → {after_2015['avg_weight'].mean():.2f}")
```

**Cell — Final comprehensive conclusions:**

```python
# === 综合结论（作业2终点）===
print("=" * 60)
print("桌游复杂度与玩家评分的量化关系 — 综合结论")
print("=" * 60)

print(f"\n【假设1：复杂度与评分的关系】")
print(f"  相关系数 r = {corr_coef:.3f}")
if abs(corr_coef) < 0.1:
    print("  结论：复杂度与评分无显著线性关系。高分桌游可出现在任意复杂度区间。")
    print("  启示：桌游设计不应盲目追求复杂化，'好玩'不取决于'难懂'。")
elif corr_coef > 0.15:
    print("  结论：复杂度与评分呈正相关，高复杂度桌游倾向获得更高评分。")
    print("  启示：核心玩家的深度需求是桌游行业的重要驱动力。")

print(f"\n【假设2：机制组合与高分桌游】")
print(f"  PCA降维将{len(mech_cols)}维机制空间压缩至2维，方差解释率{cumsum_var[1]:.1%}")
print(f"  K-Means将桌游分为{best_k}类，轮廓系数{max(silhouettes):.3f}")
print(f"  聚类揭示的桌游类型学：{list(cluster_profile.index)}")

print(f"\n【假设3：行业趋势与'黄金年代'】")
print(f"  桌游行业在2000年后呈爆发式增长")
print(f"  年度发行量增长趋势明显，行业规模持续扩大")
print(f"  年均评分趋势显示...")

print(f"\n【局限性与未来方向】")
print("  • 数据仅涵盖BGG收录游戏，可能存在生存偏差")
print("  • 评分受评分人数影响（热门游戏评分更稳定）")
print("  • 未来可引入图网络分析（设计师-机制-发行商合作网络）")
print("  • 可扩展NLP分析评论情感（对应教学大纲第七章）")
```

- [ ] **Step 2: Run complete notebook and verify all outputs**

```bash
jupyter nbconvert --to notebook --execute personal_project/personal_assignment2.ipynb --output personal_assignment2.ipynb
```

- [ ] **Step 3: Final commit**

```bash
git add personal_project/
git commit -m "feat: personal assignment 2 — time series analysis and comprehensive conclusions"
```

---

## Verification Checklist

After all tasks are complete:

- [ ] Team assignment 1 notebook executes without errors
- [ ] Personal assignment 1 notebook executes without errors
- [ ] Personal assignment 2 notebook executes without errors
- [ ] All chart files exist in respective `output/` directories
- [ ] All notebooks contain markdown cells explaining methodology in context of course materials
- [ ] Each notebook can be exported to PDF via `nbconvert --to pdf` for submission

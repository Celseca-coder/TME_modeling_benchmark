# TME benchmark 模型的 Stability Lasso 可行性与运行指南

## 1. 核心判定标准

`scripts/run_stability_lasso.py` 使用 patient-level CV、多个随机种子和
patient bootstrap，拟合的是 L1 Logistic Regression。一个模型或特征方法能否接入，
取决于它能否在每个区域上产生固定长度的纯数值向量：

```text
region_id,feature_1,feature_2,...,feature_p
region_001,0.12,1.34,...,0.08
region_002,0.07,0.91,...,0.15
```

硬性要求：

1. 一行对应一个 `region_id`，且 ID 唯一；
2. 除 `region_id` 外全部为数值列，列名唯一；
3. 特征长度和列含义在训练集、验证集间一致；
4. 当前仅支持分类任务；OS、PFS、DSS 等 survival 任务需要另外实现 L1 Cox；
5. 会学习词表、标准化参数或聚类中心的 featurizer，必须在每个 CV 训练 fold 内拟合。

`results/` 中现有 benchmark CSV、Lasso coefficient CSV 和 feature summary CSV
都是分析结果，不是 region-level 输入特征，不能作为 `--features-csv` 使用。

## 2. 可行性分类

### 2.1 可直接进行 Stability Lasso

这些方法已经接入 `run_stability_lasso.py`，并在每个训练 fold 内建立特征：

| 特征来源 | `--feature-source` | 说明 |
|---|---|---|
| 全局细胞组成 | `composition` | 细胞类型比例 |
| 全局平均表达 | `expression` | marker 平均表达 |
| 组成 + 表达 | `composition-expression` | 两类特征拼接 |
| Patch/MIL 统计聚合 | `patch` | 局部窗口特征的 mean/max/std/quantile |
| 细胞类型密度 | `density` | tissue/tumour density 和面积比例 |
| 空间距离 | `spatial-distance` | 同类/异类近邻距离 |
| Point pattern | `point-pattern` | Ripley K/L、pcf、variogram |
| Mixing/infiltration | `mixing` | entropy、局部混合和类型间邻接 |
| UTAG | `utag` | message passing、train-fold domains 或 combined |
| KRONOS | `kronos` | 冻结的 region embedding |
| Eva | `eva` | 冻结的 region embedding |

### 2.2 聚合导出后可做

以下已有产物可以由 `scripts/export_cached_model_features.py` 转成
region-by-feature CSV：

| 产物 | 导出方式 | 解释性 |
|---|---|---|
| UTAG cell × marker message-passing NPZ | 每 marker 计算 mean 和 std | 中等，保留 marker 名 |
| KRONOS region NPZ | 堆叠固定长度 `feature` | 较低 |
| Eva region NPZ | 堆叠固定长度 `feature` | 较低 |
| CytoCommunity cell-level TCN label | 计算每个 TCN 的 region 比例 | 较高 |

### 2.3 当前不能直接做

| 模型/产物 | 原因 |
|---|---|
| Attention MIL 原始 bag | 每个 region 的实例数可变，不是固定宽表 |
| Attention 权重 | 权重是模型解释输出，不是独立输入特征 |
| 模型预测概率/风险分数 | 是下游输出，不能替代原始特征做选择 |
| Space-GM | 当前只有 graph 对象和性能结果，没有 pooled region embedding |
| Cell-Graph Signature `.pt` | 是 PyG 图对象，需先经过 GNN pooling |
| Cellular Neighborhood | 当前没有可用的持久化 region 特征表，相关活跃源码缺失 |
| Random Forest/GBM 结果 | 无法从性能或 feature importance 反推 region 特征 |
| Survival 任务 | 当前 Stability Lasso 未实现 L1 Cox |

## 3. 直接运行的示例

以下命令均从项目根目录执行：

```bash
cd /autofs/nas8/tywang/tjzou/TME_modeling_benchmark
export DATA_ROOT=/autofs/bal14/zqwu/CellularTables/TME_benchmark_data
```

### 3.1 Density

```bash
python scripts/run_stability_lasso.py \
  --datasets hnc_wu2022 \
  --data-root "$DATA_ROOT" \
  --tasks primary_outcome \
  --feature-source density \
  --cell-type-col cell_type \
  --output-prefix results/lasso/hnc_density
```

Density 依赖 `tissue_polygons.geojson`；缺失 polygon 的区域可能产生无效特征。

### 3.2 Spatial distance

```bash
python scripts/run_stability_lasso.py \
  --datasets hnc_wu2022 \
  --data-root "$DATA_ROOT" \
  --tasks primary_outcome \
  --feature-source spatial-distance \
  --distance-k 1 \
  --output-prefix results/lasso/hnc_spatial_distance
```

### 3.3 Point pattern

```bash
python scripts/run_stability_lasso.py \
  --datasets hnc_wu2022 \
  --data-root "$DATA_ROOT" \
  --tasks primary_outcome \
  --feature-source point-pattern \
  --point-pattern-radii 10 20 50 100 200 \
  --point-pattern-metrics K L \
  --output-prefix results/lasso/hnc_point_pattern
```

如需按 cell type 分层：

```bash
python scripts/run_stability_lasso.py \
  --datasets hnc_wu2022 \
  --data-root "$DATA_ROOT" \
  --tasks primary_outcome \
  --feature-source point-pattern \
  --point-pattern-by-type \
  --cell-type-col cell_type \
  --output-prefix results/lasso/hnc_point_pattern_by_type
```

### 3.4 Mixing

```bash
python scripts/run_stability_lasso.py \
  --datasets hnc_wu2022 \
  --data-root "$DATA_ROOT" \
  --tasks primary_outcome \
  --feature-source mixing \
  --mixing-k 10 \
  --cell-type-col cell_type \
  --output-prefix results/lasso/hnc_mixing
```

### 3.5 UTAG domains：推荐的无泄漏运行方式

```bash
python scripts/run_stability_lasso.py \
  --datasets hnc_wu2022 \
  --data-root "$DATA_ROOT" \
  --tasks primary_outcome \
  --feature-source utag \
  --feature-mode domains \
  --n-domains 10 \
  --max-fit-cells 100000 \
  --cache-dir model_results/UTAG/message_passing_cache \
  --output-prefix results/lasso/hnc_utag_domains
```

此路径会在每个 CV 训练 fold 内重新拟合 `StandardScaler + MiniBatchKMeans`，
再用冻结的训练质心为验证区域分配 domain，因此适合正式比较。

UTAG message-passing 与 domain 同时使用：

```bash
python scripts/run_stability_lasso.py \
  --datasets hnc_wu2022 \
  --data-root "$DATA_ROOT" \
  --tasks primary_outcome \
  --feature-source utag \
  --feature-mode combined \
  --n-domains 10 \
  --cache-dir model_results/UTAG/message_passing_cache \
  --output-prefix results/lasso/hnc_utag_combined
```

注意：不要先在全数据集上拟合 UTAG domains 后导出一张 CSV，再将其用于正式 CV。
这样会让验证 fold 参与 scaler 和 KMeans 的拟合。

#### 所有数据集的 UTAG

不同数据集的 task 名称并不统一，因此不要指定
`--tasks primary_outcome`。省略 `--datasets` 和 `--tasks` 后，脚本会遍历 registry
中的所有数据集及其全部分类任务，并自动跳过 survival 任务：

```bash
nohup python scripts/run_stability_lasso.py \
  --data-root "$DATA_ROOT" \
  --feature-source utag \
  --feature-mode domains \
  --n-domains 10 \
  --max-fit-cells 100000 \
  --cache-dir model_results/UTAG/message_passing_cache \
  --continue-on-error \
  --output-prefix results/utag_all_datasets_lasso \
  --log-file log/utag_all_datasets_lasso.log \
  > lasso_utag.nohup.log 2>&1 &
```

`--continue-on-error` 会记录单个 dataset/task 的错误并继续剩余分析，适合长时间批量任务。
如果希望任何错误立即终止，以便严格排查，则去掉该参数。

### 3.6 KRONOS/Eva 在线运行

冻结 embedding 不学习临床标签，可以直接接入。脚本会优先复用相同参数签名下的缓存；
缓存缺失时可能加载 GPU 模型重新计算。

```bash
python scripts/run_stability_lasso.py \
  --datasets hnc_wu2022 \
  --data-root "$DATA_ROOT" \
  --tasks primary_outcome \
  --feature-source kronos \
  --image-mode rasterized \
  --device cuda \
  --cache-dir model_results/KRONOS/embeddings \
  --model-cache model_results/KRONOS/model_assets \
  --output-prefix results/lasso/hnc_kronos
```

```bash
python scripts/run_stability_lasso.py \
  --datasets hnc_wu2022 \
  --data-root "$DATA_ROOT" \
  --tasks primary_outcome \
  --feature-source eva \
  --image-mode rasterized \
  --device cuda \
  --cache-dir model_results/Eva/embeddings \
  --output-prefix results/lasso/hnc_eva
```

## 4. 从已有缓存聚合、导出、运行

### 4.1 UTAG message passing

`--input-root` 应指向一个确定的参数签名和 marker hash 目录，不应同时包含多个缓存版本。

```bash
python scripts/export_cached_model_features.py \
  --source utag-message-passing \
  --dataset hnc_wu2022 \
  --data-root "$DATA_ROOT" \
  --input-root model_results/UTAG/message_passing_cache/HNC-Wu2022/84b31ed6272e/c0cfa57748 \
  --output results/features/hnc_wu2022_utag_message_passing.csv

python scripts/run_stability_lasso.py \
  --datasets hnc_wu2022 \
  --data-root "$DATA_ROOT" \
  --tasks primary_outcome \
  --feature-source precomputed \
  --features-csv results/features/hnc_wu2022_utag_message_passing.csv \
  --output-prefix results/lasso/hnc_utag_message_passing
```

导出的列名为：

```text
utag_mean__<marker>
utag_std__<marker>
```

### 4.2 KRONOS

```bash
python scripts/export_cached_model_features.py \
  --source kronos \
  --dataset hnc_wu2022 \
  --data-root "$DATA_ROOT" \
  --input-root model_results/KRONOS/embeddings/rasterized/HNC-Wu2022/481fe100ab0d \
  --output results/features/hnc_wu2022_kronos.csv

python scripts/run_stability_lasso.py \
  --datasets hnc_wu2022 \
  --data-root "$DATA_ROOT" \
  --tasks primary_outcome \
  --feature-source precomputed \
  --features-csv results/features/hnc_wu2022_kronos.csv \
  --lambda-value 1.0 \
  --output-prefix results/lasso/hnc_kronos
```

### 4.3 Eva

```bash
python scripts/export_cached_model_features.py \
  --source eva \
  --dataset hnc_wu2022 \
  --data-root "$DATA_ROOT" \
  --input-root model_results/Eva/embeddings/rasterized/HNC-Wu2022/<signature> \
  --output results/features/hnc_wu2022_eva.csv

python scripts/run_stability_lasso.py \
  --datasets hnc_wu2022 \
  --data-root "$DATA_ROOT" \
  --tasks primary_outcome \
  --feature-source precomputed \
  --features-csv results/features/hnc_wu2022_eva.csv \
  --output-prefix results/lasso/hnc_eva
```

KRONOS/Eva 通常是高维特征。应比较多个 `--lambda-value`，并报告 region 数、
patient 数、有效特征数及最终非零系数数目。

### 4.4 CytoCommunity

#### 所有数据集直接运行（推荐）

脚本会读取每个数据集的 `ResultTable_*.csv`，将 cell-level `TCN_Label`
自动聚合为 region-level TCN 比例，然后运行 Stability Lasso。与 UTAG 批量分析相同，
省略 `--datasets` 和 `--tasks` 表示遍历全部数据集及其分类任务：

```bash
nohup python scripts/run_stability_lasso.py \
  --data-root "$DATA_ROOT" \
  --feature-source cytocommunity \
  --cytocommunity-root model_results/CytoCommunity/native_local_runs_cutoff02 \
  --continue-on-error \
  --output-prefix results/cytocommunity_all_datasets_lasso \
  --log-file log/cytocommunity_all_datasets_lasso.log \
  > lasso_cytocommunity.nohup.log 2>&1 &
```

每个数据集会单独确定其 TCN 标签集合并拟合 Lasso。因此，不应假设不同数据集中的
`tcn_fraction__1` 代表同一个生物学社区；系数应在各数据集内部解释。

#### 单数据集手动导出

如需保留独立的 region feature CSV，可以继续使用导出脚本：

```bash
python scripts/export_cached_model_features.py \
  --source cytocommunity \
  --dataset hnc_wu2022 \
  --data-root "$DATA_ROOT" \
  --input-root model_results/CytoCommunity/native_local_runs_cutoff02/hnc_wu2022 \
  --output results/features/hnc_wu2022_cytocommunity.csv

python scripts/run_stability_lasso.py \
  --datasets hnc_wu2022 \
  --data-root "$DATA_ROOT" \
  --tasks primary_outcome \
  --feature-source precomputed \
  --features-csv results/features/hnc_wu2022_cytocommunity.csv \
  --output-prefix results/lasso/hnc_cytocommunity
```

导出特征为每个 region 的：

```text
tcn_fraction__<TCN label>
```

## 5. 导出前后的检查

导出脚本默认允许部分 region 缺失，因为 Lasso 会按 `region_id` 与任务 metadata
取交集。如需强制完整覆盖，添加 `--strict`。

正式运行前建议检查：

1. `region_id` 是否唯一；
2. 导出覆盖了多少 metadata region 和 patient；
3. 是否存在 NaN、Inf 或常数列；
4. 不同 region 的 embedding 维度是否一致；
5. UTAG cache 是否来自同一参数签名和 marker panel；
6. KRONOS/Eva cache 是否来自同一 image mode、checkpoint 和 pooling 设置；
7. CytoCommunity 是否只使用一个 run/cutoff/`num_tcn` 版本。

## 6. 输出文件解释

每次 Stability Lasso 产生：

```text
<prefix>_fold_coefficients.csv
<prefix>_seed_summary.csv
<prefix>_feature_summary.csv
```

可选：

```text
<prefix>_bootstrap_coefficients.csv
```

重点字段：

| 字段 | 含义 |
|---|---|
| `coefficient_mean` | 标准化特征上的平均系数 |
| `fold_selection_frequency` | 跨 fold 被选中的频率 |
| `seed_selection_frequency` | 跨随机种子稳定选中的频率 |
| `ci_low`, `ci_high` | patient bootstrap 系数区间 |
| `direction` | 平均系数方向 |
| `bootstrap_ci_direction` | bootstrap 区间是否稳定为正或负 |

对于 UTAG domain、TCN composition、cell density 等可解释特征，可以进行生物学方向解释。
对于 `kronos_XXXX`、`eva_XXXX` 等匿名 embedding 维度，应将结果解释为稳定表征维度，
不能直接等同于具体 marker 或细胞过程。

## 7. 推荐执行顺序

1. 先运行 composition、expression 和 patch，作为已有可解释基线；
2. 运行 density、spatial-distance、point-pattern 和 mixing；
3. 使用 fold-safe 方式运行 UTAG domains；
4. 聚合 CytoCommunity TCN composition；
5. 最后运行 KRONOS/Eva 高维 embedding，并进行多个 lambda 的敏感性分析；
6. 不把 benchmark score、attention weight 或预测概率作为 Lasso 输入。

# 六类 TME 特征的两两组合基准

## 组合与指标

所有组合都把下面对应的两组区域级特征按列拼接，再直接进入监督模型。组合器会给列名加组名前缀，并在每个训练 fold 内重新拟合特征词表。

| 组 | 源码提取器 | 实际使用的指标 |
|---|---|---|
| composition | `CompositionFeaturizer` | 每种 cell type 的细胞数 / 区域有效细胞总数 |
| density | `CellTypeDensityFeaturizer` | tissue 中各 cell type 密度、tumour/tissue 面积比、tumour 中各 cell type 密度。注意源码实际为 `(count/1000)/area_mm2`，数值单位是千 cells/mm² |
| expression | `MeanExpressionFeaturizer` | 每个 marker 经数据集配置的 arcsinh normalization 后，在区域全部细胞上的均值 |
| distance | `SpatialDistanceFeaturizer(k=1)` | 同型和异型最近邻距离的 mean/std/min/max/median，以及最近邻同型比例；优先限制在 tissue mask 内 |
| point_pattern | `PointPatternFeaturizer` | 半径 10/20/50/100/200 μm 上的 Ripley K、Besag L、pcf、variogram；按现有 baseline 默认汇总全部 cell，而不是分 cell type |
| mixing | `MixingFeaturizer(k=10)` | 全局 Shannon entropy、归一化 entropy；局部邻域 Gini-Simpson 的 mean/std/min/max；同型邻居比例；各 cell type 有向邻接比例 |

`C(6,2)=15` 组为：

1. composition + density
2. composition + expression
3. composition + distance
4. composition + point_pattern
5. composition + mixing
6. density + expression
7. density + distance
8. density + point_pattern
9. density + mixing
10. expression + distance
11. expression + point_pattern
12. expression + mixing
13. distance + point_pattern
14. distance + mixing
15. point_pattern + mixing

每一组只改变输入特征，聚类、预测模型和验证方案完全一致，因此各组结果可以直接比较。

## 聚类、分类器与生存模型

现有六个 baseline **没有无监督聚类步骤**。cell type 是数据集中已有的注释；组合后的区域级特征不经过 k-means、层次聚类或 Leiden，而是由模型侧进行缺失值填补、删除零方差列和标准化，随后直接监督学习。

- binary / binary_classification：`LinearClassifier`，即 `class_weight="balanced"` 的 L2 Logistic Regression（`C=1.0`、LBFGS、最多 5000 次迭代）。主指标为 AUROC，同时内部还计算 average precision 和 balanced accuracy。
- multiclass / multiclass_classification：同一个平衡 L2 Logistic Regression。主指标为 balanced accuracy，同时计算 one-vs-rest macro AUROC。
- survival：`LinearCox`，即 L2/ridge penalizer 为 0.1 的 Cox proportional-hazards。主指标为 concordance index。

## 不跨队列与跨队列测试

- 不跨队列（`scheme=cv`）：按 patient 分组做 K-fold CV，默认 fold 数来自数据集 YAML（通常 5）；同一患者的多个 region 不会跨 train/validation。分类按 label、survival 按 event 尝试分层；样本太少时自动减少 fold，无法分层时退回 patient-level KFold。对 seeds 0/1/2 重复划分，汇总所有 folds 的主指标均值与样本标准差。
- 跨队列：严格读取每个数据集 YAML 的 `validation.generalization_tests`，用其 `train` cohort 全量训练，在固定 `test` cohort 上测试；每个 seed 重建模型，但不重新划分 cohort。cell-type 相关特征使用测试配置指定的 harmonized 列（通常 `cell_type_uniform`）。包含 expression 的组合只使用 train 与 test 所有相关 region 都共有的 marker，避免 panel 不一致。

单次运行全部 15 组：

```bash
python scripts/run_pairwise_feature_combinations.py \
  --data-root /path/to/TME_benchmark_data \
  --output results/pairwise_benchmark.csv
```

只跑部分数据集或组合：

```bash
python scripts/run_pairwise_feature_combinations.py \
  --datasets hnc_wu2022 luad_sorin2023 \
  --combinations composition+density distance+mixing \
  --seeds 0 1 2
```

顺序批量运行并为每组保存独立 CSV 和日志：

```bash
DATA_ROOT=/path/to/TME_benchmark_data \
  bash scripts/run_pairwise_combinations_batch.sh
```

若需要复刻旧 point-pattern baseline 的较快设置（仅 K/L），给 Python 命令增加 `--point-pattern-metrics K L`。默认值包含用户要求的 K/L/pcf/variogram，运行时间会明显更长。

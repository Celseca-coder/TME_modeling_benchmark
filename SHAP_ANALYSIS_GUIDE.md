# TME benchmark SHAP 分析指南

## 1. 分析原则

`scripts/run_shap_analysis.py` 沿用 Stability Lasso 的 patient-level CV：

1. 按患者划分训练 fold 和 held-out fold；
2. featurizer 只在训练 fold 上 `fit`；
3. 模型只在训练 fold 上拟合；
4. SHAP 只对 held-out region 计算；
5. 跨 seed、fold、patient 和 dataset 汇总。

这比在全数据训练后再解释训练样本更保守，可以减少特征构建和模型解释中的信息泄漏。

支持的模型：

| `--model` | 任务 | SHAP空间 |
|---|---|---|
| `logistic` | binary/multiclass | log-odds |
| `cox` | survival | log-risk |
| `random-forest` | binary/multiclass | probability |
| `xgboost` | binary/multiclass | raw margin/log-odds |

支持 `run_stability_lasso.py` 中的全部 tabular feature source，包括
composition、expression、patch、UTAG、KRONOS、Eva、CytoCommunity 和
`precomputed` MIL bag-level 数值向量。

原始变长 MIL bag、attention 权重、Space-GM 图和神经网络中间对象不能直接传入。
它们需要先池化为固定长度 region-by-feature CSV。DeepSHAP/GradientSHAP 属于另一条
模型内部解释流程，当前脚本不把其结果与 tabular SHAP 混合。

## 2. 环境

```bash
cd /autofs/nas8/tywang/tjzou/TME_modeling_benchmark
export DATA_ROOT=/autofs/bal14/zqwu/CellularTables/TME_benchmark_data
conda activate p3
pip install -r requirements.txt
```

省略 `--datasets` 和 `--tasks` 表示遍历所有数据集及与模型兼容的任务。

## 3. Logistic regression SHAP

以下示例解释所有数据集的 CytoCommunity TCN composition：

```bash
nohup python scripts/run_shap_analysis.py \
  --data-root "$DATA_ROOT" \
  --feature-source cytocommunity \
  --cytocommunity-root model_results/CytoCommunity/native_local_runs_cutoff02 \
  --model logistic \
  --logistic-penalty l1 \
  --C 1.0 \
  --seeds 0 1 2 \
  --top-k 20 \
  --continue-on-error \
  --output-prefix results/shap_cytocommunity_logistic \
  --log-file log/shap_cytocommunity_logistic.log \
  > shap_cytocommunity_logistic.nohup.log 2>&1 &
```

只会分析分类任务，survival 自动跳过。

### 精选17个 dataset/task/scheme

若要严格复现 `benchmark/validation/selected_tasks.py` 中的17个精选运行（包括
CV、固定 cohort transfer、分类和 survival），使用：

```bash
python scripts/run_shap_analysis.py \
  --selected-runs \
  --data-root "$DATA_ROOT" \
  --feature-source cytocommunity \
  --cytocommunity-root model_results/CytoCommunity/native_local_runs_cutoff02 \
  --model auto \
  --continue-on-error \
  --output-prefix results/shap_cytocommunity_selected_tasks \
  --log-file logs/shap_cytocommunity_selected_tasks.log
```

`--model auto` 会对分类任务使用 L1 Logistic，对 survival 任务使用 Cox。
输出增加 `scheme` 列，以区分 `cv`、`Basel_to_Zurich`、`UPMC_to_DFCI`、
`Yale_to_UQ` 和 `Yale_to_YaleExt`。

将 `--feature-source` 替换为下列值即可解释其他特征：

```text
composition
expression
composition-expression
patch
density
spatial-distance
point-pattern
mixing
utag
kronos
eva
```

## 4. Cox SHAP

`--model cox` 只分析 survival 任务，并跳过分类任务：

```bash
nohup python scripts/run_shap_analysis.py \
  --data-root "$DATA_ROOT" \
  --feature-source cytocommunity \
  --cytocommunity-root model_results/CytoCommunity/native_local_runs_cutoff02 \
  --model cox \
  --cox-penalizer 0.1 \
  --seeds 0 1 2 \
  --top-k 20 \
  --continue-on-error \
  --output-prefix results/shap_cytocommunity_cox \
  --log-file log/shap_cytocommunity_cox.log \
  > shap_cytocommunity_cox.nohup.log 2>&1 &
```

Cox SHAP 为 log-risk 贡献：

- SHAP > 0：该特征使预测风险升高；
- SHAP < 0：该特征使预测风险降低。

## 5. Random forest SHAP

```bash
nohup python scripts/run_shap_analysis.py \
  --data-root "$DATA_ROOT" \
  --feature-source cytocommunity \
  --cytocommunity-root model_results/CytoCommunity/native_local_runs_cutoff02 \
  --model random-forest \
  --n-estimators 300 \
  --seeds 0 1 2 \
  --background-size 100 \
  --top-k 20 \
  --continue-on-error \
  --output-prefix results/shap_cytocommunity_rf \
  --log-file log/shap_cytocommunity_rf.log \
  > shap_cytocommunity_rf.nohup.log 2>&1 &
```

## 6. XGBoost SHAP

```bash
nohup python scripts/run_shap_analysis.py \
  --data-root "$DATA_ROOT" \
  --feature-source cytocommunity \
  --cytocommunity-root model_results/CytoCommunity/native_local_runs_cutoff02 \
  --model xgboost \
  --n-estimators 300 \
  --learning-rate 0.05 \
  --max-depth 6 \
  --subsample 0.8 \
  --colsample-bytree 0.8 \
  --seeds 0 1 2 \
  --top-k 20 \
  --continue-on-error \
  --output-prefix results/shap_cytocommunity_xgboost \
  --log-file log/shap_cytocommunity_xgboost.log \
  > shap_cytocommunity_xgboost.nohup.log 2>&1 &
```

XGBoost 使用其原生 `pred_contribs` 精确 TreeSHAP，数值位于 raw margin 空间。

## 7. Precomputed MIL bag-level 特征

CSV 必须是一行一个 region：

```text
region_id,mil_feature_0000,mil_feature_0001,...
```

运行示例：

```bash
python scripts/run_shap_analysis.py \
  --datasets hnc_wu2022 \
  --tasks primary_outcome \
  --data-root "$DATA_ROOT" \
  --feature-source precomputed \
  --features-csv results/features/hnc_mil_bag_features.csv \
  --model logistic \
  --output-prefix results/shap_hnc_mil_logistic
```

只有已经池化为固定长度的 bag-level 数值向量适用；原始 instance bag 不适用。

## 8. 输出

每次运行生成：

| 文件 | 回答的问题 |
|---|---|
| `*_observation_values.csv` | 每个 held-out region 的特征值与 SHAP 值 |
| `*_fold_summary.csv` | 每个 seed/fold 的重要性、方向与 top-feature 状态 |
| `*_feature_summary.csv` | 每个数据集的全局重要性和跨 fold 稳定性 |
| `*_patient_summary.csv` | 哪些患者受哪些特征影响最大 |
| `*_cross_dataset_summary.csv` | 相同 task/feature 在不同数据集中的方向一致性 |
| `*_plots/*__summary.png` | SHAP summary plot |
| `*_plots/*__dependence__*.png` | top feature dependence plot |

`feature_summary` 重点列：

| 列 | 含义 |
|---|---|
| `mean_abs_shap` | 全局重要性，越大越重要 |
| `value_shap_spearman` | 特征值与 SHAP 的方向关系 |
| `direction` | 特征升高使预测升高或降低 |
| `direction_consistency` | 不同 seed/fold 的方向一致比例 |
| `top_fold_frequency` | 进入每个 fold top-K 的频率 |

`patient_summary` 按 `mean_abs_shap` 排序，可定位受某个特征影响最大的患者。

`cross_dataset_summary` 只比较 task 名、class 和 feature 名都相同的结果。
如果不同数据集的 task 语义不同或特征编号没有统一含义（例如不同数据集的 TCN 1），
不应强行进行生物学方向比较。

## 9. 计算量控制

默认每个 held-out fold 最多解释 200 个 region：

```text
--max-explain-regions-per-fold 200
```

调试时可改为 20；希望解释全部 held-out region 时可传入一个大于数据集 region 数的值。

高维 KRONOS/Eva/MIL 特征的 observation-level CSV 和图可能很大。建议先用 Lasso
确定候选特征，或先用较小的 `--max-explain-regions-per-fold` 做敏感性检查。

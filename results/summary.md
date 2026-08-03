# Benchmark Summary Update

This summary uses `Project/TME_modeling_benchmark/results` as the result directory. Files with the `_0706` suffix are treated as the newer reruns, while the same filenames without `_0706` are treated as the older comparison baselines.

## Files included in `benchmark_summary.csv`
- `celltype_density_benchmark.csv`
- `composition_benchmark.csv`
- `expression_benchmark.csv`
- `mixing_benchmark.csv`
- `patch_benchmark.csv`
- `point_pattern_benchmark.csv`
- `point_pattern_benchmark_0706.csv`
- `space_gm_benchmark.csv`
- `spatial_distance_benchmark.csv`
- `spatial_distance_benchmark_0706.csv`

## Overall ranking by mean score
| benchmark | rows | average mean | median mean | average sd |
| --- | --- | --- | --- | --- |
| composition | 32 | 0.596 | 0.596 | 0.079 |
| celltype_density | 32 | 0.592 | 0.586 | 0.075 |
| expression | 32 | 0.581 | 0.577 | 0.076 |
| patch | 32 | 0.579 | 0.579 | 0.076 |
| space_gm | 32 | 0.577 | 0.570 | 0.080 |
| point_pattern | 32 | 0.574 | 0.558 | 0.071 |
| point_pattern_0706 | 32 | 0.550 | 0.569 | 0.077 |
| spatial_distance_0706 | 32 | 0.538 | 0.548 | 0.077 |
| mixing | 32 | 0.523 | 0.553 | 0.073 |
| spatial_distance | 32 | 0.477 | 0.477 | 0.082 |

Note: this table lists `_0706` and non-suffix versions separately for `point_pattern` and `spatial_distance`, so their version differences are visible rather than overwritten.

## New 0706 version vs old non-suffix version
### point_pattern
- Matched rows: 32; 0706 improved: 8; 0706 worsened: 24; unchanged: 0.
- Mean delta `0706 - old`: -0.023; median delta: -0.027.
- Interpretation: the 0706 point-pattern rerun is lower overall than the old non-suffix result, although it improves several cross-cohort rows such as HNC UPMC-to-DFCI.

Largest 0706 gains:
| dataset | task | scheme | metric | 0706 new | old | delta |
| --- | --- | --- | --- | --- | --- | --- |
| hnc_wu2022 | primary_outcome | UPMC_to_DFCI | auc_roc | 0.601 | 0.496 | 0.105 |
| nsclc_aung2025 | immunotherapy_response | Yale_to_UQ | auc_roc | 0.616 | 0.572 | 0.043 |
| hnc_wu2022 | OS | UPMC_to_DFCI | c_index | 0.578 | 0.553 | 0.025 |
| bc_metabric_ali2020 | DSS | cv | c_index | 0.540 | 0.516 | 0.024 |
| bc_jackson2020 | response | cv | auc_roc | 0.586 | 0.562 | 0.024 |

Largest 0706 drops:
| dataset | task | scheme | metric | 0706 new | old | delta |
| --- | --- | --- | --- | --- | --- | --- |
| hnc_wu2022 | primary_outcome | cv | auc_roc | 0.570 | 0.672 | -0.103 |
| nsclc_gnn_hoebel2026 | OS | cv | c_index | 0.618 | 0.695 | -0.077 |
| nsclc_ici_monkman2024 | OS | cv | c_index | 0.473 | 0.539 | -0.066 |
| nsclc_gnn_hoebel2026 | stage_binary | cv | auc_roc | 0.684 | 0.745 | -0.062 |
| tnbc_wang2023 | pCR_arm_CI | cv | auc_roc | 0.578 | 0.638 | -0.061 |

### spatial_distance
- Matched rows: 32; 0706 improved: 25; 0706 worsened: 7; unchanged: 0.
- Mean delta `0706 - old`: 0.061; median delta: 0.062.
- Interpretation: the 0706 spatial-distance rerun is substantially better overall than the old non-suffix result, especially for several CV classification/survival rows.

Largest 0706 gains:
| dataset | task | scheme | metric | 0706 new | old | delta |
| --- | --- | --- | --- | --- | --- | --- |
| nsclc_gnn_hoebel2026 | stage_binary | cv | auc_roc | 0.755 | 0.483 | 0.272 |
| hnc_wu2022 | primary_outcome | cv | auc_roc | 0.591 | 0.410 | 0.181 |
| tnbc_wang2023 | pCR_arm_CI | cv | auc_roc | 0.635 | 0.466 | 0.169 |
| crc_wu2022 | primary_outcome | cv | auc_roc | 0.552 | 0.409 | 0.143 |
| hnc_wu2022 | hpv_status | cv | auc_roc | 0.688 | 0.547 | 0.141 |

Largest 0706 drops:
| dataset | task | scheme | metric | 0706 new | old | delta |
| --- | --- | --- | --- | --- | --- | --- |
| nsclc_aung2025 | immunotherapy_response | Yale_to_YaleExt | auc_roc | 0.572 | 0.644 | -0.072 |
| nsclc_ici_monkman2024 | immunotherapy_response | cv | auc_roc | 0.633 | 0.696 | -0.062 |
| bc_metabric_ali2020 | ER_status | cv | auc_roc | 0.556 | 0.600 | -0.044 |
| luad_sorin2023 | progression | Discovery_to_Validation | auc_roc | 0.546 | 0.580 | -0.034 |
| bc_jackson2020 | clinical_type | cv | balanced_acc | 0.287 | 0.320 | -0.033 |

## Code-level interpretation: fixed-size window local summary + MIL-style aggregation
The patch baseline divides each region into fixed square windows, computes a local feature vector inside each window, and then summarizes all window vectors into one region-level vector using fixed statistics such as mean, max, standard deviation, and quantiles. The downstream model is then a regularized linear classifier or Cox model, so this implementation is best described as handcrafted local-summary pooling with an MIL flavor, not a fully learned MIL model.

Potential reasons for weak or unstable patch/window performance:
- The code names the parameters `window_size_um` and `step_um`, but `patch_feats.py` uses `region.coordinates` directly. If a dataset stores raw coordinates in pixels rather than microns, the physical window size is inconsistent across datasets.
- Command-line arguments `--window-size`, `--step`, and `--feature-type` are parsed in `run_patch_baseline.py` but are not passed into `run()`, so changing them on the command line currently has no effect.
- The grid starts from each region's minimum x/y coordinate, so small translations, crop boundaries, or tissue-mask differences can move cells across window boundaries.
- Empty windows are ignored, which means the pooled result describes occupied windows rather than the full tissue area; this can lose information about sparse tissue or empty spatial compartments.
- With `min_cells_per_window=1`, many windows may be represented by only one or a few cells, making max and quantile summaries noisy.
- Fixed pooling loses the arrangement among windows. Two samples can have similar pooled statistics even if one has clustered immune infiltration and the other has diffuse infiltration.
- The runner currently uses composition features in practice; local expression features are exposed in the class but not actually enabled by the default `run()` implementation.

## Practical conclusion
After correcting the version direction, the main result is: the 0706 `point_pattern` version is not uniformly better and is lower on most matched rows, while the 0706 `spatial_distance` version is clearly stronger than the old non-suffix version. For the patch/window method, the concept is useful for capturing local heterogeneity, but the current implementation should be fixed for coordinate units and argument plumbing before drawing strong biological conclusions from its score.

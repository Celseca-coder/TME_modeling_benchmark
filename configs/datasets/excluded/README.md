# Excluded dataset configs

Configs in this folder are **intentionally excluded** from the benchmark. They are
kept here for provenance but are not picked up by `benchmark.utils.registry.list_datasets()`
(which globs `configs/datasets/*.yaml`, non-recursively), so the validation scripts
(`run_global_composition_baseline.py`, `run_global_expression_baseline.py`) skip them.

| Config | Reason for exclusion |
|--------|----------------------|
| `tnbc_tonic_noah2026.yaml` | No clinical labels available in the deposited data — all `clinical_response` task labels are NaN, so no task is evaluable. |

To re-enable a dataset, move its `.yaml` back up to `configs/datasets/`.

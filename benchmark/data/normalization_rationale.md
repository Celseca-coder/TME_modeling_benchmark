# Biomarker Expression Normalization — Per-Dataset Rationale

This document records the normalization choice for each benchmark dataset and the
reasoning behind it. Decisions weigh three inputs:

1. **The source paper / dataset README** — what the original authors did and what the
   data actually contains.
2. **The community standard for the imaging modality** — IMC, CODEX/PhenoCycler,
   CyCIF, MIBI-TOF, and multiplex IF each have conventional pre-processing.
3. **The empirical value distribution** of the released expression tables
   (profiled with `notebooks/profile_expression.py`).

## The pipeline: `arcsinh` (data) + standardisation (model)

The **data-level** normalization is a single, stateless variance-stabilising transform,
implemented in `benchmark/data/transforms.py::apply_normalization`:

```python
x = arcsinh(x / cofactor)          # variance stabilisation; compresses bright tails
```

This is the part you can inspect directly as a "quick view" of normalised expression.
The **only** per-dataset choice is the arcsinh **cofactor**, which sets where the
transform turns from linear (small signal) to logarithmic (bright signal) and therefore
*must match the data's dynamic range*.

**Feature z-scoring / standardisation is deliberately NOT done in the data path.** It is
a model-side step applied to the patient/region feature matrix, fit on the training
split only — every model (`LogisticRegression`, `RandomForest`, `CoxPH`, `CoxNet`)
median-imputes and `StandardScaler`-standardises features in `fit`. Tree models are
invariant to per-feature scaling so it is a harmless no-op for them; linear / Cox models
need it. Keeping it in the model avoids train→val leakage and keeps the data path simple
and stateless (no cross-region fitting, no scope to choose).

### Two cofactor modes

| Mode | Formula | When to use |
|------|---------|-------------|
| **Fixed** | single scalar for all markers | markers share a comparable dynamic range (most IMC / CODEX panels) |
| **Adaptive** (`cofactor: quantile`) | per-marker `5·p20 + 1e-5` | per-marker brightness is highly heterogeneous **and** the 20th percentile is strictly positive (marker not dominated by exact zeros) |

The adaptive cofactor is the Wu2022 default. It self-scales each marker to its own
low-signal floor, which is ideal for strictly-positive CODEX MFI. **Its failure mode**
is a marker whose 20th percentile is exactly 0 (i.e. >20% of cells are zero): the
cofactor collapses to the `1e-5` epsilon and `arcsinh(x / 1e-5)` effectively soft-
binarizes that marker, discarding quantitative signal. We therefore only select the
adaptive mode where the bottom quantile is positive for (essentially) all markers.

### Binary data

`NSCLC-GNN_Hoebel2026` expression is binary marker positivity (0/1). No intensity
transform is meaningful, so `method: none`.

## Decision summary

| Dataset | Modality | Raw value scale (sampled) | % exact zeros | Method | Cofactor |
|---------|----------|---------------------------|---------------|--------|----------|
| BC_Jackson2020 | IMC | 0–289, median 0.40 | 2.6% | arcsinh | 1.0 |
| BC-METABRIC_Ali2020 | IMC | 0–254, median 0.36 | 2.6% | arcsinh | 0.8 |
| CRC_Schürch2020 | CODEX/CyCIF | 0–4.9e4, median 79 | 15.8% | arcsinh | 150.0 |
| LUAD_Sorin2023 | IMC | 0–255, median 0 | 70.2% | arcsinh | 1.0 |
| NSCLC_Aung2025 | CODEX (rescaled 0–255) | 0–255, median 3 | 14.2% | arcsinh | 5.0 |
| NSCLC-GNN_Hoebel2026 | mIF (5-plex) | binary 0/1 | 88.3% | none | — |
| NSCLC-ICI_Monkman2024 | CODEX | 0–3.2e4, median 67 | 6.3% | arcsinh | 150.0 |
| TNBC_Wang2023 | IMC | 0–251, median ~3e-6 | 3.1% | arcsinh | **quantile** |
| TNBC-TONIC_Noah2026 | MIBI-TOF | 0–0.79, median 2.5e-4 | 8.0% | arcsinh | 0.01 |
| CRC_Wu2022 | CODEX | 0–6.4e4, median 142 | 0.0% | arcsinh | **quantile** |
| HNC_Wu2022 | CODEX | 0–4.6e4, median 85 | 0.0% | arcsinh | **quantile** |

All configs were verified to produce finite arcsinh output within a sensible range
(see "Transformed range check" below); binary Hoebel data is left as 0/1.

## Per-dataset reasoning

### CRC_Wu2022, HNC_Wu2022 — CODEX → adaptive (reference)
Raw CODEX mean fluorescence intensity (MFI), large dynamic range (up to ~6e4), and
**0% exact zeros** — every marker's 20th percentile is strictly positive (min p20 ≈
13–32). This is exactly the regime the adaptive `5·p20 + 1e-5` cofactor was designed
for, and it is the pipeline already validated on these two datasets. Kept as the
reference; `cofactor: quantile`.

### NSCLC-ICI_Monkman2024 — CODEX raw MFI → fixed 150
Same modality and the same physical TMA (4301) as the Aung cohort, but released as
**raw MFI** (0–3.2e4; per-marker maxima span 1.5k–40k). The README recommends
cofactor 150, which matches this scale, and it is the community-standard CODEX
cofactor. The adaptive mode is *not* used here because 4/26 markers have a zero 20th
percentile (6.3% zeros overall) and would soft-binarize. Fixed 150 transforms all
markers sensibly (`arcsinh(1461/150)=2.9` … `arcsinh(39966/150)=6.3`).

### NSCLC_Aung2025 — CODEX rescaled to 0–255 → fixed 5.0
Although the README (copied from the Wu2022 CODEX convention) recommends cofactor
150, the released values are **not** raw MFI: they run 0–255 in 0.5 steps and are
clipped at 255 (median 3, p99 192). This is a rescaled ~8-bit intensity. Cofactor 150
on a 0–255 scale is nearly the identity (`arcsinh(255/150)=1.3`) and wastes
resolution. A cofactor of **5** (≈ `5 × median p20`) restores a proper
linear-to-log transition (`arcsinh(3/5)=0.57`, `arcsinh(192/5)=4.3`). This is the
clearest case where the README recommendation must be overridden by the actual data
scale. Fixed (not adaptive) because 16/44 markers have p20 = 0.

### CRC_Schürch2020 — CODEX/CyCIF high-MFI → fixed 150
High-dynamic-range fluorescence intensities (0–4.9e4, median 79), comparable in scale
to HNC_Wu2022. However **15.8% of values are exact zeros and 19/58 markers have p20 =
0**, so the adaptive cofactor would soft-binarize a third of the panel. A fixed
cofactor of 150 (matched to this MFI scale, same as the other fluorescence datasets)
gives good compression (`arcsinh(79/150)=0.5`, `arcsinh(4880/150)=4.2`) without that
pathology.

### BC_Jackson2020 — IMC → fixed 1.0
IMC mean ion counts (0–289, median 0.4, only 2.6% zeros). `arcsinh` with **cofactor 1**
is the field standard for IMC (Bodenmiller-lab convention) and is what the README
notes. Values are modest and near-positive, so a fixed cofactor of 1 is appropriate;
adaptive is unnecessary (and one marker has p20 = 0).

### BC-METABRIC_Ali2020 — IMC → fixed 0.8
IMC ion counts on the same scale as Jackson (0–254, median 0.36). The README states
that **Ali et al. used arcsinh cofactor 0.8** for their single-cell clustering, so we
adopt the paper-specific value rather than the generic 1.0. All 39 markers have p20 >
0, but we defer to the published cofactor over the adaptive mode for reproducibility.

### LUAD_Sorin2023 — IMC (very sparse) → fixed 1.0
IMC signal with **70% exact zeros** (median 0; 17/18 markers have p20 = 0). The
adaptive cofactor is impossible here (it would collapse to epsilon for almost every
marker). The IMC-standard fixed **cofactor 1** handles the sparse-but-bounded values
(nonzero p90 ≈ 50, p99 ≈ 125) cleanly.

### TNBC_Wang2023 — IMC (extreme per-marker scale spread) → adaptive
This dataset is the textbook case for the adaptive cofactor. Values are extremely
right-skewed and on **wildly different per-marker scales**: per-marker medians span
~7 orders of magnitude (≈6e-7 to 1.3) and maxima 3e-5 to 251, with 95.6% of all values
below 1e-3. A single fixed cofactor cannot serve markers this heterogeneous — cofactor
1 would act as the identity for the ~7-orders-smaller markers. Crucially, **all 46
markers have p20 > 0**, so the adaptive `5·p20 + 1e-5` cofactor is safe and normalises
each marker to its own scale. `cofactor: quantile`.

### TNBC-TONIC_Noah2026 — MIBI-TOF → fixed 0.01
MIBI-TOF intensities are Rosetta-compensated and instrument-normalized upstream,
yielding a **very small dynamic range** (0–0.79, median 2.5e-4, p99 0.017). The README
recommends an empirical cofactor of 0.001–0.01. We use **0.01**, which gives a good
linear-to-log spread on this scale (`arcsinh(2.5e-4/0.01)=0.025`,
`arcsinh(0.017/0.01)=1.3`, `arcsinh(0.785/0.01)=5.05`). The paper's SpaceCat pipeline
likewise standardises features before modeling, consistent with our model-side
standardisation step. Fixed (not adaptive) because 3/37 markers have p20 = 0 and the
scale is uniform across the panel.

### NSCLC-GNN_Hoebel2026 — binary mIF → none
Expression is **binary 0/1** marker positivity from inForm classification (5 channels,
88% zero). No intensity transform is meaningful; arcsinh/z-score on 0/1 data would be
nonsensical. `method: none` — values are passed through unchanged for downstream
rule-based / GNN features.

## Transformed range check

After applying the per-dataset cofactor, the arcsinh-transformed values land in a
sensible, mutually comparable range across all datasets (sampled over 6 regions each):

| Dataset | cofactor | p50 | p99 | p99.9 | max |
|---------|----------|-----|-----|-------|-----|
| BC_Jackson2020 | 1.0 | 0.23 | 3.69 | 4.51 | 5.91 |
| BC-METABRIC_Ali2020 | 0.8 | 0.49 | 4.81 | 5.31 | 6.57 |
| CRC_Schürch2020 | 150 | 0.27 | 3.80 | 4.76 | 6.07 |
| LUAD_Sorin2023 | 1.0 | 0.00 | 5.54 | 6.00 | 6.23 |
| NSCLC_Aung2025 | 5.0 | 0.73 | 4.58 | 4.63 | 4.63 |
| NSCLC-GNN_Hoebel2026 | none | 0.00 | 1.00 | 1.00 | 1.00 |
| NSCLC-ICI_Monkman2024 | 150 | 0.26 | 3.17 | 4.27 | 5.89 |
| TNBC_Wang2023 | quantile | 0.13 | 2.63 | 4.08 | 5.64 |
| TNBC-TONIC_Noah2026 | 0.01 | 0.03 | 1.82 | 4.76 | 5.06 |
| CRC_Wu2022 | quantile | 0.24 | 2.96 | 4.22 | 6.37 |
| HNC_Wu2022 | quantile | 0.25 | 2.22 | 3.47 | 5.12 |

All minima are 0 (expression is non-negative) and all maxima sit around 5–6.6, so the
cofactors are well matched to each platform's dynamic range — no dataset is left flat
(under-compressed) or blown up (over-compressed). **No additional constant scaler is
needed.** Any residual scale offset between datasets is absorbed by the model-side
`StandardScaler` anyway. (`NSCLC-GNN_Hoebel2026` stays 0/1 as expected.)

## Reproducing the distribution profile

```bash
conda activate p3
python notebooks/profile_expression.py
```

This prints, per dataset, the global min/max/median, percentiles, fraction of exact
zeros, and the per-marker 20th-percentile statistics used to judge adaptive-cofactor
applicability.

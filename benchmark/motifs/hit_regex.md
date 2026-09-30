# Candidate regex by task

## `motif_tumor_high`

- `composition::.*{TUMOR}`
- `tissue_density::.*{TUMOR}`
- `(^|::)(tumor(?!_stroma)|epithelial|panck|pan-?ck)$`
- `tumor_area_ratio`

## `motif_cd8_high`

- `composition::.*{CD8}`
- `tissue_density::.*{CD8}`
- `(^|::)(cd8 t cells?|t cells?)$`

## `motif_cd8_clustering`

- `{CD8}.*(?:[_ ][kl]_r|ripley|pcf_r|cluster)`
- `neighbor_fraction__.*{CD8}.*__to__.*{CD8}`
- `mixing::(?:local_mixing|same_type_neighbor_fraction)`
- `spatial-distance::(?:same_neighbor_dist|proportion_same_nearest|mean_proportion_same_neighbors)`

## `motif_tumor_stroma_mixing`

- `neighbor_fraction__.*{TUMOR}.*__to__.*{STROMA}`
- `neighbor_fraction__.*{STROMA}.*__to__.*{TUMOR}`
- `{TUMOR}.*(?:[_ ][kl]_r|ripley|pcf_r)`
- `{STROMA}.*(?:[_ ][kl]_r|ripley|pcf_r)`
- `mixing::(?:local_mixing|same_type_neighbor_fraction)`
- `(min|mean)_dist_.*{TUMOR}.*_to_.*{STROMA}`
- `(min|mean)_dist_.*{STROMA}.*_to_.*{TUMOR}`
- `spatial-distance::cross_neighbor_dist`
- `spatial-distance::(?:same_neighbor_dist|proportion_same_nearest|mean_proportion_same_neighbors)`

## `motif_interface_immune`

- `tumor_density::.*{IMMUNE}`
- `neighbor_fraction__.*{IMMUNE}.*__to__.*{TUMOR}`
- `neighbor_fraction__.*{TUMOR}.*__to__.*{IMMUNE}`
- `(?:{CD8}|{T_CELL}|{B_CELL}|{MAC}|{IMMUNE}).*(?:[_ ][kl]_r|ripley|pcf_r)`
- `interface`
- `(min|mean)_dist_.*{IMMUNE}.*_to_.*{TUMOR}`
- `(min|mean)_dist_.*{TUMOR}.*_to_.*{IMMUNE}`

## `motif_immune_exclusion`

- `tumor_density::.*{CD8}`
- `neighbor_fraction__.*{CD8}.*__to__.*{TUMOR}`
- `neighbor_fraction__.*{TUMOR}.*__to__.*{CD8}`
- `{CD8}.*(?:[_ ][kl]_r|ripley|pcf_r)`
- `(min|mean)_dist_.*{CD8}.*_to_.*{TUMOR}`
- `(min|mean)_dist_.*{TUMOR}.*_to_.*{CD8}`
- `spatial-distance::cross_neighbor_dist`

## `motif_t_tumor_mixing`

- `neighbor_fraction__.*{T_CELL}.*__to__.*{TUMOR}`
- `neighbor_fraction__.*{TUMOR}.*__to__.*{T_CELL}`
- `tumor_density::.*{T_CELL}`
- `{T_CELL}.*(?:[_ ][kl]_r|ripley|pcf_r)`
- `{TUMOR}.*(?:[_ ][kl]_r|ripley|pcf_r)`
- `(min|mean)_dist_.*{T_CELL}.*_to_.*{TUMOR}`
- `(min|mean)_dist_.*{TUMOR}.*_to_.*{T_CELL}`
- `spatial-distance::cross_neighbor_dist`

## `motif_cd8_tumor_contact`

- `neighbor_fraction__.*{CD8}.*__to__.*{TUMOR}`
- `neighbor_fraction__.*{TUMOR}.*__to__.*{CD8}`
- `tumor_density::.*{CD8}`
- `{CD8}.*(?:[_ ][kl]_r|ripley|pcf_r)`
- `{TUMOR}.*(?:[_ ][kl]_r|ripley|pcf_r)`
- `(min|mean)_dist_.*{CD8}.*_to_.*{TUMOR}`
- `(min|mean)_dist_.*{TUMOR}.*_to_.*{CD8}`
- `spatial-distance::cross_neighbor_dist`

## `motif_macrophage_tumor_niche`

- `neighbor_fraction__.*{MAC}.*__to__.*{TUMOR}`
- `neighbor_fraction__.*{TUMOR}.*__to__.*{MAC}`
- `tumor_density::.*{MAC}`
- `{MAC}.*(?:[_ ][kl]_r|ripley|pcf_r)`
- `{TUMOR}.*(?:[_ ][kl]_r|ripley|pcf_r)`
- `(min|mean)_dist_.*{MAC}.*_to_.*{TUMOR}`
- `(min|mean)_dist_.*{TUMOR}.*_to_.*{MAC}`
- `spatial-distance::cross_neighbor_dist`

## `motif_apc_t_contact`

- `neighbor_fraction__.*{APC}.*__to__.*{T_CELL}`
- `neighbor_fraction__.*{T_CELL}.*__to__.*{APC}`
- `{APC}.*(?:[_ ][kl]_r|ripley|pcf_r)`
- `{T_CELL}.*(?:[_ ][kl]_r|ripley|pcf_r)`
- `(min|mean)_dist_.*{APC}.*_to_.*{T_CELL}`
- `(min|mean)_dist_.*{T_CELL}.*_to_.*{APC}`
- `spatial-distance::cross_neighbor_dist`

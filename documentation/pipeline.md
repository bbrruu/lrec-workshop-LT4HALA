# Reproducible Pipeline Specification

This document describes the analysis pipeline used in the paper. The public release contains the scripts and derived results, but not the large embedding files or the complete source corpus.

## Pipeline Overview

The conceptual workflow in the paper is implemented as eight executable stages:

```text
extract -> clean1 -> sample -> clean2 -> local -> global -> postprocess -> local_annotate
```

1. **Extract** contextual embeddings for each target character.
2. **Clean 1** remove obvious OCR errors from extracted records.
3. **Sample** stratify by document and keep at most 200 occurrences per document.
4. **Clean 2** apply additional sentence-quality filters.
5. **Local** cluster each historical period independently.
6. **Global** cluster all periods in one shared analysis (called Joint Clustering in some filenames).
7. **Postprocess** attach reviewed semantic labels and create figures.
8. **Local annotation** transfer reviewed Global labels to Local clusters by dynasty-specific majority vote.

## Canonical Entry Point

The release entry point is [`../code/pipeline.py`](../code/pipeline.py). The original source corpus and model must be supplied separately when running the extraction stages.

Example:

```bash
python3 code/pipeline.py \
  --chars 手,水,道 \
  --run-name paper_run \
  --start-stage extract \
  --end-stage local_annotate \
  --label-map-json configs/dao_shui_labels.json
```

The exact input paths and environment must be adapted to the user's local copy of the source corpus and GujiBERT-fan model.

## Parameters Used in the Paper

| Setting | Local | Global |
|---|---:|---:|
| Initial clusters (`K`) | 20 | 20 |
| Cosine merge threshold | 0.95 | 0.95 |
| Split ratio | 0.3 | 0.1 |
| Sub-cluster count | dynamic | 6 |

Clustering uses 768-dimensional contextual embeddings. MiniBatchKMeans, PCA, and silhouette sampling use `random_state=42` where the implementation exposes that setting. Local PCA is fitted jointly across all periods so that Local panels share one coordinate system.

## Output Layout

```text
results/
├── hand/
│   ├── global/       # Global assignments, statistics, and raw figures
│   ├── local/        # Dynasty-specific assignments and statistics
│   └── annotated/    # Reviewed semantic labels and annotated figures
├── water/
└── dao/
```

For each character:

- `*_global_assignments.csv` contains one row per occurrence with Global cluster IDs and PCA coordinates.
- `*_local_assignments.csv` contains one row per occurrence with Local cluster IDs and shared PCA coordinates.
- `*_global_stats.json` contains cluster counts, silhouette score, representative examples, and merge/split history.
- `*_stats.json` in `local/` contains dynasty-level cluster statistics.
- `*_global_labeled.csv` and `*_local_labeled.csv` contain `Macro_Category` and `Sub_Category` annotations.
- PNG files contain the raw and semantically annotated visualizations.

## Annotation

Global labels are defined in [`../configs/dao_shui_labels.json`](../configs/dao_shui_labels.json). Local labels are derived for each `(dynasty, Local_Cluster)` pair by joining Local and Global assignments and inheriting the majority Global label. The mapping reports preserve the per-cluster distribution and identify mixed clusters.

The paper's "Human-in-the-Loop Semantic Annotation" description (Section 3.2) covers this inheritance step implicitly: the core-sentence LLM/human annotation was performed once, on the Global (joint) clusters only. Local clusters were not separately annotated from their own core sentences; every Local cluster label shown in the Local figures is inherited from a Global label via the majority-vote procedure above, not an independent annotation. The `"local"` block in [`../configs/dao_shui_labels.json`](../configs/dao_shui_labels.json) is an unused placeholder for this reason — only the `"joint"` block is read by the pipeline.

## Source and Reuse Restrictions

The corpus is obtained from the Chinese Text Project. This release does not redistribute a complete corpus copy. Users must obtain the source data and confirm the current access and redistribution terms before recreating occurrence-level outputs. Model weights are also subject to their own distribution terms.

# Capturing Ancient Chinese Sense Induction with Automatic Pipelines

This repository accompanies the paper on automatically identifying diachronic sense patterns in Ancient Chinese. It provides the analysis code, paper figures, cluster assignments, semantic annotations, and summary statistics for three characters:

- 手 (*shǒu*, “hand”)
- 水 (*shuǐ*, “water”)
- 道 (*dào*, “way, road, principle”)

The paper and detailed reproduction notes are available in [`documentation/`](documentation/), including the [paper PDF](documentation/paper.pdf).

The pipeline runs as 9 stages, orchestrated by [`code/pipeline.py`](code/pipeline.py):

```text
extract → clean1 → sample → clean2 → local → global → label_template → global_plot → local_annotate
```

Full stage-by-stage detail (what each script reads/writes, parameters, and how semantic labels are produced) is in [`documentation/pipeline.md`](documentation/pipeline.md) — that document, not this one, is the source of truth for anything below the summary level.

## What Is Included

| Directory | Contents |
|---|---|
| [`code/`](code/) | The 9-stage pipeline (one script per stage) plus `pipeline.py` (orchestrator) and `pipeline_common.py` (shared helpers) |
| [`configs/`](configs/) | Run parameters and Global cluster semantic labels |
| [`results/shou/`](results/shou/) | Global, Local, and annotated results for 手 |
| [`results/shui/`](results/shui/) | Global, Local, and annotated results for 水 |
| [`results/dao/`](results/dao/) | Global, Local, and annotated results for 道 |
| [`results/sensitivity_analysis/`](results/sensitivity_analysis/) | Parameter sweep backing the K/merge/split sensitivity analysis in Section 3.2 |
| [`figures/`](figures/) | Figures corresponding to Figures 1–6 in the paper |
| [`documentation/`](documentation/) | Pipeline specification and reproduction checklist |

Each result directory contains CSV assignments, JSON statistics, and visualizations. The `annotated/` directories contain the semantic labels used in the paper — a human/LLM-in-the-loop step (道/水 and 手 were labeled by two different methods; see [`documentation/pipeline.md`'s Annotation section](documentation/pipeline.md#annotation)).

## Main Results

| Character | Occurrences | Global clusters | Local clusters |
|---|---:|---:|---:|
| 手 | 185,792 | 17 | dynasty-specific |
| 水 | 313,184 | 12 | dynasty-specific |
| 道 | 331,910 | 22 | dynasty-specific |

The Global results use `K=20`, a cosine merge threshold of `0.95`, a split ratio of `0.1`, and fixed `sub-K=6`. Local results use `K=20`, the same merge threshold, and a split ratio of `0.3` with dynamic sub-clustering. Clustering, PCA, and silhouette sampling use `random_state=42` where applicable.

## Reproduction

The canonical entry point is [`code/pipeline.py`](code/pipeline.py). The complete stage specification and example commands are in [`documentation/pipeline.md`](documentation/pipeline.md), with a shorter checklist in [`documentation/reproduction_checklist.md`](documentation/reproduction_checklist.md).

This public release contains the derived results and does not include the large embedding files or vector-containing PKL files required to start from raw text. To reproduce the full extraction pipeline, obtain the source texts and model according to their access and redistribution terms, then provide them as inputs to the scripts in `code/`.

## Data and Licensing Notes

The source corpus is the Chinese Text Project (CText). This repository does not redistribute a complete copy of that corpus. Users should obtain source texts directly from CText and follow its current access and redistribution terms. GujiBERT-fan model usage is subject to its own distribution terms.

The release intentionally excludes raw corpus dumps, embedding files, model weights, credentials, and private annotation material. The CSV files contain derived occurrence-level contexts; users should verify that their intended redistribution complies with the applicable source-corpus terms.

## Terminology

The paper uses **Global Clustering** for the cross-period analysis. Some implementation filenames use **Joint Clustering** for the same analysis. Local results use one PCA model fitted across all periods, allowing the dynasty panels to be compared in a shared coordinate system.

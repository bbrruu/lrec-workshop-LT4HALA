# Capturing Ancient Chinese Sense Induction with Automatic Pipelines

This repository contains the code and reproducibility materials for the study of diachronic sense induction in Ancient Chinese. The paper's six conceptual stages are implemented as eight executable stages because the two cleaning stages and the post-processing stages are kept separate:

`extract -> clean1 -> sample -> clean2 -> local -> global -> postprocess -> local_annotate`

## Canonical pipeline

Use [final_pipeline.py](final_pipeline.py) as the canonical entry point. Its defaults match the paper:

- contextual embeddings from GujiBERT-fan
- at most 200 occurrences per document during stratified sampling
- 768-dimensional embeddings for clustering
- Local: `K=20`, merge cosine threshold `0.95`, split ratio `0.3`
- Global: `K=20`, merge cosine threshold `0.95`, split ratio `0.1`, fixed `sub-K=6`
- `random_state=42` in the clustering, PCA, and silhouette sampling code

The complete stage specification and commands are documented in [final_pipeline.md](final_pipeline.md).

## Verified paper results

The Global outputs in `pipeline_runs/run_dao_shui/stage_outputs/06_joint_output/` match the appendix:

| Character | Final rows | Final clusters | Global parameters |
|---|---:|---:|---|
| 水 | 313,184 | 12 | `K=20`, `merge=0.95`, `split=0.1`, `sub-K=6` |
| 道 | 331,910 | 22 | `K=20`, `merge=0.95`, `split=0.1`, `sub-K=6` |

Both counts were checked against the corresponding `joint_stats.json` and merged CSV files.

## Public release boundary

Recommended repository contents:

- pipeline source code and execution documentation
- `config.json`, summary statistics, selected small logs, cluster assignments, labels, and merge/split histories
- final figures and the scripts that generate them
- metadata and links or identifiers for the source texts
- label templates and human annotation records, with any restricted text removed

Do not commit the following without a separate size and rights review:

- raw CText copies or complete source passages
- embedding JSONL files and vector-containing PKL files
- API keys, credentials, or private annotations
- model weights whose redistribution license is not confirmed

The source corpus should be obtained from the Chinese Text Project according to its current access and redistribution terms. The repository should record the retrieval date, source URLs or identifiers, and the exact preprocessing configuration.

## Result naming

The paper calls the cross-period analysis **Global Clustering**. The implementation file and output directory use **Joint Clustering** for the same analysis. Local output uses a shared PCA model fitted on all periods, so Local panels can be compared in one coordinate system.

Older `0212`, `0218`, `0219`, and parameter-test outputs are exploratory or historical results. They should not be presented as the canonical paper result unless their parameters are explicitly stated.

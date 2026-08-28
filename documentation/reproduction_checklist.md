# Reproduction Checklist

This checklist is for reproducing or auditing the results released with the paper.

## Before Running

- [ ] Obtain access to the Chinese Text Project source data.
- [ ] Obtain GujiBERT-fan and confirm its usage terms.
- [ ] Install Python 3 and the packages used by the scripts: NumPy, pandas, scikit-learn, PyTorch, Transformers, matplotlib, seaborn, tqdm, and adjustText.
- [ ] Confirm that the source metadata and model paths are available.
- [ ] Record the source retrieval date, source identifiers, package versions, and hardware.

## Canonical Settings

- [ ] Target characters: 手, 水, 道.
- [ ] Historical periods: pre-Qin, Qin-Han, Wei-Jin, Sui-Tang, Song-Yuan, Ming, Qing, Republican.
- [ ] Maximum sampled occurrences per document: 200.
- [ ] Embedding dimension: 768.
- [ ] Local clustering: `K=20`, merge threshold `0.95`, split ratio `0.3`, dynamic sub-clustering.
- [ ] Global clustering: `K=20`, merge threshold `0.95`, split ratio `0.1`, `sub-K=6`.
- [ ] Random state: `42` where supported by the implementation.

## Recommended Run Order

Use [`../code/pipeline.py`](../code/pipeline.py) from the repository root. For a full run, provide the source metadata and model environment required by the extraction script:

```bash
python3 code/pipeline.py \
  --chars 手,水,道 \
  --run-name paper_run \
  --start-stage extract \
  --end-stage local_annotate \
  --label-map-json configs/dao_shui_labels.json
```

For an audit of the released clustering outputs, inspect:

- [`results/hand/`](../results/hand/)
- [`results/water/`](../results/water/)
- [`results/dao/`](../results/dao/)

## Expected Result Counts

| Character | Occurrences | Global clusters |
|---|---:|---:|
| 手 | 185,792 | 17 |
| 水 | 313,184 | 12 |
| 道 | 331,910 | 22 |

The Global counts can be checked in each character's `global/*_global_stats.json` and assignment CSV. The annotated Global and Local CSV files should contain the same number of rows as the corresponding assignment files and should contain `Macro_Category` and `Sub_Category` columns.

## Audit Checks

- [ ] Global and Local assignment row counts match the expected counts.
- [ ] Annotation CSVs contain no `Unlabeled` rows in the released results.
- [ ] Global parameters match the canonical settings above.
- [ ] Local PCA coordinates are interpreted in the shared cross-period space.
- [ ] Figure files can be traced to the corresponding result directory and script.
- [ ] Any difference caused by package versions, source snapshots, or model versions is recorded.

## Public Release Boundary

The release includes code, configurations, derived assignments, statistics, annotations, and figures. It excludes raw corpus dumps, large embedding JSONL files, vector-containing PKL files, model weights, credentials, and private annotation material. These exclusions mean that a completely fresh run requires separately obtained source data and model files.

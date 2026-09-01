# Reproducible Pipeline Specification

This document describes the analysis pipeline used in the paper. The public release contains the scripts and derived results, but not the large embedding files or the complete source corpus.

## Pipeline Overview

The conceptual workflow in the paper is implemented as eight executable stages, each a standalone script in [`../code/`](../code/) and orchestrated by [`../code/pipeline.py`](../code/pipeline.py):

```text
extract -> clean1 -> sample -> clean2 -> local -> global -> postprocess -> local_annotate
```

| Stage | Script | Purpose |
|---|---|---|
| 1. Extract | `extract_keyword.py` | Read the local CText archive, keep articles that contain punctuation, split them into sentences, and compute a contextual embedding for every occurrence of each target character. |
| 2. Clean 1 | `clean_ocr_errors.py` (+ `detect_ocr_errors.py`) | Remove sentences with obvious OCR artifacts (marker glyphs, runaway length, excessive character repetition). |
| 3. Sample | `stratified_sampling.py` | Stratify by source document (`toptitle`) and reservoir-sample at most 200 occurrences per document, so no single text dominates a period. |
| 4. Clean 2 | `clean_data.py` | Apply a second, stricter sentence-quality filter (length, character diversity, digit density, abnormal patterns) — deliberately *not* punctuation-based. |
| 5. Local | `local_clustering.py` | Cluster each historical period independently (overcluster → cosine-merge → dynamic split). |
| 6. Global | `global_clustering.py` | Cluster all periods jointly in one shared embedding space (called "Joint Clustering" in some filenames/comments). |
| 7. Postprocess | (`pipeline.py`, postprocess stage) | Attach reviewed semantic labels and render the paper's figures. |
| 8. Local annotation | (`pipeline.py`, local_annotate stage) | Transfer reviewed Global labels onto Local clusters by dynasty-specific majority vote. |

## Canonical Entry Point

The release entry point is [`../code/pipeline.py`](../code/pipeline.py), run from the `paper_release/` root so relative config paths resolve:

```bash
python3 code/pipeline.py \
  --chars 手,水,道 \
  --run-name paper_run \
  --start-stage extract \
  --end-stage local_annotate \
  --label-map-json configs/dao_shui_labels.json
```

`pipeline.py` locates the six stage scripts next to itself (`code/`), independent of `--workdir`; `--workdir` (default `.`) only controls where run outputs (`--output-root`, default `final_pipeline_results/`) and label-map JSON paths are resolved. The original source corpus and model must be supplied separately when running the extraction stage (`--extract-metadata` for the CText metadata CSV, and the local archive paths hardcoded in `extract_keyword.py`, `BOOK_DATA_PATH` / `WIKI_DATA_PATTERN`).

## Stage-by-Stage Detail

### 1. Extract — `extract_keyword.py`

- `read_ctext_archive()` decompresses the local CText bulk archive (`.../book/text_book_raw.gz` and `.../wiki/text_wiki_raw_part_*`) into an in-memory `{urn: fulltext}` map. **This step loads every article, punctuated or not — there is no filtering yet.**
- For each article selected by `compact_new_metadata.csv` (filtered to the target dynasties), the article-level gate is applied before any embedding is computed:
  ```python
  if not has_punctuation(text):
      skipped_no_punct += 1
      continue
  ```
  `has_punctuation()` checks for any of `。，、；：？！「」『』`. An article without at least one of these marks is skipped entirely — it is never split into sentences and contributes no occurrences to the dataset.
- Articles that pass are split into sentences with `split_into_sentences()` (paragraph breaks on `\n\n`, then split on `。！？；`), and every occurrence of a target character within those sentences is embedded with GujiBERT‑fan (`MODEL_ID`) as a 768‑dim contextual vector.
- Output: one JSONL file per `(dynasty, character)` under `01_keyword_embeddings/`, one record per character occurrence (`sentence`, `dynasty`, `dynasty_category`, `reference_id`, `toptitle`, `vector`, …).

**This is the only point in the pipeline where punctuation is used as a filter, and it filters whole articles, not individual sentences.**

### Why two cleaning stages?

The pipeline cleans the data twice — once right after extraction (Clean 1) and again after sampling (Clean 2) — because the two passes target different kinds of problems and sit on either side of a step that depends on getting the first one right:

- **Different failure modes.** Clean 1 removes artifacts that are unambiguous scanning/OCR failures — marker glyphs, runaway sentence length, a character repeated five-plus times in a row. These are binary, rule-based defects with no judgment call involved. Clean 2 instead scores sentence *naturalness* on continuous signals (character diversity, digit density, rare-character ratio) that only make sense as a threshold, not a yes/no rule — hence the `--aggression` levels (`mild`…`ultra`) rather than a single fixed check.
- **Sequencing around Sample.** Clean 1 runs *before* the Sample stage so the per-document reservoir sampling (`--max-per-toptitle`, 200 occurrences) draws from an already-de-junked pool — otherwise sampling could burn a document's 200-slot quota on OCR garbage before the real cleaning ever ran. Clean 2 runs *after* Sample precisely because it is the more expensive, more subjective pass: applying six heuristic checks to the full pre-sample volume would be wasted work once most of it is discarded by sampling anyway, so it only runs on the reduced, already-sampled set that actually reaches clustering.

In short: Clean 1 is a cheap, deterministic pre-filter that protects the sampling step from picking bad data; Clean 2 is a pricier, threshold-tuned quality pass applied to the final candidate set right before it is used. They are not redundant passes over the same problem.

### 2. Clean 1 — `clean_ocr_errors.py` (default `--clean1-mode conservative`)

Uses `OCRErrorDetector` (`detect_ocr_errors.py`) per sentence and removes a sentence only if, in conservative mode:

- it contains an OCR marker glyph (`□〇○●◎◇◆※☆★■▪▲△▽`), or
- it is longer than 300 characters (`length_anomaly: too_long`), or
- it contains the same character repeated 5+ times in a row (`suspicious_patterns`, `\1{4,}`).

`aggressive` mode instead removes a sentence on *any* detected issue (including any non-Han/non-punctuation character or length < 5); `custom` mode is a third preset. The paper uses the default `conservative` setting. Output mirrors the input layout under `02_cleaned_embeddings/`.

### 3. Sample — `stratified_sampling.py`

- Groups occurrences by `toptitle` (source document) within each dynasty.
- Reservoir-samples at most `--max-per-toptitle` (default 200) occurrences per document, so a single long text cannot dominate a period's sample.
- Optionally caps total occurrences per character via `--baseline-max-samples` (pipeline default 10000).
- Output: one pickle per `(character, dynasty)` under `03_stratified_sampling_output_cleaned/`.

### 4. Clean 2 — `clean_data.py` (default `--aggression normal`)

A second, independent quality filter over the already-sampled records. It explicitly **does not** check punctuation density — the code comment explains why: sentences were already split on `。；？！` during extraction, so most sentences legitimately lack trailing punctuation, and a punctuation check would wrongly reject normal sentences. Instead it scores each sentence against six checks (all must pass):

1. length within `[min_length, max_length]`
2. character-uniqueness ratio ≥ threshold (catches repeated-OCR-misread text)
3. no run of the same character beyond `max_repeat_char`
4. no invalid characters (`□`, `�`, `●`, …)
5. digit density ≤ threshold
6. a set of heuristic patterns (very long sentences with zero punctuation, too many rare/uncommon characters, too many CJK-extension characters)

Thresholds are keyed by `--aggression` (`mild`/`normal`/`aggressive`/`ultra`); `normal` is the setting used in the paper. Output under `04_cleaned_output/`.

### 5. Local — `local_clustering.py`

Overclustering + auto-merge + hierarchical refinement, run **independently per dynasty**: MiniBatchKMeans with `--k-init` (20) initial clusters, merge clusters whose centroids exceed `--merge-ratio` (0.95) cosine similarity, then split any cluster larger than `--split-ratio` (0.3) of the period's occurrences into a dynamically chosen number of sub-clusters (2–6). PCA is fit **jointly across all periods** so the eight per-dynasty panels share one coordinate system. Output under `05_local_output/`.

### 6. Global — `global_clustering.py`

Same overcluster → merge → split algorithm, but run **once over all periods pooled together** (`--k` 20, `--merge-ratio` 0.95, `--split-ratio` 0.1, fixed `--sub-k` 6), so cluster identities are directly comparable across dynasties. Output under `06_joint_output/`.

### 7. Postprocess

Reads the Global/Local assignment CSVs, applies the reviewed semantic label mapping from `configs/dao_shui_labels.json` (or the hand-specific 手 label profile), and renders the paper's figures (trajectory plots, per-dynasty facets, cluster-composition bars). Output under `07_postprocess/`.

### 8. Local annotation

For each `(dynasty, Local_Cluster)` pair, joins Local and Global assignments on the sentence level and inherits the majority Global label — Local clusters are never independently annotated. Output under `08_local_annotated/`.

## Data Cleaning Summary

| Filter | Stage | Unit | Uses punctuation? |
|---|---|---|---|
| `has_punctuation()` article gate | Extract | whole article | **Yes** — the only punctuation-based filter in the pipeline |
| OCR marker / length / repetition | Clean 1 | sentence | No |
| Document-level reservoir cap | Sample | document | No |
| Length / uniqueness / digits / heuristics | Clean 2 | sentence | No (deliberately excluded, see above) |

Net effect: every contextual embedding used for clustering comes from a sentence extracted out of an article that contained punctuation, but no sentence is kept or dropped later in the pipeline because of its own punctuation.

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

## Notes for Reproducers

The release originally shipped `pipeline.py` unmodified from the internal working copy, where it called sibling scripts by their pre-release names/locations (`mainfiles/clean_ocr_errors.py`, `mainfiles/stratified_sampling_cleaned.py`, `mainfiles/0212_clean_data.py`, `0219_align_clustering.py`, `0219_joint_cluster.py`) and `clean_ocr_errors.py` imported a `detect_ocr_errors` module that was never copied into the release. Both issues are fixed as of this revision: `pipeline.py` now resolves all six stage scripts next to itself in `code/`, and `code/detect_ocr_errors.py` is included. Running `python3 code/pipeline.py` from `paper_release/` per the example above now works end-to-end given a valid CText archive and metadata file.

## Source and Reuse Restrictions

The corpus is obtained from the Chinese Text Project. This release does not redistribute a complete corpus copy. Users must obtain the source data and confirm the current access and redistribution terms before recreating occurrence-level outputs. Model weights are also subject to their own distribution terms.

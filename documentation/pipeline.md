# Reproducible Pipeline Specification

This document describes the analysis pipeline used in the paper. The public release contains the scripts and derived results, but not the large embedding files or the complete source corpus.

## Pipeline Overview

The conceptual workflow in the paper is implemented as executable stages in [`../code/`](../code/), orchestrated by [`../code/pipeline.py`](../code/pipeline.py). `pipeline.py` itself is a thin orchestrator — argument parsing and stage sequencing only, no plotting or labeling logic — so it has no dependency on pandas/numpy/matplotlib; each stage script pulls those in on its own when it actually needs them. Shared constants and label-mapping/plotting helpers used by more than one of the labeling/plotting stages live in [`../code/pipeline_common.py`](../code/pipeline_common.py), which is imported, not run directly.

```text
extract -> clean1 -> sample -> clean2 -> local -> global -> label_template -> global_plot -> local_annotate
```

| Stage | Script | Purpose |
|---|---|---|
| 1. Extract | `extract_keyword.py` | Read the local CText archive, keep articles that contain punctuation, split them into sentences, and compute a contextual embedding for every occurrence of each target character. |
| 2. Clean 1 | `clean_ocr_errors.py` (+ `detect_ocr_errors.py`) | Remove sentences with obvious OCR artifacts (marker glyphs, runaway length, excessive character repetition). |
| 3. Sample | `stratified_sampling.py` | Stratify by source document (`toptitle`) and reservoir-sample at most 200 occurrences per document, so no single text dominates a period. |
| 4. Clean 2 | `clean_data.py` | Apply a second, stricter sentence-quality filter (length, character diversity, digit density, abnormal patterns) — deliberately *not* punctuation-based. |
| 5. Local | `local_clustering.py` | Cluster each historical period independently (overcluster → cosine-merge → dynamic split). |
| 6. Global | `global_clustering.py` | Cluster all periods jointly in one shared embedding space (called "Joint Clustering" in some filenames/comments). |
| 6a. Label template | `generate_label_template.py` | Turn the cluster stats stages 5/6 already computed into fill-in-the-blank Markdown + JSON labeling material, for any keyword. Optional to act on — the pipeline runs to completion with or without filled-in labels. |
| 7. Global plot | `global_plot.py` | Attach the reviewed Global semantic label mapping if one is available, and render the paper's Global figures. Falls back to a plain, cluster-ID-colored rendering if no labels are resolved yet. |
| 8. Local annotation | `local_annotate.py` | Owns all Local output. Derives Local labels from the reviewed Global labels by dynasty-specific majority vote, applies an explicit per-`(dynasty, cluster)` override on top if one is supplied, and falls back to a plain rendering if no Global labels are resolved yet. |

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

`pipeline.py` locates every stage script next to itself (`code/`), independent of `--workdir`; `--workdir` (default `.`) only controls where run outputs (`--output-root`, default `final_pipeline_results/`) and label-map JSON paths are resolved. The original source corpus and model must be supplied separately when running the extraction stage: `--extract-metadata` for the CText metadata CSV, and `--extract-book-data-path` / `--extract-wiki-data-pattern` for the local CText archive (these default to the paths the paper was run against, `BOOK_DATA_PATH` / `WIKI_DATA_PATTERN` in `extract_keyword.py`, but can be overridden without editing source).

To stop before labeling (e.g. to generate templates for a brand-new keyword and label it before plotting):

```bash
python3 code/pipeline.py --chars 天 --run-name run_tian \
  --start-stage extract --end-stage label_template
# ... fill in stage_outputs/06a_label_templates/char_天/天_global_label_skeleton.json,
#     save it as configs/天_labels.json ...
python3 code/pipeline.py --chars 天 --run-name run_tian \
  --start-stage global_plot --end-stage local_annotate \
  --label-map-json configs/天_labels.json
```

## Stage-by-Stage Detail

### 1. Extract — `extract_keyword.py`

- `read_ctext_archive()` decompresses the local CText bulk archive (default `.../book/text_book_raw.gz` and `.../wiki/text_wiki_raw_part_*`, overridable per-run with `--book-data-path` / `--wiki-data-pattern` so the archive doesn't have to live at this repo's original path) into an in-memory `{urn: fulltext}` map. **This step loads every article, punctuated or not — there is no filtering yet.**
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

Overclustering + auto-merge + hierarchical refinement, run **independently per dynasty**: MiniBatchKMeans with `--k-init` (20) initial clusters, merge clusters whose centroids exceed `--merge-ratio` (0.95) cosine similarity, then split any cluster larger than `--split-ratio` (0.3) of the period's occurrences into a dynamically chosen number of sub-clusters (2–6). PCA is fit **jointly across all periods** so the eight per-dynasty panels share one coordinate system. Output under `05_local_output/`, including one `{dynasty}_stats.json` per period with per-cluster representative sentences (`cluster_sentences`) that stage 6a reads.

### 6. Global — `global_clustering.py`

Same overcluster → merge → split algorithm, but run **once over all periods pooled together** (`--k` 20, `--merge-ratio` 0.95, `--split-ratio` 0.1, fixed `--sub-k` 6), so cluster identities are directly comparable across dynasties. Output under `06_joint_output/`, including `joint_stats.json` with per-cluster representative sentences and dynasty distribution (`clusters`) that stage 6a reads.

### 6a. Label template — `generate_label_template.py`

Reformats the representative-sentence statistics stages 5/6 already computed — no new numerical computation — into material an LLM or human annotator can act on, for any keyword:

- Always produces `<ch>_global_label_template.md` (one section per Global cluster: size, dynasty distribution, core + boundary representative sentences) and `<ch>_global_label_skeleton.json` (`{"<ch>": {"joint": {"<cid>": ["", ""]}}}`, the exact schema `configs/*.json` already uses).
- If a Global label mapping is already resolvable, also produces `<ch>_local_label_template.md` / `<ch>_local_label_skeleton.json`, grouped by dynasty, with a majority-vote reference line per cluster (see [Annotation](#annotation) below) — **optional**: most keywords never need this half, since Local labels are inherited from Global by default (stage 8).

Output under `06a_label_templates/`. A filled-in skeleton is saved as `configs/<keyword>_labels.json` and passed to `global_plot.py` / `local_annotate.py` via `--label-map-json`.

### 7. Global plot — `global_plot.py`

Reads the Global/Joint assignment CSV, applies the reviewed semantic label mapping from `configs/dao_shui_labels.json` (or the hand-specific 手 label profile) if one is resolvable, and renders the paper's Global figures (trajectory plot, per-dynasty facets, cluster-composition bars). Falls back to a plain, cluster-ID-colored rendering with no labels if none is resolvable, so a first pass on a new keyword always produces something to look at. Output under `07_global_plot/`. Never touches Local — see stage 8.

### 8. Local annotation — `local_annotate.py`

Owns all Local output. See [Annotation](#annotation) for how Local labels are actually derived, including the difference between how 道/水 and 手 were labeled.

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

### Parameter Sensitivity Analysis

The paper (Section 3.2) justifies `K=20` over the record with a one-factor-at-a-time sweep: `K ∈ {10, 15, 20, 25, 30}`, `merge_ratio ∈ {0.90, 0.92, 0.95, 0.97}`, `split_ratio ∈ {0.1, 0.3, 0.5}`, each varied independently against the Global/Local baselines in the table above. The reported numbers — `K=25` reaching a marginally higher Global silhouette (0.054 vs. 0.049, Δ = +0.005) but collapsing to nearly the same final cluster count after merging (16 vs. 17 for `K=20`) — are reproduced in [`../results/sensitivity_analysis/`](../results/sensitivity_analysis/):

- `summary_report.md` — baselines, best configs, and top-10 tables for both Global and Local sweeps.
- `global_results.csv` / `local_results.csv` — one row per tested configuration (silhouette score, final cluster count).
- `global_silhouette.png` / `local_silhouette.png`, `global_k_distribution.png` / `local_k_distribution.png` — the corresponding charts.

These are the released, lightweight artifacts of a larger sweep; the full per-configuration clustering outputs (multiple GB of intermediate CSV/PKL/PNG per run) are not part of the release, consistent with the exclusions listed under "Public Release Boundary" in [`reproduction_checklist.md`](reproduction_checklist.md).

The sweep itself is reproducible: [`../code/parameter_sensitivity_test.py`](../code/parameter_sensitivity_test.py) drives `local_clustering.py`/`global_clustering.py` one-factor-at-a-time over the `K`/`merge_ratio`/`split_ratio` grid above and writes exactly the artifacts listed. Example:

```bash
python3 code/parameter_sensitivity_test.py \
  --input <cleaned_output_dir> \
  --char 手 \
  --output results/sensitivity_analysis
```

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

A live pipeline run additionally produces, under `stage_outputs/`:

```text
stage_outputs/
├── 05_local_output/        # local_clustering.py — per-dynasty CSVs + {dynasty}_stats.json
├── 06_joint_output/         # global_clustering.py — joint_merged.csv + joint_stats.json
├── 06a_label_templates/     # generate_label_template.py — *_template.md + *_skeleton.json
├── 07_global_plot/          # global_plot.py — Global figures + *_Labeled_Global.csv
└── 08_local_annotated/      # local_annotate.py — Local figures + *_Labeled_Local.csv
```

For each character:

- `*_global_assignments.csv` contains one row per occurrence with Global cluster IDs and PCA coordinates.
- `*_local_assignments.csv` contains one row per occurrence with Local cluster IDs and shared PCA coordinates.
- `*_global_stats.json` contains cluster counts, silhouette score, representative examples, and merge/split history.
- `*_stats.json` in `local/` contains dynasty-level cluster statistics.
- `*_global_labeled.csv` and `*_local_labeled.csv` contain `Macro_Category` and `Sub_Category` annotations.
- PNG files contain the raw and semantically annotated visualizations.

## Annotation

Global labels always require a human/LLM-in-the-loop judgment call — there is no way to automate assigning a `Macro_Category`/`Sub_Category` pair to a cluster's representative sentences. `generate_label_template.py` (stage 6a) exists to make that judgment call reproducible in *protocol*, not in exact output: it turns each character's `joint_stats.json` into the same fill-in-the-blank material regardless of keyword, so relabeling (or labeling a new keyword) starts from the same representative-sentence evidence every time. `configs/dao_shui_labels.json` and `configs/hand_labels.json` are the actual filled-in results used for the paper's figures — external, human/LLM-curated input, not something the pipeline generates on its own. They are read via `--label-map-json` by `global_plot.py` (its `"joint"` block) and `local_annotate.py` (both `"joint"` and the optional `"local"` block).

### Local labels: two different methods were actually used

Checking the released `*_local_labeled.csv` files against each character's `joint` mapping shows 道/水 and 手 were labeled differently — this is a real methodological difference in how the paper was produced, not a bug, and both are documented here rather than silently normalized to one method:

- **道 / 水** (added during camera-ready, for efficiency): every `(Macro_Category, Sub_Category)` pair appearing in `dao_local_labeled.csv` / `water_local_labeled.csv` is already present in that character's `joint` mapping — Local labels are **entirely inherited** from Global. `local_annotate.py` derives this automatically: for each `(dynasty, Local_Cluster)`, it joins Local and Global assignments on the sentence level, takes the majority-vote `Global_Cluster` match, and inherits that cluster's label wholesale (see `_derive_local_labels_from_join` in `pipeline_common.py`). The per-`(dynasty, cluster)` mapping reports (`*_label_mapping_report.json`) preserve the full distribution and flag clusters where the majority match is under 60% (`is_mixed`). `configs/dao_shui_labels.json` has no `"local"` block for either character — none is needed.
- **手** (the original, most thoroughly annotated character): Local clusters were **independently re-labeled** per `(dynasty, Local_Cluster)`, not inherited — `hand_local_labeled.csv` contains many `Sub_Category` values with no counterpart in the 17-cluster Global set (e.g. `Limbs`, `Ritual_Hold`, `Combat_Hold`, `Calligraphy`, `Etymology`), and at least one cluster (`qing`, `Body_Medical`/`Bimanual`) even has a different `Macro_Category` than its majority-matched Global cluster (`Physical_Action`/`Bimanual`). `configs/hand_labels.json`'s `"local"` block encodes this exact mapping — 85 unambiguous `(dynasty, Cluster_ID)` entries, mechanically derived from `hand_local_labeled.csv`.

`local_annotate.py` supports both methods through one mechanism: the `"local"` block of `--label-map-json`, when present, is a nested `{dynasty: {cluster_id: [Macro, Sub]}}` override — for any `(dynasty, cluster)` it lists, that value fully replaces the join-derived default (not a `Sub_Category`-only patch, since 手's own data shows `Macro_Category` can legitimately diverge too). Any `(dynasty, cluster)` not listed keeps the join-derived default. No `"local"` block at all (道/水's case) means every cluster falls back to pure inheritance. If no Global labels are resolved at all yet, `local_annotate.py` renders a plain, cluster-ID-colored fallback instead of skipping the character.

**Recommendation for a new keyword:** start with the default (no `"local"` block) — it costs nothing extra and is what 道/水 used. Only run the optional Local half of `generate_label_template.py` and fill in a `"local"` override if you specifically want 手-level dynasty-specific precision.

## Notes for Reproducers

The release originally shipped `pipeline.py` unmodified from the internal working copy, where it called sibling scripts by their pre-release names/locations (`mainfiles/clean_ocr_errors.py`, `mainfiles/stratified_sampling_cleaned.py`, `mainfiles/0212_clean_data.py`, `0219_align_clustering.py`, `0219_joint_cluster.py`) and `clean_ocr_errors.py` imported a `detect_ocr_errors` module that was never copied into the release. Both issues are fixed as of this revision: `pipeline.py` now resolves all stage scripts next to itself in `code/`, and `code/detect_ocr_errors.py` is included. Running `python3 code/pipeline.py` from `paper_release/` per the example above now works end-to-end given a valid CText archive and metadata file.

A later revision split the formerly-inline `postprocess` and `local_annotate` stages out of `pipeline.py` into their own scripts (`global_plot.py`, `local_annotate.py`) plus a shared `pipeline_common.py`, added the `label_template` stage, and corrected this document's earlier claim that Local clusters are never independently annotated (true for 道/水, not true for 手 — see [Annotation](#annotation)). `--start-stage`/`--end-stage postprocess` from older commands should be updated to `global_plot`.

## Source and Reuse Restrictions

The corpus is obtained from the Chinese Text Project. This release does not redistribute a complete corpus copy. Users must obtain the source data and confirm the current access and redistribution terms before recreating occurrence-level outputs. Model weights are also subject to their own distribution terms.

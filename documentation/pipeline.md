# Final Pipeline 完整規格文件

更新日期：2026-04-05

> **這份文件以 `final_pipeline.py` 的程式碼為唯一真相來源。**  
> 所有路徑、參數預設值、輸出檔名均來自程式碼直接核對，非舊版文件推算。

---

## 目錄

1. [快速上手](#1-快速上手)
2. [執行模式](#2-執行模式)
3. [Run 目錄結構](#3-run-目錄結構)
4. [Stage 總覽](#4-stage-總覽)
5. [各 Stage 詳細規格](#5-各-stage-詳細規格)
   - [Stage 1 — extract](#stage-1--extract)
   - [Stage 2 — clean1](#stage-2--clean1)
   - [Stage 3 — sample](#stage-3--sample)
   - [Stage 4 — clean2](#stage-4--clean2)
   - [Stage 5 — local](#stage-5--local)
   - [Stage 6 — global](#stage-6--global)
   - [Stage 7 — postprocess](#stage-7--postprocess)
   - [Stage 8 — local_annotate](#stage-8--local_annotate)
6. [CLI 參數完整列表](#6-cli-參數完整列表)
7. [語意標籤 JSON 格式](#7-語意標籤-json-格式)
8. [新字完整標注流程](#8-新字完整標注流程)
9. [常用執行配方](#9-常用執行配方)
10. [診斷與錯誤排查](#10-診斷與錯誤排查)

---

## 1. 快速上手

```bash
# 工作目錄：lrec-workshop/
cd /mnt/md0/public/dicwn/lrec-workshop

# 完整跑道（extract → local_annotate），使用語意標籤
python3 final_pipeline.py \
  --chars 道,水 \
  --run-name run_dao_shui \
  --start-stage extract \
  --end-stage local_annotate \
  --label-map-json label_templates/dao_shui_labels.json

# 互動模式（無 CLI 參數時自動進入）
python3 final_pipeline.py
```

**最低需求**：`--chars` 為必填，其餘均有預設值。

---

## 2. 執行模式

### 互動模式

```bash
python3 final_pipeline.py
```

當 **不傳任何 CLI 旗標**，或 **未傳 `--chars`** 時，程式進入互動模式，逐一提示輸入各項設定。按 Enter 接受 `[預設值]`。

### CLI 模式

所有設定透過旗標傳入，適合批次腳本或自動化執行：

```bash
python3 final_pipeline.py \
  --chars 手,道,水 \
  --run-name my_run \
  --start-stage extract \
  --end-stage local_annotate
```

### 兩段式跑法（適合新字）

Stage 8 `local_annotate` 需要人工填好語意標籤，因此新字通常分兩段跑：

```bash
# 第一段：extract → postprocess（全自動）
python3 final_pipeline.py \
  --chars 天 \
  --run-name run_tian \
  --start-stage extract \
  --end-stage postprocess

# （人工）審核 global cluster 結果，填入 labels.json 的 "天" joint 區段

# 第二段：只跑 local_annotate
python3 final_pipeline.py \
  --chars 天 \
  --run-name run_tian \
  --start-stage local_annotate \
  --end-stage local_annotate \
  --label-map-json label_templates/labels.json
```

### 斷點續跑

使用 `--start-stage` 跳到特定 stage，搭配 `--skip-existing` 略過已存在的輸出：

```bash
python3 final_pipeline.py \
  --chars 道,水 \
  --run-name run_dao_shui \
  --start-stage postprocess \
  --end-stage postprocess \
  --label-map-json label_templates/dao_shui_labels.json
```

---

## 3. Run 目錄結構

每次執行會在 `{workdir}/{output-root}/{run-name}/` 下建立以下結構：

```
final_pipeline_results/{run_name}/
├── config.json                  ← 本次 run 的所有參數快照
├── summary.json                 ← 所有 stage 的執行結果（成功/失敗/耗時）
├── logs/
│   ├── extract.log
│   ├── clean1.log
│   ├── sample.log
│   ├── clean2.log
│   ├── local.log
│   ├── global.log
│   ├── postprocess.log
│   └── local_annotate.log       ← Stage 8 專屬 log
└── stage_outputs/
    ├── 01_keyword_embeddings/   ← Stage 1 輸出
    ├── 02_cleaned_embeddings/   ← Stage 2 輸出
    ├── 03_stratified_sampling_output_cleaned/  ← Stage 3 輸出
    ├── 04_cleaned_output/       ← Stage 4 輸出
    ├── 05_local_output/         ← Stage 5 輸出
    ├── 06_joint_output/         ← Stage 6 輸出
    ├── 07_postprocess/          ← Stage 7 輸出（Global 語意圖）
    └── 08_local_annotated/      ← Stage 8 輸出（Local 語意標注圖）
```

- `config.json`：執行當下所有 `--args` 的 JSON 快照，可重現執行條件。
- `config.json` 的 `data` 區段會記錄 metadata、抽取、兩輪清理與抽樣參數；`baseline` 區段記錄 Local/Global 分群參數。
- `summary.json`：每個 stage 的 `{success, elapsed_sec, command, log_file, message}`。
- **Run 名稱未指定時**：自動產生 `run_{YYYYMMDD_HHMMSS}`。

---

## 4. Stage 總覽

| # | Stage ID          | 腳本                                       | 主要任務                                 | 輸出目錄                    |
|---|-------------------|--------------------------------------------|------------------------------------------|-----------------------------|
| 1 | `extract`         | `extract_keyword.py`                       | 抽取目標字 embedding                     | `01_keyword_embeddings/`    |
| 2 | `clean1`          | `mainfiles/clean_ocr_errors.py`            | 第一輪 OCR 清理（jsonl 層）              | `02_cleaned_embeddings/`    |
| 3 | `sample`          | `mainfiles/stratified_sampling_cleaned.py` | 分層抽樣，轉為 `.pkl`                   | `03_stratified_sampling_output_cleaned/` |
| 4 | `clean2`          | `mainfiles/0212_clean_data.py`             | 第二輪多層規則清理（pkl 層）             | `04_cleaned_output/`        |
| 5 | `local`           | `0219_align_clustering.py`                 | Local（分朝代）分群                      | `05_local_output/`          |
| 6 | `global`          | `0219_joint_cluster.py`                    | Joint（全朝代混合）分群                  | `06_joint_output/`          |
| 7 | `postprocess`     | *(內建)*                                   | 貼語意標籤 + 產出 Global 最終圖          | `07_postprocess/`           |
| 8 | `local_annotate`  | *(內建)*                                   | Join local+global → 多數決推導 Local 標籤 + 出圖 | `08_local_annotated/` |

**Stage 執行順序**（`STAGE_ORDER`）：
```
extract → clean1 → sample → clean2 → local → global → postprocess → local_annotate
```

`--start-stage` 和 `--end-stage` 可指定任意子區間，例如只跑 `local_annotate`。

---

## 5. 各 Stage 詳細規格

---

### Stage 1 — extract

| 項目 | 內容 |
|------|------|
| **腳本** | `extract_keyword.py` |
| **輸入** | `compact_new_metadata.csv`（預設；可由 `--extract-metadata` 指定） |
| **輸出根** | `stage_outputs/01_keyword_embeddings/` |

**輸出檔案結構**：
```
01_keyword_embeddings/
├── _extract_done.marker          ← 完成標記；--skip-existing 時檢查此檔
└── {dynasty}/
    └── {char}_embeddings.jsonl   ← 每個字、每個朝代一份 jsonl
```

**傳給腳本的旗標**：
```
extract_keyword.py
  --output   <01_keyword_embeddings/>
  --metadata <compact_new_metadata.csv>
  --chars    <char1> <char2> ...
  --device   <auto|cpu|cuda>
  [--dynasties <dyn1> <dyn2> ...]  ← 僅在 --dynasties 有值時加入
  [--max-texts <N>]                ← 僅在 --extract-max-texts > 0 時加入
  [--resume]                       ← 僅在 --extract-resume 時加入
```

**關鍵參數**：

| Pipeline 旗標 | 預設 | 說明 |
|--------------|------|------|
| `--extract-metadata` | `compact_new_metadata.csv` | 來源 metadata CSV |
| `--extract-device` | `auto` | `auto` / `cpu` / `cuda` |
| `--extract-max-texts` | `0`（不限） | 每朝代最多抽幾筆 text（0 = 不限） |
| `--extract-resume` | `false` | 是否續跑已有部分輸出 |
| `--dynasties` | `""` | 指定朝代（逗號或空白分隔），空白 = 全部 |

**Log**：`logs/extract.log`

---

### Stage 2 — clean1

| 項目 | 內容 |
|------|------|
| **腳本** | `mainfiles/clean_ocr_errors.py`（每個字獨立執行一次） |
| **輸入** | `stage_outputs/01_keyword_embeddings/` |
| **輸出根** | `stage_outputs/02_cleaned_embeddings/` |

**輸出檔案結構**：
```
02_cleaned_embeddings/
└── {dynasty}/
    └── {char}_embeddings.jsonl   ← 移除明顯 OCR 錯誤後的 jsonl
```

**傳給腳本的旗標**（每個字迴圈一次）：
```
mainfiles/clean_ocr_errors.py
  --char       <ch>
  --mode       <conservative|aggressive|custom>
  --input-dir  <01_keyword_embeddings/>
  --output-dir <02_cleaned_embeddings/>
```

**關鍵參數**：

| Pipeline 旗標 | 預設 | 說明 |
|--------------|------|------|
| `--clean1-mode` | `conservative` | `conservative` / `aggressive` / `custom` |

**Log**：`logs/clean1.log`（各字的輸出合併於同一個檔案，以 `### char={ch}` 區隔）

---

### Stage 3 — sample

| 項目 | 內容 |
|------|------|
| **腳本** | `mainfiles/stratified_sampling_cleaned.py` |
| **輸入** | `stage_outputs/02_cleaned_embeddings/` |
| **輸出根** | `stage_outputs/03_stratified_sampling_output_cleaned/` |

**輸出檔案結構**：
```
03_stratified_sampling_output_cleaned/
└── char_{ch}/
    ├── pre-qin.pkl
    ├── qinhan.pkl
    ├── weijin.pkl
    ├── suitang.pkl
    ├── songyuan.pkl
    ├── ming.pkl
    ├── qing.pkl
    ├── republican.pkl
    ├── {dynasty}_stats.json      ← 各朝代抽樣統計
    └── overall_stats.json        ← 整體抽樣統計
```

**傳給腳本的旗標**：
```
mainfiles/stratified_sampling_cleaned.py
  --chars               <char1,char2,...>
  --max-per-toptitle    <N>
  --baseline-max-samples <N>
  --target-chars        <char1,char2,...>   ← 與 --chars 相同
  --cleaned-root        <02_cleaned_embeddings/>
  --output              <03_stratified_sampling_output_cleaned/>
```

**關鍵參數**：

| Pipeline 旗標 | 預設 | 說明 |
|--------------|------|------|
| `--sample-max-per-toptitle` | `200` | 每個 toptitle 最多保留幾筆 |
| `--sample-baseline-max` | `10000` | baseline char 上限（target-only 流程通常不影響） |

**Log**：`logs/sample.log`

---

### Stage 4 — clean2

| 項目 | 內容 |
|------|------|
| **腳本** | `mainfiles/0212_clean_data.py`（每個字獨立執行一次） |
| **輸入** | `stage_outputs/03_stratified_sampling_output_cleaned/` |
| **輸出根** | `stage_outputs/04_cleaned_output/` |

**輸出檔案結構**：
```
04_cleaned_output/
└── char_{ch}/
    ├── pre-qin.pkl
    ├── qinhan.pkl
    ├── weijin.pkl
    ├── suitang.pkl
    ├── songyuan.pkl
    ├── ming.pkl
    ├── qing.pkl
    ├── republican.pkl
    └── cleaning_summary.json    ← 清理統計（移除筆數、原因等）
```

**傳給腳本的旗標**（每個字迴圈一次）：
```
mainfiles/0212_clean_data.py
  --char        <ch>
  --input       <03_stratified_sampling_output_cleaned/>
  --output      <04_cleaned_output/>
  --aggression  <mild|normal|aggressive|ultra>
```

**關鍵參數**：

| Pipeline 旗標 | 預設 | 說明 |
|--------------|------|------|
| `--clean2-aggression` | `normal` | `mild` / `normal` / `aggressive` / `ultra` |

**Log**：`logs/clean2.log`（各字合併，以 `### char={ch}` 區隔）

---

### Stage 5 — local

| 項目 | 內容 |
|------|------|
| **腳本** | `0219_align_clustering.py` |
| **輸入** | `stage_outputs/04_cleaned_output/` |
| **輸出根** | `stage_outputs/05_local_output/` |

**輸出目錄命名規則**：
```
05_local_output/
└── char_{ch}_Local_k{local_k}_split{local_split}_DynamicSubK/
    ├── {ch}_Local_Dynamic_Merged.csv   ← 主要輸出（Stage 8 讀取此檔）
    ├── merged_stats.json               ← per-dynasty silhouette scores（Stage 8 讀取）
    ├── *_stats.json                    ← 各 sub-cluster 統計
    ├── *_viz.png                       ← 各朝代分群視覺化
    └── facet_grid_local_dynamic.png    ← 所有朝代 facet 總覽圖
```

**命名範例**（預設參數）：
```
char_道_Local_k20_split0.3_DynamicSubK/
```

**傳給腳本的旗標**：
```
0219_align_clustering.py
  --chars       <char1,char2,...>
  --input       <04_cleaned_output/>
  --output      <05_local_output/>
  --k-init      <local_k>
  --merge-ratio <local_merge>
  --split-ratio <local_split>
```

**關鍵參數**：

| Pipeline 旗標 | 預設 | 說明 |
|--------------|------|------|
| `--local-k` | `20` | 初始 cluster 數 |
| `--local-merge` | `0.95` | merge 閾值（cosine similarity） |
| `--local-split` | `0.3` | split 閾值（size ratio） |

**Log**：`logs/local.log`

---

### Stage 6 — global

| 項目 | 內容 |
|------|------|
| **腳本** | `0219_joint_cluster.py`（每個字獨立執行一次） |
| **輸入** | `stage_outputs/04_cleaned_output/` |
| **輸出根** | `stage_outputs/06_joint_output/` |

**輸出目錄命名規則**：
```
06_joint_output/
└── char_{ch}_k{global_k}_subk{global_sub_k}_split{global_split}/
    ├── {ch}_joint_merged.csv          ← 主要輸出（Stage 8 讀取此檔）
    ├── joint_stats.json               ← 包含 silhouette score
    ├── joint_clusters_viz.png         ← 全資料分群 scatter
    ├── joint_facet_by_dynasty.png     ← 朝代 facet
    └── joint_cluster_composition.png  ← cluster 組成統計
```

**命名範例**（預設參數）：
```
char_道_k20_subk6_split0.1/
```

**傳給腳本的旗標**（每個字迴圈一次）：
```
0219_joint_cluster.py
  --input       <04_cleaned_output/>
  --char        <ch>
  --output      <06_joint_output/>
  --k           <global_k>
  --merge-ratio <global_merge>
  --sub-k       <global_sub_k>
  --split-ratio <global_split>
```

**關鍵參數**：

| Pipeline 旗標 | 預設 | 說明 |
|--------------|------|------|
| `--global-k` | `20` | 初始 cluster 數 |
| `--global-merge` | `0.95` | merge 閾值 |
| `--global-split` | `0.1` | split 閾值 |
| `--global-sub-k` | `6` | sub-cluster 數 |

**Log**：`logs/global.log`（各字合併，以 `### char={ch}` 區隔）

---

### Stage 7 — postprocess

| 項目 | 內容 |
|------|------|
| **腳本** | *(pipeline 內建，不呼叫外部腳本)* |
| **輸入** | `05_local_output/` 和 `06_joint_output/` |
| **輸出根** | `07_postprocess/local/` 和 `07_postprocess/global/` |

這個 stage 完全在 pipeline 內部用 pandas + matplotlib 完成，分為 Local 和 Joint 兩條路徑。

#### 行為決策樹

```
有傳 --label-map-json 且 JSON 中有此字的 mapping？
├─ YES → 套用語意標籤 → 輸出 0222 風格富圖（比例條、repel 標籤、彩色 macro）
└─ NO  → 無語意標籤（僅 "Unlabeled" + cluster ID） → 輸出 seaborn 簡圖
```

對 `手` 字額外有 fallback：
```
--label-map-json 無此字 AND ch == 手
└─ 使用 --hand-label-map-json 或 --hand-label-version 的內建標籤
```

> **注意**：Stage 7 的 local 路徑使用 JSON 的 `local` 區段（flat mapping，所有朝代共用 cluster ID）。這對 local clustering **並不正確**（因為各朝代 cluster ID 獨立產生），因此 local 的語意標注應交由 Stage 8 `local_annotate` 處理。

---

#### Local 後處理輸出

**路徑**：`07_postprocess/local/char_{ch}/`

| 情境 | 輸出檔案 |
|------|----------|
| **有語意標籤** | `{ch}_Labeled_Local.csv` |
| | `Final_Semantic_Evolution_Local_v2.png`（2×4 facet，各朝代彩色 + 比例條） |
| | `local_clusters_labeled_viz_v2.png`（全資料 scatter + cluster 標籤） |
| | `per_dynasties_v2/{dynasty}_Local_v2.png`（單朝代全地形圖） |
| **無語意標籤** | `{ch}_Labeled_Local.csv`（`Macro_Category=Unlabeled`） |
| | `Final_Semantic_Evolution_Local_v2.png` |
| | `per_dynasties/{dynasty}.png` |

---

#### Joint/Global 後處理輸出

**路徑**：`07_postprocess/global/char_{ch}/`

| 情境 | 輸出檔案 |
|------|----------|
| **有語意標籤** | `{ch}_Labeled_Global.csv` |
| | `Final_Semantic_Evolution_Global_v2.png`（2×4 facet，各朝代彩色 + 比例條） |
| | `joint_clusters_labeled_viz_v2.png`（全資料 scatter + cluster 標籤） |
| | `Trajectory_Global.png`（朝代質心軌跡圖） |
| | `per_dynasties_global_v2/{dynasty}_Global_v2.png`（單朝代全地形圖） |
| **無語意標籤** | `{ch}_Labeled_Global.csv`（`Macro_Category=Unlabeled`） |
| | `Final_Semantic_Evolution_Global_v2.png` |
| | `Trajectory_Global.png` |
| | `per_dynasties_global/{dynasty}.png` |

---

#### CSV 欄位說明（Stage 7 輸出）

**Local**（來自 `{ch}_Local_Dynamic_Merged.csv` 加欄）：

| 欄位 | 說明 |
|------|------|
| `dynasty` | 朝代 key（`pre-qin` / `qinhan` / ... / `republican`） |
| `sentence` | 原始語境句 |
| `title` | 來源典籍 |
| `Local_Cluster` | 分群編號（整數） |
| `PC1`, `PC2` | PCA 降維座標 |
| `Cluster_ID` | 同 Local_Cluster（後處理加入） |
| `Macro_Category` | 語意大類（由 JSON 標籤 mapping 決定） |
| `Sub_Category` | 語意小類 |
| `Label` | 同 Sub_Category |

**Joint**（來自 `{ch}_joint_merged.csv` 加欄）：

| 欄位 | 說明 |
|------|------|
| `dynasty` | 朝代 |
| `dynasty_zh` | 朝代中文名 |
| `sentence` | 原始語境句 |
| `title` | 來源典籍 |
| `Global_Cluster` | 分群編號 |
| `PC1`, `PC2` | PCA 降維座標 |
| `Cluster_ID` | 同 Global_Cluster（後處理加入） |
| `Macro_Category` | 語意大類 |
| `Sub_Category` | 語意小類 |
| `Label` | 同 Sub_Category |

**Log**：`logs/postprocess.log`

---

### Stage 8 — local_annotate

| 項目 | 內容 |
|------|------|
| **腳本** | *(pipeline 內建，不呼叫外部腳本)* |
| **輸入** | `05_local_output/`（local CSV）、`06_joint_output/`（global CSV）、`--label-map-json`（`joint` 區段） |
| **輸出根** | `stage_outputs/08_local_annotated/` |

#### 為什麼需要 Stage 8？

Pipeline Stage 7 的 local 路徑使用 flat mapping（`cluster_id → label`），但 local clustering 的 cluster ID **各朝代獨立**（`pre-qin` 的 cluster 0 和 `qinhan` 的 cluster 0 代表不同語意），因此 flat mapping 無法正確標注。

Stage 8 改用 **sentence-level join** 解決此問題：

```
local CSV ──┐
             ├─ JOIN on (dynasty, sentence)
global CSV ─┘
     ↓
對每個 (dynasty, Local_Cluster_ID)：
  統計所有句子對應哪些 Global_Cluster
  → 多數決（取佔比最高的 Global_Cluster）
  → 繼承對應的已審核 global 語意標籤
     ↓
輸出標注結果 + 圖片
```

**Global 語意標籤來源**：`--label-map-json` 的 `joint` 區段，經過 Gemini → Claude → 人工審核三層驗證後填入。

#### 輸入 CSV 路徑（由 pipeline 參數自動構建）

| 檔案 | 路徑（相對於 `stage_outputs/`） |
|------|-------------------------------|
| Local CSV | `05_local_output/char_{ch}_Local_k{local_k}_split{local_split}_DynamicSubK/{ch}_Local_Dynamic_Merged.csv` |
| Global CSV | `06_joint_output/char_{ch}_k{global_k}_subk{global_sub_k}_split{global_split}/{ch}_joint_merged.csv` |

路徑由執行時的 `--local-k`、`--local-split`、`--global-k`、`--global-sub-k`、`--global-split` 決定，**必須與 Stage 5/6 時使用的參數相同**。

#### 輸出

```
08_local_annotated/
├── char_{ch}/
│   ├── {ch}_Labeled_Local.csv               ← 每句附上 Macro/Sub_Category
│   ├── {ch}_label_mapping_report.json        ← 多數決詳細報告（含 Global_Cluster 分布比例）
│   ├── {ch}_Facet_Grid_Local.png             ← 8 朝代合一 Facet Grid（2×4，獨立座標軸）
│   └── per_dynasties/
│       ├── pre-qin_Labeled.png
│       ├── qinhan_Labeled.png
│       ├── weijin_Labeled.png
│       ├── suitang_Labeled.png
│       ├── songyuan_Labeled.png
│       ├── ming_Labeled.png
│       ├── qing_Labeled.png
│       └── republican_Labeled.png
```

合計（每個字）：**1 個 Labeled CSV、1 個 JSON 報告、1 張 Facet Grid、8 張單朝代圖**。

#### 圖片風格特點

| 特點 | 說明 |
|------|------|
| 獨立座標軸 | `sharex=False, sharey=False`，各朝代自己的 PC1/PC2 範圍 |
| 背景灰點 | 當朝代所有資料點以灰色低透明度顯示 |
| 著色方式 | 按 Macro_Category 著色（動態 palette，由 `build_color_palette_for_mapping()` 生成） |
| 標籤排版 | 使用 `adjustText` 自動避免 cluster label 重疊 |
| Silhouette | 每個朝代左下角顯示 silhouette score（從 `merged_stats.json` 動態讀取） |
| 比例條 | 右上角 stacked color bar 顯示各 Macro_Category 佔比 |

#### label_mapping_report.json 格式

每個 `(dynasty, Local_Cluster)` 的多數決細節：

```json
"pre-qin_L3": {
  "dynasty": "pre-qin",
  "local_cluster": 3,
  "total_sentences": 478,
  "assigned_label": ["Philosophy", "Political_Ethics"],
  "majority_global": 11,
  "majority_pct": 72.4,
  "is_mixed": false,
  "distribution": [
    {"global_cluster": 11, "label": ["Philosophy", "Political_Ethics"], "count": 346, "pct": 72.4},
    {"global_cluster": 10, "label": ["Philosophy", "Moral_Principle"],  "count": 132, "pct": 27.6}
  ]
}
```

- `is_mixed: true`：多數佔比 < 60%，Stage 8 log 中會列出所有 mixed cluster 清單，建議人工複查。

#### CSV 欄位說明（Stage 8 輸出）

| 欄位 | 說明 |
|------|------|
| `dynasty` | 朝代 key |
| `sentence` | 原始語境句 |
| `title` | 來源典籍 |
| `Local_Cluster` | Local 分群編號（整數） |
| `PC1`, `PC2` | PCA 降維座標（與 local CSV 相同） |
| `Cluster_ID` | 同 Local_Cluster（Stage 8 加入） |
| `Macro_Category` | 語意大類（由 global 標籤多數決繼承） |
| `Sub_Category` | 語意小類 |

**Log**：`logs/local_annotate.log`（含每個字的 join 統計、mixed cluster 清單）

**前提條件**：`--label-map-json` 的 `joint` 區段必須已填入最終審核標籤，否則該字會被 SKIP（不報錯，繼續處理下一個字）。

---

## 6. CLI 參數完整列表

```
python3 final_pipeline.py [options]
```

### 基礎設定

| 旗標 | 預設 | 說明 |
|------|------|------|
| `--chars` | `""` | **必填（CLI 模式）**。逗號分隔的目標字，例如 `手,道,水` |
| `--run-name` | 自動產生 `run_{YYYYMMDD_HHMMSS}` | Run 資料夾名稱 |
| `--workdir` | `.`（當前目錄） | lrec-workshop 根目錄的路徑 |
| `--output-root` | `final_pipeline_results` | Run 根目錄，相對於 `--workdir` |
| `--python-bin` | `sys.executable` | Python 執行檔路徑 |
| `--skip-existing` | `false` | 若 stage 輸出已存在則跳過（目前僅 extract 支援） |

### Stage 控制

| 旗標 | 預設 | 說明 |
|------|------|------|
| `--start-stage` | `extract` | 起始 stage（含）。選項：`extract clean1 sample clean2 local global postprocess local_annotate` |
| `--end-stage` | `local_annotate` | 結束 stage（含）。同上選項 |

> **注意**：預設 `--end-stage` 是 `local_annotate`（完整跑完）。若只想跑到分群，需明確設 `--end-stage global`。

### Stage 1 extract 參數

| 旗標 | 預設 | 說明 |
|------|------|------|
| `--extract-metadata` | `compact_new_metadata.csv` | 來源 metadata 檔案路徑 |
| `--extract-device` | `auto` | `auto` / `cpu` / `cuda` |
| `--extract-max-texts` | `0` | 每朝代最多幾筆（0 = 不限） |
| `--extract-resume` | `false` | 續跑現有部分輸出 |
| `--dynasties` | `""` | 限制抽取朝代，空白 = 全部 |

### Stage 2 clean1 參數

| 旗標 | 預設 | 選項 |
|------|------|------|
| `--clean1-mode` | `conservative` | `conservative` / `aggressive` / `custom` |

### Stage 3 sample 參數

| 旗標 | 預設 | 說明 |
|------|------|------|
| `--sample-max-per-toptitle` | `200` | 每個 toptitle 最多保留幾筆 |
| `--sample-baseline-max` | `10000` | baseline char 上限 |

### Stage 4 clean2 參數

| 旗標 | 預設 | 選項 |
|------|------|------|
| `--clean2-aggression` | `normal` | `mild` / `normal` / `aggressive` / `ultra` |

### Stage 5 local 參數

| 旗標 | 預設 | 說明 |
|------|------|------|
| `--local-k` | `20` | 初始 cluster 數 |
| `--local-merge` | `0.95` | merge 閾值 |
| `--local-split` | `0.3` | split 閾值 |

### Stage 6 global 參數

| 旗標 | 預設 | 說明 |
|------|------|------|
| `--global-k` | `20` | 初始 cluster 數 |
| `--global-merge` | `0.95` | merge 閾值 |
| `--global-split` | `0.1` | split 閾值 |
| `--global-sub-k` | `6` | sub-cluster 數 |

### Stage 7/8 語意標籤參數

| 旗標 | 預設 | 說明 |
|------|------|------|
| `--label-map-json` | `""` | **通用多字語意標籤 JSON**。Stage 7 讀 `joint` 區段做 global 圖；Stage 8 讀 `joint` 區段推導 local 標籤。路徑相對於 `--workdir` |
| `--hand-label-map-json` | `""` | 專用於 `手` 字的標籤 JSON（優先於 `--hand-label-version`） |
| `--hand-label-version` | `auto` | `手` 字內建標籤版本：`auto`（同 0222） / `0221` / `0222` |

---

## 7. 語意標籤 JSON 格式

標籤 JSON 透過 `--label-map-json` 指定，路徑相對於 `--workdir`。

### 格式 A：Scoped（推薦，local 和 joint 分開設定）

```json
{
  "道": {
    "local": {
      "0": ["Macro_Category_名稱", "Sub_Category_名稱"],
      "1": ["Macro_Category_名稱", "Sub_Category_名稱"]
    },
    "joint": {
      "0": ["Macro_Category_名稱", "Sub_Category_名稱"],
      "1": ["Macro_Category_名稱", "Sub_Category_名稱"]
    }
  },
  "水": {
    "local": { "...": "..." },
    "joint": { "...": "..." }
  }
}
```

### 格式 B：Flat（local 和 joint 共用同一份 mapping）

```json
{
  "道": {
    "0": ["Macro_Category_名稱", "Sub_Category_名稱"],
    "1": ["Macro_Category_名稱", "Sub_Category_名稱"]
  },
  "水": {
    "0": ["Macro_Category_名稱", "Sub_Category_名稱"]
  }
}
```

### 格式 C：單字簡化格式（只跑一個字時可用）

```json
{
  "0": ["Macro_Category_名稱", "Sub_Category_名稱"],
  "1": ["Macro_Category_名稱", "Sub_Category_名稱"]
}
```

---

### 規則與注意事項

1. **key 是字串格式的整數**（`"0"`, `"1"` …），對應 cluster 編號。
2. **value 是長度恰好為 2 的陣列**：`[Macro_Category, Sub_Category]`。
3. `Macro_Category` 決定顏色。**已知標準 macro**（對應 HAND_COLOR_PALETTE）：
   - `Body_Medical` — 藍 `#377eb8`
   - `Physical_Action` — 紅 `#e41a1c`
   - `Social_Interaction` — 橙 `#ff7f00`
   - `Text_Culture` — 綠 `#4daf4a`
   - `Power_Skill` — 紫 `#984ea3`
   - `Grammar_Suffix` — 棕 `#a65628`
   - `NOISE` — 灰 `#cccccc`（在比例條和標籤中自動抑制顯示）
   - **其他自訂 macro**：pipeline 自動從 15 色擴充調色板中配色。
4. **cluster 編號若未在 JSON 中列出**：fallback 為 `("Other", "Other")`。
5. **Stage 8 只讀 `joint` 區段**，`local` 區段可保留為 `"Unlabeled"` 佔位符（Stage 7 用）。
6. 範本檔：[label_templates/dao_shui_labels.json](label_templates/dao_shui_labels.json)（已含 `道` 和 `水` 的 joint 正確標籤）

---

### 目前各字的 Cluster 數量

| 字 | Local clusters | Joint clusters | Joint 範本 key |
|----|---------------|---------------|----------------|
| 道 | 19（0–18）    | 22（0–21）    | `"0"`–`"21"` |
| 水 | 15（0–14）    | 12（0–11）    | `"0"`–`"11"` |

> 以上為 baseline 參數（k=20）的結果；調整超參數後 cluster 數會改變，需相應更新 JSON。

---

## 8. 新字完整標注流程

給定一個新關鍵字（例如 `天`），完整流程如下：

### Step 1：跑完分群

```bash
python3 final_pipeline.py \
  --chars 天 \
  --run-name run_tian \
  --start-stage extract \
  --end-stage postprocess
```

此步驟產出（`07_postprocess/global/char_天/`）：
- 未標注的 global cluster 圖（seaborn 簡圖，cluster ID 著色）
- `天_Labeled_Global.csv`（Macro_Category = Unlabeled）

### Step 2：準備語意標注 JSON

在 `label_templates/labels.json` 加入新字的 `joint` 區段：

```json
{
  "天": {
    "local": {},
    "joint": {
      "0": ["Philosophy", "Heaven_Cosmos"],
      "1": ["Religion",   "Deity"],
      "2": ["NOISE",      "OCR_Error"],
      ...
    }
  }
}
```

標注依據（三層審核流程）：
1. **Gemini 2.5 Flash** 初步標注（根據 `joint_stats.json` 的 core sentences）
2. **Claude Sonnet** 審核
3. **人工** 最終確認

### Step 3：跑 local_annotate

```bash
python3 final_pipeline.py \
  --chars 天 \
  --run-name run_tian \
  --start-stage local_annotate \
  --end-stage local_annotate \
  --label-map-json label_templates/labels.json
```

輸出在 `final_pipeline_results/run_tian/stage_outputs/08_local_annotated/char_天/`。

---

## 9. 常用執行配方

### 道、水全流程（標籤已審核）

```bash
python3 final_pipeline.py \
  --chars 道,水 \
  --run-name run_dao_shui \
  --start-stage extract \
  --end-stage local_annotate \
  --label-map-json label_templates/dao_shui_labels.json
```

### 只重跑 postprocess（語意標籤更新後）

```bash
python3 final_pipeline.py \
  --chars 道,水 \
  --run-name run_dao_shui \
  --start-stage postprocess \
  --end-stage postprocess \
  --label-map-json label_templates/dao_shui_labels.json
```

### 只重跑 local_annotate（標籤更新後）

```bash
python3 final_pipeline.py \
  --chars 道,水 \
  --run-name run_dao_shui \
  --start-stage local_annotate \
  --end-stage local_annotate \
  --label-map-json label_templates/dao_shui_labels.json
```

### 從分群重跑到最終圖（不重抽 embedding）

```bash
python3 final_pipeline.py \
  --chars 道,水 \
  --run-name run_dao_shui_recluster \
  --start-stage local \
  --end-stage local_annotate \
  --label-map-json label_templates/dao_shui_labels.json
```

### 手字完整流程（使用內建語意標籤）

```bash
python3 final_pipeline.py \
  --chars 手 \
  --run-name run_hand \
  --start-stage extract \
  --end-stage postprocess \
  --hand-label-version 0222
```

> `手` 字目前沒有 `local_annotate` 支援（`dao_shui_labels.json` 中無 `手` 的 joint 標籤），設 `--end-stage postprocess` 即可。

### 多字同跑（含手）

```bash
python3 final_pipeline.py \
  --chars 手,道,水 \
  --run-name run_all_three \
  --start-stage extract \
  --end-stage local_annotate \
  --hand-label-version 0222 \
  --label-map-json label_templates/dao_shui_labels.json
```

> `--label-map-json` 只套用到 JSON 中有列出的字；`手` 走 `--hand-label-version` fallback。`local_annotate` 只處理 JSON 中有 `joint` 標籤的字，`手` 會被自動 SKIP。

### 調整分群超參數

```bash
python3 final_pipeline.py \
  --chars 道 \
  --run-name run_dao_k25 \
  --start-stage local \
  --end-stage global \
  --local-k 25 \
  --local-split 0.25 \
  --global-k 25 \
  --global-sub-k 8 \
  --global-split 0.15
```

> **注意**：調整超參數後 cluster 數會改變，需先跑到 `global`，確認新的 cluster 編號後更新 JSON，再重跑 `postprocess` 和 `local_annotate`。

### 限制朝代（快速測試）

```bash
python3 final_pipeline.py \
  --chars 道 \
  --run-name run_dao_test \
  --dynasties pre-qin,qinhan \
  --start-stage extract \
  --end-stage postprocess
```

---

## 10. 診斷與錯誤排查

### 確認 stage 是否成功

```bash
cat final_pipeline_results/{run_name}/summary.json | python3 -m json.tool
```

每個 stage 的 `success` 欄位應為 `true`，`message` 應為 `"ok"`。

### 查看 stage log

```bash
cat final_pipeline_results/{run_name}/logs/postprocess.log
cat final_pipeline_results/{run_name}/logs/local_annotate.log
cat final_pipeline_results/{run_name}/logs/global.log
```

### postprocess 失敗常見原因

| 錯誤訊息 | 原因 | 解法 |
|----------|------|------|
| `local csv missing: ...` | Stage 5 未執行或路徑參數與 stage 5 不一致 | 確認 `--local-k` / `--local-split` 與執行 stage 5 時相同 |
| `global csv missing: ...` | Stage 6 未執行或路徑參數不一致 | 同上，確認 `--global-k` / `--global-sub-k` / `--global-split` |
| `No cluster column found` | CSV 欄位異常 | 直接開 CSV 確認是否有 `Local_Cluster` 或 `Global_Cluster` |
| `Invalid mapping at ...` | JSON 格式錯誤 | 確認每個 value 都是 `["Macro", "Sub"]` 格式 |
| `label mapping json not found` | JSON 路徑錯誤 | 路徑相對於 `--workdir`，預設 `.`（lrec-workshop/） |

### local_annotate 常見狀況

| Log 訊息 | 意義 | 處理 |
|----------|------|------|
| `SKIP local_annotate: no joint labels` | 此字的 `joint` 區段未填或 JSON 未傳 | 填好 JSON 並確認 `--label-map-json` 路徑正確 |
| `local CSV missing` | Stage 5 輸出找不到 | 確認參數一致，或重跑 Stage 5 |
| `global CSV missing` | Stage 6 輸出找不到 | 確認參數一致，或重跑 Stage 6 |
| `join unmatched=N` | N 個句子在 global CSV 中找不到匹配 | 少量可忽略；大量表示 local/global 是不同版本資料 |
| `mixed clusters (N): [...]` | N 個 local cluster 多數 < 60%，標籤可信度較低 | 建議人工核對 `label_mapping_report.json` 中列出的 cluster |

### 確認 local_annotate 輸出

```bash
ls final_pipeline_results/{run_name}/stage_outputs/08_local_annotated/char_道/
# 應包含：
#   道_Labeled_Local.csv
#   道_label_mapping_report.json
#   道_Facet_Grid_Local.png
#   per_dynasties/          ← 8 張單朝代圖
```

### 重新確認目前 run 的超參數

```bash
cat final_pipeline_results/{run_name}/config.json
```

輸出範例：
```json
{
  "run_name": "run_dao_shui",
  "chars": ["道", "水"],
  "start_stage": "extract",
  "end_stage": "local_annotate",
  "baseline": {
    "local":  {"k": 20, "merge": 0.95, "split": 0.3},
    "global": {"k": 20, "merge": 0.95, "split": 0.1, "sub_k": 6}
  },
  "semantic": {
    "hand_label_version": "auto",
    "hand_label_map_json": "",
    "label_map_json": "label_templates/dao_shui_labels.json"
  }
}
```

# 0221 最終結果 Pipeline 與可重跑清單

> 更新日期：2026-03-13
> 適用範圍：`lrec-workshop` 目前的 0221 輸出（`0221_progress/`）

---

## 1. 一句話結論

- `0219_align_clustering.py`：做 **Local（分朝代）分群**，並輸出 local 結果與基礎圖。
- `0219_joint_cluster.py`：做 **Joint（全朝代一起）分群**，並輸出 joint 結果與基礎圖。
- `0221_progress/*/0221_label_and_plot.py`、`0222_label_moreinfo.py`、`plot_trajectory.py`：屬於 **後處理標註與最終圖像輸出**，不是重新分群。

---

## 2. 完整 Pipeline（資料來源到 0221 最終圖）

## 2.1 上游資料準備（Embedding 與清理）

1. 原始語料與 metadata 先產生 keyword embeddings。
2. OCR 清理 + 分層抽樣 + 再清理，最後得到可分群的 `.pkl`。
3. 分群程式實際吃的就是各朝代 `pkl` 內的 `record['vector']`。

常見可用輸入來源（需包含下列結構）：

- `.../char_手/pre-qin.pkl`
- `.../char_手/qinhan.pkl`
- `.../char_手/weijin.pkl`
- `.../char_手/suitang.pkl`
- `.../char_手/songyuan.pkl`
- `.../char_手/ming.pkl`
- `.../char_手/qing.pkl`
- `.../char_手/republican.pkl`

目前工作區可直接用的候選來源：

- `mainfiles/0212_cleaned_output`

---

## 2.2 0219 Local 分群（分朝代）

程式：`0219_align_clustering.py`

核心流程：

1. 每個朝代各自 overclustering。
2. auto-merge。
3. dynamic sub-clustering（依群大小動態切分）。
4. 用 global PCA 對齊到同一 2D 空間（方便跨朝代對照）。

主要輸出：

- `char_手_Local_k{K}_split{R}_DynamicSubK/{dynasty}_stats.json`
- `char_手_Local_k{K}_split{R}_DynamicSubK/{dynasty}_viz.png`
- `char_手_Local_k{K}_split{R}_DynamicSubK/facet_grid_local_dynamic.png`
- `char_手_Local_k{K}_split{R}_DynamicSubK/手_Local_Dynamic_Merged.csv`

---

## 2.3 0219 Joint 分群（全朝代混合）

程式：`0219_joint_cluster.py`

核心流程：

1. 合併全部朝代 embedding。
2. overclustering。
3. auto-merge。
4. sub-clustering（固定 `--sub-k`、以 `--split-ratio` 判定大群切分）。

主要輸出：

- `char_手_k{K}_subk{S}_split{R}/joint_stats.json`
- `char_手_k{K}_subk{S}_split{R}/手_joint_merged.csv`
- `char_手_k{K}_subk{S}_split{R}/joint_clusters_viz.png`
- `char_手_k{K}_subk{S}_split{R}/joint_facet_by_dynasty.png`
- `char_手_k{K}_subk{S}_split{R}/joint_cluster_composition.png`

---

## 2.4 0221 後處理（標註 + 最終圖）

Local 後處理：

- 讀 `0221_progress/0221_local_dynamic/char_手_Local_k20_split0.3_DynamicSubK/手_Local_Dynamic_Merged.csv`
- 產出 `手_Labeled_Local.csv`、`Final_Semantic_Evolution_Local*.png`、`per_dynasties*/*`

Joint 後處理：

- 讀 `0221_progress/0221_joint_output/char_手_k20_subk6_split0.1/手_joint_merged.csv`
- 產出 `手_Labeled_Global.csv`、`Final_Semantic_Evolution_Global*.png`、`Trajectory_Global.png`、`per_dynasties_global*/*`

補充：

- `0221_local_dynamic/merge.py` 主要是合併各朝代 `*_stats.json` 成摘要 JSON，非 embedding 分群本體。

---

## 3. 可重跑清單（對齊目前 0221 結果）

## 3.1 先確認分群輸入是否齊全

```bash
cd /mnt/md0/public/dicwn/lrec-workshop
ls mainfiles/0212_cleaned_output/char_手/*.pkl
```

若這 8 個朝代 pkl 都在，即可往下跑。

---

## 3.2 重跑 Local（要對齊 `split0.3`）

```bash
cd /mnt/md0/public/dicwn/lrec-workshop
python 0219_align_clustering.py \
  --chars 手 \
  --input mainfiles/0212_cleaned_output \
  --output 0221_progress/0221_local_dynamic \
  --k-init 20 \
  --merge-ratio 0.95 \
  --split-ratio 0.3
```

預期關鍵輸出資料夾：

- `0221_progress/0221_local_dynamic/char_手_Local_k20_split0.3_DynamicSubK/`

預期關鍵檔案：

- `手_Local_Dynamic_Merged.csv`
- `ming_stats.json`（及其餘朝代）
- `facet_grid_local_dynamic.png`

---

## 3.3 重跑 Joint（要對齊 `k20_subk6_split0.1`）

```bash
cd /mnt/md0/public/dicwn/lrec-workshop
python 0219_joint_cluster.py \
  --input mainfiles/0212_cleaned_output \
  --char 手 \
  --output 0221_progress/0221_joint_output \
  --k 20 \
  --merge-ratio 0.95 \
  --sub-k 6 \
  --split-ratio 0.1
```

預期關鍵輸出資料夾：

- `0221_progress/0221_joint_output/char_手_k20_subk6_split0.1/`

預期關鍵檔案：

- `joint_stats.json`
- `手_joint_merged.csv`
- `joint_clusters_viz.png`

---

## 3.4 重跑 0221 Local 標註與最終圖

```bash
cd /mnt/md0/public/dicwn/lrec-workshop/0221_progress/0221_local_dynamic
python 0221_label_and_plot.py
python 0222_label_moreinfo.py
```

預期重點輸出：

- `手_Labeled_Local.csv`
- `Final_Semantic_Evolution_Local.png`
- `Final_Semantic_Evolution_Local_v2.png`

---

## 3.5 重跑 0221 Joint 標註、軌跡與最終圖

```bash
cd /mnt/md0/public/dicwn/lrec-workshop/0221_progress/0221_joint_output
python 0221_label_and_plot.py
python 0222_label_moreinfo.py
python plot_trajectory.py --csv 手_Labeled_Global.csv --output Trajectory_Global.png
```

預期重點輸出：

- `手_Labeled_Global.csv`
- `Final_Semantic_Evolution_Global.png`
- `Final_Semantic_Evolution_Global_v2.png`
- `Trajectory_Global.png`

---

## 4. 比對時建議優先看哪些檔案

1. 分群本體一致性：
   - Local：`char_手_Local_k20_split0.3_DynamicSubK/*_stats.json`
   - Joint：`char_手_k20_subk6_split0.1/joint_stats.json`

2. 分群點位與標籤對照：
   - `手_Local_Dynamic_Merged.csv`
   - `手_joint_merged.csv`

3. 最終圖一致性：
   - `Final_Semantic_Evolution_Local_v2.png`
   - `Final_Semantic_Evolution_Global_v2.png`

---

## 5. 注意事項

- 只要 `--split-ratio`、`--sub-k`、`--merge-ratio` 改掉，最終 K 值與群集語意都可能不同。
- 若版本套件有差（sklearn / numpy / seaborn），Silhouette 與圖形細節可能有小幅變動。
- 0221 的 label/plot 腳本內含固定對照表（cluster -> 語意標籤），若分群 ID 改變，標註也要同步調整。

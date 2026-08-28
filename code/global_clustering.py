"""
Joint Clustering — 全朝代混合分群
=====================================
把所有朝代的 768D embeddings 混在一起，跑一次 K-Means，
輸出：
  1. joint_clusters_viz.png      — 全量散點圖，顏色 = Global Cluster
  2. joint_facet_by_dynasty.png  — 八宮格，每格一個朝代，顏色仍是 Global Cluster（可跨朝代比較）
  3. joint_cluster_composition.png — 各 Global Cluster 的朝代組成比例 (stacked bar)
  4. joint_stats.json            — 各 cluster 的代表句、大小、朝代分佈
  5. joint_merged.csv            — 所有語料 + Global_Cluster 欄位

使用方式：
    python joint_clustering.py \
        --input  ./0219_clustering_subk3_ratio0.20 \
        --char   手 \
        --output ./0219_joint_output \
        --k      20 \
        --sub-k  4 \
        --split-ratio 0.20 \
        --merge-ratio 0.95

參數說明：
    --k           初始 overclustering 的 K 值（建議 15~30，之後會自動合併）
    --merge-ratio cosine 相似度合併門檻（0.90~0.96，越高越不容易合併）
    --sub-k       大群切成幾塊（3 or 4）
    --split-ratio 超過幾 % 的 cluster 才切分（0.10~0.25）
"""

import numpy as np
import pandas as pd
import pickle
import json
import os
import argparse
import time
import warnings
from collections import Counter

warnings.filterwarnings('ignore')

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import seaborn as sns
    from sklearn.decomposition import PCA
    from sklearn.cluster import MiniBatchKMeans
    from sklearn.metrics import silhouette_score
    from sklearn.metrics.pairwise import cosine_similarity
    matplotlib.rcParams['font.sans-serif'] = [
        'Noto Sans CJK TC', 'Microsoft JhengHei', 'SimHei', 'Arial'
    ]
    matplotlib.rcParams['axes.unicode_minus'] = False
except ImportError as e:
    print(f"[ERROR] 缺少依賴: {e}")
    exit(1)

# ── 常數 ─────────────────────────────────────────────────────
DYNASTY_ORDER = [
    "pre-qin", "qinhan", "weijin", "suitang",
    "songyuan", "ming", "qing", "republican"
]
DYNASTY_NAMES = {
    "pre-qin": "先秦", "qinhan": "秦漢", "weijin": "魏晉",
    "suitang": "隋唐", "songyuan": "宋元", "ming": "明",
    "qing": "清", "republican": "民國",
}

# ── 工具函數 ──────────────────────────────────────────────────
def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def load_all_embeddings(input_dir: str, char: str):
    """載入所有朝代的 pkl，回傳 embeddings、records、dynasty 標籤"""
    all_embeddings, all_records, all_dynasties = [], [], []
    char_dir = os.path.join(input_dir, f"char_{char}")

    for dynasty in DYNASTY_ORDER:
        fpath = os.path.join(char_dir, f"{dynasty}.pkl")
        if not os.path.exists(fpath):
            # 試著從上一層的 input 目錄找原始 pkl
            fpath = os.path.join(input_dir, f"char_{char}", f"{dynasty}.pkl")
        if not os.path.exists(fpath):
            log(f"  ⚠️  找不到 {dynasty}.pkl，跳過")
            continue

        with open(fpath, 'rb') as f:
            data = pickle.load(f)

        records = data.get('records', [])
        vecs = [r['vector'] for r in records if 'vector' in r]

        if not vecs:
            log(f"  ⚠️  {dynasty} 沒有 vector 欄位，跳過")
            continue

        all_embeddings.extend(vecs)
        all_records.extend(records)
        all_dynasties.extend([dynasty] * len(vecs))
        log(f"  ✅ {DYNASTY_NAMES.get(dynasty, dynasty)}: {len(vecs)} 筆")

    return (
        np.array(all_embeddings, dtype=np.float32),
        all_records,
        np.array(all_dynasties)
    )


# ── 分群核心 ─────────────────────────────────────────────────
def overcluster(embeddings, k):
    log(f"[Overcluster] K={k}, N={len(embeddings)}")
    km = MiniBatchKMeans(
        n_clusters=k,
        batch_size=min(4000, len(embeddings)),
        n_init=10, max_iter=300, random_state=42
    )
    return km.fit_predict(embeddings)


def auto_merge(labels, embeddings, merge_ratio=0.95):
    log(f"[Auto-Merge] 門檻={merge_ratio}")
    current = labels.copy()
    history = []
    step = 0

    while True:
        unique = np.unique(current)
        if len(unique) < 2:
            break
        centroids = {int(l): embeddings[current == l].mean(axis=0) for l in unique}
        sizes = {int(l): int((current == l).sum()) for l in unique}
        llist = list(centroids.keys())
        mat = np.array([centroids[l] for l in llist])
        sim = cosine_similarity(mat)
        np.fill_diagonal(sim, -1)
        idx = np.unravel_index(np.argmax(sim), sim.shape)
        max_sim = float(sim[idx])

        if max_sim <= merge_ratio:
            break

        li, lj = llist[idx[0]], llist[idx[1]]
        current[current == lj] = li
        history.append({
            "step": step + 1, "from": int(lj), "to": int(li),
            "similarity": round(max_sim, 6),
            "size_from": sizes[lj], "size_to": sizes[li]
        })
        step += 1
        if step % 5 == 0:
            log(f"  ... 已合併 {step} 次，目前 K={len(np.unique(current))}")

    unique = np.unique(current)
    mapping = {old: new for new, old in enumerate(unique)}
    log(f"  合併完成，最終 K={len(unique)}")
    return np.array([mapping[l] for l in current]), history


def subcluster_large(embeddings, labels, k_sub=4, split_ratio=0.20):
    threshold = max(int(len(embeddings) * split_ratio), 200)
    unique, counts = np.unique(labels, return_counts=True)
    large = [(int(l), int(c)) for l, c in zip(unique, counts) if c > threshold]

    if not large:
        log(f"[Sub-cluster] 無超過閾值（>{threshold}）的群，跳過")
        return labels, {}

    log(f"[Sub-cluster] 發現 {len(large)} 個大群，各切成 {k_sub} 份")
    current = labels.copy()
    next_id = int(np.max(current)) + 1
    split_info = {}

    for target, count in large:
        mask = (current == target)
        sub_emb = embeddings[mask]
        orig_idx = np.where(mask)[0]
        n_comp = min(50, sub_emb.shape[0] - 1, sub_emb.shape[1])

        pca_local = PCA(n_components=n_comp, random_state=42)
        sub_pca = pca_local.fit_transform(sub_emb)
        km = MiniBatchKMeans(n_clusters=k_sub, n_init=10, random_state=42)
        sub_labels = km.fit_predict(sub_pca)

        new_ids = []
        for i in range(k_sub):
            current[orig_idx[sub_labels == i]] = next_id
            new_ids.append(next_id)
            next_id += 1

        split_info[target] = {"original_size": count, "new_ids": new_ids, "threshold": threshold}
        log(f"  C{target} (n={count}) → {new_ids}")

    unique = np.unique(current)
    mapping = {old: new for new, old in enumerate(unique)}
    return np.array([mapping[l] for l in current]), split_info


def extract_representative(embeddings, labels, records, dynasties, n_core=5, n_boundary=3):
    result = {}
    for lbl in np.unique(labels):
        lbl = int(lbl)
        mask = (labels == lbl)
        idx = np.where(mask)[0]
        points = embeddings[mask]
        centroid = points.mean(axis=0)
        dists = np.linalg.norm(points - centroid, axis=1)

        core = [
            {
                "sentence": records[idx[i]].get('sentence', ''),
                "source": records[idx[i]].get('toptitle', ''),
                "dynasty": DYNASTY_NAMES.get(dynasties[idx[i]], dynasties[idx[i]]),
                "dist": round(float(dists[i]), 4)
            }
            for i in np.argsort(dists)[:n_core]
        ]
        boundary = [
            {
                "sentence": records[idx[i]].get('sentence', ''),
                "source": records[idx[i]].get('toptitle', ''),
                "dynasty": DYNASTY_NAMES.get(dynasties[idx[i]], dynasties[idx[i]]),
                "dist": round(float(dists[i]), 4)
            }
            for i in np.argsort(dists)[-n_boundary:]
        ] if len(dists) > n_core + n_boundary else []

        dynasty_dist = dict(Counter(
            DYNASTY_NAMES.get(dynasties[i], dynasties[i]) for i in idx
        ))

        result[lbl] = {
            "size": int(mask.sum()),
            "dynasty_distribution": dynasty_dist,
            "core": core,
            "boundary": boundary
        }
    return result


# ── 視覺化 ────────────────────────────────────────────────────
def plot_joint_full(embed_2d, labels, output_path):
    """全量散點圖，顏色 = Global Cluster"""
    n_clusters = len(np.unique(labels))
    palette = sns.color_palette('husl', n_clusters)

    n = len(labels)
   # s = 3 if n > 50000 else (5 if n > 10000 else 10)
   # a = 0.2 if n > 50000 else (0.3 if n > 10000 else 0.5)
    s = 8
    a = 0.5

    fig, ax = plt.subplots(figsize=(16, 12))
    sns.scatterplot(
        x=embed_2d[:, 0], y=embed_2d[:, 1],
        hue=labels, palette=palette,
        s=s, alpha=a, edgecolor='none',
        legend=False, ax=ax
    )

    for lbl in np.unique(labels):
        mask = (labels == lbl)
        cx, cy = embed_2d[mask, 0].mean(), embed_2d[mask, 1].mean()
        ax.text(cx, cy, f"GC{lbl}", fontsize=8, fontweight='bold',
                ha='center', va='center', zorder=4,
                bbox=dict(facecolor='white', alpha=0.75, edgecolor='gray',
                          boxstyle='round,pad=0.2', linewidth=0.4))

    ax.set_title(f"Joint Clustering — 全朝代混合分群 (K={n_clusters}, N={n:,})",
                 fontsize=14, fontweight='bold')
    ax.set_xlabel("PC1"), ax.set_ylabel("PC2")
    ax.grid(True, linestyle='--', alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    log(f"  ✅ 儲存: {output_path}")


def plot_facet_by_dynasty(embed_2d, labels, dynasties_arr, output_path):
    """八宮格，各朝代用同一套 Global Cluster 顏色"""
    n_clusters = len(np.unique(labels))
    palette = dict(zip(np.unique(labels),
                       sns.color_palette('husl', n_clusters)))

    all_x, all_y = embed_2d[:, 0], embed_2d[:, 1]

    fig, axes = plt.subplots(2, 4, figsize=(24, 12))
    axes = axes.flatten()

    for i, dynasty in enumerate(DYNASTY_ORDER):
        ax = axes[i]
        mask = (dynasties_arr == dynasty)
        name = DYNASTY_NAMES.get(dynasty, dynasty)

        if not mask.any():
            ax.set_title(f"{name}\n(無資料)", fontsize=11)
            ax.axis('off')
            continue

        # 灰色背景（全量）
        ax.scatter(all_x, all_y, c='lightgray', s=1, alpha=0.15, zorder=1)

        # 彩色前景（本朝代，顏色 = Global Cluster）
        sub_x = all_x[mask]
        sub_y = all_y[mask]
        sub_labels = labels[mask]
        for lbl in np.unique(sub_labels):
            lmask = (sub_labels == lbl)
            ax.scatter(sub_x[lmask], sub_y[lmask],
                       c=[palette[lbl]], s=4, alpha=0.5, zorder=2, edgecolor='none')

        # 標籤
        for lbl in np.unique(sub_labels):
            lmask = (sub_labels == lbl)
            cx = sub_x[lmask].mean()
            cy = sub_y[lmask].mean()
            ax.text(cx, cy, f"GC{lbl}", fontsize=7, fontweight='bold',
                    ha='center', va='center', zorder=3,
                    bbox=dict(facecolor='white', alpha=0.75, edgecolor='gray',
                              boxstyle='round,pad=0.2', linewidth=0.4))

        ax.set_title(f"{name}  N={mask.sum():,}", fontsize=11)
        ax.set_xlabel("PC1", fontsize=8)
        ax.set_ylabel("PC2", fontsize=8)

    fig.suptitle("Joint Clustering — 各朝代分佈（同一套 Global Cluster 顏色）",
                 fontsize=15, fontweight='bold', y=1.01)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    log(f"  ✅ 儲存: {output_path}")


def plot_cluster_composition(labels, dynasties_arr, output_path):
    """各 Global Cluster 的朝代組成比例（Stacked Bar）"""
    df = pd.DataFrame({'cluster': labels, 'dynasty': dynasties_arr})
    df['dynasty_zh'] = df['dynasty'].map(DYNASTY_NAMES)

    pivot = df.groupby(['cluster', 'dynasty']).size().unstack(fill_value=0)
    # 保持朝代順序
    pivot = pivot.reindex(columns=[d for d in DYNASTY_ORDER if d in pivot.columns])
    pivot_pct = pivot.div(pivot.sum(axis=1), axis=0) * 100

    fig, ax = plt.subplots(figsize=(max(12, len(pivot) * 0.6), 7))
    dynasty_zh_cols = [DYNASTY_NAMES.get(c, c) for c in pivot_pct.columns]
    palette = sns.color_palette('tab10', len(pivot_pct.columns))

    bottom = np.zeros(len(pivot_pct))
    for j, col in enumerate(pivot_pct.columns):
        ax.bar(
            [f"GC{c}" for c in pivot_pct.index],
            pivot_pct[col].values,
            bottom=bottom,
            color=palette[j],
            label=DYNASTY_NAMES.get(col, col),
            edgecolor='white', linewidth=0.3
        )
        bottom += pivot_pct[col].values

    ax.set_xlabel("Global Cluster", fontsize=12)
    ax.set_ylabel("朝代比例 (%)", fontsize=12)
    ax.set_title("各 Global Cluster 朝代組成比例", fontsize=14, fontweight='bold')
    ax.legend(title="朝代", bbox_to_anchor=(1.01, 1), loc='upper left', fontsize=10)
    ax.set_ylim(0, 100)
    plt.xticks(rotation=45, ha='right', fontsize=9)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    log(f"  ✅ 儲存: {output_path}")


# ── 主程式 ────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Joint Clustering — 全朝代混合分群")
    parser.add_argument('--input',        type=str,   required=True,  help='輸入目錄（含 char_手/xxx.pkl）')
    parser.add_argument('--char',         type=str,   required=True,  help='要分析的字，例如: 手')
    parser.add_argument('--output',       type=str,   required=True,  help='輸出目錄')
    # ── 關鍵參數 ──
    parser.add_argument('--k',            type=int,   default=20,     help='Overcluster 初始 K（建議 15~30）')
    parser.add_argument('--merge-ratio',  type=float, default=0.95,   help='Cosine 合併門檻（建議 0.92~0.96）')
    parser.add_argument('--sub-k',        type=int,   default=4,      help='大群切成幾塊（建議 3 or 4）')
    parser.add_argument('--split-ratio',  type=float, default=0.15,   help='超過幾 %% 才切（建議 0.10~0.20）')
    parser.add_argument('--no-sub',       action='store_true',        help='不做 sub-clustering')
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)
    param_suffix = f"k{args.k}_subk{args.sub_k}_split{args.split_ratio}"
    char_dir = os.path.join(args.output, f"char_{args.char}_{param_suffix}")
    
    os.makedirs(char_dir, exist_ok=True)

    log("=" * 60)
    log(f"Joint Clustering | 字: {args.char} | K_init={args.k}")
    log(f"  merge_ratio={args.merge_ratio}, sub_k={args.sub_k}, split_ratio={args.split_ratio}")
    log("=" * 60)

    # Step 1: 載入全部資料
    embeddings, records, dynasties = load_all_embeddings(args.input, args.char)
    log(f"總計載入 {len(embeddings):,} 筆")

    # Step 2: Global PCA（只用於視覺化）
    log("[Global PCA] fitting...")
    pca = PCA(n_components=2, random_state=42)
    embed_2d = pca.fit_transform(embeddings)
    ev = pca.explained_variance_ratio_
    log(f"  PC1={ev[0]:.3f}, PC2={ev[1]:.3f}, 合計={sum(ev):.3f}")

    # Step 3: 分群
    labels_init = overcluster(embeddings, args.k)
    labels_merged, merge_hist = auto_merge(labels_init, embeddings, args.merge_ratio)

    labels_final = labels_merged
    split_info = {}
    if not args.no_sub:
        labels_final, split_info = subcluster_large(
            embeddings, labels_merged, args.sub_k, args.split_ratio
        )

    k_final = len(np.unique(labels_final))
    log(f"最終 K={k_final}")

    # Step 4: Silhouette
    log("[Silhouette] 計算中（最多取 5000 點）...")
    idx_sample = np.random.RandomState(42).choice(len(embeddings), min(5000, len(embeddings)), replace=False)
    sil = float(silhouette_score(embeddings[idx_sample], labels_final[idx_sample]))
    log(f"  Silhouette={sil:.4f}")

    # Step 5: 代表句
    log("[Representative] 抽取代表句...")
    cluster_info = extract_representative(embeddings, labels_final, records, dynasties)

    # Step 6: 儲存 JSON
    stats = {
        "char": args.char,
        "total_samples": int(len(embeddings)),
        "k_final": k_final,
        "silhouette": round(sil, 6),
        "params": {
            "k_initial": args.k,
            "merge_ratio": args.merge_ratio,
            "sub_k": args.sub_k,
            "split_ratio": args.split_ratio
        },
        "merge_history": merge_hist,
        "split_info": {str(k): v for k, v in split_info.items()},
        "clusters": cluster_info
    }

    def cvt(o):
        if isinstance(o, (np.integer,)): return int(o)
        if isinstance(o, (np.floating,)): return float(o)
        if isinstance(o, np.ndarray): return o.tolist()
        raise TypeError

    json_path = os.path.join(char_dir, "joint_stats.json")
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(stats, f, ensure_ascii=False, indent=2, default=cvt)
    log(f"  ✅ 統計儲存: {json_path}")

    # Step 7: 儲存 CSV
    df = pd.DataFrame({
        'dynasty':        dynasties,
        'dynasty_zh':     [DYNASTY_NAMES.get(d, d) for d in dynasties],
        'sentence':       [r.get('sentence', '') for r in records],
        'title':          [r.get('toptitle', '') for r in records],
        'Global_Cluster': labels_final,
        'PC1':            embed_2d[:, 0],
        'PC2':            embed_2d[:, 1],
    })
    df['dynasty'] = pd.Categorical(df['dynasty'], categories=DYNASTY_ORDER, ordered=True)
    df = df.sort_values('dynasty')

    csv_path = os.path.join(char_dir, f"{args.char}_joint_merged.csv")
    df.to_csv(csv_path, index=False, encoding='utf-8-sig')
    log(f"  ✅ CSV 儲存: {csv_path}")

    # Step 8: 畫圖
    log("[視覺化] 繪製圖表...")
    plot_joint_full(embed_2d, labels_final,
                    os.path.join(char_dir, "joint_clusters_viz.png"))
    plot_facet_by_dynasty(embed_2d, labels_final, dynasties,
                          os.path.join(char_dir, "joint_facet_by_dynasty.png"))
    plot_cluster_composition(labels_final, dynasties,
                             os.path.join(char_dir, "joint_cluster_composition.png"))

    log("=" * 60)
    log(f"完成！輸出在: {char_dir}")
    log(f"  K_final={k_final}, Silhouette={sil:.4f}")
    log("=" * 60)


if __name__ == "__main__":
    main()
"""
Overclustering + Auto-Merge + Hierarchical Refinement (V2.4.1)
Dynamic Sub-K, with per-dynasty PNG export restored.

Key changes in this revision:
1. The number of sub-clusters is now determined dynamically (2-6 groups) to
   counter over-fragmentation.
2. Restored the visualize_clusters function so each dynasty again produces
   its own _viz.png.
"""

import numpy as np
import pandas as pd
import pickle
import json
import time
import sys
import os
import argparse
from typing import List, Dict, Tuple, Optional
from tqdm import tqdm
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import silhouette_score
import warnings

warnings.filterwarnings('ignore')

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import seaborn as sns
    from sklearn.decomposition import PCA
    matplotlib.rcParams['font.sans-serif'] = ['Noto Sans CJK TC', 'Droid Sans Fallback', 'DejaVu Sans', 'Arial']
    matplotlib.rcParams['axes.unicode_minus'] = False
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    print("[WARNING] matplotlib is not installed; figure generation will be skipped")

try:
    from adjustText import adjust_text
    ADJUSTTEXT_AVAILABLE = True
except ImportError:
    ADJUSTTEXT_AVAILABLE = False

# ============================================================
# CONFIG
# ============================================================

# Note: these display names are used only for chart titles and log
# messages, not written into the released assignment CSVs. Kept as-is for
# consistency with global_clustering.py.
DYNASTY_NAMES = {
    "pre-qin": "先秦", "qinhan": "秦漢", "weijin": "魏晉",
    "suitang": "隋唐", "songyuan": "宋元", "ming": "明",
    "qing": "清", "republican": "民國",
}

CHAR_PINYIN = {
    "手": "shǒu", "天": "tiān", "地": "dì",
    "下": "xià", "日": "rì", "首": "shǒu", "道": "dào",
}

DYNASTY_ORDER = ["pre-qin", "qinhan", "weijin", "suitang", "songyuan", "ming", "qing", "republican"]

DEFAULT_K_INITIAL = 20
DEFAULT_MERGE_RATIO = 0.95
DEFAULT_SPLIT_RATIO = 0.20   # Only clusters exceeding 20% of a dynasty's total samples are split
MIN_ABSOLUTE_SIZE = 100      # Absolute floor, so very small dynasties are not split too finely
SILHOUETTE_SAMPLE_LIMIT = 5000

# ============================================================
# HELPERS
# ============================================================

def log(message):
    print(f"[{time.strftime('%H:%M:%S')}] {message}")
    sys.stdout.flush()

# ============================================================
# GLOBAL PCA
# ============================================================

def build_global_pca(input_dir: str, char: str):
    if not MATPLOTLIB_AVAILABLE: return None
    log(f"[Global PCA] Loading embeddings for every dynasty (char: {char})...")
    all_embeddings = []

    for dynasty in DYNASTY_ORDER:
        fpath = os.path.join(input_dir, f"char_{char}", f"{dynasty}.pkl")
        if not os.path.exists(fpath): continue
        try:
            with open(fpath, 'rb') as f: data = pickle.load(f)
            vecs = [r['vector'] for r in data.get('records', []) if 'vector' in r]
            all_embeddings.extend(vecs)
        except Exception as e:
            log(f"  Failed to read {dynasty}: {e}")

    if not all_embeddings: return None

    arr = np.array(all_embeddings, dtype=np.float32)
    log(f"[Global PCA] {len(arr)} records total, fitting PCA(n_components=2)...")
    pca = PCA(n_components=2, random_state=42).fit(arr)
    return pca

# ============================================================
# CLUSTERING
# ============================================================

def get_adaptive_k_initial(n_samples: int, base_k: int = 20) -> int:
    if n_samples < 800:   return max(5, base_k // 3)
    elif n_samples < 2000: return max(8, base_k // 2)
    elif n_samples < 5000: return max(12, int(base_k * 0.7))
    else: return base_k

def overcluster_768d(embeddings, K_initial, n_samples):
    kmeans = MiniBatchKMeans(
        n_clusters=K_initial, batch_size=min(2000, n_samples),
        n_init=10, max_iter=300, random_state=42, verbose=0
    )
    return kmeans.fit_predict(embeddings)

def auto_merge_clusters_cosine(labels, embeddings, merge_ratio=0.95):
    from sklearn.metrics.pairwise import cosine_similarity
    merge_history = []
    current_labels = labels.copy()
    merged = True
    step = 0

    while merged:
        merged = False
        unique_labels = np.unique(current_labels)
        if len(unique_labels) < 2: break

        centroids = {int(l): embeddings[current_labels == l].mean(axis=0) for l in unique_labels}
        sizes = {int(l): int((current_labels == l).sum()) for l in unique_labels}
        labels_list = list(centroids.keys())
        mat = np.array([centroids[l] for l in labels_list])

        sim = cosine_similarity(mat)
        np.fill_diagonal(sim, -1)
        idx = np.unravel_index(np.argmax(sim), sim.shape)
        max_sim = float(sim[idx])

        if max_sim > merge_ratio:
            li, lj = labels_list[idx[0]], labels_list[idx[1]]
            current_labels[current_labels == lj] = li
            merge_history.append({
                "step": step + 1, "from": int(lj), "to": int(li),
                "similarity": max_sim, "size_from": sizes[lj], "size_to": sizes[li]
            })
            step += 1
            merged = True

    unique = np.unique(current_labels)
    mapping = {old: new for new, old in enumerate(unique)}
    return np.array([mapping[l] for l in current_labels]), merge_history

def subcluster_all_large_groups_dynamic(embeddings, labels, split_ratio=0.20, min_absolute=100):
    threshold = max(int(len(embeddings) * split_ratio), min_absolute)
    unique_labels, counts = np.unique(labels, return_counts=True)
    large = [(int(l), int(c)) for l, c in zip(unique_labels, counts) if c > threshold]

    if not large:
        log(f"    No cluster exceeds the threshold (>{threshold}), skipping split")
        return labels, {}

    log(f"    Found {len(large)} clusters exceeding the threshold (>{threshold}), running dynamic split")
    current_labels = labels.copy()
    next_id = int(np.max(current_labels)) + 1
    split_info = {}

    for target_label, count in large:
        if count >= 20000: dyn_sub_k = 6
        elif count >= 10000: dyn_sub_k = 5
        elif count >= 5000: dyn_sub_k = 4
        elif count >= 2000: dyn_sub_k = 3
        else: dyn_sub_k = 2

        log(f"      C{target_label} (n={count}) -> dynamically split into {dyn_sub_k} parts")
        mask = (current_labels == target_label)
        sub_emb = embeddings[mask]
        orig_idx = np.where(mask)[0]

        n_comp = min(50, sub_emb.shape[0], sub_emb.shape[1])
        pca_local = PCA(n_components=n_comp, random_state=42)
        sub_pca = pca_local.fit_transform(sub_emb)

        km = MiniBatchKMeans(
            n_clusters=dyn_sub_k, batch_size=min(1024, count),
            n_init=10, random_state=42, verbose=0
        )
        sub_labels = km.fit_predict(sub_pca)

        new_ids = []
        for i in range(dyn_sub_k):
            current_labels[orig_idx[sub_labels == i]] = next_id
            new_ids.append(next_id)
            next_id += 1

        split_info[target_label] = {
            "original_size": count, "dynamic_sub_k": dyn_sub_k, "new_ids": new_ids
        }

    unique = np.unique(current_labels)
    mapping = {old: new for new, old in enumerate(unique)}
    current_labels = np.array([mapping[l] for l in current_labels])
    return current_labels, split_info

def extract_representative_samples(embeddings, labels, records, n_core=8, n_boundary=3):
    result = {}
    for label in np.unique(labels):
        label = int(label)
        mask = (labels == label)
        indices = np.where(mask)[0]
        if len(indices) == 0: continue

        points = embeddings[mask]
        centroid = points.mean(axis=0)
        dists = np.linalg.norm(points - centroid, axis=1)

        core = [{"sentence": records[indices[i]]['sentence'],
                 "toptitle": records[indices[i]].get('toptitle', ''),
                 "dist": float(dists[i])} for i in np.argsort(dists)[:n_core]]

        boundary = [{"sentence": records[indices[i]]['sentence'],
                     "toptitle": records[indices[i]].get('toptitle', ''),
                     "dist": float(dists[i])} for i in np.argsort(dists)[-n_boundary:]] if len(dists) > n_core + n_boundary else []

        result[label] = {"core": core, "boundary": boundary, "count": int(mask.sum())}
    return result

# ============================================================
# VISUALIZATION
# ============================================================

def visualize_clusters(embed_2d, labels, title, output_path):
    """Per-dynasty scatter plot."""
    if not MATPLOTLIB_AVAILABLE: return
    n_clusters = len(np.unique(labels))

    fig, ax = plt.subplots(figsize=(14, 11))
    palette = sns.color_palette('husl', n_clusters)

    sns.scatterplot(x=embed_2d[:, 0], y=embed_2d[:, 1], hue=labels,
                    palette=palette, legend='full', alpha=0.6, s=15,
                    edgecolor='none', ax=ax)

    # Add cluster labels
    for lbl in np.unique(labels):
        mask = (labels == lbl)
        center = embed_2d[mask].mean(axis=0)
        ax.text(center[0], center[1], f"C{lbl}", fontsize=11, fontweight='bold',
                bbox=dict(facecolor='white', alpha=0.85, boxstyle='circle', edgecolor='gray'))

    ax.set_title(title, fontsize=15, pad=12, fontweight='bold')
    ax.set_xlabel('PC1', fontsize=12)
    ax.set_ylabel('PC2', fontsize=12)
    ax.legend(title='Cluster ID', bbox_to_anchor=(1.02, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close(fig)

def plot_facet_grid(dynasty_results, char, output_dir, global_pca=None):
    if not MATPLOTLIB_AVAILABLE: return
    fig, axes = plt.subplots(2, 4, figsize=(24, 12))
    axes = axes.flatten()

    all_embed_2d = [r["embed_2d"] for r in dynasty_results.values() if r and r.get("embed_2d") is not None]
    if not all_embed_2d: return
    all_arr = np.vstack(all_embed_2d)

    for idx, dynasty in enumerate(DYNASTY_ORDER):
        ax = axes[idx]
        res = dynasty_results.get(dynasty)
        if not res:
            ax.set_title(f"{DYNASTY_NAMES.get(dynasty, dynasty)}\n(no data)")
            ax.axis('off')
            continue

        embed_2d, labels = res["embed_2d"], res["labels"]
        ax.scatter(all_arr[:, 0], all_arr[:, 1], c='lightgray', s=1, alpha=0.15, zorder=1)
        sns.scatterplot(x=embed_2d[:, 0], y=embed_2d[:, 1], hue=labels, palette='husl', s=8, alpha=0.6, legend=False, ax=ax, zorder=2, edgecolor='none')

        for lbl in np.unique(labels):
            mask = (labels == lbl)
            center = embed_2d[mask].mean(axis=0)
            ax.text(center[0], center[1], f"C{lbl}", fontsize=8, fontweight='bold', bbox=dict(facecolor='white', alpha=0.8, boxstyle='circle', edgecolor='gray'))

        ax.set_title(f"{DYNASTY_NAMES.get(dynasty, dynasty)} (N={len(labels)}, K={len(np.unique(labels))})", fontsize=12)

    fig.suptitle(f'"{char}" — Diachronic Semantic Evolution (Local Dynamic Clusters in Global Space)', fontsize=18, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "facet_grid_local_dynamic.png"), dpi=150, bbox_inches='tight')
    plt.close()

# ============================================================
# MAIN
# ============================================================

def process_dynasty(char, dynasty, input_dir, k_initial, merge_ratio, split_ratio, global_pca=None):
    fpath = os.path.join(input_dir, f"char_{char}", f"{dynasty}.pkl")
    if not os.path.exists(fpath): return None

    with open(fpath, 'rb') as f: data = pickle.load(f)
    records = data['records']
    n_samples = len(records)
    if n_samples < 10: return None

    log(f"  Processing {DYNASTY_NAMES.get(dynasty, dynasty)} (n={n_samples})...")
    embeddings_768d = np.array([r['vector'] for r in records], dtype=np.float32)
    raw_sentences = [r.get('sentence', '') for r in records]
    raw_titles = [r.get('toptitle', '') or r.get('work_title', '') for r in records]

    adaptive_k = get_adaptive_k_initial(n_samples, k_initial)
    labels_init = overcluster_768d(embeddings_768d, adaptive_k, n_samples)
    labels_merged, merge_hist = auto_merge_clusters_cosine(labels_init, embeddings_768d, merge_ratio)

    labels_final, split_info = subcluster_all_large_groups_dynamic(embeddings_768d, labels_merged, split_ratio=split_ratio)

    embed_2d = global_pca.transform(embeddings_768d) if global_pca else PCA(n_components=2, random_state=42).fit_transform(embeddings_768d)
    sil = silhouette_score(embeddings_768d[:5000], labels_final[:5000]) if len(embeddings_768d) > 5000 else silhouette_score(embeddings_768d, labels_final)

    return {
        "dynasty": dynasty, "n_samples": n_samples, "k_final": len(np.unique(labels_final)),
        "labels": labels_final, "silhouette": sil, "split_info": split_info,
        "embed_2d": embed_2d, "raw_sentences": raw_sentences, "raw_titles": raw_titles,
        "cluster_sentences": extract_representative_samples(embeddings_768d, labels_final, records)
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--chars', type=str, required=True)
    parser.add_argument('--input', type=str, required=True)
    parser.add_argument('--output', type=str, required=True)
    parser.add_argument('--k-init', type=int, default=20)
    parser.add_argument('--merge-ratio', type=float, default=0.95)
    parser.add_argument('--split-ratio', type=float, default=0.20)
    args = parser.parse_args()

    chars = args.chars.split(',')
    param_suffix = f"Local_k{args.k_init}_split{args.split_ratio}_DynamicSubK"

    for char in chars:
        char_dir = os.path.join(args.output, f"char_{char}_{param_suffix}")
        os.makedirs(char_dir, exist_ok=True)
        log(f"\nProcessing character: {char} (Local V2.4.1 Dynamic Sub-K)")

        global_pca = build_global_pca(args.input, char)
        dynasty_results = {}

        for dyn in DYNASTY_ORDER:
            res = process_dynasty(char, dyn, args.input, args.k_init, args.merge_ratio, args.split_ratio, global_pca)
            dynasty_results[dyn] = res

            if res:
                # Save JSON
                stats = {k: v for k, v in res.items() if k not in ['labels', 'embed_2d', 'raw_sentences', 'raw_titles']}
                with open(os.path.join(char_dir, f"{dyn}_stats.json"), 'w', encoding='utf-8') as f:
                    json.dump(stats, f, ensure_ascii=False, indent=2, default=lambda x: int(x) if isinstance(x, np.integer) else x)

                # Produce this dynasty's individual scatter plot PNG
                if MATPLOTLIB_AVAILABLE:
                    title = f'"{char}" ({DYNASTY_NAMES[dyn]}) - Local Clustering (K={res["k_final"]})'
                    out_png = os.path.join(char_dir, f"{dyn}_viz.png")
                    visualize_clusters(res["embed_2d"], res["labels"], title, out_png)
                    log(f"    Saved figure: {dyn}_viz.png")

        # Render the facet grid
        plot_facet_grid(dynasty_results, char, char_dir, global_pca)

        # Export the merged CSV
        all_rows = []
        for dyn in DYNASTY_ORDER:
            res = dynasty_results.get(dyn)
            if res:
                for i in range(res['n_samples']):
                    all_rows.append({
                        'dynasty': dyn, 'sentence': res['raw_sentences'][i], 'title': res['raw_titles'][i],
                        'Local_Cluster': res['labels'][i], 'PC1': res['embed_2d'][i, 0], 'PC2': res['embed_2d'][i, 1]
                    })
        if all_rows:
            df = pd.DataFrame(all_rows)
            df['dynasty'] = pd.Categorical(df['dynasty'], categories=DYNASTY_ORDER, ordered=True)
            df.sort_values('dynasty').to_csv(os.path.join(char_dir, f"{char}_Local_Dynamic_Merged.csv"), index=False, encoding='utf-8-sig')
            log(f"Done. All data and figures written to {char_dir}")

if __name__ == "__main__":
    main()

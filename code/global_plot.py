#!/usr/bin/env python3
"""
global_plot.py

Stage 7 of the pipeline (formerly the inline "postprocess" stage in pipeline.py).
Reads a single character's Global/Joint clustering CSV, attaches the reviewed
semantic label mapping if one is available (from --label-map-json's "joint"
block, or the hand-specific 手 fallback), and renders the paper's Global figures.

This script never decides how Local labels are produced — see local_annotate.py,
which owns all Local output (both the join-inherited default and the optional
independently-annotated override).

Runs with or without labels: if no joint mapping is resolvable, it falls back to
a plain, cluster-ID-colored rendering instead of failing, so a first pass on a
brand-new keyword still produces material to look at before anyone labels it.

Usage:
    python3 code/global_plot.py --char 道 \\
        --global-dir stage_outputs/06_joint_output/char_道_k20_subk6_split0.1 \\
        --output stage_outputs/07_global_plot \\
        --label-map-json configs/dao_shui_labels.json
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.patches as mpatches
import pandas as pd

from pipeline_common import (
    DYNASTY_ORDER,
    DYNASTY_DISPLAY,
    _char_display,
    _draw_proportion_bar,
    apply_cluster_labels,
    build_color_palette_for_mapping,
    ensure_dir,
    find_cluster_column,
    load_global_silhouette,
    plot_facet,
    plot_per_dynasty,
    resolve_hand_label_mapping,
    resolve_label_mapping_scoped,
    simple_label_df,
)

import numpy as np
import matplotlib.pyplot as plt

try:
    from adjustText import adjust_text  # noqa: F401  (unused here; kept for parity)
    _ADJUSTTEXT_AVAILABLE = True
except ImportError:
    _ADJUSTTEXT_AVAILABLE = False


# ─────────────────────────────────────────────────────────────
# Label repulsion (for the joint-scatter and single-dynasty plots)
# ─────────────────────────────────────────────────────────────
def _repel_labels(positions, data_span, offset_base_ratio=0.10,
                  min_label_dist_ratio=0.16, min_node_dist_ratio=0.12,
                  iterations=500, strength=0.06, x_bounds=None, y_bounds=None):
    keys = list(positions.keys())
    if not keys:
        return {}
    nodes = np.array([positions[k] for k in keys])
    pos = nodes.copy().astype(float)
    center = nodes.mean(axis=0)
    ob = data_span * offset_base_ratio
    mld = data_span * min_label_dist_ratio
    mnd = data_span * min_node_dist_ratio

    for i in range(len(pos)):
        d = pos[i] - center
        n = np.linalg.norm(d)
        if n < 1e-6:
            d, n = np.array([1.0, 0.0]), 1.0
        pos[i] = nodes[i] + d / n * ob

    for _ in range(iterations):
        for i in range(len(pos)):
            for j in range(len(pos)):
                if i == j:
                    continue
                delta = pos[i] - pos[j]
                dist = np.linalg.norm(delta)
                if dist < mld and dist > 1e-6:
                    pos[i] += (delta / dist) * (mld - dist) * strength
            for j in range(len(nodes)):
                delta = pos[i] - nodes[j]
                dist = np.linalg.norm(delta)
                if dist < mnd and dist > 1e-6:
                    pos[i] += (delta / dist) * (mnd - dist) * strength * 1.5

        if x_bounds is not None:
            pos[:, 0] = np.clip(pos[:, 0], x_bounds[0], x_bounds[1])
        if y_bounds is not None:
            pos[:, 1] = np.clip(pos[:, 1], y_bounds[0], y_bounds[1])

    return {k: pos[idx] for idx, k in enumerate(keys)}


# ─────────────────────────────────────────────────────────────
# Global plots (generic, character-agnostic)
# ─────────────────────────────────────────────────────────────
def plot_semantic_joint_generic(df, output_path, global_sc, ch, color_palette):
    n_total = len(df)
    fig, ax = plt.subplots(figsize=(14, 11))
    all_macros = sorted(df["Macro_Category"].unique())

    for macro in all_macros:
        msub = df[df["Macro_Category"] == macro]
        if msub.empty:
            continue
        is_noise = macro == "NOISE"
        color = color_palette.get(macro, "#888888")
        ax.scatter(msub["PC1"], msub["PC2"],
                   s=3 if is_noise else 8, color=color,
                   alpha=0.15 if is_noise else 0.45,
                   edgecolors="none", zorder=1)

    centroids = {}
    for (c_id, macro, sub_cat), grp in df.groupby(["Cluster_ID", "Macro_Category", "Sub_Category"]):
        if macro == "NOISE":
            continue
        centroids[(c_id, macro, sub_cat)] = np.array([grp["PC1"].mean(), grp["PC2"].mean()])

    data_span = max(df["PC1"].max() - df["PC1"].min(), df["PC2"].max() - df["PC2"].min())
    pad = data_span * 0.03
    x_bounds = (df["PC1"].min() - pad, df["PC1"].max() + pad)
    y_bounds = (df["PC2"].min() - pad, df["PC2"].max() + pad)
    ax.set_xlim(x_bounds)
    ax.set_ylim(y_bounds)

    label_pos = _repel_labels(centroids, data_span,
                               offset_base_ratio=0.10, min_label_dist_ratio=0.16,
                               min_node_dist_ratio=0.12, iterations=600, strength=0.06,
                               x_bounds=x_bounds, y_bounds=y_bounds)

    for key, node_xy in centroids.items():
        c_id, macro, sub_cat = key
        color = color_palette.get(macro, "#444444")
        lp = label_pos[key]
        ax.scatter(*node_xy, s=55, color=color, edgecolors="white", linewidths=1.2, zorder=4)
        ax.annotate("", xy=node_xy, xytext=lp,
                    arrowprops=dict(arrowstyle="-", color="black", lw=0.7, alpha=0.5), zorder=4)
        ax.text(*lp, f"C{int(c_id)} ({sub_cat})",
                fontsize=9.5, color=color, fontweight="bold",
                ha="center", va="center",
                bbox=dict(boxstyle="round,pad=0.2", fc="white", ec=color,
                          alpha=0.88, linewidth=0.8), zorder=5)

    sc_text = (f"Silhouette Score = {global_sc:.4f}  |  N = {n_total:,}"
               if global_sc is not None else f"N = {n_total:,}")
    ax.text(0.98, 0.03, sc_text, transform=ax.transAxes, fontsize=10, color="#333333",
            va="bottom", ha="right",
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#aaaaaa",
                      alpha=0.85, linewidth=0.7), zorder=6)

    present = df["Macro_Category"].unique()
    legend_patches = [
        mpatches.Patch(color=color_palette.get(m, "#888888"), label=m.replace("_", " "))
        for m in sorted(color_palette.keys()) if m in present
    ]
    ax.legend(handles=legend_patches, title="Macro Category",
              title_fontsize=11, fontsize=10, loc="lower left", framealpha=0.88)

    ax.set_title(f"{_char_display(ch)} — All Dynasties, Global Semantic Space  (n={n_total:,})",
                 fontsize=17, fontweight="bold", pad=14)
    ax.set_xlabel("PC1", fontsize=11)
    ax.set_ylabel("PC2", fontsize=11)
    ax.grid(True, linestyle="--", alpha=0.35)

    plt.tight_layout()
    fig.canvas.draw()
    _draw_proportion_bar(ax, df, bar_width=0.18, bar_height=0.012, fontsize=8,
                         color_palette=color_palette)
    plt.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_semantic_facet_generic(df, output_path, global_sc, ch, color_palette):
    all_macros = sorted(df["Macro_Category"].unique())
    fig, axes = plt.subplots(2, 4, figsize=(24, 12), sharex=True, sharey=True)
    axes_flat = axes.flatten()

    for idx, dynasty in enumerate(DYNASTY_ORDER):
        ax = axes_flat[idx]
        sub = df[df["dynasty"] == dynasty]
        if sub.empty:
            ax.set_visible(False)
            continue

        for macro in all_macros:
            bg = df[df["Macro_Category"] == macro]
            if bg.empty:
                continue
            color = color_palette.get(macro, "#888888")
            ax.scatter(bg["PC1"], bg["PC2"], s=3, color=color,
                       alpha=0.07, edgecolors="none", zorder=1)

        for macro in all_macros:
            msub = sub[sub["Macro_Category"] == macro]
            if msub.empty:
                continue
            is_noise = macro == "NOISE"
            color = color_palette.get(macro, "#888888")
            ax.scatter(msub["PC1"], msub["PC2"],
                       s=5 if is_noise else 16, color=color,
                       alpha=0.25 if is_noise else 0.85,
                       edgecolors="none", zorder=2)

        display = DYNASTY_DISPLAY.get(dynasty, dynasty)
        ax.set_title(f"{display}  (n={len(sub):,})", fontsize=12, fontweight="bold", pad=5)

        if global_sc is not None:
            ax.text(0.98, 0.03, f"SC = {global_sc:.4f}",
                    transform=ax.transAxes, fontsize=7, color="#444444",
                    va="bottom", ha="right",
                    bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="#aaaaaa",
                              alpha=0.8, linewidth=0.5), zorder=6)

        ax.set_xlabel("PC1", fontsize=8)
        ax.set_ylabel("PC2", fontsize=8)
        ax.tick_params(labelsize=7)

        fig.canvas.draw()
        _draw_proportion_bar(ax, sub, bar_width=0.22, bar_height=0.010, fontsize=5.5,
                             color_palette=color_palette)

    legend_patches = [
        mpatches.Patch(color=color_palette.get(m, "#888888"), label=m.replace("_", " "))
        for m in sorted(color_palette.keys()) if m in df["Macro_Category"].values
    ]
    fig.legend(handles=legend_patches, title="Macro Category",
               title_fontsize=12, fontsize=11, loc="lower center",
               ncol=len(legend_patches), bbox_to_anchor=(0.5, -0.02), frameon=True)

    fig.suptitle(f"Diachronic Semantic Evolution of {_char_display(ch)} — Global Coordinate System",
                 fontsize=17, fontweight="bold", y=1.01)
    plt.tight_layout(rect=[0, 0.04, 1, 1])
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_semantic_single_dynasty_generic(sub, df_all, dynasty, output_path,
                                         global_sc, ch, color_palette):
    fig, ax = plt.subplots(figsize=(11, 8.5))

    for macro in sorted(df_all["Macro_Category"].unique()):
        bg = df_all[df_all["Macro_Category"] == macro]
        if bg.empty:
            continue
        color = color_palette.get(macro, "#888888")
        ax.scatter(bg["PC1"], bg["PC2"], s=4, color=color,
                   alpha=0.05, edgecolors="none", zorder=1)

    for macro in sorted(sub["Macro_Category"].unique()):
        msub = sub[sub["Macro_Category"] == macro]
        if msub.empty:
            continue
        is_noise = macro == "NOISE"
        color = color_palette.get(macro, "#888888")
        ax.scatter(msub["PC1"], msub["PC2"],
                   s=8 if is_noise else 26, color=color,
                   alpha=0.30 if is_noise else 0.88,
                   edgecolors="none", zorder=2)

    data_span = max(df_all["PC1"].max() - df_all["PC1"].min(),
                    df_all["PC2"].max() - df_all["PC2"].min())
    all_centroids = {}
    for (c_id, macro, sub_cat), grp in df_all.groupby(["Cluster_ID", "Macro_Category", "Sub_Category"]):
        if macro == "NOISE":
            continue
        all_centroids[(c_id, macro, sub_cat)] = np.array([grp["PC1"].mean(), grp["PC2"].mean()])

    present_keys = set(
        (int(r["Cluster_ID"]), r["Macro_Category"], r["Sub_Category"])
        for _, r in sub.iterrows() if r["Macro_Category"] != "NOISE"
    )
    visible = {k: v for k, v in all_centroids.items()
               if (int(k[0]), k[1], k[2]) in present_keys}

    pad = data_span * 0.03
    x_bounds = (df_all["PC1"].min() - pad, df_all["PC1"].max() + pad)
    y_bounds = (df_all["PC2"].min() - pad, df_all["PC2"].max() + pad)
    ax.set_xlim(x_bounds)
    ax.set_ylim(y_bounds)

    label_pos = _repel_labels(visible, data_span,
                               offset_base_ratio=0.10, min_label_dist_ratio=0.16,
                               min_node_dist_ratio=0.12, iterations=600, strength=0.06,
                               x_bounds=x_bounds, y_bounds=y_bounds)

    for (c_id, macro, sub_cat), node_xy in visible.items():
        color = color_palette.get(macro, "#444444")
        lp = label_pos[(c_id, macro, sub_cat)]
        ax.scatter(*node_xy, s=55, color=color, edgecolors="white", linewidths=1.2, zorder=4)
        ax.annotate("", xy=node_xy, xytext=lp,
                    arrowprops=dict(arrowstyle="-", color="black", lw=0.7, alpha=0.5), zorder=4)
        ax.text(*lp, f"C{int(c_id)} ({sub_cat})",
                fontsize=10, color=color, fontweight="bold",
                ha="center", va="center",
                bbox=dict(boxstyle="round,pad=0.2", fc="white", ec=color,
                          alpha=0.88, linewidth=0.8), zorder=5)

    display = DYNASTY_DISPLAY.get(dynasty, dynasty)
    ax.set_title(f"{_char_display(ch)} — {display} on Global Terrain  (n={len(sub):,})",
                 fontsize=15, fontweight="bold", pad=10)

    if global_sc is not None:
        ax.text(0.02, 0.03, f"Silhouette Score = {global_sc:.4f}",
                transform=ax.transAxes, fontsize=9.5, color="#333333",
                va="bottom", ha="left",
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#aaaaaa",
                          alpha=0.85, linewidth=0.7), zorder=6)

    legend_patches = [
        mpatches.Patch(color=color_palette.get(m, "#888888"), label=m.replace("_", " "))
        for m in sorted(color_palette.keys()) if m in sub["Macro_Category"].values
    ]
    ax.legend(handles=legend_patches, title="Macro Category",
              title_fontsize=10, fontsize=9, loc="upper left", framealpha=0.88)

    ax.set_xlabel("PC1", fontsize=11)
    ax.set_ylabel("PC2", fontsize=11)
    ax.tick_params(labelsize=9)
    ax.grid(True, linestyle="--", alpha=0.25)

    plt.tight_layout()
    fig.canvas.draw()
    _draw_proportion_bar(ax, sub, bar_width=0.20, bar_height=0.013, fontsize=8,
                         color_palette=color_palette)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_semantic_per_dynasty_generic(df, out_dir, global_sc, ch, color_palette, suffix):
    ensure_dir(out_dir)
    for dynasty in DYNASTY_ORDER:
        sub = df[df["dynasty"] == dynasty]
        if sub.empty:
            continue
        plot_semantic_single_dynasty_generic(
            sub, df, dynasty, out_dir / f"{dynasty}{suffix}", global_sc, ch, color_palette)


def plot_global_trajectory(df, out_png, title):
    if "dynasty" not in df.columns:
        raise ValueError("Input CSV must include 'dynasty' column")
    cent = df.groupby("dynasty")[["PC1", "PC2"]].mean().reset_index()
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(cent["PC1"], cent["PC2"], marker="o")
    for _, r in cent.iterrows():
        ax.text(r["PC1"], r["PC2"], str(r["dynasty"]), fontsize=9)
    ax.set_title(title)
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_png, dpi=180)
    plt.close(fig)


# ─────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Stage 7: attach Global semantic labels (if available) and plot.")
    parser.add_argument("--char", type=str, required=True)
    parser.add_argument("--global-dir", type=str, required=True,
                        help="char_<ch>_k.../ directory from the global stage "
                             "(contains <ch>_joint_merged.csv and joint_stats.json)")
    parser.add_argument("--output", type=str, required=True,
                        help="Output root; a char_<ch>/ subdirectory is created inside it")
    parser.add_argument("--workdir", type=str, default=".",
                        help="Base directory for resolving a relative --label-map-json")

    parser.add_argument("--hand-label-version", type=str, default="auto",
                        choices=["auto", "0221", "0222"])
    parser.add_argument("--hand-label-map-json", type=str, default="")
    parser.add_argument("--label-map-json", type=str, default="")
    args = parser.parse_args()

    ch = args.char
    chars = [ch]
    workdir = Path(args.workdir).resolve()
    global_dir = Path(args.global_dir)
    out_dir = Path(args.output) / f"char_{ch}"

    global_csv = global_dir / f"{ch}_joint_merged.csv"
    if not global_csv.exists():
        print(f"[ERROR] global csv missing: {global_csv}")
        sys.exit(1)

    df_g = pd.read_csv(global_csv)
    ccg = find_cluster_column(df_g)
    ensure_dir(out_dir)

    mapping_g = resolve_label_mapping_scoped(args, workdir, ch, chars, "joint")
    if mapping_g is None and ch == "手":
        mapping_g = resolve_hand_label_mapping(args, workdir)

    if mapping_g is not None:
        df_g2 = apply_cluster_labels(df_g, ccg, mapping_g)
        pal_g = build_color_palette_for_mapping(mapping_g)
        global_sc_val = load_global_silhouette(global_dir / "joint_stats.json")
        plot_semantic_facet_generic(
            df_g2, out_dir / "Final_Semantic_Evolution_Global_v2.png",
            global_sc_val, ch, pal_g)
        plot_semantic_joint_generic(
            df_g2, out_dir / "joint_clusters_labeled_viz_v2.png",
            global_sc_val, ch, pal_g)
        plot_semantic_per_dynasty_generic(
            df_g2, out_dir / "per_dynasties_global_v2",
            global_sc_val, ch, pal_g, suffix="_Global_v2.png")
        plot_global_trajectory(
            df_g2, out_dir / "Trajectory_Global.png", f"Trajectory - {ch}")
        df_g2.to_csv(out_dir / f"{ch}_Labeled_Global.csv",
                     index=False, encoding="utf-8-sig")
        print(f"[{ch}] global semantic plot ok (labeled)")
    else:
        df_g2 = simple_label_df(df_g, ccg)
        df_g2.to_csv(out_dir / f"{ch}_Labeled_Global.csv",
                     index=False, encoding="utf-8-sig")
        plot_facet(df_g2, ccg,
                   out_dir / "Final_Semantic_Evolution_Global_v2.png",
                   f"Global - {ch}")
        plot_per_dynasty(df_g2, ccg, out_dir / "per_dynasties_global", f"Global {ch}")
        plot_global_trajectory(
            df_g2, out_dir / "Trajectory_Global.png", f"Trajectory - {ch}")
        print(f"[{ch}] global plot ok (plain — no joint labels resolved)")


if __name__ == "__main__":
    main()

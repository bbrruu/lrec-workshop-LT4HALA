#!/usr/bin/env python3
"""
local_annotate.py

Stage 8 of the pipeline. Owns *all* Local output — the single source of truth
for Local plots, symmetric with global_plot.py owning all Global output:

  - Joint (Global) labels resolvable: derive per-(dynasty, Local_Cluster) labels
    by sentence-level join + majority vote against the Global assignments
    (`_derive_local_labels_from_join`), then apply any explicit "local" override
    from --label-map-json on top (`_apply_local_labels`). The override is a full
    (Macro, Sub) replacement, not a Sub-only patch — 手's own released data shows
    Macro can legitimately diverge from the join-derived default too. No override
    given -> pure join-inheritance, which is what 道/水 actually used.
  - Joint labels not resolvable yet: plain, cluster-ID-colored fallback, so this
    stage always produces *something* to look at instead of silently skipping
    the character.

Usage:
    python3 code/local_annotate.py --char 道 \\
        --local-dir  stage_outputs/05_local_output/char_道_Local_k20_split0.3_DynamicSubK \\
        --global-dir stage_outputs/06_joint_output/char_道_k20_subk6_split0.1 \\
        --output stage_outputs/08_local_annotated \\
        --label-map-json configs/dao_shui_labels.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from pipeline_common import (
    DYNASTY_ORDER,
    _char_display,
    _apply_local_labels,
    _derive_local_labels_from_join,
    _draw_proportion_bar,
    _load_per_dynasty_silhouette,
    build_color_palette_for_mapping,
    ensure_dir,
    find_cluster_column,
    plot_facet,
    plot_per_dynasty,
    resolve_hand_label_mapping,
    resolve_label_mapping_scoped,
    resolve_local_label_overrides,
    simple_label_df,
)

try:
    from adjustText import adjust_text
    _ADJUSTTEXT_AVAILABLE = True
except ImportError:
    _ADJUSTTEXT_AVAILABLE = False


# ─────────────────────────────────────────────────────────────
# Labeled Local plots
# ─────────────────────────────────────────────────────────────
def _plot_local_annotate_facet(df: pd.DataFrame, char: str, output_path: Path,
                                color_palette, sil_scores) -> None:
    """2×4 facet grid for local annotated clusters.
    Each dynasty has independent axes (not shared). Background = gray dots of that dynasty.
    Uses adjustText for cluster label repulsion.
    """
    pinyin_display = _char_display(char)
    fig, axes = plt.subplots(2, 4, figsize=(24, 12), sharex=False, sharey=False)

    for idx, dynasty in enumerate(DYNASTY_ORDER):
        ax = axes.flatten()[idx]
        sub = df[df["dynasty"] == dynasty]
        if sub.empty:
            ax.set_visible(False)
            continue

        ax.scatter(sub["PC1"], sub["PC2"], s=5, color="lightgray",
                   alpha=0.15, edgecolors="none", zorder=1)

        for macro, mrows in sub.groupby("Macro_Category"):
            color = color_palette.get(macro, "#888888")
            is_noise = macro == "NOISE"
            ax.scatter(mrows["PC1"], mrows["PC2"],
                       s=5 if is_noise else 12,
                       color=color,
                       alpha=0.25 if is_noise else 0.65,
                       edgecolors="none", zorder=2)

        texts = []
        for (cid, macro, sub_cat), grp in sub.groupby(
                ["Cluster_ID", "Macro_Category", "Sub_Category"]):
            if macro == "NOISE":
                continue
            cx, cy = grp["PC1"].mean(), grp["PC2"].mean()
            color = color_palette.get(macro, "#444444")
            txt = ax.text(cx, cy, f"C{int(cid)} ({sub_cat})",
                          fontsize=6.5, color=color, fontweight="bold",
                          ha="center", va="center",
                          bbox=dict(boxstyle="round,pad=0.15", fc="white",
                                    ec=color, alpha=0.75, linewidth=0.6),
                          zorder=3)
            texts.append(txt)

        if _ADJUSTTEXT_AVAILABLE and texts:
            adjust_text(texts, ax=ax,
                        arrowprops=dict(arrowstyle="-", color="gray", lw=0.5))

        display = dynasty
        n = len(sub)
        sil = sil_scores.get(dynasty)
        ax.set_title(f"{display}  (n={n:,})", fontsize=12, fontweight="bold", pad=5)

        if sil is not None:
            ax.text(0.02, 0.03, f"Silhouette = {sil:.4f}",
                    transform=ax.transAxes, fontsize=7, color="#444444",
                    va="bottom", ha="left",
                    bbox=dict(boxstyle="round,pad=0.2", fc="white",
                              ec="#aaaaaa", alpha=0.8, linewidth=0.5), zorder=6)

        ax.set_xlabel("PC1", fontsize=8)
        ax.set_ylabel("PC2", fontsize=8)
        ax.tick_params(labelsize=7)

        fig.canvas.draw()
        _draw_proportion_bar(ax, sub, bar_width=0.22, bar_height=0.010,
                             fontsize=5.5, color_palette=color_palette)

    all_macros = set(df["Macro_Category"].unique())
    legend_patches = [
        mpatches.Patch(color=color_palette.get(m, "#888888"), label=m.replace("_", " "))
        for m in sorted(color_palette.keys()) if m in all_macros
    ]
    fig.legend(handles=legend_patches, title="Macro Category",
               title_fontsize=11, fontsize=10,
               loc="lower center", ncol=min(len(legend_patches), 6),
               bbox_to_anchor=(0.5, -0.02), frameon=True)

    fig.suptitle(
        f"Diachronic Semantic Evolution of {pinyin_display} — Local Clusters",
        fontsize=17, fontweight="bold", y=1.01)
    plt.tight_layout(rect=[0, 0.04, 1, 1])
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _plot_local_annotate_single(sub: pd.DataFrame, char: str, dynasty: str,
                                 output_path: Path, color_palette, sil) -> None:
    """Full-size single-dynasty plot for local annotated clusters."""
    pinyin_display = _char_display(char)
    fig, ax = plt.subplots(figsize=(11, 8.5))

    ax.scatter(sub["PC1"], sub["PC2"], s=10, color="lightgray",
               alpha=0.2, edgecolors="none", zorder=1)

    for macro, mrows in sub.groupby("Macro_Category"):
        color = color_palette.get(macro, "#888888")
        is_noise = macro == "NOISE"
        ax.scatter(mrows["PC1"], mrows["PC2"],
                   s=8 if is_noise else 22,
                   color=color,
                   alpha=0.25 if is_noise else 0.72,
                   edgecolors="none", zorder=2)

    texts = []
    for (cid, macro, sub_cat), grp in sub.groupby(
            ["Cluster_ID", "Macro_Category", "Sub_Category"]):
        if macro == "NOISE":
            continue
        cx, cy = grp["PC1"].mean(), grp["PC2"].mean()
        color = color_palette.get(macro, "#444444")
        txt = ax.text(cx, cy, f"C{int(cid)} ({sub_cat})",
                      fontsize=10, color=color, fontweight="bold",
                      ha="center", va="center",
                      bbox=dict(boxstyle="round,pad=0.22", fc="white",
                                ec=color, alpha=0.85, linewidth=0.8),
                      zorder=3)
        texts.append(txt)

    if _ADJUSTTEXT_AVAILABLE and texts:
        adjust_text(texts, ax=ax,
                    arrowprops=dict(arrowstyle="-", color="gray", lw=0.7))

    n = len(sub)
    ax.set_title(f"{pinyin_display} — {dynasty}  (n={n:,})",
                 fontsize=15, fontweight="bold", pad=10)

    if sil is not None:
        ax.text(0.02, 0.03, f"Silhouette Score = {sil:.4f}",
                transform=ax.transAxes, fontsize=9.5, color="#333333",
                va="bottom", ha="left",
                bbox=dict(boxstyle="round,pad=0.3", fc="white",
                          ec="#aaaaaa", alpha=0.85, linewidth=0.7), zorder=6)

    all_macros = set(sub["Macro_Category"].unique())
    legend_patches = [
        mpatches.Patch(color=color_palette.get(m, "#888888"), label=m.replace("_", " "))
        for m in sorted(color_palette.keys()) if m in all_macros
    ]
    ax.legend(handles=legend_patches, title="Macro Category",
              title_fontsize=10, fontsize=9, loc="upper left", framealpha=0.88)

    ax.set_xlabel("PC1", fontsize=11)
    ax.set_ylabel("PC2", fontsize=11)
    ax.tick_params(labelsize=9)

    plt.tight_layout()
    fig.canvas.draw()
    _draw_proportion_bar(ax, sub, bar_width=0.20, bar_height=0.013,
                         fontsize=8, color_palette=color_palette)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ─────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Stage 8: derive/apply Local labels (or plain fallback) and plot.")
    parser.add_argument("--char", type=str, required=True)
    parser.add_argument("--local-dir", type=str, required=True,
                        help="char_<ch>_Local_.../ directory from the local stage")
    parser.add_argument("--global-dir", type=str, required=True,
                        help="char_<ch>_k.../ directory from the global stage "
                             "(only needed when Global labels are resolvable)")
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
    local_dir = Path(args.local_dir)
    global_dir = Path(args.global_dir)
    out_ch = Path(args.output) / f"char_{ch}"
    ensure_dir(out_ch)

    local_csv = local_dir / f"{ch}_Local_Dynamic_Merged.csv"
    if not local_csv.exists():
        print(f"[ERROR] local csv missing: {local_csv}")
        sys.exit(1)
    local_df = pd.read_csv(local_csv)

    joint_mapping = resolve_label_mapping_scoped(args, workdir, ch, chars, "joint")
    if joint_mapping is None and ch == "手":
        joint_mapping = resolve_hand_label_mapping(args, workdir)

    if joint_mapping is None:
        # No Global labels to inherit from yet — plain fallback so this stage
        # always produces something, instead of skipping the character outright.
        cc = find_cluster_column(local_df)
        df_l2 = simple_label_df(local_df, cc)
        plot_facet(df_l2, cc,
                   out_ch / "Final_Semantic_Evolution_Local_v2.png",
                   f"Local - {ch}")
        plot_per_dynasty(df_l2, cc, out_ch / "per_dynasties", f"Local {ch}")
        df_l2.to_csv(out_ch / f"{ch}_Labeled_Local.csv",
                     index=False, encoding="utf-8-sig")
        print(f"[{ch}] local plot ok (plain — no joint labels resolved; "
              f"fill in --label-map-json's 'joint' section for {ch} to get semantic labels)")
        return

    global_csv = global_dir / f"{ch}_joint_merged.csv"
    if not global_csv.exists():
        print(f"[ERROR] global csv missing: {global_csv}")
        sys.exit(1)
    global_df = pd.read_csv(global_csv)

    label_map, report, unmatched = _derive_local_labels_from_join(
        local_df, global_df, joint_mapping)
    print(f"[{ch}] join unmatched={unmatched:,}, clusters_derived={len(label_map)}")

    mixed = [k for k, v in report.items() if v["is_mixed"]]
    if mixed:
        print(f"[{ch}] mixed clusters ({len(mixed)}, majority < 60%): {mixed}")

    local_overrides = resolve_local_label_overrides(args, workdir, ch)
    if local_overrides:
        print(f"[{ch}] local label overrides applied: {len(local_overrides)} "
              f"(dynasty, cluster) pairs from --label-map-json")

    labeled_df = _apply_local_labels(local_df, label_map, local_overrides)

    palette_values = list(joint_mapping.values()) + list((local_overrides or {}).values())
    color_palette = build_color_palette_for_mapping(dict(enumerate(palette_values)))

    sil_scores = _load_per_dynasty_silhouette(local_dir / "merged_stats.json")

    per_dyn = out_ch / "per_dynasties"
    ensure_dir(per_dyn)

    csv_out = out_ch / f"{ch}_Labeled_Local.csv"
    labeled_df.to_csv(csv_out, index=False, encoding="utf-8-sig")

    report_path = out_ch / f"{ch}_label_mapping_report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    facet_path = out_ch / f"{ch}_Facet_Grid_Local.png"
    _plot_local_annotate_facet(labeled_df, ch, facet_path, color_palette, sil_scores)

    for dynasty in DYNASTY_ORDER:
        sub = labeled_df[labeled_df["dynasty"] == dynasty]
        if sub.empty:
            continue
        _plot_local_annotate_single(
            sub, ch, dynasty, per_dyn / f"{dynasty}_Labeled.png",
            color_palette, sil_scores.get(dynasty))

    print(f"[{ch}] local semantic plot ok (labeled)")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
final_pipeline.py
=================
All-in-one pipeline: extract → clean1 → sample → clean2 → local → global → postprocess → local_annotate

Differences from 0331_all_in_one_pipeline.py:
- Output root:  final_pipeline_results/
- Stage 8 "local_annotate":
    Derives local cluster semantic labels from validated global labels via
    sentence-level join + majority vote, then generates annotated scatter plots.
    Requires --label-map-json with the "joint" section filled in for each character.

Usage:
    # Full pipeline (extract through local_annotate)
    python3 final_pipeline.py --chars 道,水 --run-name run_01 \\
        --label-map-json label_templates/dao_shui_labels.json

    # Only the local_annotate stage (after labels are filled in)
    python3 final_pipeline.py --chars 道,水 --run-name run_01 \\
        --start-stage local_annotate --end-stage local_annotate \\
        --label-map-json label_templates/dao_shui_labels.json

    # Extract through postprocess (skip local_annotate for now)
    python3 final_pipeline.py --chars 天 --run-name run_tian \\
        --start-stage extract --end-stage postprocess
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
import traceback
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

# Stage scripts (extract_keyword.py, clean_ocr_errors.py, stratified_sampling.py,
# clean_data.py, local_clustering.py, global_clustering.py) live next to this file
# in code/, independent of --workdir (which locates run outputs and label configs).
SCRIPT_DIR = Path(__file__).resolve().parent

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.font_manager as _fm

# ─────────────────────────────────────────────────────────────
# CJK font setup
# ─────────────────────────────────────────────────────────────
for _cjk_fp in [
    "/usr/share/fonts/opentype/NotoSansCJKtc-Regular.otf",
    "/usr/share/fonts/opentype/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf",
    "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc",
    "/usr/share/fonts/truetype/arphic/uming.ttc",
]:
    if os.path.exists(_cjk_fp):
        _fm.fontManager.addfont(_cjk_fp)
        _cjk_name = _fm.FontProperties(fname=_cjk_fp).get_name()
        matplotlib.rcParams.update({
            "font.family": "sans-serif",
            "font.sans-serif": [_cjk_name] + list(matplotlib.rcParams["font.sans-serif"]),
            "axes.unicode_minus": False,
        })
        break

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pandas as pd
import seaborn as sns

try:
    from adjustText import adjust_text
    _ADJUSTTEXT_AVAILABLE = True
except ImportError:
    _ADJUSTTEXT_AVAILABLE = False

# ─────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────
STAGE_ORDER = [
    "extract", "clean1", "sample", "clean2",
    "local", "global", "postprocess", "local_annotate",
]

DYNASTY_ORDER = [
    "pre-qin", "qinhan", "weijin", "suitang",
    "songyuan", "ming", "qing", "republican",
]
DYNASTY_DISPLAY = {
    "pre-qin":    "Pre-Qin",
    "qinhan":     "Qin-Han",
    "weijin":     "Wei-Jin",
    "suitang":    "Sui-Tang",
    "songyuan":   "Song-Yuan",
    "ming":       "Ming",
    "qing":       "Qing",
    "republican": "Republican",
}

# Semantic mapping for 手 (kept for backward compatibility)
HAND_GLOBAL_CLUSTER_LABELS_0222 = {
    0: ("Body_Medical", "Symptom"),
    1: ("Physical_Action", "Holding"),
    2: ("Body_Medical", "Meridian"),
    3: ("Grammar_Suffix", "Occupation"),
    4: ("Text_Culture", "Letter_Edict"),
    5: ("Physical_Action", "Gesture"),
    6: ("Social_Interaction", "Connection"),
    7: ("Physical_Action", "Bimanual"),
    8: ("Physical_Action", "Spatial"),
    9: ("Physical_Action", "Take_Action"),
    10: ("Power_Skill", "Agency"),
    11: ("NOISE", "OCR_Error"),
    12: ("Power_Skill", "Expertise"),
    13: ("NOISE", "OCR_Error"),
    14: ("Text_Culture", "Literary"),
    15: ("Power_Skill", "Capability"),
    16: ("Physical_Action", "State"),
}

HAND_GLOBAL_CLUSTER_LABELS_0221 = HAND_GLOBAL_CLUSTER_LABELS_0222.copy()

HAND_COLOR_PALETTE = {
    "Body_Medical":       "#377eb8",
    "Physical_Action":    "#e41a1c",
    "Social_Interaction": "#ff7f00",
    "Text_Culture":       "#4daf4a",
    "Power_Skill":        "#984ea3",
    "Grammar_Suffix":     "#a65628",
    "NOISE":              "#cccccc",
}

_CHAR_PINYIN: Dict[str, str] = {
    "手": "shǒu",
    "道": "dào",
    "水": "shuǐ",
    "天": "tiān",
    "地": "dì",
    "人": "rén",
}

_EXTRA_COLORS = [
    "#e6ab02", "#66a61e", "#a6761d", "#1b9e77", "#d95f02",
    "#7570b3", "#e7298a", "#66c2a5", "#fc8d62", "#8da0cb",
    "#17becf", "#984ea3", "#4daf4a", "#e41a1c", "#ff7f00",
]


# ─────────────────────────────────────────────────────────────
# Dataclass
# ─────────────────────────────────────────────────────────────
@dataclass
class StageResult:
    stage: str
    success: bool
    elapsed_sec: float
    command: str
    log_file: str
    message: str


# ─────────────────────────────────────────────────────────────
# General utilities
# ─────────────────────────────────────────────────────────────
def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def now_str() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def run_cmd(cmd: List[str], log_path: Path) -> StageResult:
    start = time.time()
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    elapsed = time.time() - start
    log_path.write_text(proc.stdout, encoding="utf-8")
    ok = proc.returncode == 0
    msg = "ok" if ok else f"exit={proc.returncode}"
    return StageResult(
        stage="",
        success=ok,
        elapsed_sec=elapsed,
        command=" ".join(cmd),
        log_file=str(log_path),
        message=msg,
    )


def stage_enabled(stage: str, start_stage: str, end_stage: str) -> bool:
    idx = {name: i for i, name in enumerate(STAGE_ORDER)}
    return idx[start_stage] <= idx[stage] <= idx[end_stage]


def save_json(path: Path, obj: Dict) -> None:
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def find_cluster_column(df: pd.DataFrame) -> str:
    candidates = ["Local_Cluster", "Global_Cluster", "Joint_Cluster",
                  "Cluster", "cluster", "cluster_id", "label", "labels"]
    for c in candidates:
        if c in df.columns:
            return c
    raise ValueError(f"No cluster column found in columns={list(df.columns)}")


def simple_label_df(df: pd.DataFrame, cluster_col: str) -> pd.DataFrame:
    out = df.copy()
    out["Macro_Category"] = "Unlabeled"
    out["Sub_Category"] = out[cluster_col].apply(lambda x: f"C{int(x)}")
    out["Label"] = out["Sub_Category"]
    return out


def _char_display(ch: str) -> str:
    pinyin = _CHAR_PINYIN.get(ch, ch)
    return f"'{pinyin}' {ch}"


# ─────────────────────────────────────────────────────────────
# Label mapping helpers
# ─────────────────────────────────────────────────────────────
def normalize_cluster_mapping(raw: Dict, source: str) -> Dict[int, tuple]:
    mapping: Dict[int, tuple] = {}
    for k, v in raw.items():
        cid = int(k)
        if not isinstance(v, (list, tuple)) or len(v) != 2:
            raise ValueError(f"Invalid mapping at {source} cluster={k}; expected [Macro, Sub]")
        mapping[cid] = (str(v[0]), str(v[1]))
    return mapping


def load_hand_label_mapping_from_json(mapping_path: Path) -> Dict[int, tuple]:
    raw = json.loads(mapping_path.read_text(encoding="utf-8"))
    return normalize_cluster_mapping(raw, str(mapping_path))


def load_multi_char_label_mapping_from_json(mapping_path: Path) -> Dict[str, object]:
    raw = json.loads(mapping_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict) or not raw:
        raise ValueError(f"Invalid label map json: expected non-empty object: {mapping_path}")

    if all(isinstance(v, (list, tuple)) and len(v) == 2 for v in raw.values()):
        return {"__single__": normalize_cluster_mapping(raw, str(mapping_path))}

    out: Dict[str, object] = {}
    for ch, ch_map in raw.items():
        if not isinstance(ch, str) or not ch:
            raise ValueError(f"Invalid char key in label map json: {mapping_path}")
        if not isinstance(ch_map, dict):
            raise ValueError(f"Invalid mapping for char={ch}; expected object")
        if "local" in ch_map or "joint" in ch_map:
            scoped: Dict[str, Dict[int, tuple]] = {}
            for scope in ("local", "joint"):
                if scope in ch_map:
                    scoped[scope] = normalize_cluster_mapping(
                        ch_map[scope], f"{mapping_path}:{ch}:{scope}")
            out[ch] = scoped
        else:
            out[ch] = normalize_cluster_mapping(ch_map, f"{mapping_path}:{ch}")
    return out


def _extract_scoped_mapping(full_entry: object, scope: str) -> Optional[Dict[int, tuple]]:
    if isinstance(full_entry, dict):
        if scope in full_entry:
            return full_entry[scope]  # type: ignore[return-value]
        if all(isinstance(k, int) for k in full_entry.keys()):
            return full_entry  # type: ignore[return-value]
    return None


def resolve_label_mapping_scoped(args, workdir: Path, ch: str, chars: List[str],
                                  scope: str) -> Optional[Dict[int, tuple]]:
    if not args.label_map_json:
        return None
    p = Path(args.label_map_json)
    if not p.is_absolute():
        p = (workdir / p).resolve()
    if not p.exists():
        raise FileNotFoundError(f"label mapping json not found: {p}")

    full = load_multi_char_label_mapping_from_json(p)

    if ch in full:
        return _extract_scoped_mapping(full[ch], scope)

    if "__single__" in full and len(chars) == 1:
        entry = full["__single__"]
        if isinstance(entry, dict) and all(isinstance(k, int) for k in entry.keys()):
            return entry  # type: ignore[return-value]
    return None


def resolve_hand_label_mapping(args, workdir: Path) -> Dict[int, tuple]:
    if args.hand_label_map_json:
        p = Path(args.hand_label_map_json)
        if not p.is_absolute():
            p = (workdir / p).resolve()
        if not p.exists():
            raise FileNotFoundError(f"hand label mapping json not found: {p}")
        return load_hand_label_mapping_from_json(p)
    if args.hand_label_version == "0221":
        return HAND_GLOBAL_CLUSTER_LABELS_0221
    return HAND_GLOBAL_CLUSTER_LABELS_0222


def apply_cluster_labels(df: pd.DataFrame, cluster_col: str,
                         mapping: Dict[int, tuple]) -> pd.DataFrame:
    out = df.copy()
    mapped = out[cluster_col].apply(lambda v: mapping.get(int(v), ("Other", "Other")))
    out["Cluster_ID"] = out[cluster_col]
    out["Macro_Category"] = mapped.apply(lambda x: x[0])
    out["Sub_Category"] = mapped.apply(lambda x: x[1])
    out["Label"] = out["Sub_Category"]
    return out


def build_color_palette_for_mapping(mapping: Dict[int, tuple]) -> Dict[str, str]:
    macros = sorted(set(v[0] for v in mapping.values()))
    palette: Dict[str, str] = dict(HAND_COLOR_PALETTE)
    new_macros = [m for m in macros if m not in palette]
    for i, macro in enumerate(new_macros):
        palette[macro] = _EXTRA_COLORS[i % len(_EXTRA_COLORS)]
    return palette


def load_global_silhouette(stats_path: Path) -> Optional[float]:
    if not stats_path.exists():
        return None
    try:
        stats = json.loads(stats_path.read_text(encoding="utf-8"))
        val = stats.get("silhouette")
        return float(val) if val is not None else None
    except Exception:
        return None


# ─────────────────────────────────────────────────────────────
# Shared plot helper: proportion bar (top-right corner)
# ─────────────────────────────────────────────────────────────
def _draw_proportion_bar(ax, sub_df: pd.DataFrame,
                         bar_width: float = 0.20,
                         bar_height: float = 0.013,
                         fontsize: float = 8.0,
                         color_palette: Optional[Dict[str, str]] = None) -> None:
    pal = color_palette if color_palette is not None else HAND_COLOR_PALETTE
    counts = sub_df.groupby("Macro_Category").size()
    total = counts.sum()
    if total == 0:
        return
    props = (counts / total).sort_values(ascending=False)

    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    xr = xlim[1] - xlim[0]
    yr = ylim[1] - ylim[0]
    bar_w = bar_width * xr
    bar_h = bar_height * yr
    bar_x0 = xlim[1] - bar_w - 0.01 * xr
    bar_y0 = ylim[1] - 0.02 * yr

    x_cur = bar_x0
    for macro, prop in props.items():
        color = pal.get(macro, "#888888")
        rect = plt.Rectangle((x_cur, bar_y0 - bar_h), prop * bar_w, bar_h,
                              color=color, transform=ax.transData, zorder=5)
        ax.add_patch(rect)
        x_cur += prop * bar_w

    ax.add_patch(plt.Rectangle((bar_x0, bar_y0 - bar_h), bar_w, bar_h,
                               fill=False, edgecolor="#333333", linewidth=0.6,
                               transform=ax.transData, zorder=6))

    text_y = bar_y0 - bar_h - 0.005 * yr
    line_step = 0.025 * yr
    for macro, prop in props.items():
        if macro == "NOISE" or prop < 0.01:
            continue
        color = pal.get(macro, "#888888")
        ax.text(bar_x0 + bar_w, text_y,
                f"■ {macro.replace('_', ' ')}: {prop * 100:.0f}%",
                fontsize=fontsize, color=color,
                ha="right", va="top", transform=ax.transData,
                zorder=6, fontweight="bold")
        text_y -= line_step


# ─────────────────────────────────────────────────────────────
# Label repulsion (for global postprocess plots)
# ─────────────────────────────────────────────────────────────
def _repel_labels(positions: Dict, data_span: float,
                  offset_base_ratio: float = 0.10,
                  min_label_dist_ratio: float = 0.16,
                  min_node_dist_ratio: float = 0.12,
                  iterations: int = 500,
                  strength: float = 0.06,
                  x_bounds: Optional[tuple] = None,
                  y_bounds: Optional[tuple] = None) -> Dict:
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
# Global postprocess plots (generic, character-agnostic)
# ─────────────────────────────────────────────────────────────
def plot_semantic_joint_generic(df: pd.DataFrame, output_path: Path,
                                global_sc: Optional[float], ch: str,
                                color_palette: Dict[str, str]) -> None:
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

    centroids: Dict = {}
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


def plot_semantic_facet_generic(df: pd.DataFrame, output_path: Path,
                                global_sc: Optional[float], ch: str,
                                color_palette: Dict[str, str]) -> None:
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


def plot_semantic_single_dynasty_generic(sub: pd.DataFrame, df_all: pd.DataFrame,
                                         dynasty: str, output_path: Path,
                                         global_sc: Optional[float], ch: str,
                                         color_palette: Dict[str, str]) -> None:
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
    all_centroids: Dict = {}
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


def plot_semantic_per_dynasty_generic(df: pd.DataFrame, out_dir: Path,
                                      global_sc: Optional[float], ch: str,
                                      color_palette: Dict[str, str], suffix: str) -> None:
    ensure_dir(out_dir)
    for dynasty in DYNASTY_ORDER:
        sub = df[df["dynasty"] == dynasty]
        if sub.empty:
            continue
        plot_semantic_single_dynasty_generic(
            sub, df, dynasty, out_dir / f"{dynasty}{suffix}", global_sc, ch, color_palette)


def plot_facet(df: pd.DataFrame, cluster_col: str, out_png: Path, title: str) -> None:
    if "dynasty" not in df.columns:
        raise ValueError("Input CSV must include 'dynasty' column")
    g = sns.FacetGrid(df, col="dynasty", col_wrap=4, height=3.5, sharex=True, sharey=True)
    g.map_dataframe(sns.scatterplot, x="PC1", y="PC2", hue=cluster_col,
                    s=8, alpha=0.6, linewidth=0)
    g.set_titles("{col_name}")
    g.fig.suptitle(title, y=1.02)
    g.add_legend()
    g.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close(g.fig)


def plot_per_dynasty(df: pd.DataFrame, cluster_col: str, out_dir: Path, prefix: str) -> None:
    ensure_dir(out_dir)
    for dyn, sub in df.groupby("dynasty"):
        fig, ax = plt.subplots(figsize=(7, 5))
        sns.scatterplot(data=sub, x="PC1", y="PC2", hue=cluster_col,
                        s=12, alpha=0.7, linewidth=0, ax=ax)
        ax.set_title(f"{prefix} - {dyn}")
        fig.tight_layout()
        fig.savefig(out_dir / f"{dyn}.png", dpi=150)
        plt.close(fig)


def plot_global_trajectory(df: pd.DataFrame, out_png: Path, title: str) -> None:
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
# Stage 8: local_annotate — helpers
# ─────────────────────────────────────────────────────────────
def _load_per_dynasty_silhouette(stats_path: Path) -> Dict[str, float]:
    """Read per-dynasty silhouette scores from merged_stats.json.

    Tries several common JSON layouts; returns {} if the file is missing
    or the expected structure is not found.
    """
    if not stats_path.exists():
        return {}
    try:
        data = json.loads(stats_path.read_text(encoding="utf-8"))
        dynasty_set = set(DYNASTY_ORDER)
        result: Dict[str, float] = {}

        # Layout A: {"pre-qin": {"silhouette": 0.04, ...}, ...}
        for k, v in data.items():
            if k in dynasty_set and isinstance(v, dict):
                for key in ("silhouette", "silhouette_score", "sc"):
                    if key in v:
                        try:
                            result[k] = float(v[key])
                        except (TypeError, ValueError):
                            pass
                        break
        if result:
            return result

        # Layout B: {"dynasty_silhouettes": {"pre-qin": 0.04, ...}}
        for sub_key in ("dynasty_silhouettes", "per_dynasty",
                        "silhouette_per_dynasty", "dynasties"):
            if sub_key in data and isinstance(data[sub_key], dict):
                for k, v in data[sub_key].items():
                    if k in dynasty_set:
                        try:
                            result[k] = float(v) if not isinstance(v, dict) else float(
                                v.get("silhouette", v.get("silhouette_score", 0)))
                        except (TypeError, ValueError):
                            pass
                if result:
                    return result
        return {}
    except Exception:
        return {}


def _derive_local_labels_from_join(
        local_df: pd.DataFrame,
        global_df: pd.DataFrame,
        joint_mapping: Dict[int, tuple],
) -> tuple:
    """Join local + global CSVs on (dynasty, sentence), majority-vote the global cluster
    assignment per (dynasty, Local_Cluster), and inherit the validated global label.

    Returns:
        label_map : dict[(dynasty, local_cid)] = (Macro, Sub)
        report    : dict  full breakdown per (dynasty, local_cid)
        unmatched : int   number of local sentences with no global match
    """
    # Build (dynasty, sentence) → Global_Cluster lookup
    global_lookup: Dict[tuple, int] = {}
    for _, row in global_df.iterrows():
        global_lookup[(row["dynasty"], row["sentence"])] = int(row["Global_Cluster"])

    # Annotate each local row with its matched Global_Cluster
    local_df = local_df.copy()
    local_df["_GlobalCluster"] = local_df.apply(
        lambda r: global_lookup.get((r["dynasty"], r["sentence"])), axis=1)

    unmatched = int(local_df["_GlobalCluster"].isna().sum())
    matched = local_df.dropna(subset=["_GlobalCluster"]).copy()
    matched["_GlobalCluster"] = matched["_GlobalCluster"].astype(int)

    label_map: Dict[tuple, tuple] = {}
    report: Dict[str, dict] = {}

    for (dyn, local_cid), grp in matched.groupby(["dynasty", "Local_Cluster"]):
        local_cid = int(local_cid)
        counts = grp["_GlobalCluster"].value_counts()
        total = int(counts.sum())
        majority_gcid = int(counts.idxmax())
        majority_count = int(counts.max())
        majority_pct = majority_count / total * 100

        macro, sub = joint_mapping.get(majority_gcid, ("NOISE", f"G{majority_gcid}"))
        label_map[(dyn, local_cid)] = (macro, sub)

        breakdown = [
            {
                "global_cluster": int(g),
                "label": list(joint_mapping.get(int(g), ("NOISE", f"G{g}"))),
                "count": int(c),
                "pct": round(int(c) / total * 100, 1),
            }
            for g, c in counts.items()
        ]
        report[f"{dyn}_L{local_cid}"] = {
            "dynasty":         dyn,
            "local_cluster":   local_cid,
            "total_sentences": total,
            "assigned_label":  list(joint_mapping.get(majority_gcid,
                                                       ("NOISE", f"G{majority_gcid}"))),
            "majority_global": majority_gcid,
            "majority_pct":    round(majority_pct, 1),
            "is_mixed":        majority_pct < 60.0,
            "distribution":    breakdown,
        }

    return label_map, report, unmatched


def _apply_local_labels(local_df: pd.DataFrame,
                        label_map: Dict[tuple, tuple]) -> pd.DataFrame:
    out = local_df.copy()

    def _lookup(row):
        return label_map.get((row["dynasty"], int(row["Local_Cluster"])),
                             ("NOISE", "Unmatched"))

    labels = out.apply(_lookup, axis=1)
    out["Cluster_ID"]     = out["Local_Cluster"].astype(int)
    out["Macro_Category"] = labels.apply(lambda x: x[0])
    out["Sub_Category"]   = labels.apply(lambda x: x[1])
    return out


def _plot_local_annotate_facet(df: pd.DataFrame, char: str, output_path: Path,
                                color_palette: Dict[str, str],
                                sil_scores: Dict[str, float]) -> None:
    """2×4 facet grid for local annotated clusters.
    Each dynasty has independent axes (not shared). Background = gray dots of that dynasty.
    Uses adjustText for cluster label repulsion.
    """
    pinyin = _CHAR_PINYIN.get(char, char)
    fig, axes = plt.subplots(2, 4, figsize=(24, 12), sharex=False, sharey=False)

    for idx, dynasty in enumerate(DYNASTY_ORDER):
        ax = axes.flatten()[idx]
        sub = df[df["dynasty"] == dynasty]
        if sub.empty:
            ax.set_visible(False)
            continue

        # Background gray
        ax.scatter(sub["PC1"], sub["PC2"], s=5, color="lightgray",
                   alpha=0.15, edgecolors="none", zorder=1)

        # Colored scatter per Macro_Category
        for macro, mrows in sub.groupby("Macro_Category"):
            color = color_palette.get(macro, "#888888")
            is_noise = macro == "NOISE"
            ax.scatter(mrows["PC1"], mrows["PC2"],
                       s=5 if is_noise else 12,
                       color=color,
                       alpha=0.25 if is_noise else 0.65,
                       edgecolors="none", zorder=2)

        # Cluster centroid labels
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

        display = DYNASTY_DISPLAY.get(dynasty, dynasty)
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

    # Global legend
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
        f"Diachronic Semantic Evolution of '{char}' ({pinyin}) — Local Clusters",
        fontsize=17, fontweight="bold", y=1.01)
    plt.tight_layout(rect=[0, 0.04, 1, 1])
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _plot_local_annotate_single(sub: pd.DataFrame, char: str, dynasty: str,
                                 output_path: Path, color_palette: Dict[str, str],
                                 sil: Optional[float]) -> None:
    """Full-size single-dynasty plot for local annotated clusters."""
    pinyin = _CHAR_PINYIN.get(char, char)
    fig, ax = plt.subplots(figsize=(11, 8.5))

    # Background gray
    ax.scatter(sub["PC1"], sub["PC2"], s=10, color="lightgray",
               alpha=0.2, edgecolors="none", zorder=1)

    # Colored scatter
    for macro, mrows in sub.groupby("Macro_Category"):
        color = color_palette.get(macro, "#888888")
        is_noise = macro == "NOISE"
        ax.scatter(mrows["PC1"], mrows["PC2"],
                   s=8 if is_noise else 22,
                   color=color,
                   alpha=0.25 if is_noise else 0.72,
                   edgecolors="none", zorder=2)

    # Cluster centroid labels
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

    display = DYNASTY_DISPLAY.get(dynasty, dynasty)
    n = len(sub)
    ax.set_title(f"'{char}' ({pinyin}) — {display}  (n={n:,})",
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
# Interactive prompt helpers
# ─────────────────────────────────────────────────────────────
def prompt_text(label: str, default: str = "") -> str:
    suffix = f" [{default}]" if default else ""
    raw = input(f"{label}{suffix}: ").strip()
    return raw if raw else default


def prompt_yes_no(label: str, default: bool = False) -> bool:
    default_str = "Y/n" if default else "y/N"
    raw = input(f"{label} ({default_str}): ").strip().lower()
    if not raw:
        return default
    return raw in {"y", "yes"}


def prompt_stage(label: str, default: str) -> str:
    while True:
        raw = prompt_text(label + f" options={STAGE_ORDER}", default=default)
        if raw in STAGE_ORDER:
            return raw
        print(f"Invalid stage: {raw}")


def prompt_if_needed(args):
    interactive = (len(sys.argv) == 1) or (not args.chars)
    if not interactive:
        return args

    print("\n=== Final Pipeline (Interactive) ===")
    print("直接輸入即可，按 Enter 使用預設值。\n")

    args.chars = prompt_text("要跑哪些字（逗號分隔）", default=args.chars or "手")
    args.run_name = prompt_text("Run 名稱（可空白自動生成）", default=args.run_name)
    args.workdir = prompt_text("工作目錄", default=args.workdir)
    args.output_root = prompt_text("整合輸出根目錄", default=args.output_root)

    args.start_stage = prompt_stage("起始階段", default=args.start_stage)
    args.end_stage = prompt_stage("結束階段", default=args.end_stage)

    args.dynasties = prompt_text("限定朝代（可空白；逗號或空白分隔）", default=args.dynasties)
    args.extract_device = prompt_text("embedding 裝置（auto/cpu/cuda）", default=args.extract_device)

    max_texts_raw = prompt_text("extract 每朝代最大 text 數（0=不限）",
                                default=str(args.extract_max_texts))
    try:
        args.extract_max_texts = int(max_texts_raw)
    except ValueError:
        args.extract_max_texts = 0

    args.hand_label_version = prompt_text(
        "手字標註版本（auto/0221/0222）", default=args.hand_label_version)
    args.hand_label_map_json = prompt_text(
        "手字自訂標註 JSON（可空白）", default=args.hand_label_map_json)
    args.label_map_json = prompt_text(
        "通用多字標註 JSON（可空白）", default=args.label_map_json)

    args.skip_existing = prompt_yes_no("若結果已存在是否跳過", default=args.skip_existing)

    print("\nInteractive 設定完成，開始執行...\n")
    return args


# ─────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────
def main() -> None:
    parser = argparse.ArgumentParser(description="Final all-in-one pipeline (extract → local_annotate)")
    parser.add_argument("--workdir", type=str, default=".")
    parser.add_argument("--python-bin", type=str, default=sys.executable)
    parser.add_argument("--chars", type=str, default="",
                        help="Comma-separated chars, e.g. 手,道,天")
    parser.add_argument("--dynasties", type=str, default="",
                        help="Optional space/comma list for extract stage")
    parser.add_argument("--run-name", type=str, default="")
    parser.add_argument("--output-root", type=str, default="final_pipeline_results")

    parser.add_argument("--start-stage", choices=STAGE_ORDER, default="extract")
    parser.add_argument("--end-stage",   choices=STAGE_ORDER, default="local_annotate")

    parser.add_argument("--extract-metadata",  type=str, default="compact_new_metadata.csv")
    parser.add_argument("--extract-device",    type=str, default="auto",
                        choices=["auto", "cpu", "cuda"])
    parser.add_argument("--extract-max-texts", type=int, default=0,
                        help="0 means no limit")
    parser.add_argument("--extract-resume", action="store_true")

    parser.add_argument("--clean1-mode", type=str, default="conservative",
                        choices=["conservative", "aggressive", "custom"])
    parser.add_argument("--sample-max-per-toptitle",  type=int, default=200)
    parser.add_argument("--sample-baseline-max",       type=int, default=10000)
    parser.add_argument("--clean2-aggression", type=str, default="normal",
                        choices=["mild", "normal", "aggressive", "ultra"])

    parser.add_argument("--local-k",     type=int,   default=20)
    parser.add_argument("--local-merge", type=float, default=0.95)
    parser.add_argument("--local-split", type=float, default=0.3)

    parser.add_argument("--global-k",     type=int,   default=20)
    parser.add_argument("--global-merge", type=float, default=0.95)
    parser.add_argument("--global-split", type=float, default=0.1)
    parser.add_argument("--global-sub-k", type=int,   default=6)

    parser.add_argument("--hand-label-version", type=str, default="auto",
                        choices=["auto", "0221", "0222"],
                        help="Semantic label profile for char=手")
    parser.add_argument("--hand-label-map-json", type=str, default="",
                        help="Optional JSON for char=手; overrides --hand-label-version")
    parser.add_argument("--label-map-json", type=str, default="",
                        help="Generic JSON with joint labels for all chars "
                             "(required by local_annotate stage)")

    parser.add_argument("--skip-existing", action="store_true")
    args = parser.parse_args()
    args = prompt_if_needed(args)

    if not args.chars:
        print("--chars 為必填（可使用互動模式輸入）")
        sys.exit(2)

    chars    = [c.strip() for c in args.chars.split(",") if c.strip()]
    dynasties = [d.strip() for d in args.dynasties.replace(",", " ").split() if d.strip()]
    run_name = args.run_name or f"run_{now_str()}"

    workdir  = Path(args.workdir).resolve()
    run_root = (workdir / args.output_root / run_name).resolve()
    logs_dir = run_root / "logs"
    out_dir  = run_root / "stage_outputs"
    ensure_dir(logs_dir)
    ensure_dir(out_dir)

    stage_paths = {
        "keyword_embeddings": out_dir / "01_keyword_embeddings",
        "cleaned_embeddings": out_dir / "02_cleaned_embeddings",
        "sampled":            out_dir / "03_stratified_sampling_output_cleaned",
        "clean2":             out_dir / "04_cleaned_output",
        "local":              out_dir / "05_local_output",
        "global":             out_dir / "06_joint_output",
        "post":               out_dir / "07_postprocess",
        "local_annotate":     out_dir / "08_local_annotated",
    }
    for p in stage_paths.values():
        ensure_dir(p)

    config = {
        "run_name":   run_name,
        "chars":      chars,
        "dynasties":  dynasties,
        "start_stage": args.start_stage,
        "end_stage":   args.end_stage,
        "data": {
            "extract_metadata": args.extract_metadata,
            "extract_device": args.extract_device,
            "extract_max_texts": args.extract_max_texts,
            "extract_resume": args.extract_resume,
            "clean1_mode": args.clean1_mode,
            "sample_max_per_toptitle": args.sample_max_per_toptitle,
            "sample_baseline_max": args.sample_baseline_max,
            "clean2_aggression": args.clean2_aggression,
        },
        "baseline": {
            "local":  {"k": args.local_k, "merge": args.local_merge, "split": args.local_split},
            "global": {"k": args.global_k, "merge": args.global_merge,
                       "split": args.global_split, "sub_k": args.global_sub_k},
        },
        "semantic": {
            "hand_label_version":  args.hand_label_version,
            "hand_label_map_json": args.hand_label_map_json,
            "label_map_json":      args.label_map_json,
        },
    }
    save_json(run_root / "config.json", config)

    all_stage_results: Dict[str, Dict] = {}

    def save_stage(name: str, result: StageResult) -> None:
        result.stage = name
        all_stage_results[name] = asdict(result)
        save_json(run_root / "summary.json", {"config": config, "stages": all_stage_results})

    # ── Stage: extract ────────────────────────────────────────
    if stage_enabled("extract", args.start_stage, args.end_stage):
        extract_marker = stage_paths["keyword_embeddings"] / "_extract_done.marker"
        if args.skip_existing and extract_marker.exists():
            save_stage("extract", StageResult("extract", True, 0.0, "", "", "skipped(existing)"))
        else:
            cmd = [
                args.python_bin,
                str((SCRIPT_DIR / "extract_keyword.py").resolve()),
                "--output", str(stage_paths["keyword_embeddings"]),
                "--metadata", args.extract_metadata,
                "--chars", *chars,
                "--device", args.extract_device,
            ]
            if dynasties:
                cmd += ["--dynasties", *dynasties]
            if args.extract_max_texts > 0:
                cmd += ["--max-texts", str(args.extract_max_texts)]
            if args.extract_resume:
                cmd += ["--resume"]
            res = run_cmd(cmd, logs_dir / "extract.log")
            if res.success:
                extract_marker.write_text("ok\n", encoding="utf-8")
            save_stage("extract", res)
            if not res.success:
                sys.exit(1)

    # ── Stage: clean1 ─────────────────────────────────────────
    if stage_enabled("clean1", args.start_stage, args.end_stage):
        start = time.time()
        log_file = logs_dir / "clean1.log"
        combined_log = []
        ok, msg = True, "ok"
        for ch in chars:
            cmd = [
                args.python_bin,
                str((SCRIPT_DIR / "clean_ocr_errors.py").resolve()),
                "--char", ch, "--mode", args.clean1_mode,
                "--input-dir",  str(stage_paths["keyword_embeddings"]),
                "--output-dir", str(stage_paths["cleaned_embeddings"]),
            ]
            sub = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
            combined_log.append(f"### char={ch}\n{sub.stdout}\n")
            if sub.returncode != 0:
                ok, msg = False, f"char={ch} failed exit={sub.returncode}"
                break
        log_file.write_text("\n".join(combined_log), encoding="utf-8")
        res = StageResult("clean1", ok, time.time() - start,
                          "loop clean_ocr_errors.py", str(log_file), msg)
        save_stage("clean1", res)
        if not ok:
            sys.exit(1)

    # ── Stage: sample ─────────────────────────────────────────
    if stage_enabled("sample", args.start_stage, args.end_stage):
        cmd = [
            args.python_bin,
            str((SCRIPT_DIR / "stratified_sampling.py").resolve()),
            "--chars", ",".join(chars),
            "--max-per-toptitle",    str(args.sample_max_per_toptitle),
            "--baseline-max-samples", str(args.sample_baseline_max),
            "--target-chars", ",".join(chars),
            "--cleaned-root", str(stage_paths["cleaned_embeddings"]),
            "--output",       str(stage_paths["sampled"]),
        ]
        res = run_cmd(cmd, logs_dir / "sample.log")
        save_stage("sample", res)
        if not res.success:
            sys.exit(1)

    # ── Stage: clean2 ─────────────────────────────────────────
    if stage_enabled("clean2", args.start_stage, args.end_stage):
        start = time.time()
        log_file = logs_dir / "clean2.log"
        combined_log = []
        ok, msg = True, "ok"
        for ch in chars:
            cmd = [
                args.python_bin,
                str((SCRIPT_DIR / "clean_data.py").resolve()),
                "--char", ch,
                "--input",      str(stage_paths["sampled"]),
                "--output",     str(stage_paths["clean2"]),
                "--aggression", args.clean2_aggression,
            ]
            sub = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
            combined_log.append(f"### char={ch}\n{sub.stdout}\n")
            if sub.returncode != 0:
                ok, msg = False, f"char={ch} failed exit={sub.returncode}"
                break
        log_file.write_text("\n".join(combined_log), encoding="utf-8")
        res = StageResult("clean2", ok, time.time() - start,
                          "loop 0212_clean_data.py", str(log_file), msg)
        save_stage("clean2", res)
        if not ok:
            sys.exit(1)

    # ── Stage: local ──────────────────────────────────────────
    if stage_enabled("local", args.start_stage, args.end_stage):
        cmd = [
            args.python_bin,
            str((SCRIPT_DIR / "local_clustering.py").resolve()),
            "--chars",       ",".join(chars),
            "--input",       str(stage_paths["clean2"]),
            "--output",      str(stage_paths["local"]),
            "--k-init",      str(args.local_k),
            "--merge-ratio", str(args.local_merge),
            "--split-ratio", str(args.local_split),
        ]
        res = run_cmd(cmd, logs_dir / "local.log")
        save_stage("local", res)
        if not res.success:
            sys.exit(1)

    # ── Stage: global ─────────────────────────────────────────
    if stage_enabled("global", args.start_stage, args.end_stage):
        start = time.time()
        log_file = logs_dir / "global.log"
        combined_log = []
        ok, msg = True, "ok"
        for ch in chars:
            cmd = [
                args.python_bin,
                str((SCRIPT_DIR / "global_clustering.py").resolve()),
                "--input",       str(stage_paths["clean2"]),
                "--char",        ch,
                "--output",      str(stage_paths["global"]),
                "--k",           str(args.global_k),
                "--merge-ratio", str(args.global_merge),
                "--sub-k",       str(args.global_sub_k),
                "--split-ratio", str(args.global_split),
            ]
            sub = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
            combined_log.append(f"### char={ch}\n{sub.stdout}\n")
            if sub.returncode != 0:
                ok, msg = False, f"char={ch} failed exit={sub.returncode}"
                break
        log_file.write_text("\n".join(combined_log), encoding="utf-8")
        res = StageResult("global", ok, time.time() - start,
                          "loop 0219_joint_cluster.py", str(log_file), msg)
        save_stage("global", res)
        if not ok:
            sys.exit(1)

    # ── Stage: postprocess ────────────────────────────────────
    if stage_enabled("postprocess", args.start_stage, args.end_stage):
        start = time.time()
        log_lines: List[str] = []
        ok, msg = True, "ok"

        local_post_root  = stage_paths["post"] / "local"
        global_post_root = stage_paths["post"] / "global"
        ensure_dir(local_post_root)
        ensure_dir(global_post_root)

        for ch in chars:
            try:
                # Local ──────────────────────────────────────────────────
                local_dir = (stage_paths["local"] /
                             f"char_{ch}_Local_k{args.local_k}_split{args.local_split}_DynamicSubK")
                local_csv = local_dir / f"{ch}_Local_Dynamic_Merged.csv"
                if local_csv.exists():
                    df_l = pd.read_csv(local_csv)
                    cc = find_cluster_column(df_l)
                    mapping_l = resolve_label_mapping_scoped(args, workdir, ch, chars, "local")
                    out_l_dir = local_post_root / f"char_{ch}"
                    ensure_dir(out_l_dir)

                    if mapping_l is not None:
                        df_l2   = apply_cluster_labels(df_l, cc, mapping_l)
                        pal_l   = build_color_palette_for_mapping(mapping_l)
                        local_sc = load_global_silhouette(local_dir / "local_stats.json")
                        plot_semantic_facet_generic(
                            df_l2, out_l_dir / "Final_Semantic_Evolution_Local_v2.png",
                            local_sc, ch, pal_l)
                        plot_semantic_joint_generic(
                            df_l2, out_l_dir / "local_clusters_labeled_viz_v2.png",
                            local_sc, ch, pal_l)
                        plot_semantic_per_dynasty_generic(
                            df_l2, out_l_dir / "per_dynasties_v2",
                            local_sc, ch, pal_l, suffix="_Local_v2.png")
                        log_lines.append(f"local semantic postprocess ok: {ch}")
                    else:
                        df_l2 = simple_label_df(df_l, cc)
                        plot_facet(df_l2, cc,
                                   out_l_dir / "Final_Semantic_Evolution_Local_v2.png",
                                   f"Local - {ch}")
                        plot_per_dynasty(df_l2, cc, out_l_dir / "per_dynasties", f"Local {ch}")
                        log_lines.append(f"local postprocess ok: {ch}")

                    df_l2.to_csv(out_l_dir / f"{ch}_Labeled_Local.csv",
                                 index=False, encoding="utf-8-sig")
                else:
                    log_lines.append(f"local csv missing: {local_csv}")

                # Global/Joint ───────────────────────────────────────────
                global_dir = (stage_paths["global"] /
                              f"char_{ch}_k{args.global_k}_subk{args.global_sub_k}_split{args.global_split}")
                global_csv = global_dir / f"{ch}_joint_merged.csv"
                if global_csv.exists():
                    df_g  = pd.read_csv(global_csv)
                    ccg   = find_cluster_column(df_g)
                    out_g_dir = global_post_root / f"char_{ch}"
                    ensure_dir(out_g_dir)

                    mapping_g = resolve_label_mapping_scoped(args, workdir, ch, chars, "joint")
                    if mapping_g is None and ch == "手":
                        mapping_g = resolve_hand_label_mapping(args, workdir)

                    if mapping_g is not None:
                        df_g2        = apply_cluster_labels(df_g, ccg, mapping_g)
                        pal_g        = build_color_palette_for_mapping(mapping_g)
                        global_sc_val = load_global_silhouette(global_dir / "joint_stats.json")
                        plot_semantic_facet_generic(
                            df_g2, out_g_dir / "Final_Semantic_Evolution_Global_v2.png",
                            global_sc_val, ch, pal_g)
                        plot_semantic_joint_generic(
                            df_g2, out_g_dir / "joint_clusters_labeled_viz_v2.png",
                            global_sc_val, ch, pal_g)
                        plot_semantic_per_dynasty_generic(
                            df_g2, out_g_dir / "per_dynasties_global_v2",
                            global_sc_val, ch, pal_g, suffix="_Global_v2.png")
                        plot_global_trajectory(
                            df_g2, out_g_dir / "Trajectory_Global.png", f"Trajectory - {ch}")
                        df_g2.to_csv(out_g_dir / f"{ch}_Labeled_Global.csv",
                                     index=False, encoding="utf-8-sig")
                        log_lines.append(f"global semantic postprocess ok: {ch}")
                    else:
                        df_g2 = simple_label_df(df_g, ccg)
                        df_g2.to_csv(out_g_dir / f"{ch}_Labeled_Global.csv",
                                     index=False, encoding="utf-8-sig")
                        plot_facet(df_g2, ccg,
                                   out_g_dir / "Final_Semantic_Evolution_Global_v2.png",
                                   f"Global - {ch}")
                        plot_per_dynasty(df_g2, ccg,
                                         out_g_dir / "per_dynasties_global", f"Global {ch}")
                        plot_global_trajectory(
                            df_g2, out_g_dir / "Trajectory_Global.png", f"Trajectory - {ch}")
                        log_lines.append(f"global postprocess ok: {ch}")
                else:
                    log_lines.append(f"global csv missing: {global_csv}")

            except Exception as e:
                ok, msg = False, f"char={ch} postprocess failed: {e}"
                log_lines.append(msg)
                log_lines.append(traceback.format_exc())
                break

        log_path = logs_dir / "postprocess.log"
        log_path.write_text("\n".join(log_lines), encoding="utf-8")
        res = StageResult("postprocess", ok, time.time() - start,
                          "internal pandas+matplotlib", str(log_path), msg)
        save_stage("postprocess", res)
        if not ok:
            sys.exit(1)

    # ── Stage: local_annotate ─────────────────────────────────
    if stage_enabled("local_annotate", args.start_stage, args.end_stage):
        start = time.time()
        log_lines = []
        ok, msg = True, "ok"

        for ch in chars:
            try:
                # Resolve joint label mapping — required for this stage
                joint_mapping = resolve_label_mapping_scoped(
                    args, workdir, ch, chars, "joint")
                if joint_mapping is None:
                    log_lines.append(
                        f"[{ch}] SKIP local_annotate: no joint labels found in "
                        f"--label-map-json. Fill in the 'joint' section first.")
                    continue

                # Input CSV paths (constructed from pipeline parameters)
                local_dir = (stage_paths["local"] /
                             f"char_{ch}_Local_k{args.local_k}_split{args.local_split}_DynamicSubK")
                local_csv = local_dir / f"{ch}_Local_Dynamic_Merged.csv"
                global_dir = (stage_paths["global"] /
                              f"char_{ch}_k{args.global_k}_subk{args.global_sub_k}_split{args.global_split}")
                global_csv = global_dir / f"{ch}_joint_merged.csv"

                if not local_csv.exists():
                    log_lines.append(f"[{ch}] local CSV missing: {local_csv}")
                    continue
                if not global_csv.exists():
                    log_lines.append(f"[{ch}] global CSV missing: {global_csv}")
                    continue

                local_df  = pd.read_csv(local_csv)
                global_df = pd.read_csv(global_csv)
                log_lines.append(
                    f"[{ch}] local={len(local_df):,} rows, global={len(global_df):,} rows")

                # Derive labels via join + majority vote
                label_map, report, unmatched = _derive_local_labels_from_join(
                    local_df, global_df, joint_mapping)
                log_lines.append(
                    f"[{ch}] join unmatched={unmatched:,}, clusters_derived={len(label_map)}")

                mixed = [k for k, v in report.items() if v["is_mixed"]]
                if mixed:
                    log_lines.append(
                        f"[{ch}] mixed clusters ({len(mixed)}, majority < 60%): {mixed}")

                labeled_df    = _apply_local_labels(local_df, label_map)
                color_palette = build_color_palette_for_mapping(joint_mapping)

                # Per-dynasty silhouette (read from merged_stats.json if available)
                sil_scores = _load_per_dynasty_silhouette(
                    local_dir / "merged_stats.json")

                # Output directories
                out_ch    = stage_paths["local_annotate"] / f"char_{ch}"
                per_dyn   = out_ch / "per_dynasties"
                ensure_dir(per_dyn)

                # Save labeled CSV
                csv_out = out_ch / f"{ch}_Labeled_Local.csv"
                labeled_df.to_csv(csv_out, index=False, encoding="utf-8-sig")
                log_lines.append(f"[{ch}] labeled CSV saved: {csv_out.name}")

                # Save label mapping report
                report_path = out_ch / f"{ch}_label_mapping_report.json"
                report_path.write_text(
                    json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
                log_lines.append(f"[{ch}] report saved: {report_path.name}")

                # Facet grid (2×4, all dynasties)
                facet_path = out_ch / f"{ch}_Facet_Grid_Local.png"
                _plot_local_annotate_facet(
                    labeled_df, ch, facet_path, color_palette, sil_scores)
                log_lines.append(f"[{ch}] facet grid saved")

                # Per-dynasty single plots
                for dynasty in DYNASTY_ORDER:
                    sub = labeled_df[labeled_df["dynasty"] == dynasty]
                    if sub.empty:
                        continue
                    _plot_local_annotate_single(
                        sub, ch, dynasty,
                        per_dyn / f"{dynasty}_Labeled.png",
                        color_palette, sil_scores.get(dynasty))
                log_lines.append(f"[{ch}] per-dynasty plots saved")

            except Exception as e:
                ok, msg = False, f"char={ch} local_annotate failed: {e}"
                log_lines.append(msg)
                log_lines.append(traceback.format_exc())
                break

        log_path = logs_dir / "local_annotate.log"
        log_path.write_text("\n".join(log_lines), encoding="utf-8")
        res = StageResult("local_annotate", ok, time.time() - start,
                          "internal pandas+matplotlib", str(log_path), msg)
        save_stage("local_annotate", res)
        if not ok:
            sys.exit(1)

    print("\nPipeline finished.")
    print(f"Run root: {run_root}")
    print(f"Summary:  {run_root / 'summary.json'}")


if __name__ == "__main__":
    main()

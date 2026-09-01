#!/usr/bin/env python3
"""
pipeline_common.py

Shared constants and helpers used by generate_label_template.py, global_plot.py,
and local_annotate.py: CJK font setup, dynasty/color constants, label-mapping-JSON
resolution (including the local-override nested schema), and the local-label
join/majority-vote derivation used both to render final labels and to compute
majority-vote hints in the label templates.

This module has no `main()` — it is imported, not run directly.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
import numpy as np
import pandas as pd

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
import seaborn as sns

# ─────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────
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
# General utilities
# ─────────────────────────────────────────────────────────────
def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


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


def plot_facet(df: pd.DataFrame, cluster_col: str, out_png: Path, title: str) -> None:
    """Plain (unlabeled) faceted scatter, colored by raw cluster id. Used as the
    fallback rendering by both global_plot.py and local_annotate.py when no
    semantic label mapping is resolvable yet."""
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
    """Plain (unlabeled) per-dynasty scatter, colored by raw cluster id."""
    ensure_dir(out_dir)
    for dyn, sub in df.groupby("dynasty"):
        fig, ax = plt.subplots(figsize=(7, 5))
        sns.scatterplot(data=sub, x="PC1", y="PC2", hue=cluster_col,
                        s=12, alpha=0.7, linewidth=0, ax=ax)
        ax.set_title(f"{prefix} - {dyn}")
        fig.tight_layout()
        fig.savefig(out_dir / f"{dyn}.png", dpi=150)
        plt.close(fig)


def _char_display(ch: str) -> str:
    pinyin = _CHAR_PINYIN.get(ch, ch)
    return f"'{pinyin}' {ch}"


# ─────────────────────────────────────────────────────────────
# Label mapping helpers — joint (Global) scope
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


# ─────────────────────────────────────────────────────────────
# Label mapping helpers — local (dynasty-scoped) override
# ─────────────────────────────────────────────────────────────
def normalize_local_cluster_mapping(raw: Dict) -> Dict[Tuple[str, int], tuple]:
    """Parse the nested `{dynasty: {cluster_id: [Macro, Sub]}}` shape used by the
    "local" block of a --label-map-json file (see configs/hand_labels.json)."""
    mapping: Dict[Tuple[str, int], tuple] = {}
    for dyn, cluster_map in raw.items():
        if not isinstance(cluster_map, dict):
            raise ValueError(
                f"Invalid local mapping for dynasty={dyn}; expected an object of "
                f"{{cluster_id: [Macro, Sub]}}")
        for cid, v in cluster_map.items():
            if not isinstance(v, (list, tuple)) or len(v) != 2:
                raise ValueError(
                    f"Invalid local mapping at dynasty={dyn} cluster={cid}; "
                    f"expected [Macro, Sub]")
            mapping[(dyn, int(cid))] = (str(v[0]), str(v[1]))
    return mapping


def resolve_local_label_overrides(args, workdir: Path,
                                   ch: str) -> Optional[Dict[Tuple[str, int], tuple]]:
    """Read the optional "local" override block of --label-map-json for `ch`.

    Returns None when no override is supplied (the caller should then fall back to
    join-derived inheritance from the Global labels) — this is the default path used
    by 道/水. A non-empty result fully replaces the (Macro, Sub) pair for the listed
    (dynasty, cluster_id) keys, as used by 手.
    """
    if not args.label_map_json:
        return None
    p = Path(args.label_map_json)
    if not p.is_absolute():
        p = (workdir / p).resolve()
    if not p.exists():
        raise FileNotFoundError(f"label mapping json not found: {p}")

    raw = json.loads(p.read_text(encoding="utf-8"))
    ch_entry = raw.get(ch)
    if not isinstance(ch_entry, dict):
        return None
    local_raw = ch_entry.get("local")
    if not isinstance(local_raw, dict) or not local_raw:
        return None
    return normalize_local_cluster_mapping(local_raw)


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
# Local-label join + majority-vote derivation (stage 8's core algorithm)
# ─────────────────────────────────────────────────────────────
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
                        label_map: Dict[tuple, tuple],
                        overrides: Optional[Dict[tuple, tuple]] = None) -> pd.DataFrame:
    """Attach Macro_Category/Sub_Category to each local row.

    `overrides` (optional), when given, takes precedence per (dynasty, Local_Cluster):
    it is a full (Macro, Sub) replacement — not a Sub-only patch, since a manually/
    LLM-labeled Local cluster can legitimately diverge on Macro too (verified for 手).
    Any (dynasty, cluster) not present in `overrides` falls back to `label_map`
    (the join-derived default), so characters with no override behave unchanged.
    """
    out = local_df.copy()
    overrides = overrides or {}

    def _lookup(row):
        key = (row["dynasty"], int(row["Local_Cluster"]))
        if key in overrides:
            return overrides[key]
        return label_map.get(key, ("NOISE", "Unmatched"))

    labels = out.apply(_lookup, axis=1)
    out["Cluster_ID"]     = out["Local_Cluster"].astype(int)
    out["Macro_Category"] = labels.apply(lambda x: x[0])
    out["Sub_Category"]   = labels.apply(lambda x: x[1])
    return out

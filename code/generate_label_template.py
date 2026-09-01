#!/usr/bin/env python3
"""
generate_label_template.py

Stage 6a. Turns the cluster stats already computed by stage 5 (local_clustering.py)
and stage 6 (global_clustering.py) into fill-in-the-blank labeling material for any
keyword — no new numerical computation here, pure reformatting of joint_stats.json
and each {dynasty}_stats.json.

Produces, per character:
  - <ch>_global_label_template.md / <ch>_global_label_skeleton.json   (always)
  - <ch>_local_label_template.md  / <ch>_local_label_skeleton.json    (optional —
    only needed if you want dynasty-specific Local labels instead of the default
    join-inherited-from-Global behavior; see local_annotate.py)

The skeleton JSON files use exactly the schema configs/*.json already uses, so a
filled-in copy can be saved as configs/<keyword>_labels.json and passed straight
to global_plot.py / local_annotate.py via --label-map-json with no code changes.

Usage:
    python3 code/generate_label_template.py --char 道 \\
        --global-dir stage_outputs/06_joint_output/char_道_k20_subk6_split0.1 \\
        --local-dir  stage_outputs/05_local_output/char_道_Local_k20_split0.3_DynamicSubK \\
        --output stage_outputs/06a_label_templates
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Optional

import pandas as pd

from pipeline_common import (
    DYNASTY_ORDER,
    DYNASTY_DISPLAY,
    _char_display,
    _derive_local_labels_from_join,
    ensure_dir,
    resolve_hand_label_mapping,
    resolve_label_mapping_scoped,
)


def generate_global_label_template(global_dir: Path, ch: str, out_dir: Path) -> Optional[Path]:
    stats_path = global_dir / "joint_stats.json"
    if not stats_path.exists():
        return None
    stats = json.loads(stats_path.read_text(encoding="utf-8"))
    clusters = stats.get("clusters", {}) or {}
    if not clusters:
        return None

    lines = [
        f"# Global Label Template — {_char_display(ch)}",
        "",
        "Assign a Macro_Category and Sub_Category to each cluster below, based on the",
        "representative sentences. Reuse a Macro/Sub_Category pair across clusters where",
        "the underlying sense is the same; keep Sub_Category names short and stable",
        "(e.g. `Holding`, not `Act_Of_Holding_Something`). Use `NOISE` / `OCR_Error` for",
        "clusters made of garbled/OCR text.",
        "",
        "Fill in the matching *_global_label_skeleton.json instead of this document, then",
        "save the result as configs/<keyword>_labels.json and pass it to global_plot.py",
        "and local_annotate.py via --label-map-json.",
        "",
    ]
    skeleton: Dict[str, list] = {}
    for cid, info in sorted(clusters.items(), key=lambda kv: int(kv[0])):
        size = info.get("size", "?")
        dyn_dist = info.get("dynasty_distribution", {}) or {}
        dyn_str = ", ".join(
            f"{d} {c}" for d, c in sorted(dyn_dist.items(), key=lambda kv: -kv[1]))
        lines.append(f"## Cluster {cid}  (n={size}, dynasties: {dyn_str})")
        lines.append("")
        lines.append("Core sentences (closest to cluster center):")
        for s in info.get("core", []):
            lines.append(
                f"- [{s.get('dynasty', '')}] {s.get('sentence', '')}  "
                f"(source: {s.get('source', '')})")
        boundary = info.get("boundary", [])
        if boundary:
            lines.append("")
            lines.append("Boundary sentences (edge of cluster):")
            for s in boundary:
                lines.append(
                    f"- [{s.get('dynasty', '')}] {s.get('sentence', '')}  "
                    f"(source: {s.get('source', '')})")
        lines.append("")
        lines.append("Macro_Category: ")
        lines.append("Sub_Category: ")
        lines.append("")
        lines.append("---")
        lines.append("")
        skeleton[str(int(cid))] = ["", ""]

    ensure_dir(out_dir)
    md_path = out_dir / f"{ch}_global_label_template.md"
    md_path.write_text("\n".join(lines), encoding="utf-8")

    json_path = out_dir / f"{ch}_global_label_skeleton.json"
    json_path.write_text(
        json.dumps({ch: {"joint": skeleton}}, ensure_ascii=False, indent=2),
        encoding="utf-8")
    return md_path


def generate_local_label_template(local_dir: Path, global_dir: Path, ch: str,
                                   joint_mapping: Optional[Dict[int, tuple]],
                                   out_dir: Path) -> Optional[Path]:
    hints: Dict[tuple, dict] = {}
    if joint_mapping is not None:
        local_csv = local_dir / f"{ch}_Local_Dynamic_Merged.csv"
        global_csv = global_dir / f"{ch}_joint_merged.csv"
        if local_csv.exists() and global_csv.exists():
            local_df = pd.read_csv(local_csv)
            global_df = pd.read_csv(global_csv)
            _, report, _ = _derive_local_labels_from_join(local_df, global_df, joint_mapping)
            for info in report.values():
                hints[(info["dynasty"], info["local_cluster"])] = info

    lines = [
        f"# Local Label Template — {_char_display(ch)}",
        "",
        "OPTIONAL: only fill this in if you want dynasty-specific Local labels (as was",
        "done for 手). By default Local labels are inherited from the Global mapping via",
        "majority vote — leave this template unfilled and local_annotate.py falls back",
        "to that automatically; no extra step is required for most keywords.",
        "",
        "The 'majority vote reference' line, where shown, is a hint only, not a forced",
        "value — assign whatever Macro_Category / Sub_Category actually fits this",
        "dynasty's usage, reusing the Global label where it genuinely applies.",
        "",
    ]
    skeleton: Dict[str, Dict[str, list]] = {}

    for dyn in DYNASTY_ORDER:
        stats_path = local_dir / f"{dyn}_stats.json"
        if not stats_path.exists():
            continue
        stats = json.loads(stats_path.read_text(encoding="utf-8"))
        cluster_sentences = stats.get("cluster_sentences", {}) or {}
        if not cluster_sentences:
            continue

        lines.append(f"## Dynasty: {DYNASTY_DISPLAY.get(dyn, dyn)}")
        lines.append("")
        skeleton[dyn] = {}

        for cid, info in sorted(cluster_sentences.items(), key=lambda kv: int(kv[0])):
            cid_int = int(cid)
            lines.append(f"### Cluster {cid} (n={info.get('count', '?')})")
            hint = hints.get((dyn, cid_int))
            if hint:
                macro, sub = hint["assigned_label"]
                flag = ("  [MIXED — majority vote is unreliable here]"
                        if hint["is_mixed"] else "")
                lines.append(
                    f"Majority vote reference: {hint['majority_pct']}% of sentences match "
                    f"Global cluster {hint['majority_global']} ({macro} / {sub}).{flag}")
            lines.append("")
            lines.append("Core sentences:")
            for s in info.get("core", []):
                lines.append(f"- {s.get('sentence', '')}  (source: {s.get('toptitle', '')})")
            boundary = info.get("boundary", [])
            if boundary:
                lines.append("")
                lines.append("Boundary sentences:")
                for s in boundary:
                    lines.append(f"- {s.get('sentence', '')}  (source: {s.get('toptitle', '')})")
            lines.append("")
            lines.append("Macro_Category: ")
            lines.append("Sub_Category: ")
            lines.append("")
            lines.append("---")
            lines.append("")
            skeleton[dyn][str(cid_int)] = ["", ""]

    if not skeleton:
        return None

    ensure_dir(out_dir)
    md_path = out_dir / f"{ch}_local_label_template.md"
    md_path.write_text("\n".join(lines), encoding="utf-8")

    json_path = out_dir / f"{ch}_local_label_skeleton.json"
    json_path.write_text(
        json.dumps({ch: {"local": skeleton}}, ensure_ascii=False, indent=2),
        encoding="utf-8")
    return md_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Stage 6a: turn stage 5/6 cluster stats into fill-in-the-blank "
                     "labeling material (Markdown + JSON skeleton) for any keyword.")
    parser.add_argument("--char", type=str, required=True)
    parser.add_argument("--global-dir", type=str, required=True,
                        help="char_<ch>_k.../ directory from the global stage")
    parser.add_argument("--local-dir", type=str, required=True,
                        help="char_<ch>_Local_.../ directory from the local stage")
    parser.add_argument("--output", type=str, required=True,
                        help="Output root; a char_<ch>/ subdirectory is created inside it")
    parser.add_argument("--workdir", type=str, default=".",
                        help="Base directory for resolving a relative --label-map-json")

    parser.add_argument("--hand-label-version", type=str, default="auto",
                        choices=["auto", "0221", "0222"])
    parser.add_argument("--hand-label-map-json", type=str, default="")
    parser.add_argument("--label-map-json", type=str, default="",
                        help="Optional; only used to compute majority-vote hints in "
                             "the Local template")
    args = parser.parse_args()

    ch = args.char
    chars = [ch]
    workdir = Path(args.workdir).resolve()
    global_dir = Path(args.global_dir)
    local_dir = Path(args.local_dir)
    out_ch = Path(args.output) / f"char_{ch}"

    md_g = generate_global_label_template(global_dir, ch, out_ch)
    if md_g:
        print(f"[{ch}] global label template: {md_g.name}")
    else:
        print(f"[{ch}] SKIP global template: joint_stats.json missing under {global_dir}")

    joint_mapping = resolve_label_mapping_scoped(args, workdir, ch, chars, "joint")
    if joint_mapping is None and ch == "手":
        joint_mapping = resolve_hand_label_mapping(args, workdir)

    md_l = generate_local_label_template(local_dir, global_dir, ch, joint_mapping, out_ch)
    if md_l:
        hint_note = ("" if joint_mapping is not None else
                     " (no majority-vote hints — joint labels not yet resolved)")
        print(f"[{ch}] local label template: {md_l.name}{hint_note}")
    else:
        print(f"[{ch}] SKIP local template: no per-dynasty stats found under {local_dir}")


if __name__ == "__main__":
    main()

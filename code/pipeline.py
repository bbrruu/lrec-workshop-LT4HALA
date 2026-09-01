#!/usr/bin/env python3
"""
pipeline.py

All-in-one pipeline orchestrator: extract → clean1 → sample → clean2 → local →
global → label_template → global_plot → local_annotate.

Every stage is a subprocess call to its own script next to this file in code/ —
this file only does argument parsing, stage sequencing, and logging. It carries
no plotting or labeling logic itself (see pipeline_common.py, generate_label_template.py,
global_plot.py, local_annotate.py) and therefore has no dependency on pandas/
numpy/matplotlib — only stages that actually plot or crunch data need those,
and each pulls them in on its own.

Output root: final_pipeline_results/ (see --output-root)

Stage 6a "label_template":
    Turns the cluster stats already computed by stages 5/6 into fill-in-the-blank
    labeling material (Markdown + JSON skeleton) for any keyword. Optional to act
    on — running it just produces material to hand to an LLM/human annotator.

Stage 8 "local_annotate":
    Derives Local cluster semantic labels from validated Global labels via
    sentence-level join + majority vote, unless an explicit "local" override is
    supplied in --label-map-json (see configs/hand_labels.json for an example).
    With no Global labels resolved yet, it falls back to a plain rendering
    instead of skipping the character.

Usage:
    # Full pipeline (extract through local_annotate)
    python3 code/pipeline.py --chars 道,水 --run-name run_01 \\
        --label-map-json configs/dao_shui_labels.json

    # Only the local_annotate stage (after labels are filled in)
    python3 code/pipeline.py --chars 道,水 --run-name run_01 \\
        --start-stage local_annotate --end-stage local_annotate \\
        --label-map-json configs/dao_shui_labels.json

    # Extract through global_plot (skip local_annotate for now)
    python3 code/pipeline.py --chars 天 --run-name run_tian \\
        --start-stage extract --end-stage global_plot
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List

# Stage scripts (extract_keyword.py, clean_ocr_errors.py, stratified_sampling.py,
# clean_data.py, local_clustering.py, global_clustering.py, generate_label_template.py,
# global_plot.py, local_annotate.py) live next to this file in code/, independent of
# --workdir (which locates run outputs and label configs).
SCRIPT_DIR = Path(__file__).resolve().parent

# ─────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────
STAGE_ORDER = [
    "extract", "clean1", "sample", "clean2",
    "local", "global", "label_template", "global_plot", "local_annotate",
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


def semantic_flags(args) -> List[str]:
    """Common --label-map-json / --hand-label-* flags shared by label_template,
    global_plot, and local_annotate (all three read the same semantic config)."""
    flags = ["--hand-label-version", args.hand_label_version]
    if args.hand_label_map_json:
        flags += ["--hand-label-map-json", args.hand_label_map_json]
    if args.label_map_json:
        flags += ["--label-map-json", args.label_map_json]
    return flags


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
    print("Type a value and press Enter, or press Enter alone to accept the default.\n")

    args.chars = prompt_text("Characters to run (comma-separated)", default=args.chars or "手")
    args.run_name = prompt_text("Run name (leave blank to auto-generate)", default=args.run_name)
    args.workdir = prompt_text("Working directory", default=args.workdir)
    args.output_root = prompt_text("Combined output root directory", default=args.output_root)

    args.start_stage = prompt_stage("Start stage", default=args.start_stage)
    args.end_stage = prompt_stage("End stage", default=args.end_stage)

    args.dynasties = prompt_text("Restrict to dynasties (blank for all; comma- or space-separated)", default=args.dynasties)
    args.extract_device = prompt_text("Embedding device (auto/cpu/cuda)", default=args.extract_device)

    max_texts_raw = prompt_text("Max texts per dynasty for extract (0 = unlimited)",
                                default=str(args.extract_max_texts))
    try:
        args.extract_max_texts = int(max_texts_raw)
    except ValueError:
        args.extract_max_texts = 0

    args.hand_label_version = prompt_text(
        "Label profile for char=手 (auto/0221/0222)", default=args.hand_label_version)
    args.hand_label_map_json = prompt_text(
        "Custom label JSON for char=手 (blank to skip)", default=args.hand_label_map_json)
    args.label_map_json = prompt_text(
        "Generic multi-character label JSON (blank to skip)", default=args.label_map_json)

    args.skip_existing = prompt_yes_no("Skip stages whose output already exists", default=args.skip_existing)

    print("\nInteractive setup complete, starting run...\n")
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
    parser.add_argument("--extract-book-data-path", type=str, default="",
                        help="Override extract_keyword.py's default CText book archive path")
    parser.add_argument("--extract-wiki-data-pattern", type=str, default="",
                        help="Override extract_keyword.py's default CText wiki archive glob pattern")

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
                        help="Generic JSON with joint (and optionally local) labels "
                             "for all chars; read by label_template, global_plot, "
                             "and local_annotate")

    parser.add_argument("--skip-existing", action="store_true")
    args = parser.parse_args()
    args = prompt_if_needed(args)

    if not args.chars:
        print("--chars is required (or use interactive mode to enter it)")
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
        "label_template":     out_dir / "06a_label_templates",
        "global_plot":        out_dir / "07_global_plot",
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
            "extract_book_data_path": args.extract_book_data_path,
            "extract_wiki_data_pattern": args.extract_wiki_data_pattern,
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

    def local_dir_for(ch: str) -> Path:
        return (stage_paths["local"] /
                f"char_{ch}_Local_k{args.local_k}_split{args.local_split}_DynamicSubK")

    def global_dir_for(ch: str) -> Path:
        return (stage_paths["global"] /
                f"char_{ch}_k{args.global_k}_subk{args.global_sub_k}_split{args.global_split}")

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
            if args.extract_book_data_path:
                cmd += ["--book-data-path", args.extract_book_data_path]
            if args.extract_wiki_data_pattern:
                cmd += ["--wiki-data-pattern", args.extract_wiki_data_pattern]
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
                          "loop clean_data.py", str(log_file), msg)
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
                          "loop global_clustering.py", str(log_file), msg)
        save_stage("global", res)
        if not ok:
            sys.exit(1)

    # ── Stage: label_template ─────────────────────────────────
    if stage_enabled("label_template", args.start_stage, args.end_stage):
        start = time.time()
        log_file = logs_dir / "label_template.log"
        combined_log = []
        ok, msg = True, "ok"
        for ch in chars:
            cmd = [
                args.python_bin,
                str((SCRIPT_DIR / "generate_label_template.py").resolve()),
                "--char",       ch,
                "--global-dir", str(global_dir_for(ch)),
                "--local-dir",  str(local_dir_for(ch)),
                "--output",     str(stage_paths["label_template"]),
                "--workdir",    str(workdir),
            ] + semantic_flags(args)
            sub = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
            combined_log.append(f"### char={ch}\n{sub.stdout}\n")
            if sub.returncode != 0:
                ok, msg = False, f"char={ch} failed exit={sub.returncode}"
                break
        log_file.write_text("\n".join(combined_log), encoding="utf-8")
        res = StageResult("label_template", ok, time.time() - start,
                          "loop generate_label_template.py", str(log_file), msg)
        save_stage("label_template", res)
        if not ok:
            sys.exit(1)

    # ── Stage: global_plot ────────────────────────────────────
    if stage_enabled("global_plot", args.start_stage, args.end_stage):
        start = time.time()
        log_file = logs_dir / "global_plot.log"
        combined_log = []
        ok, msg = True, "ok"
        for ch in chars:
            cmd = [
                args.python_bin,
                str((SCRIPT_DIR / "global_plot.py").resolve()),
                "--char",       ch,
                "--global-dir", str(global_dir_for(ch)),
                "--output",     str(stage_paths["global_plot"]),
                "--workdir",    str(workdir),
            ] + semantic_flags(args)
            sub = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
            combined_log.append(f"### char={ch}\n{sub.stdout}\n")
            if sub.returncode != 0:
                ok, msg = False, f"char={ch} failed exit={sub.returncode}"
                break
        log_file.write_text("\n".join(combined_log), encoding="utf-8")
        res = StageResult("global_plot", ok, time.time() - start,
                          "loop global_plot.py", str(log_file), msg)
        save_stage("global_plot", res)
        if not ok:
            sys.exit(1)

    # ── Stage: local_annotate ─────────────────────────────────
    if stage_enabled("local_annotate", args.start_stage, args.end_stage):
        start = time.time()
        log_file = logs_dir / "local_annotate.log"
        combined_log = []
        ok, msg = True, "ok"
        for ch in chars:
            cmd = [
                args.python_bin,
                str((SCRIPT_DIR / "local_annotate.py").resolve()),
                "--char",       ch,
                "--local-dir",  str(local_dir_for(ch)),
                "--global-dir", str(global_dir_for(ch)),
                "--output",     str(stage_paths["local_annotate"]),
                "--workdir",    str(workdir),
            ] + semantic_flags(args)
            sub = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
            combined_log.append(f"### char={ch}\n{sub.stdout}\n")
            if sub.returncode != 0:
                ok, msg = False, f"char={ch} failed exit={sub.returncode}"
                break
        log_file.write_text("\n".join(combined_log), encoding="utf-8")
        res = StageResult("local_annotate", ok, time.time() - start,
                          "loop local_annotate.py", str(log_file), msg)
        save_stage("local_annotate", res)
        if not ok:
            sys.exit(1)

    print("\nPipeline finished.")
    print(f"Run root: {run_root}")
    print(f"Summary:  {run_root / 'summary.json'}")


if __name__ == "__main__":
    main()

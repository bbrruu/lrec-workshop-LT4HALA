#!/usr/bin/env python3
"""
Parameter sensitivity test runner for local_clustering.py / global_clustering.py.

Design:
1. One-factor-at-a-time tests only.
2. Global baseline: K=20, merge=0.95, split=0.1
3. Local baseline:  K=20, merge=0.95, split=0.3
4. Local overall silhouette uses dynasty-sample weighted average.

Outputs:
- CSV tables for each test group
- PNG charts for silhouette and cluster-count distributions
- A markdown summary report with best config vs baseline
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


# Clustering scripts live next to this file in code/, independent of --workdir
# (which locates --input/--output data), same convention as pipeline.py's SCRIPT_DIR.
SCRIPT_DIR = Path(__file__).resolve().parent
GLOBAL_SCRIPT = "global_clustering.py"
LOCAL_SCRIPT = "local_clustering.py"


@dataclass(frozen=True)
class Config:
    k: int
    merge_ratio: float
    split_ratio: float


@dataclass
class RunResult:
    mode: str  # global | local
    run_id: str
    config: Config
    group: str
    group_value: str
    success: bool
    elapsed_sec: float
    silhouette: Optional[float]
    final_k: Optional[float]
    total_samples: Optional[int]
    out_dir: str
    error: str


def parse_num_list(raw: str, cast):
    return [cast(x.strip()) for x in raw.split(",") if x.strip()]


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def make_run_id(prefix: str, cfg: Config) -> str:
    return (
        f"{prefix}_k{cfg.k}_m{cfg.merge_ratio:.2f}_s{cfg.split_ratio:.2f}"
        .replace(".", "p")
    )


def run_command(cmd: List[str], log_path: Path) -> Tuple[bool, str, float]:
    start = time.time()
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    elapsed = time.time() - start
    log_path.write_text(proc.stdout, encoding="utf-8")
    ok = proc.returncode == 0
    err = "" if ok else f"exit={proc.returncode}"
    return ok, err, elapsed


def read_json(path: Path) -> Dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def build_one_factor_configs(
    baseline: Config,
    k_values: List[int],
    merge_values: List[float],
    split_values: List[float],
) -> Dict[str, List[Config]]:
    cfgs = {
        "K": [Config(k=v, merge_ratio=baseline.merge_ratio, split_ratio=baseline.split_ratio) for v in k_values],
        "merge_ratio": [Config(k=baseline.k, merge_ratio=v, split_ratio=baseline.split_ratio) for v in merge_values],
        "split_ratio": [Config(k=baseline.k, merge_ratio=baseline.merge_ratio, split_ratio=v) for v in split_values],
    }
    return cfgs


def write_csv(path: Path, rows: List[Dict], columns: List[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=columns)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def plot_group_metric(
    rows: List[RunResult],
    metric_name: str,
    title: str,
    output_png: Path,
) -> None:
    grouped: Dict[str, List[Tuple[str, float]]] = {"K": [], "merge_ratio": [], "split_ratio": []}
    for r in rows:
        if not r.success:
            continue
        value = r.silhouette if metric_name == "silhouette" else r.final_k
        if value is None:
            continue
        grouped[r.group].append((r.group_value, float(value)))

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    order = ["K", "merge_ratio", "split_ratio"]

    for ax, group in zip(axes, order):
        pts = grouped[group]
        if not pts:
            ax.set_title(f"{group} (no data)")
            ax.axis("off")
            continue
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        ax.plot(xs, ys, marker="o")
        ax.set_title(group)
        ax.set_xlabel(group)
        ax.set_ylabel(metric_name)
        ax.grid(alpha=0.3)

    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(output_png, dpi=160)
    plt.close(fig)


def find_best(rows: List[RunResult]) -> Optional[RunResult]:
    ok_rows = [r for r in rows if r.success and r.silhouette is not None]
    if not ok_rows:
        return None
    ok_rows.sort(key=lambda r: (float(r.silhouette), -float(r.final_k or 1e9)), reverse=True)
    return ok_rows[0]


def find_baseline(rows: List[RunResult], baseline: Config) -> Optional[RunResult]:
    for r in rows:
        if (
            r.success
            and r.config.k == baseline.k
            and abs(r.config.merge_ratio - baseline.merge_ratio) < 1e-9
            and abs(r.config.split_ratio - baseline.split_ratio) < 1e-9
        ):
            return r
    return None


def local_char_dir(base_out: Path, char: str, cfg: Config) -> Path:
    return base_out / f"char_{char}_Local_k{cfg.k}_split{cfg.split_ratio}_DynamicSubK"


def global_char_dir(base_out: Path, char: str, cfg: Config, sub_k: int) -> Path:
    return base_out / f"char_{char}_k{cfg.k}_subk{sub_k}_split{cfg.split_ratio}"


def parse_local_metrics(char_dir: Path) -> Tuple[Optional[float], Optional[float], Optional[int], str]:
    stat_files = sorted(char_dir.glob("*_stats.json"))
    if not stat_files:
        return None, None, None, "no *_stats.json"

    weighted_sil_num = 0.0
    weighted_k_num = 0.0
    total_n = 0

    for p in stat_files:
        data = read_json(p)
        sil = data.get("silhouette")
        k_final = data.get("k_final")
        n = data.get("n_samples")
        if sil is None or k_final is None or n is None:
            continue
        weighted_sil_num += float(sil) * int(n)
        weighted_k_num += float(k_final) * int(n)
        total_n += int(n)

    if total_n == 0:
        return None, None, None, "local stats missing silhouette/k_final/n_samples"

    return weighted_sil_num / total_n, weighted_k_num / total_n, total_n, ""


def parse_global_metrics(char_dir: Path) -> Tuple[Optional[float], Optional[float], Optional[int], str]:
    stats_path = char_dir / "joint_stats.json"
    if not stats_path.exists():
        return None, None, None, "joint_stats.json not found"
    data = read_json(stats_path)
    sil = data.get("silhouette")
    k_final = data.get("k_final")
    n = data.get("total_samples")
    if sil is None or k_final is None or n is None:
        return None, None, None, "joint_stats missing silhouette/k_final/total_samples"
    return float(sil), float(k_final), int(n), ""


def run_global(
    args,
    cfg: Config,
    group: str,
    group_value: str,
    unique_run_root: Path,
    skip_existing: bool,
) -> RunResult:
    run_id = make_run_id("global", cfg)
    run_root = unique_run_root / run_id
    ensure_dir(run_root)
    log_path = run_root / "run.log"

    cmd = [
        args.python_bin,
        str((SCRIPT_DIR / GLOBAL_SCRIPT).resolve()),
        "--input", args.input,
        "--char", args.char,
        "--output", str(run_root),
        "--k", str(cfg.k),
        "--merge-ratio", str(cfg.merge_ratio),
        "--sub-k", str(args.global_sub_k),
        "--split-ratio", str(cfg.split_ratio),
    ]

    already_dir = global_char_dir(run_root, args.char, cfg, args.global_sub_k)
    if skip_existing and (already_dir / "joint_stats.json").exists():
        ok, err, elapsed = True, "", 0.0
    else:
        ok, err, elapsed = run_command(cmd, log_path)

    sil, final_k, total_n, parse_err = parse_global_metrics(already_dir)
    if parse_err and ok:
        ok = False
        err = parse_err

    return RunResult(
        mode="global",
        run_id=run_id,
        config=cfg,
        group=group,
        group_value=group_value,
        success=ok,
        elapsed_sec=elapsed,
        silhouette=sil,
        final_k=final_k,
        total_samples=total_n,
        out_dir=str(already_dir),
        error=err,
    )


def run_local(
    args,
    cfg: Config,
    group: str,
    group_value: str,
    unique_run_root: Path,
    skip_existing: bool,
) -> RunResult:
    run_id = make_run_id("local", cfg)
    run_root = unique_run_root / run_id
    ensure_dir(run_root)
    log_path = run_root / "run.log"

    cmd = [
        args.python_bin,
        str((SCRIPT_DIR / LOCAL_SCRIPT).resolve()),
        "--chars", args.char,
        "--input", args.input,
        "--output", str(run_root),
        "--k-init", str(cfg.k),
        "--merge-ratio", str(cfg.merge_ratio),
        "--split-ratio", str(cfg.split_ratio),
    ]

    already_dir = local_char_dir(run_root, args.char, cfg)
    if skip_existing and any(already_dir.glob("*_stats.json")):
        ok, err, elapsed = True, "", 0.0
    else:
        ok, err, elapsed = run_command(cmd, log_path)

    sil, final_k, total_n, parse_err = parse_local_metrics(already_dir)
    if parse_err and ok:
        ok = False
        err = parse_err

    return RunResult(
        mode="local",
        run_id=run_id,
        config=cfg,
        group=group,
        group_value=group_value,
        success=ok,
        elapsed_sec=elapsed,
        silhouette=sil,
        final_k=final_k,
        total_samples=total_n,
        out_dir=str(already_dir),
        error=err,
    )


def rows_to_dicts(rows: List[RunResult]) -> List[Dict]:
    out = []
    for r in rows:
        out.append(
            {
                "mode": r.mode,
                "run_id": r.run_id,
                "group": r.group,
                "group_value": r.group_value,
                "k": r.config.k,
                "merge_ratio": r.config.merge_ratio,
                "split_ratio": r.config.split_ratio,
                "success": r.success,
                "elapsed_sec": round(r.elapsed_sec, 3),
                "silhouette": "" if r.silhouette is None else f"{r.silhouette:.6f}",
                "final_k": "" if r.final_k is None else f"{r.final_k:.4f}",
                "total_samples": "" if r.total_samples is None else r.total_samples,
                "out_dir": r.out_dir,
                "error": r.error,
            }
        )
    return out


def make_markdown_table(rows: List[Dict], columns: List[str]) -> str:
    lines = []
    lines.append("| " + " | ".join(columns) + " |")
    lines.append("| " + " | ".join(["---"] * len(columns)) + " |")
    for r in rows:
        vals = [str(r.get(c, "")) for c in columns]
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines)


def generate_report(
    out_root: Path,
    global_rows: List[RunResult],
    local_rows: List[RunResult],
    global_baseline: Config,
    local_baseline: Config,
) -> None:
    best_global = find_best(global_rows)
    base_global = find_baseline(global_rows, global_baseline)
    best_local = find_best(local_rows)
    base_local = find_baseline(local_rows, local_baseline)

    report_path = out_root / "summary_report.md"
    lines: List[str] = []
    lines.append("# 0331 Parameter Test Summary")
    lines.append("")
    lines.append("## Baselines")
    lines.append(f"- Global baseline: K={global_baseline.k}, merge_ratio={global_baseline.merge_ratio}, split_ratio={global_baseline.split_ratio}")
    lines.append(f"- Local baseline: K={local_baseline.k}, merge_ratio={local_baseline.merge_ratio}, split_ratio={local_baseline.split_ratio}")
    lines.append("")

    lines.append("## Best Configs")
    if best_global:
        lines.append(
            f"- Global best: run_id={best_global.run_id}, "
            f"silhouette={best_global.silhouette:.6f}, final_k={best_global.final_k:.2f}, "
            f"cfg=(K={best_global.config.k}, merge={best_global.config.merge_ratio}, split={best_global.config.split_ratio})"
        )
    else:
        lines.append("- Global best: N/A")

    if best_local:
        lines.append(
            f"- Local best (weighted by dynasty n): run_id={best_local.run_id}, "
            f"silhouette={best_local.silhouette:.6f}, weighted_final_k={best_local.final_k:.2f}, "
            f"cfg=(K={best_local.config.k}, merge={best_local.config.merge_ratio}, split={best_local.config.split_ratio})"
        )
    else:
        lines.append("- Local best: N/A")
    lines.append("")

    lines.append("## Baseline vs Best")
    if best_global and base_global and best_global.silhouette is not None and base_global.silhouette is not None:
        lines.append(f"- Global silhouette delta: {best_global.silhouette - base_global.silhouette:+.6f}")
    else:
        lines.append("- Global silhouette delta: N/A")

    if best_local and base_local and best_local.silhouette is not None and base_local.silhouette is not None:
        lines.append(f"- Local silhouette delta: {best_local.silhouette - base_local.silhouette:+.6f}")
    else:
        lines.append("- Local silhouette delta: N/A")
    lines.append("")

    lines.append("## Artifacts")
    lines.append("- global_results.csv")
    lines.append("- local_results.csv")
    lines.append("- global_silhouette.png")
    lines.append("- global_k_distribution.png")
    lines.append("- local_silhouette.png")
    lines.append("- local_k_distribution.png")
    lines.append("")

    top_cols = ["run_id", "group", "group_value", "k", "merge_ratio", "split_ratio", "silhouette", "final_k"]
    g_rows = sorted(
        [r for r in rows_to_dicts(global_rows) if r["success"]],
        key=lambda x: float(x["silhouette"] or -1),
        reverse=True,
    )[:10]
    l_rows = sorted(
        [r for r in rows_to_dicts(local_rows) if r["success"]],
        key=lambda x: float(x["silhouette"] or -1),
        reverse=True,
    )[:10]

    lines.append("## Top 10 Global")
    lines.append(make_markdown_table(g_rows, top_cols) if g_rows else "No successful runs.")
    lines.append("")
    lines.append("## Top 10 Local")
    lines.append(make_markdown_table(l_rows, top_cols) if l_rows else "No successful runs.")
    lines.append("")

    report_path.write_text("\n".join(lines), encoding="utf-8")


def dedupe_configs(cfgs_by_group: Dict[str, List[Config]]) -> List[Tuple[str, str, Config]]:
    seen: Dict[Config, str] = {}
    expanded: List[Tuple[str, str, Config]] = []
    for group, cfgs in cfgs_by_group.items():
        for cfg in cfgs:
            if group == "K":
                val = str(cfg.k)
            elif group == "merge_ratio":
                val = str(cfg.merge_ratio)
            else:
                val = str(cfg.split_ratio)
            expanded.append((group, val, cfg))
            if cfg not in seen:
                seen[cfg] = make_run_id("tmp", cfg)
    return expanded


def main() -> None:
    parser = argparse.ArgumentParser(description="One-factor parameter sensitivity tests for local_clustering.py / global_clustering.py")
    parser.add_argument("--workdir", type=str, default=".", help="Directory containing --input data and where --output is written (clustering scripts are always resolved next to this file)")
    parser.add_argument("--input", type=str, default="mainfiles/0212_cleaned_output", help="Input folder for clustering scripts")
    parser.add_argument("--char", type=str, default="手", help="Target character")
    parser.add_argument("--output", type=str, default="0331_parameter_test_output", help="Output folder for test artifacts")
    parser.add_argument("--python-bin", type=str, default=sys.executable, help="Python interpreter for subprocess runs")
    parser.add_argument("--global-sub-k", type=int, default=6, help="Fixed sub-k for global script (not compared)")

    parser.add_argument("--k-values", type=str, default="10,15,20,25,30")
    parser.add_argument("--merge-values", type=str, default="0.9,0.92,0.95,0.97")
    parser.add_argument("--split-values", type=str, default="0.1,0.3,0.5")

    parser.add_argument("--global-baseline-k", type=int, default=20)
    parser.add_argument("--global-baseline-merge", type=float, default=0.95)
    parser.add_argument("--global-baseline-split", type=float, default=0.1)

    parser.add_argument("--local-baseline-k", type=int, default=20)
    parser.add_argument("--local-baseline-merge", type=float, default=0.95)
    parser.add_argument("--local-baseline-split", type=float, default=0.3)

    parser.add_argument("--skip-existing", action="store_true", help="Skip rerun when expected stats file already exists")
    args = parser.parse_args()

    k_values = parse_num_list(args.k_values, int)
    merge_values = parse_num_list(args.merge_values, float)
    split_values = parse_num_list(args.split_values, float)

    global_baseline = Config(args.global_baseline_k, args.global_baseline_merge, args.global_baseline_split)
    local_baseline = Config(args.local_baseline_k, args.local_baseline_merge, args.local_baseline_split)

    out_root = (Path(args.workdir) / args.output).resolve()
    ensure_dir(out_root)
    runs_root = out_root / "runs"
    ensure_dir(runs_root)
    global_runs_root = runs_root / "global"
    local_runs_root = runs_root / "local"
    ensure_dir(global_runs_root)
    ensure_dir(local_runs_root)

    global_cfgs = build_one_factor_configs(global_baseline, k_values, merge_values, split_values)
    local_cfgs = build_one_factor_configs(local_baseline, k_values, merge_values, split_values)

    # Expand group rows for reporting; execute each unique config once per mode.
    global_expanded = dedupe_configs(global_cfgs)
    local_expanded = dedupe_configs(local_cfgs)

    global_unique = sorted({cfg for _, _, cfg in global_expanded}, key=lambda c: (c.k, c.merge_ratio, c.split_ratio))
    local_unique = sorted({cfg for _, _, cfg in local_expanded}, key=lambda c: (c.k, c.merge_ratio, c.split_ratio))

    print("=" * 70)
    print("0331 parameter test")
    print("=" * 70)
    print(f"Global runs (unique): {len(global_unique)}")
    print(f"Local runs (unique):  {len(local_unique)}")
    print(f"Output root: {out_root}")
    print("=" * 70)

    global_by_cfg: Dict[Config, RunResult] = {}
    for i, cfg in enumerate(global_unique, 1):
        print(f"[Global {i}/{len(global_unique)}] K={cfg.k}, merge={cfg.merge_ratio}, split={cfg.split_ratio}")
        rr = run_global(args, cfg, group="", group_value="", unique_run_root=global_runs_root, skip_existing=args.skip_existing)
        global_by_cfg[cfg] = rr

    local_by_cfg: Dict[Config, RunResult] = {}
    for i, cfg in enumerate(local_unique, 1):
        print(f"[Local  {i}/{len(local_unique)}] K={cfg.k}, merge={cfg.merge_ratio}, split={cfg.split_ratio}")
        rr = run_local(args, cfg, group="", group_value="", unique_run_root=local_runs_root, skip_existing=args.skip_existing)
        local_by_cfg[cfg] = rr

    # Rebuild grouped views (K / merge_ratio / split_ratio), reusing unique run results.
    global_rows: List[RunResult] = []
    for group, value, cfg in global_expanded:
        base = global_by_cfg[cfg]
        global_rows.append(
            RunResult(
                mode=base.mode,
                run_id=base.run_id,
                config=base.config,
                group=group,
                group_value=value,
                success=base.success,
                elapsed_sec=base.elapsed_sec,
                silhouette=base.silhouette,
                final_k=base.final_k,
                total_samples=base.total_samples,
                out_dir=base.out_dir,
                error=base.error,
            )
        )

    local_rows: List[RunResult] = []
    for group, value, cfg in local_expanded:
        base = local_by_cfg[cfg]
        local_rows.append(
            RunResult(
                mode=base.mode,
                run_id=base.run_id,
                config=base.config,
                group=group,
                group_value=value,
                success=base.success,
                elapsed_sec=base.elapsed_sec,
                silhouette=base.silhouette,
                final_k=base.final_k,
                total_samples=base.total_samples,
                out_dir=base.out_dir,
                error=base.error,
            )
        )

    # Save result tables.
    cols = [
        "mode", "run_id", "group", "group_value", "k", "merge_ratio", "split_ratio",
        "success", "elapsed_sec", "silhouette", "final_k", "total_samples", "out_dir", "error"
    ]
    write_csv(out_root / "global_results.csv", rows_to_dicts(global_rows), cols)
    write_csv(out_root / "local_results.csv", rows_to_dicts(local_rows), cols)

    # Charts.
    plot_group_metric(
        global_rows,
        metric_name="silhouette",
        title="Global silhouette by one-factor tests",
        output_png=out_root / "global_silhouette.png",
    )
    plot_group_metric(
        global_rows,
        metric_name="final_k",
        title="Global final cluster count by one-factor tests",
        output_png=out_root / "global_k_distribution.png",
    )
    plot_group_metric(
        local_rows,
        metric_name="silhouette",
        title="Local weighted silhouette by one-factor tests",
        output_png=out_root / "local_silhouette.png",
    )
    plot_group_metric(
        local_rows,
        metric_name="final_k",
        title="Local weighted cluster count by one-factor tests",
        output_png=out_root / "local_k_distribution.png",
    )

    # Summary report.
    generate_report(out_root, global_rows, local_rows, global_baseline, local_baseline)

    print("\nDone.")
    print(f"- global table: {out_root / 'global_results.csv'}")
    print(f"- local table:  {out_root / 'local_results.csv'}")
    print(f"- summary:      {out_root / 'summary_report.md'}")


if __name__ == "__main__":
    main()

"""
Stratified Sampling

Purpose:
1. Stratified sampling for a given character (at most max_per_toptitle records per source document).
2. Uses reservoir sampling to keep memory usage bounded.
3. Supports an overall max_samples cap for baseline (comparison) characters.
4. Saves the sampled records together with summary statistics.

Usage:
python stratified_sampling.py --chars 手,天,地,下
python stratified_sampling.py --chars 手 --max-per-toptitle 250
python stratified_sampling.py --chars 天,地,下 --max-samples 10000
"""

import json
import numpy as np
import pickle
import time
import sys
import os
import argparse
import random
from typing import List, Dict, Tuple, Optional
from collections import defaultdict
from tqdm import tqdm

# ============================================================
# CONFIG
# ============================================================

# JSONL file locations for the eight dynastic periods.
# Format: /mnt/md0/public/dicwn/lrec-workshop/cleaned_embeddings/{dynasty}/{char}_embeddings.jsonl
# Note: only the directory is defined here; the actual file path is built at
# runtime.
DYNASTY_DIRS = {
    "pre-qin": "/mnt/md0/public/dicwn/lrec-workshop/cleaned_embeddings/pre-qin",
    "qinhan": "/mnt/md0/public/dicwn/lrec-workshop/cleaned_embeddings/qinhan",
    "weijin": "/mnt/md0/public/dicwn/lrec-workshop/cleaned_embeddings/weijin",
    "suitang": "/mnt/md0/public/dicwn/lrec-workshop/cleaned_embeddings/suitang",
    "songyuan": "/mnt/md0/public/dicwn/lrec-workshop/cleaned_embeddings/songyuan",
    "ming": "/mnt/md0/public/dicwn/lrec-workshop/cleaned_embeddings/ming",
    "qing": "/mnt/md0/public/dicwn/lrec-workshop/cleaned_embeddings/qing",
    "republican": "/mnt/md0/public/dicwn/lrec-workshop/cleaned_embeddings/republican",
}

DYNASTY_NAMES = {
    "pre-qin": "Pre-Qin",
    "qinhan": "Qin-Han",
    "weijin": "Wei-Jin",
    "suitang": "Sui-Tang",
    "songyuan": "Song-Yuan",
    "ming": "Ming",
    "qing": "Qing",
    "republican": "Republican",
}

DYNASTY_ORDER = ["pre-qin", "qinhan", "weijin", "suitang", "songyuan", "ming", "qing", "republican"]

# Defaults
DEFAULT_MAX_PER_TOPTITLE = 200
DEFAULT_OUTPUT_DIR = "stratified_sampling_output_cleaned"

# ============================================================
# HELPER FUNCTIONS
# ============================================================

def log(message):
    """Print with timestamp."""
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] {message}")
    sys.stdout.flush()


def build_dynasty_dirs(cleaned_root: Optional[str]) -> Dict[str, str]:
    """Build dynasty->directory mapping from a cleaned embeddings root.

    If cleaned_root is None, keep backward-compatible hardcoded paths.
    """
    if not cleaned_root:
        return DYNASTY_DIRS.copy()

    root = os.path.abspath(cleaned_root)
    return {dynasty: os.path.join(root, dynasty) for dynasty in DYNASTY_ORDER}


def compute_shannon_entropy(distribution: Dict[str, int]) -> float:
    """Compute Shannon entropy as a measure of source-document diversity."""
    total = sum(distribution.values())
    if total == 0:
        return 0.0

    probs = np.array([count / total for count in distribution.values()])
    # Avoid log(0)
    probs = probs[probs > 0]
    entropy = -np.sum(probs * np.log2(probs))
    return float(entropy)


# ============================================================
# STRATIFIED SAMPLING (reservoir sampling)
# ============================================================

def stratified_sample_reservoir(
    dynasty_dir: str,
    target_char: str,
    max_per_toptitle: int = 200,
    max_samples: Optional[int] = None,
    random_seed: int = 42
) -> Tuple[List[Dict], Dict[str, Dict]]:
    """
    Stratified sampling via the reservoir sampling algorithm.

    Args:
        dynasty_dir: path to the dynasty's embeddings directory
        target_char: target character
        max_per_toptitle: maximum records to keep per source document (toptitle)
        max_samples: overall sample cap (None means no cap)
        random_seed: random seed

    Returns:
        (sampled_records, statistics)
        - sampled_records: List[Dict] of sampled records
        - statistics: Dict[str, Dict] of summary statistics
    """
    random.seed(random_seed)

    # Build the file path: {dynasty_dir}/{char}_embeddings.jsonl
    jsonl_path = os.path.join(dynasty_dir, f"{target_char}_embeddings.jsonl")

    # Maintain one reservoir per source document (toptitle)
    toptitle_reservoirs = defaultdict(lambda: {
        'records': [],
        'count': 0  # Total records seen so far (used by the reservoir algorithm)
    })

    seen_sentences = set()  # Deduplication
    total_scanned = 0
    total_duplicates = 0
    total_matched = 0  # Total records matching target_char

    dynasty_name = os.path.basename(dynasty_dir)

    if not os.path.exists(jsonl_path):
        log(f"  File not found: {jsonl_path}")
        return [], {}

    # Get the file size (used for the progress bar)
    file_size = os.path.getsize(jsonl_path)

    log(f"  Scanning {dynasty_name}/{target_char}_embeddings.jsonl...")

    with open(jsonl_path, 'r', encoding='utf-8') as f:
        pbar = tqdm(
            f,
            desc=f"    Scanning {dynasty_name}",
            unit=" lines",
            total=file_size // 100,  # Estimated line count
            leave=False,
            ncols=120
        )

        for line in pbar:
            total_scanned += 1

            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue

            # Check whether this record matches the target character
            if record.get('word') != target_char:
                continue

            total_matched += 1

            sentence = record.get('sentence', '')

            # Deduplicate
            if sentence in seen_sentences:
                total_duplicates += 1
                continue
            seen_sentences.add(sentence)

            # Get the source document (toptitle)
            toptitle = record.get('toptitle', 'unknown')
            reservoir = toptitle_reservoirs[toptitle]
            reservoir['count'] += 1
            k = reservoir['count']

            # Reservoir sampling
            if len(reservoir['records']) < max_per_toptitle:
                # Reservoir not yet full, append directly
                reservoir['records'].append(record)
            else:
                # Reservoir full, replace with decreasing probability
                j = random.randint(1, k)
                if j <= max_per_toptitle:
                    reservoir['records'][j - 1] = record

            # Update the progress bar
            current_total = sum(len(r['records']) for r in toptitle_reservoirs.values())
            pbar.set_postfix({
                "matched": total_matched,
                "dedup": total_duplicates,
                "sampled": current_total,
                "docs": len(toptitle_reservoirs)
            })

        pbar.close()

    # Merge all reservoirs
    all_records = []
    for toptitle, reservoir in toptitle_reservoirs.items():
        all_records.extend(reservoir['records'])

    # If max_samples is set, apply one final sampling pass
    if max_samples is not None and len(all_records) > max_samples:
        log(f"    Total samples {len(all_records)} exceeds cap {max_samples}, applying final sampling...")
        all_records = random.sample(all_records, max_samples)

    # Summary statistics
    statistics = {
        "dynasty": dynasty_name,
        "total_scanned": total_scanned,
        "total_matched": total_matched,
        "total_duplicates": total_duplicates,
        "n_toptitles": len(toptitle_reservoirs),
        "n_samples_final": len(all_records),
        "toptitle_distribution": {},
        "toptitle_stats": {}
    }

    # Per-toptitle statistics
    final_toptitle_counts = defaultdict(int)
    for record in all_records:
        toptitle = record.get('toptitle', 'unknown')
        final_toptitle_counts[toptitle] += 1

    for toptitle, reservoir in toptitle_reservoirs.items():
        original_count = reservoir['count']
        sampled_count = final_toptitle_counts.get(toptitle, 0)

        statistics["toptitle_stats"][toptitle] = {
            "original": original_count,
            "sampled": sampled_count,
            "ratio": sampled_count / original_count if original_count > 0 else 0
        }

    # Final distribution
    statistics["toptitle_distribution"] = dict(final_toptitle_counts)

    # Source-document diversity
    statistics["diversity"] = compute_shannon_entropy(final_toptitle_counts)

    # Top 10 source documents
    top_10 = sorted(final_toptitle_counts.items(), key=lambda x: x[1], reverse=True)[:10]
    statistics["top_10_toptitles"] = [
        {"toptitle": t, "count": c, "percentage": c / len(all_records) * 100}
        for t, c in top_10
    ]

    log(f"  {dynasty_name}: scanned {total_scanned:,} lines, matched {total_matched:,}, "
        f"deduplicated {total_duplicates:,}, final {len(all_records):,} records, {len(toptitle_reservoirs)} source documents")

    return all_records, statistics


# ============================================================
# PROCESS ALL DYNASTIES
# ============================================================

def process_char_all_dynasties(
    char: str,
    max_per_toptitle: int,
    max_samples: Optional[int],
    output_dir: str,
    dynasty_dirs: Dict[str, str],
) -> Dict:
    """
    Process every dynasty for a single character.

    Each dynasty is saved and released from memory as soon as it finishes.
    """
    log(f"\n{'='*60}")
    log(f"Processing character: {char}")
    log(f"  max_per_toptitle: {max_per_toptitle}")
    log(f"  max_samples: {max_samples if max_samples else 'unlimited'}")
    log(f"{'='*60}")

    # Create the character's output directory
    char_dir = os.path.join(output_dir, f"char_{char}")
    os.makedirs(char_dir, exist_ok=True)

    overall_stats = {
        "char": char,
        "max_per_toptitle": max_per_toptitle,
        "max_samples": max_samples,
        "dynasties": {}
    }

    # Progress bar over dynasties
    dynasty_pbar = tqdm(
        DYNASTY_ORDER,
        desc=f"Processing dynasties for {char}",
        unit=" dynasty",
        ncols=120
    )

    for dynasty in dynasty_pbar:
        dynasty_name = DYNASTY_NAMES.get(dynasty, dynasty)
        dynasty_pbar.set_postfix({"current": dynasty_name})

        dynasty_dir = dynasty_dirs.get(dynasty)

        if not dynasty_dir or not os.path.exists(dynasty_dir):
            log(f"  {dynasty_name}: directory not found ({dynasty_dir})")
            continue

        # Run stratified sampling
        sampled_records, statistics = stratified_sample_reservoir(
            dynasty_dir,
            char,
            max_per_toptitle=max_per_toptitle,
            max_samples=max_samples
        )

        if len(sampled_records) == 0:
            log(f"  {dynasty_name}: no records sampled")
            continue

        # Save this dynasty's data immediately (avoid accumulating memory)
        dynasty_output = {
            "char": char,
            "dynasty": dynasty,
            "dynasty_name": dynasty_name,
            "n_samples": len(sampled_records),
            "statistics": statistics,
            "records": sampled_records
        }

        dynasty_file = os.path.join(char_dir, f"{dynasty}.pkl")
        with open(dynasty_file, 'wb') as f:
            pickle.dump(dynasty_output, f)

        # Also save the statistics as JSON for easy inspection
        stats_file = os.path.join(char_dir, f"{dynasty}_stats.json")
        stats_json = {
            "char": char,
            "dynasty": dynasty,
            "dynasty_name": dynasty_name,
            "n_samples": len(sampled_records),
            "statistics": statistics
        }
        with open(stats_file, 'w', encoding='utf-8') as f:
            json.dump(stats_json, f, ensure_ascii=False, indent=2)

        log(f"    Saved: {dynasty_file}")

        # Record into the overall summary
        overall_stats["dynasties"][dynasty] = {
            "dynasty_name": dynasty_name,
            "n_samples": len(sampled_records),
            "n_toptitles": statistics["n_toptitles"],
            "diversity": statistics["diversity"],
            "top_3_toptitles": statistics["top_10_toptitles"][:3]
        }

        # Free memory
        del sampled_records
        del dynasty_output

    dynasty_pbar.close()

    # Save the overall summary
    overall_file = os.path.join(char_dir, "overall_stats.json")
    with open(overall_file, 'w', encoding='utf-8') as f:
        json.dump(overall_stats, f, ensure_ascii=False, indent=2)

    log(f"\nFinished processing '{char}'. Overall statistics saved: {overall_file}")

    # Print a summary
    log(f"\n{'='*60}")
    log(f"Summary for '{char}':")
    log(f"{'='*60}")
    for dynasty in DYNASTY_ORDER:
        if dynasty in overall_stats["dynasties"]:
            info = overall_stats["dynasties"][dynasty]
            log(f"  {info['dynasty_name']:10s}: {info['n_samples']:7,} records, "
                f"{info['n_toptitles']:4} source documents, "
                f"diversity={info['diversity']:.2f}")

    return overall_stats


# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser(description='Stratified Sampling')
    parser.add_argument('--chars', type=str, required=True,
                       help='Characters to process, comma-separated (e.g. 手,天,地,下)')
    parser.add_argument('--max-per-toptitle', type=int, default=DEFAULT_MAX_PER_TOPTITLE,
                       help=f'Maximum records to keep per source document (default {DEFAULT_MAX_PER_TOPTITLE})')
    parser.add_argument('--max-samples', type=int, default=None,
                       help='Overall sample cap (default None, unlimited). Recommended as 10000 for baseline characters')
    parser.add_argument('--output', type=str, default=DEFAULT_OUTPUT_DIR,
                       help=f'Output directory (default {DEFAULT_OUTPUT_DIR})')
    parser.add_argument('--baseline-max-samples', type=int, default=10000,
                       help='max_samples used for baseline characters (default 10000)')
    parser.add_argument('--target-chars', type=str, default='手',
                       help='Target characters, comma-separated; these characters are not subject to max_samples (default: 手)')
    parser.add_argument('--cleaned-root', type=str, default=None,
                       help='Root directory of cleaned_embeddings (default: use the built-in paths)')

    args = parser.parse_args()

    # Parse arguments
    chars = [c.strip() for c in args.chars.split(',')]
    target_chars_set = set(c.strip() for c in args.target_chars.split(','))
    output_dir = args.output
    max_per_toptitle = args.max_per_toptitle
    dynasty_dirs = build_dynasty_dirs(args.cleaned_root)

    log("=" * 60)
    log("Stratified Sampling")
    log("=" * 60)
    log(f"Characters: {chars}")
    log(f"Target characters (no max_samples cap): {list(target_chars_set)}")
    log(f"max_per_toptitle: {max_per_toptitle}")
    log(f"Baseline max_samples: {args.baseline_max_samples}")
    log(f"cleaned_root: {args.cleaned_root if args.cleaned_root else 'built-in default paths'}")
    log(f"Output directory: {output_dir}")
    log("=" * 60)

    # Create the output directory
    os.makedirs(output_dir, exist_ok=True)

    # Process each character
    all_chars_stats = {}

    for char in chars:
        # Determine whether this is a target character or a baseline character
        if char in target_chars_set:
            max_samples = None  # Target characters have no cap
            log(f"\n[Target character] {char} (no max_samples cap)")
        else:
            max_samples = args.baseline_max_samples  # Baseline characters are capped
            log(f"\n[Baseline character] {char} (max_samples = {max_samples})")

        char_stats = process_char_all_dynasties(
            char,
            max_per_toptitle=max_per_toptitle,
            max_samples=max_samples,
            output_dir=output_dir,
            dynasty_dirs=dynasty_dirs
        )

        all_chars_stats[char] = char_stats

    # Save the summary across all characters
    summary_file = os.path.join(output_dir, "summary.json")
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(all_chars_stats, f, ensure_ascii=False, indent=2)

    log("\n" + "=" * 60)
    log("Done.")
    log("=" * 60)
    log(f"Output directory: {output_dir}/")
    log("  - char_{char}/: sampling results for each character")
    log("    - {dynasty}.pkl: sampled data for that dynasty (includes the full records)")
    log("    - {dynasty}_stats.json: statistics for that dynasty")
    log("    - overall_stats.json: overall statistics for that character")
    log("  - summary.json: overall statistics across all characters")
    log("\nUsage examples:")
    log("  # Process all four characters")
    log("  python stratified_sampling.py --chars 手,天,地,下")
    log("")
    log("  # Custom parameters")
    log("  python stratified_sampling.py --chars 手,天,地,下 \\")
    log("    --max-per-toptitle 250 \\")
    log("    --baseline-max-samples 15000")


if __name__ == "__main__":
    main()

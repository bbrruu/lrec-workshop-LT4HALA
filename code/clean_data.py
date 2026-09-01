#!/usr/bin/env python3
"""
Classical Chinese data cleaning tool.

Applies a multi-layer filter targeted at OCR quality issues:
1. Sentence length (both too long and too short are problematic)
2. Punctuation density (OCR errors often drop punctuation) -- computed but not used, see below
3. Character repetition rate (an OCR error signature)
4. Consecutive identical characters (an unambiguous error)
5. Invalid characters (square glyph, replacement character, etc.)
6. Character diversity (uniqueness ratio)
7. Digit density (classical text should not contain too many digits)

Usage:
python clean_data.py \
  --input stratified_sampling_output_cleaned \
  --output stratified_sampling_output_ULTRA_cleaned \
  --char 手

Optional arguments:
  --aggression [mild|normal|aggressive|ultra]  # Cleaning aggression level
  --show-examples  # Show examples of filtered sentences
  --dry-run        # Report statistics only, without writing output
"""

import pickle
import json
import os
import re
import argparse
from collections import defaultdict, Counter
from typing import List, Dict, Tuple
import numpy as np
from tqdm import tqdm

# ============================================================
# FILTER RULE CONFIGURATION
# ============================================================

FILTER_CONFIGS = {
    "mild": {
        "min_length": 3,
        "max_length": 300,
        "min_punct_density": 0.01,  # kept for compatibility, unused
        "min_unique_ratio": 0.30,  # raised slightly to compensate for not checking punctuation
        "max_repeat_char": 7,
        "max_digit_density": 0.25,
        "description": "Mild cleaning (retains more data)"
    },
    "normal": {
        "min_length": 4,
        "max_length": 200,
        "min_punct_density": 0.015,  # kept for compatibility, unused
        "min_unique_ratio": 0.35,  # raised to compensate for not checking punctuation
        "max_repeat_char": 5,     # stricter
        "max_digit_density": 0.20,  # stricter
        "description": "Standard cleaning (recommended, no punctuation check)"
    },
    "aggressive": {
        "min_length": 5,
        "max_length": 150,
        "min_punct_density": 0.02,
        "min_unique_ratio": 0.40,  # stricter
        "max_repeat_char": 4,
        "max_digit_density": 0.15,
        "description": "Aggressive cleaning (removes substantially more data)"
    },
    "ultra": {
        "min_length": 5,
        "max_length": 120,
        "min_punct_density": 0.025,
        "min_unique_ratio": 0.45,  # very strict
        "max_repeat_char": 3,
        "max_digit_density": 0.10,
        "description": "Ultra-aggressive cleaning (retains only the highest-quality data)"
    }
}

# Common punctuation marks
CHINESE_PUNCTUATION = r'[。，、；：？！「」『』（）《》〈〉【】〔〕…—·]'

# Invalid characters
INVALID_CHARS = r'[�□▪■●○◎★☆◇◆△▲▽▼]'

# Digits
DIGITS = r'[0-9０-９一二三四五六七八九十百千萬億兆壹貳參肆伍陸柒捌玖拾佰仟]'

# ============================================================
# QUALITY CHECK FUNCTIONS
# ============================================================

def check_length(sentence: str, config: dict) -> Tuple[bool, str]:
    """Check sentence length."""
    length = len(sentence)
    if length < config['min_length']:
        return False, f"too_short({length})"
    if length > config['max_length']:
        return False, f"too_long({length})"
    return True, "OK"


def check_punctuation_density(sentence: str, config: dict) -> Tuple[bool, str]:
    """Check punctuation density."""
    if len(sentence) == 0:
        return False, "empty_sentence"

    punct_count = len(re.findall(CHINESE_PUNCTUATION, sentence))
    density = punct_count / len(sentence)

    if density < config['min_punct_density']:
        return False, f"low_punctuation_density({density:.3f})"

    return True, "OK"


def check_unique_ratio(sentence: str, config: dict) -> Tuple[bool, str]:
    """Check character diversity (uniqueness ratio)."""
    if len(sentence) == 0:
        return False, "empty_sentence"

    unique_chars = len(set(sentence))
    ratio = unique_chars / len(sentence)

    if ratio < config['min_unique_ratio']:
        return False, f"low_uniqueness_ratio({ratio:.3f})"

    return True, "OK"


def check_repeat_chars(sentence: str, config: dict) -> Tuple[bool, str]:
    """Check for consecutively repeated characters."""
    # Find the longest run of a repeated character
    max_repeat = 0
    current_char = None
    current_count = 0

    for char in sentence:
        if char == current_char:
            current_count += 1
            max_repeat = max(max_repeat, current_count)
        else:
            current_char = char
            current_count = 1

    if max_repeat > config['max_repeat_char']:
        return False, f"excessive_repetition({max_repeat})"

    return True, "OK"


def check_invalid_chars(sentence: str) -> Tuple[bool, str]:
    """Check for invalid characters."""
    invalid = re.findall(INVALID_CHARS, sentence)
    if invalid:
        return False, f"invalid_chars({','.join(set(invalid))})"
    return True, "OK"


def check_digit_density(sentence: str, config: dict) -> Tuple[bool, str]:
    """Check digit density."""
    if len(sentence) == 0:
        return False, "empty_sentence"

    digit_count = len(re.findall(DIGITS, sentence))
    density = digit_count / len(sentence)

    if density > config['max_digit_density']:
        return False, f"high_digit_density({density:.3f})"

    return True, "OK"


def check_abnormal_patterns(sentence: str) -> Tuple[bool, str]:
    """
    Check for abnormal patterns (additional heuristic rules).

    v2: loosened thresholds to reduce false positives.
    """

    # 1. Check whether the sentence is entirely punctuation
    if re.match(r'^[' + CHINESE_PUNCTUATION + r']+$', sentence):
        return False, "all_punctuation"

    # 2. Check for excessive whitespace
    if sentence.count(' ') > len(sentence) * 0.1:
        return False, "excessive_whitespace"

    # 3. Check for abnormal character patterns (e.g. aaabbbccc)
    pattern = r'(.)\1{2,}'
    matches = re.findall(pattern, sentence)
    if len(matches) >= 3:
        return False, "abnormal_repeat_pattern"

    # 4. Loosened: overly long with no punctuation at all (not just the
    #    sentence-splitting marks). Raised from 100 to 150 characters and
    #    now checks against the full punctuation set.
    if len(sentence) > 150:
        punct_count = len(re.findall(CHINESE_PUNCTUATION, sentence))
        if punct_count == 0:
            return False, "overlong_no_punctuation_ocr_merge"

    # 5. Loosened: excessive rare characters. Threshold lowered so a segment
    #    only counts as low-common if under 10% of it is common characters
    #    (was 20%).
    common_chars = set('的一是不了人我有他這為之大來以個中上們到說國和地也子時道出而要於就下得可你年生自會那後能對著事其裡所去行過家十用發天如然作方成者多日都三小軍二無同麼經法當起與好看學進種將還分此心前面又定見只主沒公從')

    if len(sentence) >= 10:
        segments = [sentence[i:i+10] for i in range(0, len(sentence)-9, 10)]
        low_common_count = 0

        for seg in segments:
            common_ratio = sum(1 for c in seg if c in common_chars) / len(seg)
            if common_ratio < 0.10:  # threshold lowered to 10% (was 20%)
                low_common_count += 1

        # Now requires over 80% of segments to be low-common (was 50%) before
        # flagging as abnormal.
        if low_common_count > len(segments) * 0.8:
            return False, "low_common_char_ratio_possible_ocr_error"

    # 6. Unchanged: excessive rare/uncommon characters
    rare_char_count = 0
    for char in sentence:
        if '㐀' <= char <= '䶿' or '\U00020000' <= char <= '\U0002A6DF':
            rare_char_count += 1

    if len(sentence) > 0 and rare_char_count / len(sentence) > 0.3:
        return False, "excessive_rare_chars"

    return True, "OK"


def quality_score(sentence: str, config: dict) -> Tuple[float, Dict[str, str]]:
    """
    Compute an overall quality score (0-1).

    Note: punctuation density is intentionally not used as a check.
    Reason: sentences are already split on "。；？！" during embedding
    extraction, so many otherwise-normal sentences lack trailing
    punctuation (it was consumed as the sentence delimiter).

    Returns:
        score: quality score in [0, 1]
        reasons: result of each individual check
    """
    checks = {
        "length": check_length(sentence, config),
        # "punctuation": check_punctuation_density(sentence, config),  # not used, see docstring
        "uniqueness": check_unique_ratio(sentence, config),
        "repeat": check_repeat_chars(sentence, config),
        "invalid_chars": check_invalid_chars(sentence),
        "digits": check_digit_density(sentence, config),
        "patterns": check_abnormal_patterns(sentence)
    }

    # Fraction of checks passed
    passed = sum(1 for ok, _ in checks.values() if ok)
    score = passed / len(checks)

    # Collect the reasons for any failed checks
    reasons = {name: reason for name, (ok, reason) in checks.items() if not ok}

    return score, reasons


# ============================================================
# CLEANING FUNCTIONS
# ============================================================

def clean_records(
    records: List[Dict],
    config: dict,
    show_examples: bool = False,
    max_examples: int = 10
) -> Tuple[List[Dict], Dict]:
    """
    Clean a list of records.

    Returns:
        cleaned_records: the retained records
        stats: summary statistics
    """
    cleaned = []
    filtered_reasons = defaultdict(int)
    filtered_examples = defaultdict(list)

    for record in tqdm(records, desc="    Filtering", leave=False):
        sentence = record.get('sentence', '')

        score, reasons = quality_score(sentence, config)

        if score == 1.0:  # Passed every check
            cleaned.append(record)
        else:
            # Record the filtering reasons
            for reason_key, reason_msg in reasons.items():
                filtered_reasons[reason_key] += 1

                # Keep an example for reporting
                if len(filtered_examples[reason_key]) < max_examples:
                    filtered_examples[reason_key].append({
                        'sentence': sentence,
                        'reason': reason_msg,
                        'score': score
                    })

    stats = {
        'total': len(records),
        'kept': len(cleaned),
        'filtered': len(records) - len(cleaned),
        'keep_ratio': len(cleaned) / len(records) if records else 0,
        'filter_reasons': dict(filtered_reasons),
        'examples': dict(filtered_examples) if show_examples else {}
    }

    return cleaned, stats


def process_dynasty(
    input_file: str,
    output_file: str,
    dynasty: str,
    config: dict,
    show_examples: bool = False,
    dry_run: bool = False
) -> Dict:
    """Process a single dynasty."""

    if not os.path.exists(input_file):
        return None

    print(f"\n  Processing {dynasty}...")

    # Load
    with open(input_file, 'rb') as f:
        data = pickle.load(f)

    original_records = data['records']
    print(f"    Original: {len(original_records):,} sentences")

    # Clean
    cleaned_records, stats = clean_records(
        original_records,
        config,
        show_examples
    )

    print(f"    Kept: {len(cleaned_records):,} sentences ({stats['keep_ratio']:.1%})")
    print(f"    Filtered: {stats['filtered']:,} sentences")

    # Print filter reason statistics
    if stats['filter_reasons']:
        print("    Filter reasons:")
        for reason, count in sorted(stats['filter_reasons'].items(),
                                    key=lambda x: -x[1]):
            print(f"      - {reason}: {count:,} sentences")

    # Print example sentences
    if show_examples and stats['examples']:
        print("\n    Filtered examples (up to 3 per category):")
        for reason, examples in stats['examples'].items():
            print(f"\n      [{reason}]")
            for ex in examples[:3]:
                print(f"        ({ex['score']:.2f}) {ex['sentence'][:50]}...")
                print(f"             -> {ex['reason']}")

    # Save (unless this is a dry run)
    if not dry_run:
        os.makedirs(os.path.dirname(output_file), exist_ok=True)

        # Update the record statistics
        data['records'] = cleaned_records
        data['statistics']['n_samples'] = len(cleaned_records)

        # Attach the cleaning provenance
        data['cleaning_info'] = {
            'config': config,
            'stats': {k: v for k, v in stats.items() if k != 'examples'}
        }

        with open(output_file, 'wb') as f:
            pickle.dump(data, f)

        print(f"    Saved: {output_file}")

    return stats


# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description='Classical Chinese data cleaning tool (no punctuation check)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Filter rules (punctuation check removed):

  Punctuation density is not used as a check.
  Reason: sentences were already split on "。；？！" during embedding
  extraction, so many normal sentences lack trailing punctuation.

  The 6 checks actually used:

  1. Length: sentences that are too short or too long (possible split errors)
  2. Character diversity: repetition rate too high (repeated OCR misreads)
     - unique_ratio = unique character count / total character count
     - normal sentences are typically > 0.35
  3. Consecutive repetition: the same character repeated too many times (an unambiguous error)
  4. Invalid characters: square glyph, replacement character, and other invalid characters
  5. Digit density: too many digits (e.g. page numbers)
  6. Abnormal patterns: heuristic rules
     - overly long with no punctuation (OCR segmentation merge)
     - too many rare characters (possible OCR misrecognition)
     - too many uncommon characters
     - common-character ratio too low

Aggression levels (v2, loosened thresholds):
  mild:       mild cleaning, retains ~92% of data
  normal:     standard cleaning, retains ~88% of data (recommended)
  aggressive: aggressive cleaning, retains ~80% of data
  ultra:      ultra-aggressive, retains only ~70% of the highest-quality data

Examples:
  # Standard cleaning (recommended)
  python clean_data.py --char 手 --aggression normal

  # Preview which sentences would be filtered
  python clean_data.py --char 手 --aggression normal --show-examples --dry-run

  # Report statistics only, without writing output
  python clean_data.py --char 手 --dry-run
        """
    )

    parser.add_argument('--char', type=str, required=True,
                       help='Character to process')
    parser.add_argument('--input', type=str,
                       default='stratified_sampling_output_cleaned',
                       help='Input directory')
    parser.add_argument('--output', type=str,
                       default='stratified_sampling_output_ULTRA_cleaned',
                       help='Output directory')
    parser.add_argument('--aggression', type=str,
                       default='normal',
                       choices=['mild', 'normal', 'aggressive', 'ultra'],
                       help='Cleaning aggression level')
    parser.add_argument('--show-examples', action='store_true',
                       help='Show examples of filtered sentences')
    parser.add_argument('--dry-run', action='store_true',
                       help='Report statistics only, without writing output')
    parser.add_argument('--dynasties', type=str, default=None,
                       help='Restrict to specific dynasties (comma-separated), e.g. ming,qing')

    args = parser.parse_args()

    # Setup
    config = FILTER_CONFIGS[args.aggression]
    char = args.char

    print("=" * 60)
    print("Classical Chinese Data Cleaning Tool")
    print("=" * 60)
    print(f"Character: {char}")
    print(f"Aggression: {args.aggression} - {config['description']}")
    print(f"Input: {args.input}")
    print(f"Output: {args.output}")
    if args.dry_run:
        print("Mode: DRY RUN (no files will be written)")
    print("=" * 60)

    print("\nFilter rules:")
    print("  Punctuation density is not checked (sentences were already split on punctuation during embedding extraction)")
    print(f"  Length: {config['min_length']}-{config['max_length']} characters")
    print(f"  Minimum character diversity: {config['min_unique_ratio']:.3f}")
    print(f"  Maximum consecutive repetition: {config['max_repeat_char']} characters")
    print(f"  Maximum digit density: {config['max_digit_density']:.3f}")
    print("  Abnormal patterns: overly long with no punctuation, too many rare characters, too many uncommon characters, etc.")

    # Determine which dynasties to process
    if args.dynasties:
        dynasties = [d.strip() for d in args.dynasties.split(',')]
    else:
        dynasties = ["pre-qin", "qinhan", "weijin", "suitang",
                    "songyuan", "ming", "qing", "republican"]

    # Process each dynasty
    all_stats = {}

    for dynasty in dynasties:
        input_file = os.path.join(args.input, f"char_{char}", f"{dynasty}.pkl")
        output_file = os.path.join(args.output, f"char_{char}", f"{dynasty}.pkl")

        stats = process_dynasty(
            input_file, output_file, dynasty, config,
            args.show_examples, args.dry_run
        )

        if stats:
            all_stats[dynasty] = stats

    # Summary
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)

    total_original = sum(s['total'] for s in all_stats.values())
    total_kept = sum(s['kept'] for s in all_stats.values())
    total_filtered = sum(s['filtered'] for s in all_stats.values())

    print("\nTotal:")
    print(f"  Original: {total_original:,} sentences")
    print(f"  Kept: {total_kept:,} sentences ({total_kept/total_original:.1%})")
    print(f"  Filtered: {total_filtered:,} sentences ({total_filtered/total_original:.1%})")

    # Per-dynasty statistics
    print("\nRetention rate by dynasty:")
    for dynasty in dynasties:
        if dynasty in all_stats:
            s = all_stats[dynasty]
            print(f"  {dynasty:10s}: {s['kept']:6,}/{s['total']:6,} "
                  f"({s['keep_ratio']:5.1%}) - "
                  f"filtered {s['filtered']:,}")

    # Aggregate filter reasons
    print("\nFilter reason summary:")
    reason_totals = defaultdict(int)
    for stats in all_stats.values():
        for reason, count in stats['filter_reasons'].items():
            reason_totals[reason] += count

    for reason, count in sorted(reason_totals.items(), key=lambda x: -x[1]):
        pct = count / total_filtered * 100 if total_filtered > 0 else 0
        print(f"  {reason:15s}: {count:6,} ({pct:5.1f}%)")

    if not args.dry_run:
        # Save statistics
        summary_file = os.path.join(args.output, f"char_{char}", "cleaning_summary.json")
        os.makedirs(os.path.dirname(summary_file), exist_ok=True)

        summary = {
            'char': char,
            'aggression': args.aggression,
            'config': config,
            'total': {
                'original': total_original,
                'kept': total_kept,
                'filtered': total_filtered,
                'keep_ratio': total_kept / total_original if total_original > 0 else 0
            },
            'by_dynasty': {
                dynasty: {k: v for k, v in stats.items() if k != 'examples'}
                for dynasty, stats in all_stats.items()
            },
            'filter_reasons': dict(reason_totals)
        }

        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)

        print(f"\nStatistics saved: {summary_file}")

    print("\n" + "=" * 60)
    if args.dry_run:
        print("DRY RUN complete - no files were written")
    else:
        print("Cleaning complete.")
    print("=" * 60)


if __name__ == "__main__":
    main()

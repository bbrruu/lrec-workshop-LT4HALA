#!/usr/bin/env python3
"""
Remove OCR-error records from extracted embedding data.

Strategy:
1. Conservative filtering: only remove unambiguous OCR errors (marker glyphs, overly long sentences).
2. Aggressive filtering: also remove rare characters, overly short sentences, and other flagged cases.
3. Preserve the original data and write results to a new directory.

Usage:
  python clean_ocr_errors.py --char 手 --mode conservative
  python clean_ocr_errors.py --char 手 --mode aggressive --output cleaned_embeddings/
"""

import json
import re
import sys
from pathlib import Path
from collections import Counter
import argparse

# Ensure project root is importable when script is run from mainfiles/
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from detect_ocr_errors import OCRErrorDetector


class DataCleaner:
    def __init__(self, mode='conservative'):
        """
        mode: 'conservative', 'aggressive', or 'custom'.
        """
        self.mode = mode
        self.detector = OCRErrorDetector()
        self.stats = {
            'total': 0,
            'removed': 0,
            'reasons': Counter()
        }

    def should_remove(self, sentence):
        """Decide whether a record should be removed."""
        errors = self.detector.detect_errors(sentence)

        if not errors:
            return False, None

        # Conservative mode: remove only unambiguous OCR errors
        if self.mode == 'conservative':
            # 1. OCR marker glyphs present -> remove
            if 'ocr_markers' in errors:
                return True, 'ocr_markers'

            # 2. Sentence too long (likely a segmentation error) -> remove
            if 'length_anomaly' in errors:
                anomaly = errors['length_anomaly']
                if 'too_long' in anomaly:
                    length = int(anomaly.split('(')[1].strip(')'))
                    if length > 300:  # Over 300 characters is treated as anomalous
                        return True, 'too_long'

            # 3. Excessive character repetition -> remove
            if 'suspicious_patterns' in errors:
                for pattern, matches in errors['suspicious_patterns']:
                    if r'\1{4,}' in pattern:  # Same character repeated 5+ times
                        return True, 'excessive_repetition'

            return False, None

        # Aggressive mode: remove any record with a detected issue
        elif self.mode == 'aggressive':
            reason = list(errors.keys())[0]  # Use the first detected error type as the reason
            return True, reason

        # Custom mode: decided by finer-grained rules
        elif self.mode == 'custom':
            # Additional rules can be added here, e.g. allow certain rare
            # characters while still removing OCR marker glyphs.
            if 'ocr_markers' in errors:
                return True, 'ocr_markers'

            if 'length_anomaly' in errors:
                anomaly = errors['length_anomaly']
                if 'too_short' in anomaly:
                    length = int(anomaly.split('(')[1].strip(')'))
                    if length < 5:  # Under 5 characters may be a segmentation issue
                        return True, 'too_short'
                elif 'too_long' in anomaly:
                    length = int(anomaly.split('(')[1].strip(')'))
                    if length > 200:
                        return True, 'too_long'

            if 'suspicious_patterns' in errors:
                return True, 'suspicious_patterns'

            return False, None

    def clean_file(self, input_path, output_path):
        """Clean a single embeddings file."""
        file_stats = {
            'total': 0,
            'removed': 0,
            'kept': 0,
            'reasons': Counter()
        }

        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(input_path, 'r', encoding='utf-8') as fin, \
             open(output_path, 'w', encoding='utf-8') as fout:

            for line in fin:
                try:
                    data = json.loads(line)
                    sentence = data.get('sentence', '')

                    file_stats['total'] += 1
                    self.stats['total'] += 1

                    should_del, reason = self.should_remove(sentence)

                    if should_del:
                        file_stats['removed'] += 1
                        file_stats['reasons'][reason] += 1
                        self.stats['removed'] += 1
                        self.stats['reasons'][reason] += 1
                    else:
                        file_stats['kept'] += 1
                        fout.write(line)

                except Exception as e:
                    print(f"  Error processing record: {e}")
                    continue

        return file_stats


def main():
    parser = argparse.ArgumentParser(description='Remove OCR-error records from embedding data')
    parser.add_argument('--char', type=str, default='手', help='Character to clean')
    parser.add_argument('--dynasty', type=str, help='Restrict to a single dynasty (default: all)')
    parser.add_argument('--mode', type=str, default='conservative',
                        choices=['conservative', 'aggressive', 'custom'],
                        help='Cleaning mode: conservative, aggressive, or custom')
    parser.add_argument('--input-dir', type=str, default='keyword_embeddings',
                        help='Input directory')
    parser.add_argument('--output-dir', type=str, default='cleaned_embeddings',
                        help='Output directory')
    parser.add_argument('--dry-run', action='store_true',
                        help='Dry-run mode (report statistics without writing files)')

    args = parser.parse_args()

    cleaner = DataCleaner(mode=args.mode)
    input_base = Path(args.input_dir)
    output_base = Path(args.output_dir)

    # Determine which dynasties to process
    if args.dynasty:
        dynasties = [args.dynasty]
    else:
        dynasties = ['pre-qin', 'qinhan', 'weijin', 'suitang', 'songyuan', 'ming', 'qing', 'republican']

    print(f"Cleaning OCR errors for character '{args.char}'...")
    print(f"Mode: {args.mode}")
    print(f"{'[Dry run - no files will be written]' if args.dry_run else ''}")
    print("=" * 80)

    dynasty_results = {}

    for dynasty in dynasties:
        input_path = input_base / dynasty / f"{args.char}_embeddings.jsonl"
        output_path = output_base / dynasty / f"{args.char}_embeddings.jsonl"

        if not input_path.exists():
            print(f"File not found: {input_path}")
            continue

        print(f"\nDynasty: {dynasty}")
        print("-" * 80)

        if args.dry_run:
            # Dry-run mode: compute statistics without writing output
            stats = {'total': 0, 'removed': 0, 'kept': 0, 'reasons': Counter()}
            with open(input_path, 'r', encoding='utf-8') as f:
                for line in f:
                    try:
                        data = json.loads(line)
                        sentence = data.get('sentence', '')
                        stats['total'] += 1
                        should_del, reason = cleaner.should_remove(sentence)
                        if should_del:
                            stats['removed'] += 1
                            stats['reasons'][reason] += 1
                        else:
                            stats['kept'] += 1
                    except:
                        continue
        else:
            stats = cleaner.clean_file(input_path, output_path)

        dynasty_results[dynasty] = stats

        # Print statistics
        removal_rate = (stats['removed'] / stats['total'] * 100) if stats['total'] > 0 else 0
        print(f"Total: {stats['total']}")
        print(f"Removed: {stats['removed']} ({removal_rate:.2f}%)")
        print(f"Kept: {stats['kept']} ({100-removal_rate:.2f}%)")

        if stats.get('reasons'):
            print("Removal reasons:")
            for reason, count in stats['reasons'].most_common():
                print(f"  - {reason}: {count}")

        if not args.dry_run:
            print(f"Saved to: {output_path}")

    # Summary
    print("\n" + "=" * 80)
    print("Summary")
    print("=" * 80)

    total_all = sum(r['total'] for r in dynasty_results.values())
    removed_all = sum(r['removed'] for r in dynasty_results.values())
    kept_all = sum(r['kept'] for r in dynasty_results.values())
    overall_removal_rate = (removed_all / total_all * 100) if total_all > 0 else 0

    print(f"Total: {total_all}")
    print(f"Removed: {removed_all} ({overall_removal_rate:.2f}%)")
    print(f"Kept: {kept_all} ({100-overall_removal_rate:.2f}%)")

    print("\nRemoval rate by dynasty:")
    for dynasty, stats in sorted(dynasty_results.items()):
        if stats['total'] > 0:
            rate = stats['removed'] / stats['total'] * 100
            print(f"  {dynasty:15s}: {rate:6.2f}% ({stats['removed']}/{stats['total']})")

    print("\nRemoval reason distribution:")
    all_reasons = Counter()
    for stats in dynasty_results.values():
        all_reasons.update(stats.get('reasons', {}))
    for reason, count in all_reasons.most_common():
        print(f"  - {reason}: {count}")

    if args.dry_run:
        print("\nThis was a dry run. Remove --dry-run to write the cleaned data.")
    else:
        print(f"\nDone. Cleaned data saved to: {args.output_dir}/")


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""
Detect OCR errors in extracted embedding records.

Common OCR error signatures:
1. Rare character combinations
2. Anomalous glyphs (e.g. common OCR failure markers such as square, circle, asterisk)
3. Non-Han characters mixed into the text (aside from standard punctuation)
4. Excessive character repetition
5. Abnormal sentence length
"""

import json
import re
from pathlib import Path
from collections import Counter
import argparse


class OCRErrorDetector:
    def __init__(self):
        # Common OCR error marker glyphs
        self.ocr_error_markers = set('□〇○●◎◇◆※☆★■▪▲△▽')

        # Allowed punctuation
        self.allowed_punctuation = set('，。、；：！？「」『』（）〈〉《》【】…—·')

        # Suspicious character patterns
        self.suspicious_patterns = [
            r'(.)\1{4,}',  # Same character repeated 5+ times
            r'[A-Za-z]{2,}',  # 2+ consecutive Latin letters
            r'\d{6,}',  # 6+ consecutive digits
        ]

    def check_ocr_markers(self, text):
        """Check whether the text contains OCR error marker glyphs."""
        markers_found = [char for char in text if char in self.ocr_error_markers]
        return markers_found

    def check_suspicious_patterns(self, text):
        """Check the text against suspicious character patterns."""
        found = []
        for pattern in self.suspicious_patterns:
            matches = re.findall(pattern, text)
            if matches:
                found.append((pattern, matches))
        return found

    def check_unusual_chars(self, text):
        """Check for characters that are neither Han nor standard punctuation."""
        unusual = []
        for char in text:
            # Skip the Han character range
            if '\u4e00' <= char <= '\u9fff':
                continue
            # Skip standard punctuation
            if char in self.allowed_punctuation:
                continue
            # Skip digits (a small number of digits is normal)
            if char.isdigit():
                continue
            # Skip whitespace
            if char.isspace():
                continue

            unusual.append(char)
        return unusual

    def check_length_anomaly(self, text, min_len=5, max_len=500):
        """Check for abnormal sentence length."""
        length = len(text)
        if length < min_len:
            return f"too_short({length})"
        if length > max_len:
            return f"too_long({length})"
        return None

    def detect_errors(self, sentence):
        """Run the full set of OCR error checks on a sentence."""
        errors = {}

        # OCR marker check
        markers = self.check_ocr_markers(sentence)
        if markers:
            errors['ocr_markers'] = markers

        # Suspicious pattern check
        patterns = self.check_suspicious_patterns(sentence)
        if patterns:
            errors['suspicious_patterns'] = patterns

        # Unusual character check
        unusual = self.check_unusual_chars(sentence)
        if unusual:
            errors['unusual_chars'] = unusual

        # Length check
        length_issue = self.check_length_anomaly(sentence)
        if length_issue:
            errors['length_anomaly'] = length_issue

        return errors


def analyze_file(filepath, detector, max_samples=None):
    """Analyze a single embeddings file."""
    results = {
        'total': 0,
        'has_errors': 0,
        'error_types': Counter(),
        'examples': []
    }

    with open(filepath, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if max_samples and i >= max_samples:
                break

            try:
                data = json.loads(line)
                sentence = data.get('sentence', '')
                results['total'] += 1

                errors = detector.detect_errors(sentence)
                if errors:
                    results['has_errors'] += 1

                    # Record error type counts
                    for error_type in errors.keys():
                        results['error_types'][error_type] += 1

                    # Keep the first 10 error examples
                    if len(results['examples']) < 10:
                        results['examples'].append({
                            'sentence': sentence,
                            'errors': errors,
                            'toptitle': data.get('toptitle', 'N/A')
                        })
            except Exception as e:
                print(f"Error processing line {i}: {e}")

    return results


def main():
    parser = argparse.ArgumentParser(description='Detect OCR errors in embedding records')
    parser.add_argument('--char', type=str, default='手', help='Character to inspect (default: 手)')
    parser.add_argument('--dynasty', type=str, help='Restrict to a single dynasty (default: all)')
    parser.add_argument('--max-samples', type=int, help='Maximum number of records to inspect per file')
    parser.add_argument('--output', type=str, help='Path to write the report to')
    parser.add_argument('--input-dir', type=str, default='keyword_embeddings', help='Input directory (default: keyword_embeddings)')

    args = parser.parse_args()

    detector = OCRErrorDetector()
    base_path = Path(args.input_dir)

    # Determine which dynasties to inspect
    if args.dynasty:
        dynasties = [args.dynasty]
    else:
        dynasties = ['pre-qin', 'qinhan', 'weijin', 'suitang', 'songyuan', 'ming', 'qing', 'republican']

    all_results = {}

    print(f"Detecting OCR errors for character '{args.char}'...")
    print("=" * 80)

    for dynasty in dynasties:
        filepath = base_path / dynasty / f"{args.char}_embeddings.jsonl"

        if not filepath.exists():
            print(f"File not found: {filepath}")
            continue

        print(f"\nDynasty: {dynasty}")
        print("-" * 80)

        results = analyze_file(filepath, detector, args.max_samples)
        all_results[dynasty] = results

        # Print statistics
        error_rate = (results['has_errors'] / results['total'] * 100) if results['total'] > 0 else 0
        print(f"Total sentences: {results['total']}")
        print(f"With errors: {results['has_errors']} ({error_rate:.2f}%)")

        if results['error_types']:
            print("Error type distribution:")
            for error_type, count in results['error_types'].most_common():
                print(f"  - {error_type}: {count}")

        # Print examples
        if results['examples']:
            print(f"\nError examples (first {len(results['examples'])}):")
            for idx, example in enumerate(results['examples'], 1):
                print(f"\n  [{idx}] {example['toptitle']}")
                print(f"      Sentence: {example['sentence'][:100]}...")
                print(f"      Errors: {example['errors']}")

    # Summary
    print("\n" + "=" * 80)
    print("Summary")
    print("=" * 80)

    total_all = sum(r['total'] for r in all_results.values())
    errors_all = sum(r['has_errors'] for r in all_results.values())
    overall_rate = (errors_all / total_all * 100) if total_all > 0 else 0

    print(f"Total: {total_all} sentences")
    print(f"With errors: {errors_all} ({overall_rate:.2f}%)")

    # Error rate by dynasty
    print("\nError rate by dynasty:")
    for dynasty, results in sorted(all_results.items()):
        if results['total'] > 0:
            rate = results['has_errors'] / results['total'] * 100
            print(f"  {dynasty:15s}: {rate:6.2f}% ({results['has_errors']}/{results['total']})")

    # Write report
    if args.output:
        with open(args.output, 'w', encoding='utf-8') as f:
            f.write(f"# OCR Error Report — {args.char}\n\n")
            f.write("## Summary\n\n")
            f.write(f"- Total sentences: {total_all}\n")
            f.write(f"- With errors: {errors_all} ({overall_rate:.2f}%)\n\n")

            f.write("## Statistics by Dynasty\n\n")
            f.write("| Dynasty | Total | Errors | Error rate |\n")
            f.write("|---------|-------|--------|------------|\n")
            for dynasty, results in sorted(all_results.items()):
                if results['total'] > 0:
                    rate = results['has_errors'] / results['total'] * 100
                    f.write(f"| {dynasty} | {results['total']} | {results['has_errors']} | {rate:.2f}% |\n")

            f.write("\n## Error Type Distribution\n\n")
            all_error_types = Counter()
            for results in all_results.values():
                all_error_types.update(results['error_types'])

            for error_type, count in all_error_types.most_common():
                f.write(f"- {error_type}: {count}\n")

            f.write("\n## Error Examples\n\n")
            for dynasty, results in sorted(all_results.items()):
                if results['examples']:
                    f.write(f"\n### {dynasty}\n\n")
                    for idx, example in enumerate(results['examples'][:3], 1):
                        f.write(f"{idx}. **{example['toptitle']}**\n")
                        f.write(f"   - Sentence: {example['sentence']}\n")
                        f.write(f"   - Errors: {example['errors']}\n\n")

        print(f"\nReport saved to: {args.output}")


if __name__ == '__main__':
    main()

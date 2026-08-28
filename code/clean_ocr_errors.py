#!/usr/bin/env python3
"""
清理 embedding 数据中的 OCR 错误

策略：
1. 保守过滤：只删除明显的OCR错误（如OCR标记符、过长句子）
2. 宽容过滤：同时删除罕见字符、过短句子等
3. 保留原始数据，输出到新目录

使用示例：
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
        mode: 'conservative' (保守) 或 'aggressive' (激进)
        """
        self.mode = mode
        self.detector = OCRErrorDetector()
        self.stats = {
            'total': 0,
            'removed': 0,
            'reasons': Counter()
        }
    
    def should_remove(self, sentence):
        """判断是否应该删除这条数据"""
        errors = self.detector.detect_errors(sentence)
        
        if not errors:
            return False, None
        
        # 保守模式：只删除明确的OCR错误
        if self.mode == 'conservative':
            # 1. 有OCR标记符（□、○等） → 删除
            if 'ocr_markers' in errors:
                return True, 'ocr_markers'
            
            # 2. 句子过长（可能是分段错误） → 删除
            if 'length_anomaly' in errors:
                anomaly = errors['length_anomaly']
                if 'too_long' in anomaly:
                    length = int(anomaly.split('(')[1].strip(')'))
                    if length > 300:  # 超过300字认为异常
                        return True, 'too_long'
            
            # 3. 重复字符过多 → 删除
            if 'suspicious_patterns' in errors:
                for pattern, matches in errors['suspicious_patterns']:
                    if r'\1{4,}' in pattern:  # 同字符重复5次以上
                        return True, 'excessive_repetition'
            
            return False, None
        
        # 激进模式：删除所有有疑问的数据
        elif self.mode == 'aggressive':
            # 所有检测到的错误都删除
            reason = list(errors.keys())[0]  # 取第一个错误类型作为原因
            return True, reason
        
        # 自定义模式：根据配置决定
        elif self.mode == 'custom':
            # 可以在这里添加更细致的规则
            # 例如：允许某些罕见字但删除OCR标记
            if 'ocr_markers' in errors:
                return True, 'ocr_markers'
            
            if 'length_anomaly' in errors:
                anomaly = errors['length_anomaly']
                if 'too_short' in anomaly:
                    length = int(anomaly.split('(')[1].strip(')'))
                    if length < 5:  # 少于5字删除（可能是断句问题）
                        return True, 'too_short'
                elif 'too_long' in anomaly:
                    length = int(anomaly.split('(')[1].strip(')'))
                    if length > 200:
                        return True, 'too_long'
            
            if 'suspicious_patterns' in errors:
                return True, 'suspicious_patterns'
            
            return False, None
    
    def clean_file(self, input_path, output_path):
        """清理单个文件"""
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
                    print(f"  ⚠️  处理错误: {e}")
                    continue
        
        return file_stats


def main():
    parser = argparse.ArgumentParser(description='清理embedding数据中的OCR错误')
    parser.add_argument('--char', type=str, default='手', help='要清理的字')
    parser.add_argument('--dynasty', type=str, help='指定朝代（默认：所有）')
    parser.add_argument('--mode', type=str, default='conservative',
                        choices=['conservative', 'aggressive', 'custom'],
                        help='清理模式：conservative（保守）、aggressive（激进）、custom（自定义）')
    parser.add_argument('--input-dir', type=str, default='keyword_embeddings',
                        help='输入目录')
    parser.add_argument('--output-dir', type=str, default='cleaned_embeddings',
                        help='输出目录')
    parser.add_argument('--dry-run', action='store_true',
                        help='演示模式（不实际写入文件）')
    
    args = parser.parse_args()
    
    cleaner = DataCleaner(mode=args.mode)
    input_base = Path(args.input_dir)
    output_base = Path(args.output_dir)
    
    # 确定要处理的朝代
    if args.dynasty:
        dynasties = [args.dynasty]
    else:
        dynasties = ['pre-qin', 'qinhan', 'weijin', 'suitang', 'songyuan', 'ming', 'qing', 'republican']
    
    print(f"开始清理字「{args.char}」的OCR错误...")
    print(f"清理模式: {args.mode}")
    print(f"{'[演示模式 - 不写入文件]' if args.dry_run else ''}")
    print("=" * 80)
    
    dynasty_results = {}
    
    for dynasty in dynasties:
        input_path = input_base / dynasty / f"{args.char}_embeddings.jsonl"
        output_path = output_base / dynasty / f"{args.char}_embeddings.jsonl"
        
        if not input_path.exists():
            print(f"⚠️  文件不存在: {input_path}")
            continue
        
        print(f"\n处理朝代: {dynasty}")
        print("-" * 80)
        
        if args.dry_run:
            # 演示模式：只统计不写入
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
        
        # 输出统计
        removal_rate = (stats['removed'] / stats['total'] * 100) if stats['total'] > 0 else 0
        print(f"总数: {stats['total']}")
        print(f"删除: {stats['removed']} ({removal_rate:.2f}%)")
        print(f"保留: {stats['kept']} ({100-removal_rate:.2f}%)")
        
        if stats.get('reasons'):
            print(f"删除原因:")
            for reason, count in stats['reasons'].most_common():
                print(f"  - {reason}: {count}")
        
        if not args.dry_run:
            print(f"✅ 已保存到: {output_path}")
    
    # 汇总统计
    print("\n" + "=" * 80)
    print("汇总统计")
    print("=" * 80)
    
    total_all = sum(r['total'] for r in dynasty_results.values())
    removed_all = sum(r['removed'] for r in dynasty_results.values())
    kept_all = sum(r['kept'] for r in dynasty_results.values())
    overall_removal_rate = (removed_all / total_all * 100) if total_all > 0 else 0
    
    print(f"总数: {total_all}")
    print(f"删除: {removed_all} ({overall_removal_rate:.2f}%)")
    print(f"保留: {kept_all} ({100-overall_removal_rate:.2f}%)")
    
    print("\n各朝代删除率:")
    for dynasty, stats in sorted(dynasty_results.items()):
        if stats['total'] > 0:
            rate = stats['removed'] / stats['total'] * 100
            print(f"  {dynasty:15s}: {rate:6.2f}% ({stats['removed']}/{stats['total']})")
    
    print("\n删除原因分布:")
    all_reasons = Counter()
    for stats in dynasty_results.values():
        all_reasons.update(stats.get('reasons', {}))
    for reason, count in all_reasons.most_common():
        print(f"  - {reason}: {count}")
    
    if args.dry_run:
        print("\n💡 这是演示模式。要实际清理数据，请去掉 --dry-run 参数")
    else:
        print(f"\n✅ 清理完成！数据已保存到: {args.output_dir}/")


if __name__ == '__main__':
    main()

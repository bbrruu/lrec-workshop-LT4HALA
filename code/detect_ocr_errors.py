#!/usr/bin/env python3
"""
检测 embedding 数据中的 OCR 错误

常见 OCR 错误特征：
1. 罕见字符组合
2. 异常符号（如 □、〇、※等常见OCR失败标记）
3. 非汉字字符混入（除了常见标点）
4. 重复字符过多
5. 句子长度异常
"""

import json
import re
from pathlib import Path
from collections import Counter
import argparse


class OCRErrorDetector:
    def __init__(self):
        # OCR错误常见标记符号
        self.ocr_error_markers = set('□〇○●◎◇◆※☆★■▪▲△▽')
        
        # 允许的标点符号
        self.allowed_punctuation = set('，。、；：！？「」『』（）〈〉《》【】…—·')
        
        # 异常字符模式
        self.suspicious_patterns = [
            r'(.)\1{4,}',  # 同一字符重复5次以上
            r'[A-Za-z]{2,}',  # 连续2个以上英文字母
            r'\d{6,}',  # 连续6个以上数字
        ]
    
    def check_ocr_markers(self, text):
        """检查是否包含OCR错误标记符号"""
        markers_found = [char for char in text if char in self.ocr_error_markers]
        return markers_found
    
    def check_suspicious_patterns(self, text):
        """检查可疑模式"""
        found = []
        for pattern in self.suspicious_patterns:
            matches = re.findall(pattern, text)
            if matches:
                found.append((pattern, matches))
        return found
    
    def check_unusual_chars(self, text):
        """检查异常字符（非汉字、非常见标点）"""
        unusual = []
        for char in text:
            # 跳过汉字范围
            if '\u4e00' <= char <= '\u9fff':
                continue
            # 跳过常见标点
            if char in self.allowed_punctuation:
                continue
            # 跳过数字（少量数字是正常的）
            if char.isdigit():
                continue
            # 跳过空白
            if char.isspace():
                continue
            
            unusual.append(char)
        return unusual
    
    def check_length_anomaly(self, text, min_len=5, max_len=500):
        """检查句子长度异常"""
        length = len(text)
        if length < min_len:
            return f"too_short({length})"
        if length > max_len:
            return f"too_long({length})"
        return None
    
    def detect_errors(self, sentence):
        """综合检测OCR错误"""
        errors = {}
        
        # 检查OCR标记
        markers = self.check_ocr_markers(sentence)
        if markers:
            errors['ocr_markers'] = markers
        
        # 检查可疑模式
        patterns = self.check_suspicious_patterns(sentence)
        if patterns:
            errors['suspicious_patterns'] = patterns
        
        # 检查异常字符
        unusual = self.check_unusual_chars(sentence)
        if unusual:
            errors['unusual_chars'] = unusual
        
        # 检查长度
        length_issue = self.check_length_anomaly(sentence)
        if length_issue:
            errors['length_anomaly'] = length_issue
        
        return errors


def analyze_file(filepath, detector, max_samples=None):
    """分析单个文件"""
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
                    
                    # 记录错误类型
                    for error_type in errors.keys():
                        results['error_types'][error_type] += 1
                    
                    # 保存前10个错误样本
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
    parser = argparse.ArgumentParser(description='检测embedding数据中的OCR错误')
    parser.add_argument('--char', type=str, default='手', help='要检测的字（默认：手）')
    parser.add_argument('--dynasty', type=str, help='指定朝代（默认：所有）')
    parser.add_argument('--max-samples', type=int, help='每个文件最多检测的样本数')
    parser.add_argument('--output', type=str, help='输出报告文件路径')
    parser.add_argument('--input-dir', type=str, default='keyword_embeddings', help='輸入目錄（默認：keyword_embeddings）')
    
    args = parser.parse_args()
    
    detector = OCRErrorDetector()
    base_path = Path(args.input_dir)
    
    # 确定要检测的朝代
    if args.dynasty:
        dynasties = [args.dynasty]
    else:
        dynasties = ['pre-qin', 'qinhan', 'weijin', 'suitang', 'songyuan', 'ming', 'qing', 'republican']
    
    all_results = {}
    
    print(f"开始检测字「{args.char}」的OCR错误...")
    print("=" * 80)
    
    for dynasty in dynasties:
        filepath = base_path / dynasty / f"{args.char}_embeddings.jsonl"
        
        if not filepath.exists():
            print(f"⚠️  文件不存在: {filepath}")
            continue
        
        print(f"\n检测朝代: {dynasty}")
        print("-" * 80)
        
        results = analyze_file(filepath, detector, args.max_samples)
        all_results[dynasty] = results
        
        # 输出统计
        error_rate = (results['has_errors'] / results['total'] * 100) if results['total'] > 0 else 0
        print(f"总句数: {results['total']}")
        print(f"有错误: {results['has_errors']} ({error_rate:.2f}%)")
        
        if results['error_types']:
            print(f"错误类型分布:")
            for error_type, count in results['error_types'].most_common():
                print(f"  - {error_type}: {count}")
        
        # 显示样本
        if results['examples']:
            print(f"\n错误样本 (前{len(results['examples'])}个):")
            for idx, example in enumerate(results['examples'], 1):
                print(f"\n  [{idx}] {example['toptitle']}")
                print(f"      句子: {example['sentence'][:100]}...")
                print(f"      错误: {example['errors']}")
    
    # 汇总统计
    print("\n" + "=" * 80)
    print("汇总统计")
    print("=" * 80)
    
    total_all = sum(r['total'] for r in all_results.values())
    errors_all = sum(r['has_errors'] for r in all_results.values())
    overall_rate = (errors_all / total_all * 100) if total_all > 0 else 0
    
    print(f"总计: {total_all} 句")
    print(f"有错误: {errors_all} 句 ({overall_rate:.2f}%)")
    
    # 各朝代错误率
    print("\n各朝代错误率:")
    for dynasty, results in sorted(all_results.items()):
        if results['total'] > 0:
            rate = results['has_errors'] / results['total'] * 100
            print(f"  {dynasty:15s}: {rate:6.2f}% ({results['has_errors']}/{results['total']})")
    
    # 保存报告
    if args.output:
        with open(args.output, 'w', encoding='utf-8') as f:
            f.write(f"# OCR错误检测报告 — 字「{args.char}」\n\n")
            f.write(f"## 汇总\n\n")
            f.write(f"- 总句数: {total_all}\n")
            f.write(f"- 有错误: {errors_all} ({overall_rate:.2f}%)\n\n")
            
            f.write(f"## 各朝代统计\n\n")
            f.write(f"| 朝代 | 总数 | 错误数 | 错误率 |\n")
            f.write(f"|------|------|--------|--------|\n")
            for dynasty, results in sorted(all_results.items()):
                if results['total'] > 0:
                    rate = results['has_errors'] / results['total'] * 100
                    f.write(f"| {dynasty} | {results['total']} | {results['has_errors']} | {rate:.2f}% |\n")
            
            f.write(f"\n## 错误类型分布\n\n")
            all_error_types = Counter()
            for results in all_results.values():
                all_error_types.update(results['error_types'])
            
            for error_type, count in all_error_types.most_common():
                f.write(f"- {error_type}: {count}\n")
            
            f.write(f"\n## 错误样本示例\n\n")
            for dynasty, results in sorted(all_results.items()):
                if results['examples']:
                    f.write(f"\n### {dynasty}\n\n")
                    for idx, example in enumerate(results['examples'][:3], 1):
                        f.write(f"{idx}. **{example['toptitle']}**\n")
                        f.write(f"   - 句子: {example['sentence']}\n")
                        f.write(f"   - 错误: {example['errors']}\n\n")
        
        print(f"\n✅ 报告已保存到: {args.output}")


if __name__ == '__main__':
    main()

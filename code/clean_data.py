#!/usr/bin/env python3
"""
古漢語資料清理工具 - 激進版

針對 OCR 品質問題的多層過濾：
1. 句長檢查（太長太短都有問題）
2. 標點密度（OCR 錯誤常導致標點消失）
3. 字元重複率（OCR 錯誤特徵）
4. 連續同字（明顯錯誤）
5. 異常字元（□、�、等）
6. 字元多樣性（uniqueness ratio）
7. 數字密度（古文中數字不應太多）

使用：
python clean_data.py \
  --input stratified_sampling_output_cleaned \
  --output stratified_sampling_output_ULTRA_cleaned \
  --char 手

可選參數：
  --aggression [mild|normal|aggressive|ultra]  # 清理激進程度
  --show-examples  # 顯示被過濾的例句
  --dry-run        # 只統計，不實際寫入
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
# 過濾規則配置
# ============================================================

FILTER_CONFIGS = {
    "mild": {
        "min_length": 3,
        "max_length": 300,
        "min_punct_density": 0.01,  # 保留（雖然不用，但避免錯誤）
        "min_unique_ratio": 0.30,  # 稍微提高（補償沒有標點檢查）
        "max_repeat_char": 7,
        "max_digit_density": 0.25,
        "description": "溫和清理（保留較多數據）"
    },
    "normal": {
        "min_length": 4,
        "max_length": 200,
        "min_punct_density": 0.015,  # 保留（雖然不用）
        "min_unique_ratio": 0.35,  # 提高！補償沒有標點檢查
        "max_repeat_char": 5,     # 更嚴格！
        "max_digit_density": 0.20,  # 更嚴格！
        "description": "標準清理（推薦，不使用標點檢查）"
    },
    "aggressive": {
        "min_length": 5,
        "max_length": 150,
        "min_punct_density": 0.02,
        "min_unique_ratio": 0.40,  # 更嚴格
        "max_repeat_char": 4,
        "max_digit_density": 0.15,
        "description": "激進清理（大幅刪除）"
    },
    "ultra": {
        "min_length": 5,
        "max_length": 120,
        "min_punct_density": 0.025,
        "min_unique_ratio": 0.45,  # 非常嚴格
        "max_repeat_char": 3,
        "max_digit_density": 0.10,
        "description": "超激進清理（只保留高品質）"
    }
}

# 常見標點符號
CHINESE_PUNCTUATION = r'[。，、；：？！「」『』（）《》〈〉【】〔〕…—·]'

# 異常字元
INVALID_CHARS = r'[�□▪■●○◎★☆◇◆△▲▽▼]'

# 數字
DIGITS = r'[0-9０-９一二三四五六七八九十百千萬億兆壹貳參肆伍陸柒捌玖拾佰仟]'

# ============================================================
# 品質檢查函數
# ============================================================

def check_length(sentence: str, config: dict) -> Tuple[bool, str]:
    """檢查句長"""
    length = len(sentence)
    if length < config['min_length']:
        return False, f"太短({length})"
    if length > config['max_length']:
        return False, f"太長({length})"
    return True, "OK"


def check_punctuation_density(sentence: str, config: dict) -> Tuple[bool, str]:
    """檢查標點密度"""
    if len(sentence) == 0:
        return False, "空句子"
    
    punct_count = len(re.findall(CHINESE_PUNCTUATION, sentence))
    density = punct_count / len(sentence)
    
    if density < config['min_punct_density']:
        return False, f"標點過少({density:.3f})"
    
    return True, "OK"


def check_unique_ratio(sentence: str, config: dict) -> Tuple[bool, str]:
    """檢查字元多樣性（uniqueness ratio）"""
    if len(sentence) == 0:
        return False, "空句子"
    
    unique_chars = len(set(sentence))
    ratio = unique_chars / len(sentence)
    
    if ratio < config['min_unique_ratio']:
        return False, f"重複率過高({ratio:.3f})"
    
    return True, "OK"


def check_repeat_chars(sentence: str, config: dict) -> Tuple[bool, str]:
    """檢查連續重複字元"""
    # 找出最長的連續重複字元
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
        return False, f"連續重複({max_repeat})"
    
    return True, "OK"


def check_invalid_chars(sentence: str) -> Tuple[bool, str]:
    """檢查異常字元"""
    invalid = re.findall(INVALID_CHARS, sentence)
    if invalid:
        return False, f"異常字元({','.join(set(invalid))})"
    return True, "OK"


def check_digit_density(sentence: str, config: dict) -> Tuple[bool, str]:
    """檢查數字密度"""
    if len(sentence) == 0:
        return False, "空句子"
    
    digit_count = len(re.findall(DIGITS, sentence))
    density = digit_count / len(sentence)
    
    if density > config['max_digit_density']:
        return False, f"數字過多({density:.3f})"
    
    return True, "OK"


def check_abnormal_patterns(sentence: str) -> Tuple[bool, str]:
    """
    檢查異常模式（額外的啟發式規則）
    
    v2: 放寬檢查，減少誤判
    """
    
    # 1. 檢查是否全是標點
    if re.match(r'^[' + CHINESE_PUNCTUATION + r']+$', sentence):
        return False, "全是標點"
    
    # 2. 檢查是否有過多空白
    if sentence.count(' ') > len(sentence) * 0.1:
        return False, "空白過多"
    
    # 3. 檢查是否有異常的字元模式（如：aaabbbccc）
    pattern = r'(.)\1{2,}'
    matches = re.findall(pattern, sentence)
    if len(matches) >= 3:
        return False, "異常重複模式"
    
    # 4. ✨ 放寬：超長且完全無任何標點（不只「。；？！」）
    # 改成 150 字（原 100），且檢查所有標點
    if len(sentence) > 150:
        punct_count = len(re.findall(CHINESE_PUNCTUATION, sentence))
        if punct_count == 0:
            return False, "超長且無標點(OCR黏接)"
    
    # 5. ✨ 放寬：罕見字過多 - 降低閾值
    # 改成：常見字 < 10%（原 20%）才算異常
    common_chars = set('的一是不了人我有他這為之大來以個中上們到說國和地也子時道出而要於就下得可你年生自會那後能對著事其裡所去行過家十用發天如然作方成者多日都三小軍二無同麼經法當起與好看學進種將還分此心前面又定見只主沒公從')
    
    if len(sentence) >= 10:
        segments = [sentence[i:i+10] for i in range(0, len(sentence)-9, 10)]
        low_common_count = 0
        
        for seg in segments:
            common_ratio = sum(1 for c in seg if c in common_chars) / len(seg)
            if common_ratio < 0.10:  # ← 改成 10%（原 20%）
                low_common_count += 1
        
        # 改成：超過 80% 的段落（原 50%）常見字都很少才算異常
        if low_common_count > len(segments) * 0.8:
            return False, "罕見字過多(可能OCR誤識)"
    
    # 6. ✨ 保持：生僻字過多（這個應該 OK）
    rare_char_count = 0
    for char in sentence:
        if '\u3400' <= char <= '\u4DBF' or '\U00020000' <= char <= '\U0002A6DF':
            rare_char_count += 1
    
    if len(sentence) > 0 and rare_char_count / len(sentence) > 0.3:
        return False, "生僻字過多"
    
    return True, "OK"


def quality_score(sentence: str, config: dict) -> Tuple[float, Dict[str, str]]:
    """
    綜合品質評分 (0-1)
    
    注意：不使用標點密度檢查！
    原因：句子在 embedding 提取時已用「。；？！」切分，
    導致很多正常句子沒有標點（句尾標點被用於切分）
    
    返回：
        score: 0-1 的品質分數
        reasons: 各項檢查的結果
    """
    checks = {
        "length": check_length(sentence, config),
        # "punctuation": check_punctuation_density(sentence, config),  # ← 不使用！
        "uniqueness": check_unique_ratio(sentence, config),
        "repeat": check_repeat_chars(sentence, config),
        "invalid_chars": check_invalid_chars(sentence),
        "digits": check_digit_density(sentence, config),
        "patterns": check_abnormal_patterns(sentence)
    }
    
    # 計算通過的比例
    passed = sum(1 for ok, _ in checks.values() if ok)
    score = passed / len(checks)
    
    # 整理未通過的原因
    reasons = {name: reason for name, (ok, reason) in checks.items() if not ok}
    
    return score, reasons


# ============================================================
# 清理函數
# ============================================================

def clean_records(
    records: List[Dict],
    config: dict,
    show_examples: bool = False,
    max_examples: int = 10
) -> Tuple[List[Dict], Dict]:
    """
    清理記錄
    
    返回：
        cleaned_records: 清理後的記錄
        stats: 統計資訊
    """
    cleaned = []
    filtered_reasons = defaultdict(int)
    filtered_examples = defaultdict(list)
    
    for record in tqdm(records, desc="    過濾中", leave=False):
        sentence = record.get('sentence', '')
        
        score, reasons = quality_score(sentence, config)
        
        if score == 1.0:  # 完全通過所有檢查
            cleaned.append(record)
        else:
            # 記錄過濾原因
            for reason_key, reason_msg in reasons.items():
                filtered_reasons[reason_key] += 1
                
                # 保存例句（用於展示）
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
    """處理單一朝代"""
    
    if not os.path.exists(input_file):
        return None
    
    print(f"\n  處理 {dynasty}...")
    
    # 載入
    with open(input_file, 'rb') as f:
        data = pickle.load(f)
    
    original_records = data['records']
    print(f"    原始: {len(original_records):,} 句")
    
    # 清理
    cleaned_records, stats = clean_records(
        original_records, 
        config, 
        show_examples
    )
    
    print(f"    保留: {len(cleaned_records):,} 句 ({stats['keep_ratio']:.1%})")
    print(f"    過濾: {stats['filtered']:,} 句")
    
    # 顯示過濾原因統計
    if stats['filter_reasons']:
        print(f"    過濾原因:")
        for reason, count in sorted(stats['filter_reasons'].items(), 
                                    key=lambda x: -x[1]):
            print(f"      - {reason}: {count:,} 句")
    
    # 顯示例句
    if show_examples and stats['examples']:
        print(f"\n    過濾例句（每類最多 3 個）:")
        for reason, examples in stats['examples'].items():
            print(f"\n      [{reason}]")
            for ex in examples[:3]:
                print(f"        ({ex['score']:.2f}) {ex['sentence'][:50]}...")
                print(f"             → {ex['reason']}")
    
    # 儲存（如果不是 dry run）
    if not dry_run:
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        
        # 更新統計資訊
        data['records'] = cleaned_records
        data['statistics']['n_samples'] = len(cleaned_records)
        
        # 加入清理資訊
        data['cleaning_info'] = {
            'config': config,
            'stats': {k: v for k, v in stats.items() if k != 'examples'}
        }
        
        with open(output_file, 'wb') as f:
            pickle.dump(data, f)
        
        print(f"    ✓ 已儲存: {output_file}")
    
    return stats


# ============================================================
# 主函數
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description='古漢語資料清理工具（優化版：不使用標點檢查）',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
過濾規則（已移除標點檢查）：
  
  ⚠️  不使用標點密度檢查！
  原因：句子在提取 embedding 時已用「。；？！」切分，
       導致很多正常句子沒有句尾標點。
  
  實際使用的 6 個過濾規則：
  
  1. 句長: 過短/過長的句子（可能是分句錯誤）
  2. 字元多樣性: 重複率過高（OCR 重複識別）
     - unique_ratio = 不重複字數 / 總字數
     - 正常句子通常 > 0.35
  3. 連續重複: 同字連續出現過多次（明顯錯誤）
  4. 異常字元: □、�、等無效字元
  5. 數字密度: 數字過多（可能是頁碼等）
  6. 異常模式: 啟發式規則
     - 超長無標點（OCR 黏接）
     - 罕見字過多（OCR 誤識）
     - 生僻字過多
     - 常見字比例過低

激進程度（調整後，v2 放寬版）：
  mild:       溫和清理，保留 ~92% 數據
  normal:     標準清理，保留 ~88% 數據（推薦）
  aggressive: 激進清理，保留 ~80% 數據
  ultra:      超激進，只保留 ~70% 高品質數據

範例：
  # 標準清理（推薦）
  python clean_data.py --char 手 --aggression normal
  
  # 查看會過濾哪些句子
  python clean_data.py --char 手 --aggression normal --show-examples --dry-run
  
  # 只統計不寫入
  python clean_data.py --char 手 --dry-run
        """
    )
    
    parser.add_argument('--char', type=str, required=True,
                       help='要處理的字')
    parser.add_argument('--input', type=str, 
                       default='stratified_sampling_output_cleaned',
                       help='輸入目錄')
    parser.add_argument('--output', type=str,
                       default='stratified_sampling_output_ULTRA_cleaned',
                       help='輸出目錄')
    parser.add_argument('--aggression', type=str, 
                       default='normal',
                       choices=['mild', 'normal', 'aggressive', 'ultra'],
                       help='清理激進程度')
    parser.add_argument('--show-examples', action='store_true',
                       help='顯示被過濾的例句')
    parser.add_argument('--dry-run', action='store_true',
                       help='只統計，不實際寫入檔案')
    parser.add_argument('--dynasties', type=str, default=None,
                       help='只處理特定朝代（逗號分隔），如：ming,qing')
    
    args = parser.parse_args()
    
    # 設定
    config = FILTER_CONFIGS[args.aggression]
    char = args.char
    
    print("=" * 60)
    print("古漢語資料清理工具")
    print("=" * 60)
    print(f"字: {char}")
    print(f"激進程度: {args.aggression} - {config['description']}")
    print(f"輸入: {args.input}")
    print(f"輸出: {args.output}")
    if args.dry_run:
        print("模式: DRY RUN（不寫入檔案）")
    print("=" * 60)
    
    print("\n過濾規則:")
    print(f"  ⚠️  不使用標點密度檢查（因 embedding 提取時已按標點切分）")
    print(f"  句長: {config['min_length']}-{config['max_length']} 字")
    print(f"  最小字元多樣性: {config['min_unique_ratio']:.3f}")
    print(f"  最大連續重複: {config['max_repeat_char']} 字")
    print(f"  最大數字密度: {config['max_digit_density']:.3f}")
    print(f"  異常模式: 超長無標點、罕見字過多、生僻字過多等")
    
    # 決定要處理的朝代
    if args.dynasties:
        dynasties = [d.strip() for d in args.dynasties.split(',')]
    else:
        dynasties = ["pre-qin", "qinhan", "weijin", "suitang", 
                    "songyuan", "ming", "qing", "republican"]
    
    # 處理每個朝代
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
    
    # 總結
    print("\n" + "=" * 60)
    print("總結")
    print("=" * 60)
    
    total_original = sum(s['total'] for s in all_stats.values())
    total_kept = sum(s['kept'] for s in all_stats.values())
    total_filtered = sum(s['filtered'] for s in all_stats.values())
    
    print(f"\n總計:")
    print(f"  原始: {total_original:,} 句")
    print(f"  保留: {total_kept:,} 句 ({total_kept/total_original:.1%})")
    print(f"  過濾: {total_filtered:,} 句 ({total_filtered/total_original:.1%})")
    
    # 各朝代統計
    print(f"\n各朝代保留率:")
    for dynasty in dynasties:
        if dynasty in all_stats:
            s = all_stats[dynasty]
            print(f"  {dynasty:10s}: {s['kept']:6,}/{s['total']:6,} "
                  f"({s['keep_ratio']:5.1%}) - "
                  f"過濾 {s['filtered']:,}")
    
    # 彙總過濾原因
    print(f"\n過濾原因彙總:")
    reason_totals = defaultdict(int)
    for stats in all_stats.values():
        for reason, count in stats['filter_reasons'].items():
            reason_totals[reason] += count
    
    for reason, count in sorted(reason_totals.items(), key=lambda x: -x[1]):
        pct = count / total_filtered * 100 if total_filtered > 0 else 0
        print(f"  {reason:15s}: {count:6,} ({pct:5.1f}%)")
    
    if not args.dry_run:
        # 儲存統計
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
        
        print(f"\n✓ 統計已儲存: {summary_file}")
    
    print("\n" + "=" * 60)
    if args.dry_run:
        print("DRY RUN 完成 - 未寫入任何檔案")
    else:
        print("清理完成！")
    print("=" * 60)


if __name__ == "__main__":
    main()
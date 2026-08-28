"""
0202
Stratified Sampling - 分層抽樣程式

功能：
1. 對指定字進行分層抽樣（每個 toptitle 最多 max_per_toptitle 筆）
2. 使用蓄水池抽樣（Reservoir Sampling）確保記憶體安全
3. 支援對 Baseline 字設定 max_samples 上限
4. 儲存抽樣結果和統計資訊

使用方式：
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

# 八個朝代的 JSONL 檔案路徑
# 格式：/mnt/md0/public/dicwn/lrec-workshop/cleaned_embeddings/{朝代}/{字}_embeddings.jsonl
# 注意：這裡只定義目錄，實際檔案路徑會在程式中動態生成
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
    "pre-qin": "先秦",
    "qinhan": "秦漢",
    "weijin": "魏晉",
    "suitang": "隋唐",
    "songyuan": "宋元",
    "ming": "明",
    "qing": "清",
    "republican": "民國",
}

DYNASTY_ORDER = ["pre-qin", "qinhan", "weijin", "suitang", "songyuan", "ming", "qing", "republican"]

# 預設參數
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
    """計算 Shannon entropy（文獻多樣性指標）"""
    total = sum(distribution.values())
    if total == 0:
        return 0.0
    
    probs = np.array([count / total for count in distribution.values()])
    # 避免 log(0)
    probs = probs[probs > 0]
    entropy = -np.sum(probs * np.log2(probs))
    return float(entropy)


# ============================================================
# STRATIFIED SAMPLING (蓄水池抽樣)
# ============================================================

def stratified_sample_reservoir(
    dynasty_dir: str,
    target_char: str,
    max_per_toptitle: int = 200,
    max_samples: Optional[int] = None,
    random_seed: int = 42
) -> Tuple[List[Dict], Dict[str, Dict]]:
    """
    分層抽樣（蓄水池演算法）
    
    Args:
        dynasty_dir: 朝代目錄路徑
        target_char: 目標字
        max_per_toptitle: 每個 toptitle 最多取幾筆
        max_samples: 總樣本上限（None 表示無上限）
        random_seed: 隨機種子
    
    Returns:
        (sampled_records, statistics)
        - sampled_records: List[Dict] 抽樣後的記錄
        - statistics: Dict[str, Dict] 統計資訊
    """
    random.seed(random_seed)
    
    # 構建檔案路徑：{dynasty_dir}/{char}_embeddings.jsonl
    jsonl_path = os.path.join(dynasty_dir, f"{target_char}_embeddings.jsonl")
    
    # 每個 toptitle 維護一個水庫
    toptitle_reservoirs = defaultdict(lambda: {
        'records': [],
        'count': 0  # 總共看過多少筆（用於蓄水池演算法）
    })
    
    seen_sentences = set()  # 去重
    total_scanned = 0
    total_duplicates = 0
    total_matched = 0  # 符合 target_char 的總數
    
    dynasty_name = os.path.basename(dynasty_dir)
    
    if not os.path.exists(jsonl_path):
        log(f"  ⚠️ 檔案不存在: {jsonl_path}")
        return [], {}
    
    # 取得檔案大小（用於進度條）
    file_size = os.path.getsize(jsonl_path)
    
    log(f"  開始掃描 {dynasty_name}/{target_char}_embeddings.jsonl...")
    
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        pbar = tqdm(
            f,
            desc=f"    掃描 {dynasty_name}",
            unit=" lines",
            total=file_size // 100,  # 估計行數
            leave=False,
            ncols=120
        )
        
        for line in pbar:
            total_scanned += 1
            
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            
            # 檢查是否為目標字
            if record.get('word') != target_char:
                continue
            
            total_matched += 1
            
            sentence = record.get('sentence', '')
            
            # 去重
            if sentence in seen_sentences:
                total_duplicates += 1
                continue
            seen_sentences.add(sentence)
            
            # 取得 toptitle
            toptitle = record.get('toptitle', 'unknown')
            reservoir = toptitle_reservoirs[toptitle]
            reservoir['count'] += 1
            k = reservoir['count']
            
            # 蓄水池演算法
            if len(reservoir['records']) < max_per_toptitle:
                # 水庫未滿，直接加入
                reservoir['records'].append(record)
            else:
                # 水庫已滿，隨機替換
                j = random.randint(1, k)
                if j <= max_per_toptitle:
                    reservoir['records'][j - 1] = record
            
            # 更新進度條
            current_total = sum(len(r['records']) for r in toptitle_reservoirs.values())
            pbar.set_postfix({
                "匹配": total_matched,
                "去重": total_duplicates,
                "已抽": current_total,
                "文獻": len(toptitle_reservoirs)
            })
        
        pbar.close()
    
    # 合併所有水庫
    all_records = []
    for toptitle, reservoir in toptitle_reservoirs.items():
        all_records.extend(reservoir['records'])
    
    # 如果設定了 max_samples，再進行最終抽樣
    if max_samples is not None and len(all_records) > max_samples:
        log(f"    總樣本 {len(all_records)} 超過上限 {max_samples}，進行最終抽樣...")
        all_records = random.sample(all_records, max_samples)
    
    # 統計資訊
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
    
    # 統計每個 toptitle 的資訊
    final_toptitle_counts = defaultdict(int)
    for record in all_records:
        toptitle = record.get('toptitle', 'unknown')
        final_toptitle_counts[toptitle] += 1
    
    # 計算統計
    for toptitle, reservoir in toptitle_reservoirs.items():
        original_count = reservoir['count']
        sampled_count = final_toptitle_counts.get(toptitle, 0)
        
        statistics["toptitle_stats"][toptitle] = {
            "original": original_count,
            "sampled": sampled_count,
            "ratio": sampled_count / original_count if original_count > 0 else 0
        }
    
    # 最終分布
    statistics["toptitle_distribution"] = dict(final_toptitle_counts)
    
    # 計算文獻多樣性
    statistics["diversity"] = compute_shannon_entropy(final_toptitle_counts)
    
    # 前 10 大文獻
    top_10 = sorted(final_toptitle_counts.items(), key=lambda x: x[1], reverse=True)[:10]
    statistics["top_10_toptitles"] = [
        {"toptitle": t, "count": c, "percentage": c / len(all_records) * 100}
        for t, c in top_10
    ]
    
    log(f"  ✓ {dynasty_name}: 掃描 {total_scanned:,} 行，匹配 {total_matched:,} 筆，"
        f"去重 {total_duplicates:,} 筆，最終 {len(all_records):,} 筆，{len(toptitle_reservoirs)} 個文獻")
    
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
    處理一個字的所有朝代
    
    每個朝代處理完立即儲存並釋放記憶體
    """
    log(f"\n{'='*60}")
    log(f"處理字: 「{char}」")
    log(f"  max_per_toptitle: {max_per_toptitle}")
    log(f"  max_samples: {max_samples if max_samples else '無上限'}")
    log(f"{'='*60}")
    
    # 建立字的輸出目錄
    char_dir = os.path.join(output_dir, f"char_{char}")
    os.makedirs(char_dir, exist_ok=True)
    
    overall_stats = {
        "char": char,
        "max_per_toptitle": max_per_toptitle,
        "max_samples": max_samples,
        "dynasties": {}
    }
    
    # 朝代進度條
    dynasty_pbar = tqdm(
        DYNASTY_ORDER,
        desc=f"處理「{char}」各朝代",
        unit=" dynasty",
        ncols=120
    )
    
    for dynasty in dynasty_pbar:
        dynasty_name = DYNASTY_NAMES.get(dynasty, dynasty)
        dynasty_pbar.set_postfix({"當前": dynasty_name})
        
        dynasty_dir = dynasty_dirs.get(dynasty)
        
        if not dynasty_dir or not os.path.exists(dynasty_dir):
            log(f"  ⚠️ {dynasty_name}: 目錄不存在 ({dynasty_dir})")
            continue
        
        # 執行分層抽樣
        sampled_records, statistics = stratified_sample_reservoir(
            dynasty_dir,
            char,
            max_per_toptitle=max_per_toptitle,
            max_samples=max_samples
        )
        
        if len(sampled_records) == 0:
            log(f"  ⚠️ {dynasty_name}: 沒有抽樣到任何資料")
            continue
        
        # 立即儲存這個朝代的資料（避免記憶體累積）
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
        
        # 也儲存統計資訊（JSON 格式，方便查看）
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
        
        log(f"    ✓ 已儲存: {dynasty_file}")
        
        # 記錄到總體統計
        overall_stats["dynasties"][dynasty] = {
            "dynasty_name": dynasty_name,
            "n_samples": len(sampled_records),
            "n_toptitles": statistics["n_toptitles"],
            "diversity": statistics["diversity"],
            "top_3_toptitles": statistics["top_10_toptitles"][:3]
        }
        
        # 釋放記憶體
        del sampled_records
        del dynasty_output
    
    dynasty_pbar.close()
    
    # 儲存總體統計
    overall_file = os.path.join(char_dir, "overall_stats.json")
    with open(overall_file, 'w', encoding='utf-8') as f:
        json.dump(overall_stats, f, ensure_ascii=False, indent=2)
    
    log(f"\n✓ 「{char}」處理完成，總體統計已儲存: {overall_file}")
    
    # 顯示摘要
    log(f"\n{'='*60}")
    log(f"「{char}」摘要:")
    log(f"{'='*60}")
    for dynasty in DYNASTY_ORDER:
        if dynasty in overall_stats["dynasties"]:
            info = overall_stats["dynasties"][dynasty]
            log(f"  {info['dynasty_name']:6s}: {info['n_samples']:7,} 筆, "
                f"{info['n_toptitles']:4} 個文獻, "
                f"多樣性={info['diversity']:.2f}")
    
    return overall_stats


# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser(description='Stratified Sampling - 分層抽樣')
    parser.add_argument('--chars', type=str, required=True,
                       help='要處理的字（用逗號分隔，例如：手,天,地,下）')
    parser.add_argument('--max-per-toptitle', type=int, default=DEFAULT_MAX_PER_TOPTITLE,
                       help=f'每個 toptitle 最多取幾筆（預設 {DEFAULT_MAX_PER_TOPTITLE}）')
    parser.add_argument('--max-samples', type=int, default=None,
                       help='總樣本上限（預設 None，無上限）。Baseline 字建議設為 10000')
    parser.add_argument('--output', type=str, default=DEFAULT_OUTPUT_DIR,
                       help=f'輸出目錄（預設 {DEFAULT_OUTPUT_DIR}）')
    parser.add_argument('--baseline-max-samples', type=int, default=10000,
                       help='Baseline 字的 max_samples（預設 10000）')
    parser.add_argument('--target-chars', type=str, default='手',
                       help='目標字（用逗號分隔），這些字不設 max_samples 上限（預設：手）')
    parser.add_argument('--cleaned-root', type=str, default=None,
                       help='cleaned_embeddings 根目錄（預設使用腳本內建路徑）')
    
    args = parser.parse_args()
    
    # 解析參數
    chars = [c.strip() for c in args.chars.split(',')]
    target_chars_set = set(c.strip() for c in args.target_chars.split(','))
    output_dir = args.output
    max_per_toptitle = args.max_per_toptitle
    dynasty_dirs = build_dynasty_dirs(args.cleaned_root)
    
    log("=" * 60)
    log("Stratified Sampling - 分層抽樣")
    log("=" * 60)
    log(f"處理字: {chars}")
    log(f"目標字（無 max_samples 上限）: {list(target_chars_set)}")
    log(f"max_per_toptitle: {max_per_toptitle}")
    log(f"Baseline 字的 max_samples: {args.baseline_max_samples}")
    log(f"cleaned_root: {args.cleaned_root if args.cleaned_root else '內建預設路徑'}")
    log(f"輸出目錄: {output_dir}")
    log("=" * 60)
    
    # 建立輸出目錄
    os.makedirs(output_dir, exist_ok=True)
    
    # 處理每個字
    all_chars_stats = {}
    
    for char in chars:
        # 判斷是目標字還是 baseline 字
        if char in target_chars_set:
            max_samples = None  # 目標字無上限
            log(f"\n[目標字] 「{char}」（無 max_samples 上限）")
        else:
            max_samples = args.baseline_max_samples  # Baseline 字有上限
            log(f"\n[Baseline 字] 「{char}」（max_samples = {max_samples}）")
        
        char_stats = process_char_all_dynasties(
            char,
            max_per_toptitle=max_per_toptitle,
            max_samples=max_samples,
            output_dir=output_dir,
            dynasty_dirs=dynasty_dirs
        )
        
        all_chars_stats[char] = char_stats
    
    # 儲存所有字的總體統計
    summary_file = os.path.join(output_dir, "summary.json")
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(all_chars_stats, f, ensure_ascii=False, indent=2)
    
    log("\n" + "=" * 60)
    log("完成！")
    log("=" * 60)
    log(f"輸出目錄: {output_dir}/")
    log(f"  - char_{{字}}/: 各字的抽樣結果")
    log(f"    - {{dynasty}}.pkl: 該朝代的抽樣資料（包含完整 records）")
    log(f"    - {{dynasty}}_stats.json: 該朝代的統計資訊")
    log(f"    - overall_stats.json: 該字的總體統計")
    log(f"  - summary.json: 所有字的總體統計")
    log("\n使用方式範例:")
    log("  # 處理所有四個字")
    log("  python stratified_sampling.py --chars 手,天,地,下")
    log("")
    log("  # 自訂參數")
    log("  python stratified_sampling.py --chars 手,天,地,下 \\")
    log("    --max-per-toptitle 250 \\")
    log("    --baseline-max-samples 15000")


if __name__ == "__main__":
    main()
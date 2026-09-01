#!/usr/bin/env python3
"""
Extract keyword embeddings from CText corpus using GujiBERT.

This script:
1. Scans all texts in the corpus
2. Finds sentences containing target characters (手 + 16 baseline chars)
3. Computes GujiBERT embeddings ONLY for those target characters
4. Outputs by dynasty_category and character

Usage:
    python extract_keyword_embeddings.py --output ../lrec-workshop/keyword_embeddings/
    python extract_keyword_embeddings.py --dynasties qinhan suitang --output ./output/
    python extract_keyword_embeddings.py --chars 手 上 下 --output ./output/

Output structure:
    output_dir/
    ├── pre-qin/
    │   ├── 手_embeddings.jsonl
    │   ├── 上_embeddings.jsonl
    │   └── ...
    ├── qinhan/
    │   └── ...
    └── ...
"""

import torch
import json
import re
import pandas as pd
import time
import sys
import gc
import os
import gzip
import glob
import argparse
from typing import List, Dict, Optional, Tuple, Set
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm

# ============================================================
# CONFIGURATION
# ============================================================

# Model
MODEL_ID = "hsc748NLP/GujiBERT_fan"

# Target characters
TARGET_KEYWORD = "道"
# BASELINE_CHARS = ["上", "下", "中", "東", "天", "地", "首",
# "一", "二", "三", "五", "十", "大", "小", "日", "月"]
# BASELINE_CHARS = ["天", "地", "小", "一", "下"]
BASELINE_CHARS = []  # Empty list - only extract TARGET_KEYWORD
ALL_TARGET_CHARS = [TARGET_KEYWORD] + BASELINE_CHARS

# Dynasty groupings (excluding Liao, Jin)
DYNASTY_GROUPS = {
    'pre-qin': [
        'Western Zhou', 'Eastern Zhou', 'Zhou',
        'Spring and Autumn', 'Warring States'
    ],
    'qinhan': [
        'Qin', 'Han', 'Western Han', 'Eastern Han', 'Xin'
    ],
    'weijin': [
        'Three Kingdoms', 'Western Jin', 'Eastern Jin',
        'Northern and Southern', 'Liang', 'Chen',
        'Southern Qi', 'Northern Qi', 'Northern Zhou'
    ],
    'suitang': [
        'Sui', 'Tang', 'Five Dynasties and Ten Kingdoms'
    ],
    'songyuan': [
        'Song', 'Northern Song', 'Southern Song', 'Yuan'
    ],
    'ming': ['Ming'],
    'qing': ['Qing'],
    'republican': ['Republican era']
}

# Create reverse mapping: dynasty_name -> dynasty_category
DYNASTY_TO_CATEGORY = {}
for category, dynasties in DYNASTY_GROUPS.items():
    for dynasty in dynasties:
        DYNASTY_TO_CATEGORY[dynasty] = category

# Local CText data paths (defaults; override with --book-data-path / --wiki-data-pattern)
BOOK_DATA_PATH = "/mnt/md0/corpus/SinDia/ctext/data/raw/book/text_book_raw.gz"
WIKI_DATA_PATTERN = "/mnt/md0/corpus/SinDia/ctext/data/raw/wiki/text_wiki_raw_part_*"

# Metadata file path
METADATA_FILE = "compact_new_metadata.csv"

# Context window (0 = no context, just the sentence)
CONTEXT_WINDOW = 0

# Memory management
CLEAR_MEMORY_EVERY = 10
CHECKPOINT_EVERY = 100

# ============================================================
# ARGUMENT PARSING
# ============================================================

parser = argparse.ArgumentParser(description='Extract keyword embeddings using GujiBERT')
parser.add_argument('--output', type=str, default='../lrec-workshop/keyword_embeddings/',
                    help='Output directory')
parser.add_argument('--metadata', type=str, default=METADATA_FILE,
                    help='Metadata CSV file')
parser.add_argument('--dynasties', nargs='+', default=None,
                    help='Dynasty categories to process (default: all)')
parser.add_argument('--chars', nargs='+', default=None,
                    help='Characters to extract (default: all 17)')
parser.add_argument('--device', type=str, default='auto',
                    choices=['auto', 'cuda', 'cpu'],
                    help='Device to use')
parser.add_argument('--max-texts', type=int, default=None,
                    help='Max texts to process per dynasty (for testing)')
parser.add_argument('--resume', action='store_true',
                    help='Resume from checkpoint')
parser.add_argument('--book-data-path', type=str, default=BOOK_DATA_PATH,
                    help='Path to the CText book archive (text_book_raw.gz)')
parser.add_argument('--wiki-data-pattern', type=str, default=WIKI_DATA_PATTERN,
                    help='Glob pattern for the CText wiki archive parts')

# ============================================================
# HELPER FUNCTIONS
# ============================================================

def log(message):
    """Print with timestamp."""
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] {message}")
    sys.stdout.flush()


def clear_memory(device: torch.device):
    """Clear memory cache."""
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def has_punctuation(text: str) -> bool:
    """Check if text contains Chinese punctuation marks."""
    punctuation_marks = ['。', '，', '、', '；', '：', '？', '！', '「', '」', '『', '』']
    return any(p in text for p in punctuation_marks)


def split_into_sentences(text: str) -> List[str]:
    """Split text into sentences based on Chinese punctuation."""
    text = text.replace("{", "").replace("}", "").replace("\u3000", "").replace("●", "")
    paragraphs = text.split('\n\n')
    
    sentences = []
    for para in paragraphs:
        para_sentences = re.split(r'[。！？；]', para)
        sentences.extend(para_sentences)
    
    sentences = [s.strip() for s in sentences if s.strip()]
    return sentences


def _extract_target_char_spans(sentence: str, target_chars: Set[str]) -> List[Tuple[str, int, int]]:
    """Extract target characters with their positions."""
    return [(ch, i, i+1) for i, ch in enumerate(sentence) if ch in target_chars]


def _overlap_len(a0, a1, b0, b1):
    """Calculate overlap length between two spans."""
    return max(0, min(a1, b1) - max(a0, b0))


def read_ctext_archive(book_path: str, wiki_pattern: str) -> Dict[str, str]:
    """
    Read all CText archive files and create URN -> text mapping.
    """
    import tarfile
    
    texts = {}
    
    def extract_urn_from_filename(filename: str) -> Optional[str]:
        try:
            basename = os.path.basename(filename)
            basename = basename.replace('.json', '')
            parts = basename.split('_')
            if len(parts) >= 3:
                urn = parts[2]
                if urn.startswith('wb'):
                    urn = urn[2:]
                return urn
        except:
            pass
        return None
    
    # Read book data
    log(f"Reading book archive: {book_path}")
    if os.path.exists(book_path):
        try:
            with tarfile.open(book_path, 'r:gz') as tar:
                members = tar.getmembers()
                log(f"Found {len(members)} files in book archive")
                
                count = 0
                for member in tqdm(members, desc="Loading books", ncols=100, unit="file"):
                    if member.isfile() and member.name.endswith('.json'):
                        try:
                            urn = extract_urn_from_filename(member.name)
                            if not urn:
                                continue
                            
                            f = tar.extractfile(member)
                            if f:
                                content = f.read().decode('utf-8', errors='ignore')
                                obj = json.loads(content)
                                
                                fulltext = ''
                                if isinstance(obj, dict):
                                    fulltext = obj.get('fulltext', '').strip()
                                elif isinstance(obj, list):
                                    for item in obj:
                                        if isinstance(item, dict):
                                            text_part = (item.get('fulltext', '') or 
                                                       item.get('text', '') or 
                                                       item.get('content', ''))
                                            fulltext += text_part
                                        elif isinstance(item, str):
                                            fulltext += item
                                    fulltext = fulltext.strip()
                                
                                if fulltext:
                                    texts[urn] = fulltext
                                    count += 1
                        except:
                            continue
                
                log(f"Loaded {count} texts from books")
        except Exception as e:
            log(f"Error reading book archive: {e}")
    
    # Read wiki data parts
    wiki_files = sorted(glob.glob(wiki_pattern))
    log(f"Found {len(wiki_files)} wiki archive parts")
    
    if wiki_files:
        try:
            import subprocess
            import tempfile
            
            log("Combining wiki parts into temporary file...")
            
            with tempfile.NamedTemporaryFile(suffix='.tar.gz', delete=False) as tmp:
                tmp_path = tmp.name
            
            # Show merge progress with a progress bar
            log("Merging wiki parts...")
            subprocess.run(['cat'] + wiki_files, stdout=open(tmp_path, 'wb'), check=True)
            
            log(f"Reading combined wiki archive...")
            
            count = 0
            with tarfile.open(tmp_path, 'r:gz') as tar:
                members = tar.getmembers()
                log(f"Found {len(members)} files in wiki archive")
                
                for member in tqdm(members, desc="Loading wiki", ncols=100, unit="file"):
                    if member.isfile() and member.name.endswith('.json'):
                        try:
                            urn = extract_urn_from_filename(member.name)
                            if not urn:
                                continue
                            
                            f = tar.extractfile(member)
                            if f:
                                content = f.read().decode('utf-8', errors='ignore')
                                obj = json.loads(content)
                                
                                fulltext = ''
                                if isinstance(obj, dict):
                                    fulltext = obj.get('fulltext', '').strip()
                                elif isinstance(obj, list):
                                    for item in obj:
                                        if isinstance(item, dict):
                                            text_part = (item.get('fulltext', '') or 
                                                       item.get('text', '') or 
                                                       item.get('content', ''))
                                            fulltext += text_part
                                        elif isinstance(item, str):
                                            fulltext += item
                                    fulltext = fulltext.strip()
                                
                                if fulltext:
                                    texts[urn] = fulltext
                                    count += 1
                        except:
                            continue
            
            log(f"Loaded {count} texts from wiki parts")
            
            try:
                os.remove(tmp_path)
            except:
                pass
                
        except Exception as e:
            log(f"Error reading wiki archive: {e}")
    
    log(f"Total loaded: {len(texts)} unique texts from local archive")
    return texts


def compute_keyword_embeddings(
    sentences: List[str],
    tokenizer: AutoTokenizer,
    model: AutoModel,
    device: torch.device,
    target_chars: Set[str],
    dynasty: str,
    dynasty_category: str,
    reference_id: str,
    toptitle: str,
    output_dir: str,
    file_handles: Dict[str, any],
    clear_every: int = 10,
    show_progress: bool = False
) -> Dict[str, int]:
    """
    Compute embeddings only for target characters in sentences.
    Returns count of records per character.
    """
    char_counts = {ch: 0 for ch in target_chars}
    
    # Wrap sentence iteration with a progress bar
    sentence_iter = enumerate(sentences)
    if show_progress:
        sentence_iter = tqdm(sentence_iter, total=len(sentences), 
                           desc=f"    Sentences ({toptitle[:15]}...)" if len(toptitle) > 15 else f"    Sentences ({toptitle})",
                           leave=False, ncols=100)
    
    for sent_idx, sentence in sentence_iter:
        # Find target characters in this sentence
        char_spans = _extract_target_char_spans(sentence, target_chars)
        
        if not char_spans:
            continue  # Skip sentences without target chars
        
        # Tokenize
        enc = tokenizer(
            sentence,
            padding=False,
            truncation=True,
            max_length=512,
            return_offsets_mapping=True,
            return_attention_mask=True,
            return_tensors="pt"
        )
        
        input_ids = enc["input_ids"].to(device)
        attention_mask = enc["attention_mask"].to(device)
        
        # Get embeddings
        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        
        hidden_states = outputs.last_hidden_state[0].cpu()
        offset_mapping = enc["offset_mapping"][0].tolist()
        attn_mask = enc["attention_mask"][0].tolist()
        
        del input_ids, attention_mask, outputs, enc
        
        # Map token positions to vectors
        token_vec_by_span = {}
        for t_idx, (m, (a, d)) in enumerate(zip(attn_mask, offset_mapping)):
            if not m or a == d:
                continue
            key = (int(a), int(d))
            token_vec_by_span[key] = hidden_states[t_idx]
        
        # Extract embeddings for each target character
        for char, c0, c1 in char_spans:
            vecs, weights = [], []
            for (a, d), tv in token_vec_by_span.items():
                ov = _overlap_len(c0, c1, a, d)
                if ov > 0:
                    vecs.append(tv)
                    weights.append(ov / max(1, d - a))
            
            if not vecs:
                continue
            
            V = torch.stack(vecs, 0)
            W = torch.tensor(weights).unsqueeze(1)
            char_vec = (V * W).sum(0) / W.sum(0).clamp(min=1e-9)
            
            char_vec_list = char_vec.detach().float().tolist()
            
            record = {
                "word": char,
                "reference_id": reference_id,
                "dynasty": dynasty,
                "dynasty_category": dynasty_category,
                "toptitle": toptitle,
                "media": "ctext",
                "sentence": sentence,
                "sentence_index": sent_idx,
                "vector": char_vec_list,
            }
            
            # Write to appropriate file
            if char in file_handles:
                file_handles[char].write(json.dumps(record, ensure_ascii=False) + '\n')
                char_counts[char] += 1
            
            del V, W, char_vec, char_vec_list, record
        
        del hidden_states, token_vec_by_span, offset_mapping, attn_mask
        
        # Clear memory periodically
        if (sent_idx + 1) % clear_every == 0:
            clear_memory(device)
    
    return char_counts


def load_checkpoint(checkpoint_file: str) -> Set[str]:
    """Load set of already processed texts."""
    if not os.path.exists(checkpoint_file):
        return set()
    
    processed = set()
    with open(checkpoint_file, 'r', encoding='utf-8') as f:
        for line in f:
            processed.add(line.strip())
    
    log(f"Loaded checkpoint: {len(processed)} texts already processed")
    return processed


def save_checkpoint(checkpoint_file: str, text_id: str):
    """Append processed text to checkpoint."""
    with open(checkpoint_file, 'a', encoding='utf-8') as f:
        f.write(f"{text_id}\n")


def main():
    args = parser.parse_args()
    
    log("=" * 60)
    log("Keyword Embedding Extraction using GujiBERT")
    log("=" * 60)
    
    # Determine target characters
    if args.chars:
        target_chars = set(args.chars)
    else:
        target_chars = set(ALL_TARGET_CHARS)
    
    log(f"Target characters: {sorted(target_chars)}")
    
    # Determine dynasty categories
    if args.dynasties:
        dynasty_categories = args.dynasties
    else:
        dynasty_categories = list(DYNASTY_GROUPS.keys())
    
    log(f"Dynasty categories: {dynasty_categories}")
    
    # Setup device
    if args.device == 'auto':
        if torch.cuda.is_available():
            device = torch.device("cuda")
            log("Using CUDA device")
        else:
            device = torch.device("cpu")
            log("Using CPU device")
    else:
        device = torch.device(args.device)
        log(f"Using device: {args.device}")
    
    # Load metadata
    log(f"Loading metadata from {args.metadata}")
    metadata_df = pd.read_csv(args.metadata, low_memory=False)
    log(f"Total texts in metadata: {len(metadata_df)}")
    
    # Filter to valid dynasties only
    valid_dynasties = set()
    for cat in dynasty_categories:
        if cat in DYNASTY_GROUPS:
            valid_dynasties.update(DYNASTY_GROUPS[cat])
    
    metadata_df = metadata_df[metadata_df['fromDynastyName'].isin(valid_dynasties)].copy()
    log(f"Texts in selected dynasties: {len(metadata_df)}")
    
    # Add dynasty_category column
    metadata_df['dynasty_category'] = metadata_df['fromDynastyName'].map(DYNASTY_TO_CATEGORY)
    
    # Read local CText archive
    log("Reading local CText archive files...")
    log("   This may take 5-10 minutes...")
    archive_texts = read_ctext_archive(args.book_data_path, args.wiki_data_pattern)
    
    # Load model
    log(f"Loading model: {MODEL_ID}")
    log("   This may take a few minutes...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, use_fast=True)
    model = AutoModel.from_pretrained(MODEL_ID, torch_dtype=torch.float32)
    model = model.to(device)
    model.eval()
    log("Model loaded successfully")
    
    # Create output directories
    output_base = args.output
    for cat in dynasty_categories:
        os.makedirs(os.path.join(output_base, cat), exist_ok=True)
    
    # Process each dynasty category
    total_stats = {ch: 0 for ch in target_chars}
    
    for dynasty_cat in dynasty_categories:
        log(f"\n{'='*60}")
        log(f"Processing dynasty category: {dynasty_cat}")
        log(f"{'='*60}")
        
        # Filter metadata for this dynasty category
        cat_df = metadata_df[metadata_df['dynasty_category'] == dynasty_cat].copy()
        
        if args.max_texts:
            cat_df = cat_df.head(args.max_texts)
        
        log(f"Texts to process: {len(cat_df)}")
        
        # Checkpoint file
        checkpoint_file = os.path.join(output_base, dynasty_cat, '_checkpoint.txt')
        processed_texts = load_checkpoint(checkpoint_file) if args.resume else set()
        
        # Open file handles for each character
        file_handles = {}
        for char in target_chars:
            filepath = os.path.join(output_base, dynasty_cat, f'{char}_embeddings.jsonl')
            # Append mode if resuming, write mode otherwise
            mode = 'a' if args.resume else 'w'
            file_handles[char] = open(filepath, mode, encoding='utf-8')
        
        # Stats for this dynasty
        dynasty_stats = {ch: 0 for ch in target_chars}
        processed_count = 0
        skipped_checkpoint = 0
        skipped_not_found = 0
        skipped_no_punct = 0
        
        # Process each text
        for idx, row in tqdm(cat_df.iterrows(), total=len(cat_df), 
                           desc=f"{dynasty_cat}", ncols=100, unit="text"):
            title = row['title']
            toptitle = row['toptitle']
            dynasty = row['fromDynastyName']
            topurn = row['topurn']
            
            # Create unique ID for checkpoint
            text_id = f"{topurn}_{title}"
            
            # Skip if already processed
            if text_id in processed_texts:
                skipped_checkpoint += 1
                continue
            
            # Extract URN
            urn = topurn.split(':')[-1] if ':' in topurn else topurn
            if urn.startswith('wb'):
                urn = urn[2:]
            
            # Find text in archive
            if urn not in archive_texts:
                skipped_not_found += 1
                continue
            
            text = archive_texts[urn]
            
            # Check punctuation
            if not has_punctuation(text):
                skipped_no_punct += 1
                continue
            
            # Split into sentences
            sentences = split_into_sentences(text)
            if not sentences:
                continue
            
            # Create reference_id
            ref_match = re.search(r'\d+', topurn)
            reference_id = f"ctext_{ref_match.group(0)}" if ref_match else f"ctext_{urn}"
            
            # Compute embeddings
            char_counts = compute_keyword_embeddings(
                sentences=sentences,
                tokenizer=tokenizer,
                model=model,
                device=device,
                target_chars=target_chars,
                dynasty=dynasty,
                dynasty_category=dynasty_cat,
                reference_id=reference_id,
                toptitle=toptitle,
                output_dir=output_base,
                file_handles=file_handles,
                clear_every=CLEAR_MEMORY_EVERY,
                show_progress=True
            )
            
            # Update stats
            for ch, count in char_counts.items():
                dynasty_stats[ch] += count
                total_stats[ch] += count
            
            processed_count += 1
            
            # Save checkpoint
            save_checkpoint(checkpoint_file, text_id)
            
            # Clear memory
            del text, sentences
            clear_memory(device)
            
            # Progress update
            if processed_count % 50 == 0:
                log(f"  Processed: {processed_count} texts")
        
        # Close file handles
        for fh in file_handles.values():
            fh.close()
        
        # Dynasty summary
        log(f"\n--- {dynasty_cat} Summary ---")
        log(f"Texts processed: {processed_count}")
        log(f"Skipped (checkpoint): {skipped_checkpoint}")
        log(f"Skipped (not found): {skipped_not_found}")
        log(f"Skipped (no punct): {skipped_no_punct}")
        log("Character counts:")
        for ch in sorted(target_chars):
            log(f"  {ch}: {dynasty_stats[ch]:,}")
    
    # Final summary
    log(f"\n{'='*60}")
    log("FINAL SUMMARY")
    log(f"{'='*60}")
    log("Total embeddings per character:")
    for ch in sorted(total_stats.keys()):
        log(f"  {ch}: {total_stats[ch]:,}")
    log(f"\nOutput directory: {output_base}")
    log("Done.")


if __name__ == "__main__":
    main()
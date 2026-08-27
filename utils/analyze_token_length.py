#!/usr/bin/env python3
"""
Token & Text Length Scanner Utility.
Scans an input dataset CSV file, analyzes word count and subword token length distributions,
and calculates recommended text and token length limits (default 95% coverage without truncation).
Optionally updates config.yaml with the recommended max_seq_length.
"""

import argparse
import os
import sys
from typing import List, Dict, Any, Tuple

# Disable pyarrow_hotfix vulnerability patch conflict with modern pyarrow
sys.modules['pyarrow_hotfix'] = type('pyarrow_hotfix', (), {'install': lambda *args, **kwargs: None})()

import numpy as np
import pandas as pd

try:
    from transformers import AutoTokenizer
except ImportError:
    AutoTokenizer = None

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    SentenceTransformer = None

# Add parent directory to path to allow importing src
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.config import load_config, save_config, Config
from src.utils import generate_synthetic_csv


def get_tokenizer(model_name: str):
    """Instantiates a HuggingFace or SentenceTransformers tokenizer."""
    if AutoTokenizer is not None:
        try:
            return AutoTokenizer.from_pretrained(model_name)
        except Exception:
            pass

    if SentenceTransformer is not None:
        try:
            st_model = SentenceTransformer(model_name)
            return st_model.tokenizer
        except Exception as e:
            raise RuntimeError(f"Failed to load tokenizer for model '{model_name}': {e}")

    raise ImportError("Hugging Face transformers or sentence-transformers package is required.")


def analyze_lengths(
    texts: List[str],
    tokenizer: Any,
    percentile: float = 95.0,
) -> Dict[str, Any]:
    """Analyzes word and token length distributions across a list of text entries."""
    if not texts:
        raise ValueError("Text list is empty. Cannot analyze length distributions.")

    word_counts = np.array([len(text.split()) for text in texts], dtype=np.int32)
    char_counts = np.array([len(text) for text in texts], dtype=np.int32)

    print(f"[Scanner] Tokenizing {len(texts)} text entries...")
    token_counts = []
    for i, text in enumerate(texts):
        # Encode tokens without truncation
        tokens = tokenizer.encode(text, truncation=False, add_special_tokens=True)
        token_counts.append(len(tokens))

    token_counts = np.array(token_counts, dtype=np.int32)

    # Compute percentiles
    percentiles = [50.0, 75.0, 90.0, 95.0, 99.0, 100.0]
    word_stats = {f"p{int(p) if p.is_integer() else p}": int(np.percentile(word_counts, p)) for p in percentiles}
    token_stats = {f"p{int(p) if p.is_integer() else p}": int(np.percentile(token_counts, p)) for p in percentiles}

    # Recommended values for the target percentile
    rec_word_len = int(np.percentile(word_counts, percentile))
    rec_token_len = int(np.percentile(token_counts, percentile))

    # Standard transformer max length rounding (e.g. nearest multiple of 16 or 32)
    rec_token_len_rounded = int(np.ceil(rec_token_len / 16.0) * 16)

    results = {
        "num_samples": len(texts),
        "target_percentile": percentile,
        "recommended_word_length": rec_word_len,
        "recommended_token_length": rec_token_len,
        "recommended_token_length_rounded": rec_token_len_rounded,
        "word_counts": {
            "min": int(np.min(word_counts)),
            "max": int(np.max(word_counts)),
            "mean": float(np.mean(word_counts)),
            "median": int(np.median(word_counts)),
            "percentiles": word_stats,
        },
        "token_counts": {
            "min": int(np.min(token_counts)),
            "max": int(np.max(token_counts)),
            "mean": float(np.mean(token_counts)),
            "median": int(np.median(token_counts)),
            "percentiles": token_stats,
        },
    }

    return results


def main():
    parser = argparse.ArgumentParser(description="Scan CSV text column and calculate recommended token/text length.")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to YAML configuration file")
    parser.add_argument("--csv", type=str, default=None, help="Path to input CSV file to scan")
    parser.add_argument("--column", type=str, default=None, help="Text column header name to analyze")
    parser.add_argument("--percentile", type=float, default=95.0, help="Target percentile coverage (default: 95.0%)")
    parser.add_argument("--update_config", action="store_true", help="Automatically update max_seq_length in config YAML file")
    parser.add_argument("--generate_synthetic", action="store_true", help="Generate synthetic CSV dataset if file to analyze is missing")
    args = parser.parse_args()

    config_path = args.config

    print("=" * 70)
    print("      MindSpace-CLIP Dataset Token & Text Length Scanner")
    print("=" * 70)

    # 1. Load Configuration
    config = load_config(config_path)
    csv_path = args.csv or config.dataset.train_csv_path
    text_column = args.column or config.dataset.text_column
    model_name = config.text_encoder.model_name
    target_pct = args.percentile

    print(f"[1/3] Configuration loaded from '{config_path}'")
    print(f"      Target Model: '{model_name}'")
    print(f"      Target CSV Path: '{csv_path}' (Column: '{text_column}')")
    print(f"      Target Coverage Percentile: {target_pct}%")

    # 2. Check or Generate Dataset
    if not os.path.exists(csv_path):
        if args.generate_synthetic:
            print(f"[Synthetic Data] CSV dataset '{csv_path}' not found. Generating synthetic dataset (N=500)...")
            generate_synthetic_csv(
                output_csv_path=csv_path,
                text_column=text_column,
                pca_columns=config.dataset.pca_columns,
                num_samples=500,
                seed=42,
            )
        else:
            print(f"[Dataset Error] Dataset file to analyze not found at '{csv_path}'.")
            print("Please check '--csv' or 'dataset.train_csv_path' in 'config.yaml' or pass '--generate_synthetic' to create a test dataset.")
            sys.exit(1)

    # 3. Read CSV File
    df = pd.read_csv(csv_path)
    if text_column not in df.columns:
        print(f"[Error] Column '{text_column}' not found in '{csv_path}'. Available columns: {list(df.columns)}")
        sys.exit(1)

    texts = df[text_column].fillna("").astype(str).tolist()
    print(f"[2/3] Loaded {len(texts)} text records from '{csv_path}'")

    # 4. Load Tokenizer & Analyze Lengths
    print(f"[3/3] Loading tokenizer for '{model_name}'...")
    tokenizer = get_tokenizer(model_name)

    stats = analyze_lengths(texts=texts, tokenizer=tokenizer, percentile=target_pct)

    # 5. Display Formatted Distribution Summary
    rec_tokens = stats["recommended_token_length_rounded"]
    curr_tokens = config.text_encoder.max_seq_length

    print("\n" + "=" * 70)
    print(f"        TEXT & TOKEN LENGTH DISTRIBUTION SUMMARY (N={stats['num_samples']})")
    print("=" * 70)
    print(f" Target Percentile Coverage: {target_pct}%")
    print("-" * 70)
    print(" WORD COUNT STATS (Words per Entry):")
    print(f"   - Min / Max:    {stats['word_counts']['min']} / {stats['word_counts']['max']} words")
    print(f"   - Mean / Med:   {stats['word_counts']['mean']:.1f} / {stats['word_counts']['median']} words")
    print(f"   - Percentiles:  P50={stats['word_counts']['percentiles']['p50']} | P75={stats['word_counts']['percentiles']['p75']} | P90={stats['word_counts']['percentiles']['p90']} | P95={stats['word_counts']['percentiles']['p95']} | P99={stats['word_counts']['percentiles']['p99']}")
    print("-" * 70)
    print(" SUBWORD TOKEN STATS (Tokens per Entry):")
    print(f"   - Min / Max:    {stats['token_counts']['min']} / {stats['token_counts']['max']} tokens")
    print(f"   - Mean / Med:   {stats['token_counts']['mean']:.1f} / {stats['token_counts']['median']} tokens")
    print(f"   - Percentiles:  P50={stats['token_counts']['percentiles']['p50']} | P75={stats['token_counts']['percentiles']['p75']} | P90={stats['token_counts']['percentiles']['p90']} | P95={stats['token_counts']['percentiles']['p95']} | P99={stats['token_counts']['percentiles']['p99']}")
    print("-" * 70)
    print(" 💡 RECOMMENDATION:")
    print(f"   - Current max_seq_length in Config: {curr_tokens} tokens")
    print(f"   - Recommended max_seq_length ({target_pct}% no-truncation): {rec_tokens} tokens")
    print(f"   - Recommended Max Word Limit ({target_pct}% coverage): {stats['recommended_word_length']} words")
    print("=" * 70)

    # 6. Optionally Update Config File
    if args.update_config:
        config.text_encoder.max_seq_length = rec_tokens
        save_config(config, config_path)
        print(f"[Config Updated] Successfully set max_seq_length = {rec_tokens} in '{config_path}'")


if __name__ == "__main__":
    main()

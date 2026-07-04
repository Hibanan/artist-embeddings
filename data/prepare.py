"""Load, merge, and deduplicate training pairs for artist embeddings.

Returns ``datasets.DatasetDict`` objects compatible with the v3
``SentenceTransformerTrainer`` API.

Usage:
    python data/prepare.py    # prints dataset stats
"""

import sys
from pathlib import Path

# Ensure project root is on sys.path when run as `python data/prepare.py`
_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import random
from collections import Counter

from datasets import Dataset, DatasetDict

from config import MAX_PAIRS, MIN_MATCH_SCORE
from data.collect import load_cache
from data.synthetic import generate_synthetic_pairs


def load_training_pairs() -> list[tuple[str, str]]:
    """Merge synthetic + Last.fm pairs, deduplicate, cap at MAX_PAIRS.

    Returns raw (anchor, positive) tuples.  Use :func:`create_datasets`
    to convert these into the ``datasets.Dataset`` format expected by the
    v3 SentenceTransformerTrainer.
    """
    # --- Synthetic (minimal — just enough to bootstrap if no cache) ---
    synthetic = generate_synthetic_pairs()
    pairs: list[tuple[str, str]] = []
    source_counts: Counter = Counter()

    # --- Last.fm (if cache exists) ---
    cached = load_cache()
    if cached is not None:
        # Generate pairs directly from cache — no need to re-run BFS
        lastfm_pairs = []
        for artist, similars in cached.items():
            for name, score in similars:
                if score >= MIN_MATCH_SCORE:
                    lastfm_pairs.append((artist, name))
        source_counts["lastfm"] = len(lastfm_pairs)
        pairs.extend(lastfm_pairs)
        # Only add synthetic if Last.fm data is sparse
        if len(lastfm_pairs) < 1000:
            source_counts["synthetic"] = len(synthetic)
            pairs.extend(synthetic)
            print("[prepare] Last.fm data sparse — supplementing with synthetic pairs.")

    # --- Deduplicate by sorted tuple key ---
    seen: set[tuple[str, str]] = set()
    deduped: list[tuple[str, str]] = []
    for pair in pairs:
        key = tuple(sorted(pair))
        if key not in seen:
            seen.add(key)
            deduped.append(pair)
    pairs = deduped

    # --- Cap at MAX_PAIRS ---
    if len(pairs) > MAX_PAIRS:
        pairs = random.sample(pairs, MAX_PAIRS)

    # --- Stats ---
    all_artists = get_all_artists(pairs)
    print(f"[prepare] Total pairs: {len(pairs)}")
    print(f"[prepare] Unique artists: {len(all_artists)}")
    for source, count in source_counts.most_common():
        print(f"[prepare]   {source}: {count}")
    if "lastfm" not in source_counts:
        print(
            "[prepare] WARNING: Training on synthetic data only. "
            "Results will reflect synthetic-quality genre+era clustering, "
            "not real listening patterns."
        )

    return pairs


def create_datasets(
    pairs: list[tuple[str, str]], split: float = 0.9
) -> DatasetDict:
    """Convert raw pairs into a ``datasets.DatasetDict`` with train/eval splits.

    The returned dataset has columns ``["anchor", "positive"]``, matching
    the 2-input format expected by ``MultipleNegativesRankingLoss``.
    """
    random.shuffle(pairs)
    split_idx = int(len(pairs) * split)

    anchors, positives = [], []
    for a, p in pairs:
        anchors.append(a)
        positives.append(p)

    train_dataset = Dataset.from_dict({
        "anchor": anchors[:split_idx],
        "positive": positives[:split_idx],
    })
    eval_dataset = Dataset.from_dict({
        "anchor": anchors[split_idx:],
        "positive": positives[split_idx:],
    })

    return DatasetDict({"train": train_dataset, "eval": eval_dataset})


def get_all_artists(pairs: list[tuple[str, str]]) -> list[str]:
    """Return deduplicated sorted list of every artist appearing in pairs."""
    return sorted({artist for pair in pairs for artist in pair})


# ---------------------------------------------------------------------------
# Standalone: print dataset stats
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    pairs = load_training_pairs()
    datasets = create_datasets(pairs)
    print(f"[prepare] Train: {len(datasets['train'])} examples")
    print(f"[prepare] Eval:  {len(datasets['eval'])} examples")
    print(f"[prepare] Columns: {datasets['train'].column_names}")
    print(f"[prepare] Ready for SentenceTransformerTrainer.")

"""Demo script for artist similarity search.

Queries a diverse set of artists and prints top similar results.

Usage:
    python demo.py

If model / index hasn't been built yet, run:
    python train.py && python build_index.py
"""

from __future__ import annotations

import textwrap

from config import TOP_K
from inference import ArtistSearch


def main() -> None:
    # A diverse set spanning pop, hip-hop, metal, electronic, country, jazz
    query_artists = [
        "Taylor Swift",
        "Maroon 5",
        "Kendrick Lamar",
        "Metallica",
        "Daft Punk",
    ]

    try:
        searcher = ArtistSearch()
    except FileNotFoundError:
        print(
            textwrap.dedent("""\
                Model or index not found.

                To train and build from scratch:
                    python train.py && python build_index.py
                """)
        )
        return

    header = f"{'Demo: Artist Similarity Search':^60}"
    sep = "=" * 60
    print(f"\n{header}")
    print(sep)

    for artist in query_artists:
        results = searcher.query(artist, top_k=TOP_K)
        print(f"\nQuery: {artist}")
        print("-" * len(f"Query: {artist}"))
        if not results:
            print("  (no results)")
        else:
            for rank, (name, score) in enumerate(results, 1):
                print(f"  {rank:>2}. {name:<40s} ({score:.4f})")

    # Summary
    print(f"\n{sep}")
    print("Index Summary")
    print(sep)
    print(f"  Total artists:    {searcher.index.ntotal}")
    print(f"  Embedding dim:    {searcher.index.d}")
    print(f"  Model:            {searcher.model_name}")
    print(f"  Top-K queried:    {TOP_K}")
    print(sep)


if __name__ == "__main__":
    main()
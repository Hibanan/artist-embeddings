"""Artist similarity search via FAISS-indexed SentenceTransformer embeddings.

Usage:
    from inference import ArtistSearch
    searcher = ArtistSearch()
    results = searcher.query("Taylor Swift", top_k=5)
"""

from __future__ import annotations

import json
from pathlib import Path

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from config import INDEX_DIR, MODEL_DIR, TOP_K


class ArtistSearch:
    """Load a pre-trained bi-encoder and FAISS index, then query for similar artists."""

    def __init__(
        self,
        model_path: str | Path | None = None,
        index_dir: str | Path | None = None,
    ):
        model_path = Path(model_path or MODEL_DIR)
        index_dir = Path(index_dir or INDEX_DIR)

        index_file = index_dir / "artist_index.faiss"
        names_file = index_dir / "artist_names.json"

        # --- Load model ---
        if not model_path.exists() or not any(model_path.iterdir()):
            raise FileNotFoundError(
                f"Model not found at {model_path.resolve()}\n"
                "Run: python train.py && python build_index.py"
            )
        self.model = SentenceTransformer(str(model_path))
        self.model_name = model_path.name
        print(f"Loaded model: {self.model_name}")

        # --- Load FAISS index ---
        if not index_file.exists():
            raise FileNotFoundError(
                f"FAISS index not found at {index_file.resolve()}\n"
                "Run: python train.py && python build_index.py"
            )
        self.index = faiss.read_index(str(index_file))
        print(f"Loaded index: {index_file} ({self.index.ntotal} vectors, dim {self.index.d})")

        # --- Load artist name list ---
        if not names_file.exists():
            raise FileNotFoundError(
                f"Artist list not found at {names_file.resolve()}\n"
                "Run: python train.py && python build_index.py"
            )
        with open(names_file) as f:
            self.artists: list[str] = json.load(f)
        print(f"Loaded {len(self.artists)} artist names")

        if self.index.ntotal != len(self.artists):
            raise ValueError(
                f"Mismatch: index has {self.index.ntotal} vectors "
                f"but artist list has {len(self.artists)} entries"
            )

    def _embed_and_normalize(self, texts: list[str]) -> np.ndarray:
        """Embed texts and L2-normalize for cosine similarity via inner product."""
        embeddings = self.model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
        faiss.normalize_L2(embeddings)
        return embeddings

    def query(self, artist_name: str, top_k: int = TOP_K) -> list[tuple[str, float]]:
        """Return top-K similar artists with (name, cosine_similarity) scores.

        Filters out the query artist itself from results.
        """
        vec = self._embed_and_normalize([artist_name])
        scores, indices = self.index.search(vec, min(top_k + 1, self.index.ntotal))

        results: list[tuple[str, float]] = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue
            name = self.artists[idx]
            # Skip self-match (case-insensitive, strip whitespace)
            if name.strip().lower() == artist_name.strip().lower():
                continue
            results.append((name, float(score)))
            if len(results) == top_k:
                break

        return results

    def query_batch(
        self, names: list[str], top_k: int = TOP_K
    ) -> list[list[tuple[str, float]]]:
        """Batch query: embed all names, search, return per-name result lists."""
        vecs = self._embed_and_normalize(names)
        # Need k+1 to do self-match removal per row
        scores, indices = self.index.search(vecs, min(top_k + 1, self.index.ntotal))

        all_results: list[list[tuple[str, float]]] = []
        for row_idx in range(len(names)):
            query_name = names[row_idx]
            results: list[tuple[str, float]] = []
            for score, idx in zip(scores[row_idx], indices[row_idx]):
                if idx == -1:
                    continue
                name = self.artists[idx]
                if name.strip().lower() == query_name.strip().lower():
                    continue
                results.append((name, float(score)))
                if len(results) == top_k:
                    break
            all_results.append(results)

        return all_results

    def get_artist_list(self) -> list[str]:
        """Return the full deduplicated list of indexed artist names."""
        return list(self.artists)


if __name__ == "__main__":
    import textwrap

    try:
        searcher = ArtistSearch()
    except FileNotFoundError as e:
        print(textwrap.dedent(f"""\
            {e}

            To train and build the index:
                1. (Optional) Collect Last.fm data:  python collect.py
                2. Train:                            python train.py
                3. Build index:                      python build_index.py
            """))
        raise SystemExit(1) from e

    print("\n--- Quick test queries ---")
    test_artists = ["Taylor Swift", "Metallica", "Daft Punk"]
    for artist in test_artists:
        results = searcher.query(artist, top_k=3)
        print(f"\nQuery: {artist}")
        for i, (name, score) in enumerate(results, 1):
            print(f"  {i}. {name} ({score:.4f})")
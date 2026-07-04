"""Build a FAISS index over artist embeddings for fast nearest-neighbor search.

Loads the fine-tuned SentenceTransformer, embeds every known artist, L2-normalises,
builds an IndexFlatIP (inner product == cosine similarity on unit vectors), and
persists both the index and the artist-name list to INDEX_DIR.
"""

import json
import logging
import sys

import numpy as np

from config import INDEX_DIR, MODEL_DIR, TOP_K
from data.prepare import get_all_artists, load_training_pairs

from sentence_transformers import SentenceTransformer

logging.basicConfig(
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    level=logging.INFO,
    stream=sys.stdout,
)
logger = logging.getLogger("build_index")


def main():
    # --- Load model ---
    model_path = MODEL_DIR / "final"
    if not model_path.exists():
        model_path = MODEL_DIR
    logger.info("Loading model from %s", model_path)
    model = SentenceTransformer(str(model_path), device="cuda" if __import__("torch").cuda.is_available() else "cpu")
    device = model.device
    logger.info("Model loaded on %s", device)

    # --- Load artist list from training pairs ---
    logger.info("Loading training pairs and building artist list...")
    pairs = load_training_pairs()
    artists = get_all_artists(pairs)
    logger.info("Loaded %d unique artists", len(artists))

    if not artists:
        logger.error("No artists found -- cannot build index. Run data collection first.")
        sys.exit(1)

    # --- Embed all artists in batches ---
    batch_size = 256
    all_embeddings = []

    logger.info("Encoding %d artists in batches of %d...", len(artists), batch_size)
    for start in range(0, len(artists), batch_size):
        batch = artists[start : start + batch_size]
        emb = model.encode(batch, convert_to_numpy=True, show_progress_bar=False)
        all_embeddings.append(emb)
        if (start // batch_size) % 10 == 0:
            logger.info("  Encoded %d / %d", start + len(batch), len(artists))

    embeddings = np.concatenate(all_embeddings, axis=0)
    logger.info("Embedding matrix shape: %s", embeddings.shape)

    # --- L2-normalise so inner product = cosine similarity ---
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)  # guard against zero-norm vectors
    embeddings = embeddings / norms

    # --- Build FAISS index ---
    import faiss

    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    logger.info("Built FAISS IndexFlatIP (dim=%d)", dim)

    index.add(embeddings.astype(np.float32))
    logger.info("Added %d vectors to index", index.ntotal)

    # --- Save ---
    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    index_path = INDEX_DIR / "artist_index.faiss"
    names_path = INDEX_DIR / "artist_names.json"

    faiss.write_index(index, str(index_path))
    with open(names_path, "w", encoding="utf-8") as f:
        json.dump(artists, f, ensure_ascii=False)

    logger.info("Index saved to  %s", index_path)
    logger.info("Names saved to  %s", names_path)
    logger.info("Index size:     %d artists, dim=%d", index.ntotal, dim)

    # --- Quick sanity: nearest neighbours for a few seeds ---
    seed_artists = [
        "Taylor Swift",
        "Metallica",
        "Daft Punk",
        "Miles Davis",
        "Kendrick Lamar",
        "Adele",
    ]
    # Build a lookup mapping for fast O(1) access by name
    name_to_idx = {name: i for i, name in enumerate(artists)}

    logger.info("\nNearest neighbour samples (top-%d):", TOP_K)
    for seed in seed_artists:
        idx = name_to_idx.get(seed)
        if idx is None:
            logger.info("  '%s' not found in artist list -- skipping", seed)
            continue

        query_vec = embeddings[idx : idx + 1].astype(np.float32)
        scores, indices = index.search(query_vec, TOP_K + 1)

        neighbours = []
        for rank, (neighbor_idx, score) in enumerate(zip(indices[0], scores[0])):
            if rank == 0 and artists[neighbor_idx] == seed:
                continue  # skip self-match
            neighbours.append(f"{artists[neighbor_idx]} ({score:.4f})")

        logger.info("  %s: %s", seed, ", ".join(neighbours[:TOP_K]))


if __name__ == "__main__":
    main()
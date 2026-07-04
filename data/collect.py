"""Last.fm API collector for collaborative filtering artist pairs.

Collects similar-artist data via BFS from seed artists, caching results
incrementally to ``data/lastfm_cache.json``.

Exports:
    collect_from_lastfm(seed_artists, depth=2, max_per=15) -> list[tuple[str, str]]
    fetch_similar(artist, limit=30) -> list[tuple[str, float]]
    load_cache() -> dict | None
"""

from __future__ import annotations
import sys
from pathlib import Path  # noqa: F401 — used for sys.path bootstrapping

# Allow ``python data/collect.py`` from the project root to find ``config``
_project_root = Path(__file__).resolve().parent.parent
if _project_root not in sys.path:
    sys.path.insert(0, str(_project_root))

import json
import logging
import os
import time
from collections import deque
from pathlib import Path
from typing import Any

import requests

# Project config — use the same import pattern as siblings
from config import (
    DATA_DIR,
    LASTFM_API_KEY,
    LASTFM_BASE_URL,
    LASTFM_RATE_LIMIT,
    MIN_MATCH_SCORE,
)

log = logging.getLogger(__name__)

CACHE_PATH: Path = DATA_DIR / "lastfm_cache.json"

# ---------------------------------------------------------------------------
# HTTP helpers
# ---------------------------------------------------------------------------

_last_request: float = 0.0  # monotonic timestamp of last API call


def _rate_limit() -> None:
    """Sleep if necessary to honour ``LASTFM_RATE_LIMIT``."""
    global _last_request
    elapsed = time.monotonic() - _last_request
    if elapsed < LASTFM_RATE_LIMIT:
        time.sleep(LASTFM_RATE_LIMIT - elapsed)
    _last_request = time.monotonic()


def fetch_similar(artist: str, limit: int = 30) -> list[tuple[str, float]]:
    """Fetch similar artists from Last.fm via ``artist.getSimilar``.

    Returns a list of ``(similar_name, match_score)`` tuples sorted by
    descending score.  On any error (network, missing artist, malformed
    response) the error is logged and an empty list is returned.
    """
    key = LASTFM_API_KEY or os.getenv("LASTFM_API_KEY", "")
    if not key:
        log.warning("LASTFM_API_KEY is not configured — cannot fetch similar artists")
        return []

    _rate_limit()

    try:
        resp = requests.get(
            LASTFM_BASE_URL,
            params={
                "method": "artist.getSimilar",
                "artist": artist,
                "limit": str(limit),
                "api_key": key,
                "format": "json",
            },
            timeout=15,
        )
        resp.raise_for_status()
    except requests.RequestException as exc:
        log.warning("Failed to fetch similar for %r: %s", artist, exc)
        return []

    try:
        data: dict[str, Any] = resp.json()
    except ValueError:
        log.warning("Non-JSON response for %r", artist)
        return []

    # Handle API-level error responses
    if "error" in data:
        log.warning("Last.fm API error for %r: %s", artist, data.get("message", data["error"]))
        return []

    similar_artists = data.get("similarartists", {}).get("artist")
    if not similar_artists:
        # artist exists but has no similar artists listed
        return []

    results: list[tuple[str, float]] = []
    for entry in similar_artists:
        name = entry.get("name", "")
        match_str = entry.get("match", "0")
        try:
            score = float(match_str)
        except (ValueError, TypeError):
            score = 0.0
        if name:
            results.append((name, score))

    results.sort(key=lambda x: x[1], reverse=True)
    return results


# ---------------------------------------------------------------------------
# Cache I/O
# ---------------------------------------------------------------------------


def load_cache() -> dict[str, list[list[Any]]] | None:
    """Return the cached artist-similar map, or ``None`` if unavailable."""
    if not CACHE_PATH.exists():
        return None
    try:
        with CACHE_PATH.open("r", encoding="utf-8") as fh:
            return json.load(fh)
    except (json.JSONDecodeError, OSError) as exc:
        log.warning("Corrupt or unreadable cache %s: %s", CACHE_PATH, exc)
        return None


def _save_cache(cache: dict[str, list[list[Any]]]) -> None:
    """Write the cache dict to disk, retrying on Windows permission errors."""
    import time as _time
    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp = CACHE_PATH.with_suffix(".tmp")
    with tmp.open("w", encoding="utf-8") as fh:
        json.dump(cache, fh, ensure_ascii=False, indent=2)
    for attempt in range(5):
        try:
            tmp.replace(CACHE_PATH)
            return
        except PermissionError:
            if attempt < 4:
                _time.sleep(0.5 * (attempt + 1))
    # Last resort: overwrite directly
    CACHE_PATH.write_text(tmp.read_text(encoding="utf-8"), encoding="utf-8")

# ---------------------------------------------------------------------------
# BFS collection
# ---------------------------------------------------------------------------


def collect_from_lastfm(
    seed_artists: list[str],
    depth: int = 2,
    max_per: int = 15,
) -> list[tuple[str, str]]:
    """BFS from *seed_artists* via Last.fm similar-artist lookups.

    Parameters
    ----------
    seed_artists:
        Artist names to start the BFS from.
    depth:
        Number of BFS layers to expand (``0`` = seeds only, no expansion).
    max_per:
        Maximum number of similar artists to emit per source artist.

    Returns
    -------
    list[tuple[str, str]]
        ``(anchor, positive)`` pairs (up to *max_per* per anchor).
    """
    key = LASTFM_API_KEY or os.getenv("LASTFM_API_KEY", "")
    if not key:
        print("=" * 60)
        print("  No Last.fm API key configured.")
        print()
        print("  1. Get a free key at:  https://www.last.fm/api/account/create")
        print("  2. Set the environment variable:")
        print('       set LASTFM_API_KEY=your_key_here     (Windows)')
        print('       export LASTFM_API_KEY=your_key_here  (macOS/Linux)')
        print("  3. Rerun this script.")
        print("=" * 60)
        return []

    cache: dict[str, list[list[Any]]] = load_cache() or {}

    # Track which artists we've expanded (queued for BFS) vs just seen
    expanded: set[str] = set()
    queue: deque[tuple[str, int]] = deque()

    # Always enqueue seeds — even if cached, we need their pairs
    for s in seed_artists:
        queue.append((s, 0))

    # Also process any cached artists NOT reachable from seeds
    # (from previous runs at different depths)
    remaining_cached = set(cache.keys()) - {s for s, _ in queue}
    for artist in sorted(remaining_cached):
        queue.append((artist, 1))  # treat as depth 1 (won't expand further if depth=1)

    pairs: list[tuple[str, str]] = []
    new_fetches = 0
    total_processed = 0

    while queue:
        artist, level = queue.popleft()
        if artist in expanded:
            continue
        expanded.add(artist)
        total_processed += 1

        # Fetch (or retrieve from cache)
        if artist in cache:
            similar = cache[artist]
        else:
            raw = fetch_similar(artist, limit=max_per * 2)
            similar = [[name, score] for name, score in raw]
            cache[artist] = similar
            _save_cache(cache)
            new_fetches += 1
            if new_fetches % 50 == 0:
                log.info("Fetched %d new artists so far (%d total processed, %d pairs)",
                         new_fetches, total_processed, len(pairs))

        # Yield pairs (no 'seen' guard — we want pairs even for known artists)
        for name, score in similar:
            if score >= MIN_MATCH_SCORE:
                pairs.append((artist, name))

        # Enqueue for next BFS level
        if level < depth:
            for name, score in similar[:max_per]:
                if name not in expanded and score >= MIN_MATCH_SCORE:
                    queue.append((name, level + 1))

    log.info("Collection complete — %d fetches, %d total processed, %d pairs",
             new_fetches, total_processed, len(pairs))


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    # Import here so the module can be imported without side-effects
    from config import SEED_ARTISTS, BFS_DEPTH, BFS_MAX_PER

    pairs = collect_from_lastfm(SEED_ARTISTS, depth=BFS_DEPTH, max_per=BFS_MAX_PER)
    print(f"Collected {len(pairs)} (anchor, positive) pairs")
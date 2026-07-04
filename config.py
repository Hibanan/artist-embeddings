"""Central config for artist embeddings project.

Collaborative filtering approach: train a bi-encoder so that artists
frequently co-listened-to have high cosine similarity. Query returns top-K.

Training targets RunPod or similar GPU instances; CPU fallback works for
small-scale iteration with synthetic data.
"""

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
MODEL_DIR = ROOT / "model"
INDEX_DIR = ROOT / "index"

# --- Model ---
BASE_MODEL = os.getenv("MODEL_NAME", "sentence-transformers/all-mpnet-base-v2")
# Auto-detect embedding dim from model name
_model_lower = BASE_MODEL.lower()
EMBEDDING_DIM = 768 if "mpnet" in _model_lower or "bert-base" in _model_lower else 384

# --- Training ---
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "32"))
EPOCHS = int(os.getenv("EPOCHS", "15"))
EARLY_STOPPING_PATIENCE = 0  # 0 = disabled (overfit intentionally)
LEARNING_RATE = float(os.getenv("LEARNING_RATE", "5e-5"))
WARMUP_STEPS = int(os.getenv("WARMUP_STEPS", "400"))
EVAL_STEPS = int(os.getenv("EVAL_STEPS", "200"))
MAX_PAIRS = int(os.getenv("MAX_PAIRS", "1000000"))

# --- Retrieval ---
TOP_K = int(os.getenv("TOP_K", "10"))

LASTFM_API_KEY = os.getenv("LASTFM_API_KEY", "")
LASTFM_BASE_URL = "https://ws.audioscrobbler.com/2.0/"
LASTFM_RATE_LIMIT = 0.15  # seconds between requests (~6.7/sec, safe for free tier)
BFS_DEPTH = 2              # BFS expansion depth for data collection
BFS_MAX_PER = 20           # max similar artists to use per source artist
MIN_MATCH_SCORE = 0.25     # minimum Last.fm match score for pair generation

# --- Seed artists for bootstrapping data collection ---
SEED_ARTISTS = [
    "Taylor Swift", "Maroon 5", "Arctic Monkeys", "Kendrick Lamar",
    "Adele", "Drake", "Billie Eilish", "Ed Sheeran",
    "The Weeknd", "Dua Lipa", "Coldplay", "Bruno Mars",
    "Lana Del Rey", "Post Malone", "Ariana Grande", "Kanye West",
    "Rihanna", "Eminem", "Beyonce", "Imagine Dragons",
    "Harry Styles", "Olivia Rodrigo", "Doja Cat", "Bad Bunny",
    "Metallica", "Nirvana", "Radiohead", "Queen",
    "Daft Punk", "Calvin Harris", "Marshmello", "David Guetta",
    "Luke Bryan", "Morgan Wallen", "Carrie Underwood", "Johnny Cash",
    "BTS", "BLACKPINK", "Bad Bunny", "J Balvin",
    "Miles Davis", "John Coltrane", "Frank Sinatra", "Ella Fitzgerald",
    "Tame Impala", "Mac DeMarco", "Beach House", "The Strokes",
]

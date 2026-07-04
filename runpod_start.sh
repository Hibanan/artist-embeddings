#!/bin/bash
# RunPod entrypoint — full training pipeline
set -e

echo "=== Artist Embeddings — RunPod Training ==="
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "Model: ${MODEL_NAME:-sentence-transformers/all-mpnet-base-v2}"
echo "Epochs: ${EPOCHS:-15}"
echo "Batch: ${BATCH_SIZE:-32}"
echo ""

# Ensure workspace exists for cache/output
# Symlink workspace dirs — create targets first, then link
mkdir -p /workspace/model /workspace/index
rm -rf /app/model /app/index
ln -sf /workspace/model /app/model
ln -sf /workspace/index /app/index
# Ensure the symlink targets are real (broken symlinks break Path.mkdir)
mkdir -p /app/model /app/index
echo "[1/4] Verifying CUDA..."
python -c "import torch; assert torch.cuda.is_available(), 'CUDA not found!'; print(f'  {torch.cuda.get_device_name(0)} — {torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB')"

echo "[2/4] Preparing dataset..."
python data/prepare.py

echo "[3/4] Training..."
python train.py

echo "[4/4] Building FAISS index..."
python build_index.py

echo ""
echo "=== Training complete ==="
echo "Model: /workspace/model/final"
echo "Index: /workspace/index/"
echo ""

# Run a quick demo query
python -c "
from inference import ArtistSearch
s = ArtistSearch(model_path='/workspace/model/final', index_dir='/workspace/index')
for q in ['Taylor Swift', 'Lana Del Rey', 'Kendrick Lamar', 'Metallica']:
    results = s.query(q, top_k=5)
    print(f'\n{q}:')
    for r, (name, score) in enumerate(results, 1):
        print(f'  {r}. {name} ({score:.4f})')
"

echo ""
echo "[5/5] Pushing to GitHub release..."
if [ -n "${GH_TOKEN:-}" ]; then
    echo "$GH_TOKEN" | gh auth login --with-token
    ARCHIVE="artist-embeddings-$(date +%Y%m%d-%H%M%S).tar.gz"
    tar -czf "/workspace/$ARCHIVE" -C /workspace model/final index/
    echo "Archive: /workspace/$ARCHIVE ($(du -h /workspace/$ARCHIVE | cut -f1))"
    gh release create "run-$(date +%Y%m%d-%H%M%S)" "/workspace/$ARCHIVE" \
        --repo Hibanan/artist-embeddings \
        --title "Training run $(date +%Y-%m-%d\ %H:%M)" \
        --notes "Automated upload from RunPod training pod."
    echo "Release pushed to GitHub."
else
    echo "GH_TOKEN not set — skipping release upload."
fi

echo ""
echo "Keeping pod alive — press Ctrl+C to stop."
# Sleep indefinitely so the pod stays alive for interactive use
exec sleep infinity

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
# Symlink workspace dirs — remove existing dirs first to avoid nested symlinks
rm -rf /app/model /app/index
ln -sf /workspace/model /app/model
ln -sf /workspace/index /app/index

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
echo "Keeping pod alive — press Ctrl+C to stop."
# Sleep indefinitely so the pod stays alive for interactive use
exec sleep infinity

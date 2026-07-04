"""Quick CUDA dry-run: one training epoch to verify GPU pipeline."""
import sys
import traceback
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

try:
    print("1. importing sentence_transformers...")
    from sentence_transformers import SentenceTransformer, SentenceTransformerTrainer
    from sentence_transformers.losses import MultipleNegativesRankingLoss
    from sentence_transformers.training_args import SentenceTransformerTrainingArguments, BatchSamplers
    print("   OK")

    print("2. importing torch...")
    import torch
    print(f"   OK — {torch.__version__}, CUDA={torch.cuda.is_available()}")

    print("3. importing data.prepare...")
    from data.prepare import load_training_pairs, create_datasets
    print("   OK")

    print("4. loading training pairs...")
    pairs = load_training_pairs()
    print(f"   OK — {len(pairs)} pairs")

    print("5. creating datasets...")
    datasets = create_datasets(pairs)
    print(f"   OK — train={len(datasets['train'])}, eval={len(datasets['eval'])}")

    print("6. loading model on GPU...")
    model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2', device='cuda')
    loss = MultipleNegativesRankingLoss(model)
    print(f"   OK — {model.device}")

    print("7. setting up trainer...")
    args = SentenceTransformerTrainingArguments(
        output_dir='model/test',
        num_train_epochs=1,
        per_device_train_batch_size=64,
        per_device_eval_batch_size=64,
        eval_strategy='steps',
        eval_steps=200,
        save_strategy='no',
        logging_steps=100,
        report_to='none',
        fp16=True,
        batch_sampler=BatchSamplers.NO_DUPLICATES,
    )
    trainer = SentenceTransformerTrainer(
        model=model, args=args,
        train_dataset=datasets['train'],
        eval_dataset=datasets['eval'],
        loss=loss,
    )
    print("   OK")

    print(f"8. GPU: {torch.cuda.get_device_name(0)}, VRAM: {torch.cuda.get_device_properties(0).total_memory/1e9:.1f}GB")
    print("9. Starting 1-epoch dry run...")
    trainer.train()
    final = trainer.state.log_history[-1]
    print(f"10. Final loss: {final.get('loss', final)}")
    print("CUDA dry run PASSED.")

except Exception:
    traceback.print_exc()
    sys.exit(1)

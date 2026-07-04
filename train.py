"""Train a SentenceTransformer bi-encoder on artist similarity pairs.

Uses MultipleNegativesRankingLoss with BatchSamplers.NO_DUPLICATES.
Saves the fine-tuned model to MODEL_DIR.

Usage:
    python train.py
"""

import logging
import sys

from sentence_transformers import (
    SentenceTransformer,
    SentenceTransformerTrainer,
    SentenceTransformerTrainingArguments,
)
from sentence_transformers.evaluation import TripletEvaluator
from sentence_transformers.losses import MultipleNegativesRankingLoss
from sentence_transformers.training_args import BatchSamplers
from transformers import EarlyStoppingCallback

import torch

from config import (
    BASE_MODEL, BATCH_SIZE, EARLY_STOPPING_PATIENCE, EPOCHS, EVAL_STEPS,
    LEARNING_RATE, MODEL_DIR, WARMUP_STEPS,
)
from data.prepare import create_datasets, load_training_pairs

logging.basicConfig(
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    level=logging.INFO,
    stream=sys.stdout,
)
logger = logging.getLogger("train")


def build_triplet_eval(model, eval_pairs: list[tuple[str, str]]):
    """Create a TripletEvaluator from (anchor, positive) pairs.

    Generates random negatives from the pool of all artists in the eval set.
    """
    all_artists = list({artist for pair in eval_pairs for artist in pair})

    anchors, positives, negatives = [], [], []
    for anchor, positive in eval_pairs:
        pool = [a for a in all_artists if a not in (anchor, positive)]
        if pool:
            negatives.append(pool[hash(anchor + positive) % len(pool)])
            anchors.append(anchor)
            positives.append(positive)

    if not negatives:
        logger.warning("No valid triplets could be built from eval pairs")
        return None

    return TripletEvaluator(
        anchors=anchors,
        positives=positives,
        negatives=negatives,
        name="artist-eval",
        show_progress_bar=False,
    )


def main():
    # --- Device ---
    if torch.cuda.is_available():
        device = torch.device("cuda")
        logger.info("CUDA available: %s", torch.cuda.get_device_name(0))
    else:
        device = torch.device("cpu")
        logger.warning(
            "CUDA not available -- training on CPU. "
            "Consider a GPU instance (RunPod, Colab, etc.) for real workloads."
        )

    # --- Data ---
    logger.info("Loading training pairs...")
    pairs = load_training_pairs()
    logger.info("Loaded %d training pairs", len(pairs))
    datasets = create_datasets(pairs)
    train_dataset = datasets["train"]
    eval_dataset = datasets["eval"]
    logger.info("Train: %d examples, Eval: %d examples", len(train_dataset), len(eval_dataset))

    # --- Model ---
    logger.info("Loading base model: %s", BASE_MODEL)
    model = SentenceTransformer(BASE_MODEL, device=device)
    loss = MultipleNegativesRankingLoss(model)

    # --- Evaluator: use raw eval pairs (tuples), not the dataset object ---
    split_idx = int(len(pairs) * 0.9)
    eval_pairs_raw = pairs[split_idx:]
    evaluator = build_triplet_eval(model, eval_pairs_raw)

    # --- Training arguments ---
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    args = SentenceTransformerTrainingArguments(
        output_dir=str(MODEL_DIR),
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        learning_rate=LEARNING_RATE,
        warmup_steps=WARMUP_STEPS,
        eval_strategy="steps",
        eval_steps=EVAL_STEPS,
        save_strategy="steps",
        save_steps=EVAL_STEPS,
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        fp16=torch.cuda.is_available(),
        batch_sampler=BatchSamplers.NO_DUPLICATES,
        report_to="none",
        dataloader_drop_last=False,
    )

    # --- Trainer ---
    trainer = SentenceTransformerTrainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        loss=loss,
        evaluator=evaluator,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=EARLY_STOPPING_PATIENCE)] if EARLY_STOPPING_PATIENCE > 0 else None,
    )

    # --- Train ---
    logger.info("Starting training for %d epochs", EPOCHS)
    trainer.train()
    logger.info("Training complete")

    # --- Save final model ---
    final_path = MODEL_DIR / "final"
    model.save_pretrained(str(final_path))
    logger.info("Final model saved to %s", final_path)

    # --- Manual sanity check ---
    logger.info("Manual embedding similarity check:")
    test_artists = [
        "Taylor Swift", "Kendrick Lamar", "Adele",
        "Metallica", "Daft Punk", "Miles Davis",
    ]
    embeddings = model.encode(
        test_artists, convert_to_tensor=True, normalize_embeddings=True
    )
    similarities = embeddings @ embeddings.T
    for i in range(len(test_artists)):
        for j in range(i + 1, len(test_artists)):
            logger.info(
                "  %s <-> %s: %.4f",
                test_artists[i], test_artists[j],
                similarities[i, j].item(),
            )


if __name__ == "__main__":
    main()

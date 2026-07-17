from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from app.config import settings
from training.dataset_slue import label_names, load_slue_voxceleb


def main() -> None:
    parser = argparse.ArgumentParser(description="Fine-tune Wav2Vec2 for voice emotion recognition")
    parser.add_argument("--base-model", default="facebook/wav2vec2-base")
    parser.add_argument("--output", default=settings.wav2vec_emotion_model_path)
    parser.add_argument("--epochs", type=float, default=3)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    args = parser.parse_args()

    import evaluate
    from transformers import (
        AutoFeatureExtractor,
        AutoModelForAudioClassification,
        Trainer,
        TrainingArguments,
    )

    splits = load_slue_voxceleb()
    labels = label_names(splits.train)
    label_to_id = {label: idx for idx, label in enumerate(labels)}
    feature_extractor = AutoFeatureExtractor.from_pretrained(args.base_model)

    def preprocess(batch):
        audio = batch["audio"]
        inputs = feature_extractor(
            audio["array"],
            sampling_rate=audio["sampling_rate"],
            max_length=settings.audio_sample_rate * 8,
            truncation=True,
        )
        batch["input_values"] = inputs["input_values"][0]
        label_value = batch.get("label", batch.get("sentiment", batch.get("emotion")))
        batch["labels"] = label_value if isinstance(label_value, int) else label_to_id[str(label_value).lower()]
        return batch

    train_ds = splits.train.map(preprocess, remove_columns=splits.train.column_names)
    val_ds = splits.validation.map(preprocess, remove_columns=splits.validation.column_names)
    accuracy = evaluate.load("accuracy")
    f1 = evaluate.load("f1")

    def compute_metrics(pred):
        predictions = np.argmax(pred.predictions, axis=1)
        return {
            "accuracy": accuracy.compute(predictions=predictions, references=pred.label_ids)["accuracy"],
            "f1": f1.compute(predictions=predictions, references=pred.label_ids, average="weighted")["f1"],
        }

    model = AutoModelForAudioClassification.from_pretrained(
        args.base_model,
        num_labels=len(labels),
        label2id=label_to_id,
        id2label={idx: label for label, idx in label_to_id.items()},
    )
    training_args = TrainingArguments(
        output_dir=args.output,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        num_train_epochs=args.epochs,
        fp16=False,
        gradient_checkpointing=True,
        report_to=[],
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        tokenizer=feature_extractor,
        compute_metrics=compute_metrics,
    )
    trainer.train()
    trainer.save_model(args.output)
    feature_extractor.save_pretrained(args.output)
    print(f"Saved fine-tuned Wav2Vec2 model to {args.output}")


if __name__ == "__main__":
    main()

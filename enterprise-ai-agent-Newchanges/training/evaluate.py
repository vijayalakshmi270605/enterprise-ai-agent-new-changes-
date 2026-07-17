from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from app.config import settings
from app.nlp.text_sentiment import TextSentimentPipeline
from training.dataset_slue import iter_text_examples, load_slue_voxceleb


def _save_plots(y_true, y_pred, labels, probabilities, output_dir: Path) -> None:
    import matplotlib.pyplot as plt
    import seaborn as sns
    from sklearn.metrics import ConfusionMatrixDisplay, PrecisionRecallDisplay, RocCurveDisplay, confusion_matrix
    from sklearn.preprocessing import label_binarize

    output_dir.mkdir(parents=True, exist_ok=True)
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    ConfusionMatrixDisplay(cm, display_labels=labels).plot(xticks_rotation=45)
    plt.tight_layout()
    plt.savefig(output_dir / "confusion_matrix.png")
    plt.close()

    sns.countplot(x=y_true, order=labels)
    plt.title("Emotion/Sentiment Distribution")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(output_dir / "emotion_distribution.png")
    plt.close()

    if probabilities and len(labels) > 1:
        y_bin = label_binarize(y_true, classes=labels)
        prob_matrix = np.asarray([[row.get(label, 0.0) for label in labels] for row in probabilities])
        for index, label in enumerate(labels):
            RocCurveDisplay.from_predictions(y_bin[:, index], prob_matrix[:, index], name=label)
        plt.tight_layout()
        plt.savefig(output_dir / "roc.png")
        plt.close()

        for index, label in enumerate(labels):
            PrecisionRecallDisplay.from_predictions(y_bin[:, index], prob_matrix[:, index], name=label)
        plt.tight_layout()
        plt.savefig(output_dir / "precision_recall.png")
        plt.close()


def plot_training_history(checkpoint_path: str, output_dir: Path) -> None:
    import matplotlib.pyplot as plt
    import torch

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    history = checkpoint.get("history", {})
    output_dir.mkdir(parents=True, exist_ok=True)
    if history.get("loss"):
        plt.plot(history["loss"])
        plt.title("Loss Curve")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.tight_layout()
        plt.savefig(output_dir / "loss_curve.png")
        plt.close()
    if history.get("accuracy"):
        plt.plot(history["accuracy"])
        plt.title("Accuracy Curve")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.tight_layout()
        plt.savefig(output_dir / "accuracy_curve.png")
        plt.close()


def plot_feature_distribution(feature_values: list[float], output_dir: Path, name: str = "energy") -> None:
    import matplotlib.pyplot as plt

    output_dir.mkdir(parents=True, exist_ok=True)
    plt.hist(feature_values, bins=40)
    plt.title(f"{name.title()} Feature Distribution")
    plt.xlabel(name)
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(output_dir / f"{name}_feature_distribution.png")
    plt.close()


def evaluate_text(output_dir: Path) -> dict:
    from sklearn.metrics import accuracy_score, classification_report, precision_recall_fscore_support

    splits = load_slue_voxceleb()
    texts, labels = iter_text_examples(splits.test)
    model = TextSentimentPipeline(settings.text_sentiment_model_path).load()
    predictions = []
    probabilities = []
    start = time.perf_counter()
    for text in texts:
        result = model.predict(text)
        predictions.append(result.label)
        probabilities.append(result.probabilities)
    latency_ms = ((time.perf_counter() - start) * 1000) / max(len(texts), 1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average="weighted", zero_division=0)
    label_order = sorted(set(labels) | set(predictions))
    _save_plots(labels, predictions, label_order, probabilities, output_dir)
    metrics = {
        "model": "tfidf_multinomial_naive_bayes",
        "accuracy": accuracy_score(labels, predictions),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "avg_inference_time_ms": latency_ms,
        "classification_report": classification_report(labels, predictions, zero_division=0),
    }
    (output_dir / "text_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate trained sentiment/emotion models and generate plots")
    parser.add_argument("--task", choices=["text"], default="text")
    parser.add_argument("--output-dir", default="./reports/evaluation")
    parser.add_argument("--cnn-checkpoint", default=None, help="Optional CNN-MFCC checkpoint for loss/accuracy plots")
    args = parser.parse_args()
    output_dir = Path(args.output_dir)
    metrics = evaluate_text(output_dir)
    if args.cnn_checkpoint:
        plot_training_history(args.cnn_checkpoint, output_dir)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.config import settings
from training.sentiment_classifier import (
    MultinomialNaiveBayes,
    load_sentiment_dataset,
)


def train_sentiment_model(
    train_data_path: str,
    output_path: str,
    alpha: float = 1.0,
) -> MultinomialNaiveBayes:
    texts, labels = load_sentiment_dataset(train_data_path)
    classifier = MultinomialNaiveBayes(alpha=alpha).fit(texts, labels)
    classifier.save(output_path)
    return classifier


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train the Multinomial Naive Bayes sentiment classifier"
    )
    parser.add_argument(
        "--train-data",
        default=settings.sentiment_dataset_path,
        help="JSON dataset containing text and label fields",
    )
    parser.add_argument(
        "--output",
        default=settings.sentiment_model_path,
        help="Path for the trained JSON model",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=settings.sentiment_alpha,
        help="Laplace smoothing value",
    )
    args = parser.parse_args()

    model = train_sentiment_model(args.train_data, args.output, args.alpha)
    print(
        f"Trained sentiment model with labels {', '.join(model.labels)} "
        f"and saved it to {args.output}"
    )

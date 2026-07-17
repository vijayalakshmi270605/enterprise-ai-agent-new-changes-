import logging
from pathlib import Path
from typing import Dict

from app.config import settings
from app.nlp.text_sentiment import TextSentimentPipeline
from training.sentiment_classifier import (
    MultinomialNaiveBayes,
    load_sentiment_dataset,
)


logger = logging.getLogger(__name__)


class SentimentService:
    classifier: MultinomialNaiveBayes = None
    pipeline: TextSentimentPipeline = None
    neutral_indicators = {
        "meeting",
        "scheduled",
        "schedule",
        "appointment",
        "report",
        "submitted",
        "received",
        "available",
        "arrived",
        "delivered",
        "today",
        "tomorrow",
        "yesterday",
        "time",
        "date",
        "office",
        "school",
        "college",
        "project",
        "status",
        "information",
    }
    positive_words = {
        "amazing",
        "awesome",
        "best",
        "excellent",
        "good",
        "great",
        "happy",
        "helpful",
        "impressed",
        "love",
        "perfect",
        "satisfied",
        "smooth",
        "wonderful",
    }
    negative_words = {
        "angry",
        "bad",
        "crash",
        "crashing",
        "disappointed",
        "disappointing",
        "frustrated",
        "hate",
        "poor",
        "sad",
        "terrible",
        "unhappy",
        "unsatisfied",
        "worst",
    }

    @classmethod
    def initialize(cls) -> None:
        pipeline_path = Path(settings.text_sentiment_model_path)
        if pipeline_path.exists():
            cls.pipeline = TextSentimentPipeline(str(pipeline_path)).load()
            logger.info("Loaded TF-IDF sentiment model from %s", pipeline_path)
            return

        model_path = Path(settings.sentiment_model_path)
        if model_path.exists():
            cls.classifier = MultinomialNaiveBayes.load(str(model_path))
            logger.info("Loaded sentiment model from %s", model_path)
            return

        dataset_path = Path(settings.sentiment_dataset_path)
        texts, labels = load_sentiment_dataset(str(dataset_path))
        cls.classifier = MultinomialNaiveBayes(alpha=settings.sentiment_alpha)
        cls.classifier.fit(texts, labels)
        cls.classifier.save(str(model_path))
        logger.info(
            "Trained sentiment model with %d records and saved it to %s",
            len(texts),
            model_path,
        )

    @classmethod
    def analyze(cls, text: str) -> Dict[str, object]:
        if cls.pipeline is not None:
            result = cls.pipeline.predict(text)
            label, confidence, probabilities = cls._neutral_adjustment(
                text,
                result.label,
                result.confidence,
                result.probabilities,
            )
            return {
                "text": text,
                "sentiment": label,
                "confidence": round(confidence, 6),
                "probabilities": probabilities,
                "model": result.model,
            }
        if cls.classifier is None:
            cls.initialize()
        if cls.pipeline is not None:
            result = cls.pipeline.predict(text)
            label, confidence, probabilities = cls._neutral_adjustment(
                text,
                result.label,
                result.confidence,
                result.probabilities,
            )
            return {
                "text": text,
                "sentiment": label,
                "confidence": round(confidence, 6),
                "probabilities": probabilities,
                "model": result.model,
            }
        label, confidence, probabilities = cls.classifier.predict(text)
        label, confidence, probabilities = cls._neutral_adjustment(
            text,
            label,
            confidence,
            probabilities,
        )
        return {
            "text": text,
            "sentiment": label,
            "confidence": round(confidence, 6),
            "probabilities": {
                name: round(probability, 6)
                for name, probability in probabilities.items()
            },
            "model": "multinomial_naive_bayes",
        }

    @classmethod
    def _neutral_adjustment(
        cls,
        text: str,
        label: str,
        confidence: float,
        probabilities: Dict[str, float],
    ) -> tuple[str, float, Dict[str, float]]:
        normalized = text.strip().lower()
        tokens = set(normalized.replace(".", " ").replace(",", " ").split())
        if "neutral" in probabilities:
            sentiment_override = cls._lexicon_override(normalized, tokens, probabilities)
            if sentiment_override is not None:
                return sentiment_override

        has_neutral_indicator = bool(tokens.intersection(cls.neutral_indicators))
        has_emotional_word = bool(tokens.intersection(cls.positive_words | cls.negative_words))
        ordered = sorted(probabilities.items(), key=lambda item: item[1], reverse=True)
        margin = ordered[0][1] - ordered[1][1] if len(ordered) > 1 else ordered[0][1]

        if "neutral" in probabilities:
            if has_neutral_indicator and not has_emotional_word:
                probabilities = dict(probabilities)
                probabilities["neutral"] = max(probabilities["neutral"], 0.72)
                total = sum(probabilities.values()) or 1.0
                probabilities = {key: round(value / total, 6) for key, value in probabilities.items()}
                return "neutral", probabilities["neutral"], probabilities
            if label != "neutral" and confidence < 0.58 and margin < 0.18:
                probabilities = dict(probabilities)
                probabilities["neutral"] = max(probabilities["neutral"], confidence + 0.05)
                total = sum(probabilities.values()) or 1.0
                probabilities = {key: round(value / total, 6) for key, value in probabilities.items()}
                return "neutral", probabilities["neutral"], probabilities

        return label, confidence, probabilities

    @classmethod
    def _lexicon_override(
        cls,
        normalized: str,
        tokens: set[str],
        probabilities: Dict[str, float],
    ) -> tuple[str, float, Dict[str, float]] | None:
        negative_phrases = (
            "not happy",
            "not satisfied",
            "not good",
            "not helpful",
            "not impressed",
            "not perfect",
        )
        positive_phrases = (
            "very happy",
            "so happy",
            "happy with",
            "satisfied with",
            "love this",
            "good service",
            "great service",
            "excellent service",
        )
        if any(phrase in normalized for phrase in negative_phrases) or tokens.intersection(cls.negative_words):
            return cls._force_label(probabilities, "negative", 0.82)
        if any(phrase in normalized for phrase in positive_phrases) or tokens.intersection(cls.positive_words):
            return cls._force_label(probabilities, "positive", 0.82)
        return None

    @staticmethod
    def _force_label(
        probabilities: Dict[str, float],
        label: str,
        minimum: float,
    ) -> tuple[str, float, Dict[str, float]]:
        adjusted = dict(probabilities)
        other_labels = [key for key in adjusted if key != label]
        remaining = max(0.0, 1.0 - minimum)
        other_total = sum(adjusted[key] for key in other_labels) or 1.0
        adjusted[label] = minimum
        for key in other_labels:
            adjusted[key] = remaining * (adjusted[key] / other_total)
        adjusted = {key: round(value, 6) for key, value in adjusted.items()}
        return label, adjusted[label], adjusted

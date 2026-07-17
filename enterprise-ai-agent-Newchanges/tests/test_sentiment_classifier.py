import json

import pytest

from app.api.v1.schemas import LeadScoringRequest, SentimentResponse
from app.services.lead_scoring_service import LeadScoringService
from app.services.sentiment_service import SentimentService
from training.sentiment_classifier import MultinomialNaiveBayes, tokenize


TRAIN_TEXTS = [
    "I love this excellent service",
    "This product is wonderful",
    "I hate this terrible service",
    "This product is awful",
    "The package arrived today",
    "The meeting starts at noon",
]
TRAIN_LABELS = [
    "positive",
    "positive",
    "negative",
    "negative",
    "neutral",
    "neutral",
]


def test_tokenizer_marks_words_after_negation():
    assert tokenize("not very good today") == [
        "not",
        "not_very",
        "not_good",
        "not_today",
    ]


def test_classifier_predicts_each_sentiment():
    model = MultinomialNaiveBayes().fit(TRAIN_TEXTS, TRAIN_LABELS)

    assert model.predict("excellent wonderful service")[0] == "positive"
    assert model.predict("awful terrible product")[0] == "negative"
    assert model.predict("the meeting arrived at noon")[0] == "neutral"


def test_probabilities_sum_to_one():
    model = MultinomialNaiveBayes().fit(TRAIN_TEXTS, TRAIN_LABELS)
    probabilities = model.predict_proba("I love the product")

    assert sum(probabilities.values()) == pytest.approx(1.0)


def test_unseen_vocabulary_falls_back_to_neutral():
    model = MultinomialNaiveBayes().fit(TRAIN_TEXTS, TRAIN_LABELS)

    assert model.predict("xyzzy plugh")[0] == "neutral"


def test_model_round_trip(tmp_path):
    model_path = tmp_path / "sentiment.json"
    original = MultinomialNaiveBayes().fit(TRAIN_TEXTS, TRAIN_LABELS)
    original.save(str(model_path))

    loaded = MultinomialNaiveBayes.load(str(model_path))

    assert loaded.predict("excellent service")[0] == "positive"
    assert json.loads(model_path.read_text())["model_type"] == (
        "multinomial_naive_bayes"
    )


def test_sentiment_service_response_shape():
    SentimentService.classifier = MultinomialNaiveBayes().fit(
        TRAIN_TEXTS,
        TRAIN_LABELS,
    )

    result = SentimentService.analyze("terrible and awful")

    assert result["sentiment"] == "negative"
    assert result["model"] == "multinomial_naive_bayes"
    assert 0.0 <= result["confidence"] <= 1.0
    assert set(result["probabilities"]) == {"negative", "neutral", "positive"}


def test_sentiment_response_schema():
    SentimentService.classifier = MultinomialNaiveBayes().fit(
        TRAIN_TEXTS,
        TRAIN_LABELS,
    )

    response = SentimentResponse(
        **SentimentService.analyze("wonderful excellent product")
    )

    assert isinstance(response, SentimentResponse)
    assert response.sentiment == "positive"


def test_lead_scoring_returns_category_and_scores():
    result = LeadScoringService.score(
        "I need a demo for enterprise integration and I am ready to buy this quarter",
        voice_emotion="happy",
        emotion_confidence=0.84,
    )

    assert result["conversion_category"] in {"hot", "warm", "cold"}
    assert 0.0 <= result["conversion_probability"] <= 1.0
    assert set(result["scores"]) == {
        "icp_score",
        "intent_score",
        "engagement_score",
        "qualification_score",
        "buying_signal_score",
        "relationship_score",
    }


def test_lead_intelligence_service_exposes_full_pipeline():
    result = LeadScoringService.score(
        "We need a demo for our enterprise team and want pricing this quarter",
        voice_emotion="happy",
        emotion_confidence=0.88,
    )

    assert result["lead_score"] >= 0.0
    assert result["lead_category"]
    assert result["recommendation"]
    assert len(result["engineered_features"]) > 100
    assert result["feature_importance"]
    assert result["icp_score"] >= 0.0
    assert result["intent_score"] >= 0.0
    assert result["engagement_score"] >= 0.0
    assert result["qualification_score"] >= 0.0
    assert result["buying_signal_score"] >= 0.0
    assert result["relationship_score"] >= 0.0
    assert result["conversion_probability_percent"] >= 0.0
    assert result["xgboost_probability_percent"] >= 0.0


def test_lead_scoring_includes_report_payload():
    SentimentService.classifier = MultinomialNaiveBayes().fit(
        TRAIN_TEXTS,
        TRAIN_LABELS,
    )

    result = LeadScoringService.score(
        "We need budget approval from our VP and want to start this quarter",
        voice_emotion="happy",
        emotion_confidence=0.75,
    )

    report = result["report"]

    assert report["summary"]
    assert report["bant"]["budget"]["summary"]
    assert report["bant"]["authority"]["summary"]
    assert report["bant"]["need"]["summary"]
    assert report["bant"]["timeline"]["summary"]
    assert report["metrics"]["intent_score"] >= 0.0
    assert result["sentiment"]["text"]["sentiment"] in {"positive", "neutral", "negative"}


def test_meddic_scoring_returns_discovery_scores_and_followup():
    result = LeadScoringService.score(
        "Our CFO owns the budget. We need to reduce manual processing by 30% and save $100k. "
        "Security and integration are required; procurement and legal approval follow the pilot. "
        "The current manual workflow causes delays and lost revenue.",
    )

    assert 0.0 <= result["meddic_score"] <= 100.0
    assert set(result["meddic_scores"]) == {
        "metrics", "economic_buyer", "decision_criteria", "decision_process", "identify_pain", "champion"
    }
    assert result["meddic_category"]
    assert result["meddic_recommendation"]
    assert result["report"]["meddic"]["metrics"]["summary"]


def test_lead_scoring_request_rejects_invalid_confidence():
    with pytest.raises(ValueError):
        LeadScoringRequest(text="Need a demo", emotion_confidence=1.1)

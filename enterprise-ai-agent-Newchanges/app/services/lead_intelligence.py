import asyncio
import json
import logging
import math
import re
from datetime import datetime, timezone
from collections import Counter
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

try:
    from xgboost import XGBClassifier
except ImportError:  # pragma: no cover - optional dependency fallback
    XGBClassifier = None

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

try:  # pragma: no cover - optional dependency fallback
    import shap  # type: ignore
except ImportError:  # pragma: no cover
    shap = None

from app.services.memory_service import MemoryService
from app.services.sentiment_service import SentimentService

logger = logging.getLogger(__name__)


class LeadIntelligenceService:
    model = None
    feature_names: List[str] = []
    customer_profiles: Dict[str, Dict[str, Any]] = {}
    lead_history: Dict[str, List[Dict[str, Any]]] = {}

    @classmethod
    def _ensure_model(cls):
        if cls.model is not None:
            return cls.model

        X, y = cls._build_training_data()
        if XGBClassifier is not None:
            cls.model = XGBClassifier(
                n_estimators=120,
                max_depth=3,
                learning_rate=0.18,
                random_state=42,
                objective="binary:logistic",
                eval_metric="logloss",
            )
        else:
            cls.model = make_pipeline(
                SimpleImputer(strategy="median"),
                StandardScaler(),
                GradientBoostingClassifier(random_state=42),
            )

        cls.model.fit(X, y)
        cls.feature_names = cls._build_feature_names()
        logger.info("Lead intelligence model initialized using %s", type(cls.model).__name__)
        return cls.model

    @classmethod
    def calculate_lead_scores(
        cls,
        text: str,
        voice_emotion: Optional[str] = None,
        emotion_confidence: Optional[float] = None,
        history: Optional[List[Dict[str, Any]]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        crm_context: Optional[Dict[str, Any]] = None,
        business_docs: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        normalized_text = cls._normalize_text(text or "")
        history_text = " ".join(
            item.get("content", "")
            for item in (history or [])
            if isinstance(item, dict) and item.get("content")
        )
        combined_text = f"{normalized_text} {history_text}".strip()
        tokens = re.findall(r"[a-zA-Z0-9_]+", combined_text.lower())
        token_counts = Counter(tokens)
        sentiment = cls._analyze_text_sentiment(
            combined_text,
            voice_emotion=voice_emotion,
            emotion_confidence=emotion_confidence,
        )

        icp_score = cls._score_icp(combined_text, metadata or {}, crm_context or {})
        intent_score = cls._score_intent(combined_text, token_counts)
        engagement_score = cls._score_engagement(combined_text, token_counts, history or [])
        qualification_score = cls._score_qualification(combined_text, metadata or {}, crm_context or {})
        buying_signal_score = cls._score_buying_signal(combined_text)
        relationship_score = cls._score_relationship(
            combined_text,
            voice_emotion,
            emotion_confidence,
            history or [],
            metadata or {},
        )
        bant_scores = cls._score_bant(combined_text, history or [], metadata or {}, crm_context or {})
        meddic_scores = cls._score_meddic(combined_text, history or [], metadata or {}, crm_context or {})

        weighted_score = (
            0.2 * icp_score
            + 0.2 * intent_score
            + 0.2 * qualification_score
            + 0.15 * buying_signal_score
            + 0.15 * engagement_score
            + 0.1 * relationship_score
        )
        lead_score = cls._calibrated_lead_score(combined_text, weighted_score * 100.0)
        return {
            "icp_score": round(icp_score * 100.0, 2),
            "intent_score": round(intent_score * 100.0, 2),
            "engagement_score": round(engagement_score * 100.0, 2),
            "qualification_score": round(qualification_score * 100.0, 2),
            "buying_signal_score": round(buying_signal_score * 100.0, 2),
            "relationship_score": round(relationship_score * 100.0, 2),
            "lead_score": round(lead_score, 2),
            "bant_score": round(bant_scores["bant_score"] * 100.0, 2),
            "bant_scores": bant_scores["bant_scores"],
            "bant_category": bant_scores["bant_category"],
            "bant_recommendation": bant_scores["bant_recommendation"],
            "meddic_score": round(meddic_scores["meddic_score"] * 100.0, 2),
            "meddic_scores": meddic_scores["meddic_scores"],
            "meddic_category": meddic_scores["meddic_category"],
            "meddic_recommendation": meddic_scores["meddic_recommendation"],
            "sentiment": sentiment,
        }

    @classmethod
    def predict_conversion(
        cls,
        text: str,
        voice_emotion: Optional[str] = None,
        emotion_confidence: Optional[float] = None,
        history: Optional[List[Dict[str, Any]]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        crm_context: Optional[Dict[str, Any]] = None,
        business_docs: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        cls._ensure_model()
        features = cls._build_engineered_features(
            text,
            voice_emotion=voice_emotion,
            emotion_confidence=emotion_confidence,
            history=history,
            metadata=metadata,
            crm_context=crm_context,
            business_docs=business_docs,
        )
        vector = np.array([features[name] for name in cls.feature_names], dtype=float)
        vector = cls._handle_missing_and_normalize(vector)
        probability = float(cls.model.predict_proba([vector])[0][1])
        return {
            "conversion_probability": round(probability, 4),
            "conversion_probability_percent": round(probability * 100.0, 2),
            "feature_importance": cls._feature_importance(),
            "engineered_features": features,
            "shap_available": shap is not None,
        }

    @classmethod
    def categorize_lead(cls, lead_score: float) -> str:
        if lead_score >= 75:
            return "Hot Lead"
        if lead_score >= 40:
            return "Warm Lead"
        return "Cold Lead"

    @classmethod
    def generate_next_action(
        cls,
        lead_score: float,
        scores: Optional[Dict[str, Any]] = None,
        text: Optional[str] = None,
    ) -> str:
        if lead_score >= 75:
            return "Send a tailored proposal and confirm implementation timeline."
        if text and any(term in (text or "").lower() for term in ["profit", "profits", "revenue", "growth", "sales", "margin"]):
            return "Clarify the main profit lever: pricing, conversion, retention, or cost reduction."
        if lead_score >= 40:
            return "Provide a value summary and ask for a follow-up meeting."
        if text and any(term in (text or "").lower() for term in ["pricing", "demo", "buy", "need"]):
            return "Re-qualify the need and offer a short discovery call."
        return "Continue nurturing the lead with helpful content and periodic follow-up."

    @classmethod
    def update_customer_profile(
        cls,
        session_id: str,
        profile: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        merged = dict(cls.customer_profiles.get(session_id, {}))
        if profile:
            merged.update(profile)
        if metadata:
            merged.update(metadata)
        merged.setdefault("returning_customer", bool(merged.get("returning_customer", False)))
        merged.setdefault("customer_lifetime", 0)
        cls.customer_profiles[session_id] = merged
        return merged

    @classmethod
    def store_lead_history(
        cls,
        session_id: str,
        payload: Dict[str, Any],
        text: Optional[str] = None,
    ) -> Dict[str, Any]:
        history = list(cls.lead_history.get(session_id, []))
        record = {
            "timestamp": payload.get("timestamp") or "now",
            "text": text or payload.get("text", ""),
            **payload,
        }
        history.append(record)
        cls.lead_history[session_id] = history[-20:]
        try:
            loop = asyncio.get_running_loop()
            loop.create_task(MemoryService.append_user_message(session_id, text or payload.get("text", "")))
        except RuntimeError:
            asyncio.run(MemoryService.append_user_message(session_id, text or payload.get("text", "")))
        except Exception as exc:  # pragma: no cover - best effort persistence
            logger.debug("Lead history persistence skipped: %s", exc)
        return record

    @classmethod
    def score(
        cls,
        text: str,
        voice_emotion: Optional[str] = None,
        emotion_confidence: Optional[float] = None,
        history: Optional[List[Dict[str, Any]]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        crm_context: Optional[Dict[str, Any]] = None,
        business_docs: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        scores = cls.calculate_lead_scores(
            text,
            voice_emotion=voice_emotion,
            emotion_confidence=emotion_confidence,
            history=history,
            metadata=metadata,
            crm_context=crm_context,
            business_docs=business_docs,
        )
        lead_score = scores["lead_score"]
        prediction = cls._fast_conversion_prediction(lead_score)
        lead_category = cls.categorize_lead(lead_score)
        recommendation = cls.generate_next_action(lead_score, scores=scores, text=text)
        conversion_category = "hot" if lead_score >= 75 else "warm" if lead_score >= 40 else "cold"
        detailed_features = cls._build_engineered_features(
            text,
            voice_emotion=voice_emotion,
            emotion_confidence=emotion_confidence,
            history=history,
            metadata=metadata,
            crm_context=crm_context,
            business_docs=business_docs,
        )
        report = cls._build_lead_report(
            text=text,
            scores=scores,
            lead_score=lead_score,
            lead_category=lead_category,
            recommendation=recommendation,
            conversion_probability=prediction["conversion_probability_percent"],
            history=history or [],
            metadata=metadata or {},
            crm_context=crm_context or {},
            business_docs=business_docs or [],
        )

        return {
            "text": text,
            "scores": {
                "icp_score": scores["icp_score"],
                "intent_score": scores["intent_score"],
                "engagement_score": scores["engagement_score"],
                "qualification_score": scores["qualification_score"],
                "buying_signal_score": scores["buying_signal_score"],
                "relationship_score": scores["relationship_score"],
            },
            "icp_score": scores["icp_score"],
            "intent_score": scores["intent_score"],
            "engagement_score": scores["engagement_score"],
            "qualification_score": scores["qualification_score"],
            "buying_signal_score": scores["buying_signal_score"],
            "relationship_score": scores["relationship_score"],
            "bant_score": scores["bant_score"],
            "bant_scores": scores["bant_scores"],
            "bant_category": scores["bant_category"],
            "bant_recommendation": scores["bant_recommendation"],
            "meddic_score": scores["meddic_score"],
            "meddic_scores": scores["meddic_scores"],
            "meddic_category": scores["meddic_category"],
            "meddic_recommendation": scores["meddic_recommendation"],
            "sentiment": scores["sentiment"],
            "conversion_probability": prediction["conversion_probability"],
            "conversion_probability_percent": prediction["conversion_probability_percent"],
            "xgboost_probability": prediction["conversion_probability"],
            "xgboost_probability_percent": prediction["conversion_probability_percent"],
            "conversion_category": conversion_category,
            "lead_score": lead_score,
            "lead_category": lead_category,
            "recommendation": recommendation,
            "engineered_features": detailed_features,
            "feature_importance": prediction["feature_importance"],
            "report": report,
            "model": "realtime_rules",
            "probabilities": {
                "cold": round(max(0.0, 1.0 - prediction["conversion_probability"]), 4),
                "warm": round(min(0.99, max(0.0, abs(prediction["conversion_probability"] - 0.5))), 4),
                "hot": round(prediction["conversion_probability"], 4),
            },
            "shap_available": prediction["shap_available"],
        }

    @classmethod
    def _build_feature_names(cls) -> List[str]:
        signal_terms = [
            "demo",
            "pricing",
            "implementation",
            "timeline",
            "discount",
            "roi",
            "urgent",
            "today",
            "quarter",
            "month",
            "finalize",
            "procurement",
            "contract",
            "schedule",
            "compare",
            "feature",
            "security",
            "compliance",
            "automation",
            "integration",
            "team",
            "enterprise",
            "company",
            "organization",
            "budget",
            "authority",
            "stakeholder",
            "decision",
            "need",
            "pain",
            "cost",
            "trial",
            "support",
            "scale",
            "revenue",
            "growth",
            "buy",
            "purchase",
            "ready",
            "start",
            "reliable",
            "immediately",
            "critical",
            "workflow",
        ]
        base = [f"{term}_density" for term in signal_terms]
        base += [f"{term}_present" for term in signal_terms]
        base += [
            "sentiment_score",
            "emotion_strength",
            "question_count",
            "conversation_length",
            "response_frequency",
            "avg_reply_time",
            "voice_energy",
            "speech_rate",
            "interruption_count",
            "continuity_score",
            "industry_match",
            "company_size_score",
            "annual_revenue_score",
            "tech_stack_match",
            "decision_maker_score",
            "country_match",
            "business_requirement_match",
            "current_software_match",
            "need_match",
            "ideal_customer_similarity",
            "returning_customer",
            "customer_lifetime",
            "demo_requested",
            "pricing_requested",
            "authority_mentioned",
            "budget_mentioned",
            "timeline_mentioned",
            "need_mentioned",
            "technical_question",
            "business_pain_point",
            "decision_confidence",
            "positive_sentiment_trend",
        ]
        return base

    @classmethod
    def _build_engineered_features(
        cls,
        text: str,
        voice_emotion: Optional[str] = None,
        emotion_confidence: Optional[float] = None,
        history: Optional[List[Dict[str, Any]]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        crm_context: Optional[Dict[str, Any]] = None,
        business_docs: Optional[List[str]] = None,
    ) -> Dict[str, float]:
        normalized_text = cls._normalize_text(text or "")
        history_text = " ".join(
            item.get("content", "")
            for item in (history or [])
            if isinstance(item, dict) and item.get("content")
        )
        combined_text = f"{normalized_text} {history_text}".strip()
        tokens = re.findall(r"[a-zA-Z0-9_]+", combined_text.lower())
        token_counts = Counter(tokens)
        total_tokens = max(1, len(tokens))

        features: Dict[str, float] = {}
        signal_terms = [
            "demo",
            "pricing",
            "implementation",
            "timeline",
            "discount",
            "roi",
            "urgent",
            "today",
            "quarter",
            "month",
            "finalize",
            "procurement",
            "contract",
            "schedule",
            "compare",
            "feature",
            "security",
            "compliance",
            "automation",
            "integration",
            "team",
            "enterprise",
            "company",
            "organization",
            "budget",
            "authority",
            "stakeholder",
            "decision",
            "need",
            "pain",
            "cost",
            "trial",
            "support",
            "scale",
            "revenue",
            "growth",
            "buy",
            "purchase",
            "ready",
            "start",
            "reliable",
            "immediately",
            "critical",
            "workflow",
        ]
        for term in signal_terms:
            features[f"{term}_density"] = token_counts.get(term, 0) / total_tokens
            features[f"{term}_present"] = 1.0 if term in combined_text else 0.0

        sentiment_payload = cls._analyze_text_sentiment(
            combined_text,
            voice_emotion=voice_emotion,
            emotion_confidence=emotion_confidence,
        )
        sentiment_score = float(sentiment_payload["blended_score"])
        features["sentiment_score"] = sentiment_score
        features["emotion_strength"] = cls._emotion_strength(voice_emotion, emotion_confidence)
        features["question_count"] = 1.0 if "?" in normalized_text else 0.0
        features["conversation_length"] = min(1.0, len(tokens) / 80.0)
        features["response_frequency"] = min(1.0, len(history or []) / 10.0)
        features["avg_reply_time"] = min(1.0, 0.25 + (len(history or []) * 0.04))
        features["voice_energy"] = 0.7 if voice_emotion in {"happy", "excited", "calm"} else 0.4
        features["speech_rate"] = min(1.0, len(tokens) / 120.0)
        features["interruption_count"] = 1.0 if any(term in combined_text for term in ["wait", "hold on", "sorry", "actually"]) else 0.0
        features["continuity_score"] = 1.0 if len(history or []) >= 2 else 0.0

        metadata = metadata or {}
        crm_context = crm_context or {}
        business_docs = business_docs or []
        features["industry_match"] = float(metadata.get("industry_match", 0.5 if crm_context else 0.4))
        features["company_size_score"] = float(metadata.get("company_size_score", 0.5))
        features["annual_revenue_score"] = float(metadata.get("annual_revenue_score", 0.5))
        features["tech_stack_match"] = float(metadata.get("tech_stack_match", 0.5))
        features["decision_maker_score"] = float(metadata.get("decision_maker_score", 0.5))
        features["country_match"] = float(metadata.get("country_match", 0.5))
        features["business_requirement_match"] = float(metadata.get("business_requirement_match", 0.5))
        features["current_software_match"] = float(metadata.get("current_software_match", 0.5))
        features["need_match"] = float(metadata.get("need_match", 0.5))
        features["ideal_customer_similarity"] = float(metadata.get("ideal_customer_similarity", 0.5))
        features["returning_customer"] = 1.0 if bool(metadata.get("returning_customer", False)) or bool(crm_context.get("returning_customer", False)) else 0.0
        features["customer_lifetime"] = min(1.0, float(metadata.get("customer_lifetime", crm_context.get("customer_lifetime", 0))) / 10.0)
        features["demo_requested"] = 1.0 if any(term in combined_text for term in ["demo", "demo request", "book a demo"]) else 0.0
        features["pricing_requested"] = 1.0 if any(term in combined_text for term in ["pricing", "quote", "cost", "budget"]) else 0.0
        features["authority_mentioned"] = 1.0 if any(term in combined_text for term in ["authority", "decision maker", "stakeholder", "cto", "ceo", "vp"]) else 0.0
        features["budget_mentioned"] = 1.0 if any(term in combined_text for term in ["budget", "cost", "price", "pricing"]) else 0.0
        features["timeline_mentioned"] = 1.0 if any(term in combined_text for term in ["timeline", "this quarter", "this month", "next month", "today", "urgent"]) else 0.0
        features["need_mentioned"] = 1.0 if any(term in combined_text for term in ["need", "requirement", "use case", "pain point", "problem"]) else 0.0
        features["technical_question"] = 1.0 if any(term in combined_text for term in ["integration", "api", "security", "compliance", "automation"]) else 0.0
        features["business_pain_point"] = 1.0 if any(term in combined_text for term in ["pain", "issue", "problem", "bottleneck", "slow"]) else 0.0
        features["decision_confidence"] = 0.8 if any(term in combined_text for term in ["decision", "authority", "stakeholder", "approved"]) else 0.4
        features["positive_sentiment_trend"] = 1.0 if any(term in combined_text for term in ["great", "excellent", "happy", "thanks", "appreciate", "love"]) else 0.0
        features["business_docs_count"] = min(1.0, len(business_docs) / 5.0)
        return features

    @classmethod
    def _handle_missing_and_normalize(cls, vector: np.ndarray) -> np.ndarray:
        filled = np.nan_to_num(vector.astype(float), nan=0.0, posinf=1.0, neginf=0.0)
        return np.clip(filled, 0.0, 1.0)

    @classmethod
    def _feature_importance(cls) -> Dict[str, float]:
        model = cls._ensure_model()
        if hasattr(model, "feature_importances_"):
            importances = model.feature_importances_
        else:
            importances = np.zeros(len(cls.feature_names), dtype=float)
        ranked = sorted(
            zip(cls.feature_names, importances.tolist()),
            key=lambda item: item[1],
            reverse=True,
        )
        return {name: round(float(score), 4) for name, score in ranked[:10]}

    @classmethod
    def _build_training_data(cls) -> Tuple[np.ndarray, List[int]]:
        rows = []
        labels = []
        for row_payload, label in [
            ("cold lead with no urgency and no demo request", 0),
            ("just browsing and asking questions about the product", 0),
            ("we have a small team and need general information", 0),
            ("looking for basic support and no immediate need", 0),
            ("we need a demo for our enterprise team and want pricing this quarter", 1),
            ("we are ready to buy and need implementation for our company", 1),
            ("can we schedule a demo and discuss budget and timeline", 1),
            ("our team needs automation and we need this immediately", 1),
        ]:
            features = cls._build_engineered_features(row_payload)
            rows.append([features[name] for name in cls._build_feature_names()])
            labels.append(label)
        return np.array(rows, dtype=float), labels

    @staticmethod
    def _normalize_text(text: str) -> str:
        normalized = re.sub(r"\s+", " ", text.strip().lower())
        replacements = {
            "rising": "pricing",
            "priceing": "pricing",
            "prizing": "pricing",
            "life demo": "live demo",
            "lime demo": "live demo",
            "leave demo": "live demo",
            "implement timeline": "implementation timeline",
        }
        for source, target in replacements.items():
            normalized = re.sub(rf"\b{re.escape(source)}\b", target, normalized)
        return normalized

    @classmethod
    def _calibrated_lead_score(cls, text: str, weighted_score: float) -> float:
        bucket = cls._signal_bucket(text)
        if bucket == "hot":
            return round(max(weighted_score, 85.0), 2)
        if bucket == "warm":
            return round(min(max(weighted_score, 55.0), 74.0), 2)
        if bucket == "cold":
            return round(min(weighted_score, 29.0), 2)
        return round(weighted_score, 2)

    @classmethod
    def _signal_bucket(cls, text: str) -> str:
        cold_blockers = [
            "just checking",
            "just browsing",
            "general information",
            "no budget",
            "no timeline",
            "not ready",
            "later",
            "maybe later",
            "no immediate need",
        ]
        blocker_hits = sum(1 for term in cold_blockers if term in text)
        if blocker_hits >= 2:
            return "cold"

        signals = {
            "ready_to_buy": any(term in text for term in ["ready to buy", "want to buy", "ready to purchase", "approved"]),
            "demo": "demo" in text,
            "pricing": any(term in text for term in ["pricing", "price", "quote", "cost", "budget"]),
            "timeline": any(term in text for term in ["this month", "this quarter", "today", "urgent", "timeline", "implementation"]),
            "authority": any(term in text for term in ["decision maker", "stakeholder", "cto", "ceo", "vp", "approved"]),
            "enterprise_fit": any(term in text for term in ["enterprise", "team", "automation", "workflow", "security", "integration"]),
        }
        hot_count = sum(1 for value in signals.values() if value)
        if signals["ready_to_buy"] and signals["demo"] and signals["timeline"]:
            return "hot"
        if signals["ready_to_buy"] and signals["pricing"] and hot_count >= 3:
            return "hot"
        if hot_count >= 5:
            return "hot"

        warm_terms = ["exploring", "consider", "may", "maybe", "case study", "features", "options", "next month"]
        if any(term in text for term in warm_terms) and hot_count >= 2:
            return "warm"
        if hot_count >= 2:
            return "warm"
        return "neutral"

    @staticmethod
    def _fast_conversion_prediction(lead_score: float) -> Dict[str, Any]:
        probability = min(0.98, max(0.03, lead_score / 100.0))
        return {
            "conversion_probability": round(probability, 4),
            "conversion_probability_percent": round(probability * 100.0, 2),
            "engineered_features": {"lead_score": round(lead_score / 100.0, 4)},
            "feature_importance": {
                "buying_signal_score": 0.35,
                "intent_score": 0.25,
                "qualification_score": 0.2,
                "icp_score": 0.1,
                "engagement_score": 0.06,
                "relationship_score": 0.04,
            },
            "shap_available": False,
        }

    @staticmethod
    def _score_icp(text: str, metadata: Dict[str, Any], crm_context: Dict[str, Any]) -> float:
        score = 0.15
        if any(term in text for term in ["enterprise", "team", "company", "organization", "large"]):
            score += 0.2
        if any(term in text for term in ["integration", "workflow", "automation", "compliance", "security"]):
            score += 0.2
        if metadata.get("industry_match") or crm_context.get("industry_match"):
            score += 0.2
        if metadata.get("company_size_score") or crm_context.get("company_size_score"):
            score += 0.15
        if metadata.get("tech_stack_match") or crm_context.get("tech_stack_match"):
            score += 0.1
        return min(1.0, round(score, 4))

    @staticmethod
    def _score_intent(text: str, token_counts: Counter) -> float:
        strong_terms = [
            "demo",
            "pricing",
            "buy",
            "purchase",
            "ready",
            "need",
            "implement",
            "trial",
            "quote",
            "roi",
            "discount",
            "urgent",
        ]
        matches = sum(token_counts.get(term, 0) for term in strong_terms)
        score = 0.2 + min(0.7, matches * 0.08)
        if any(term in text for term in ["need", "buy", "ready", "pricing", "demo"]):
            score += 0.08
        return min(1.0, round(score, 4))

    @staticmethod
    def _score_engagement(text: str, token_counts: Counter, history: List[Dict[str, Any]]) -> float:
        if not text:
            return 0.0
        base = min(1.0, len(text.split()) / 45.0)
        if "?" in text:
            base += 0.1
        if len(history) >= 2:
            base += 0.08
        if any(term in text for term in ["please", "tell me", "can you", "what", "why"]):
            base += 0.08
        return min(1.0, round(base, 4))

    @staticmethod
    def _score_qualification(text: str, metadata: Dict[str, Any], crm_context: Dict[str, Any]) -> float:
        score = 0.1
        if any(term in text for term in ["budget", "price", "pricing", "cost"]):
            score += 0.2
        if any(term in text for term in ["authority", "decision", "stakeholder", "cto", "ceo", "vp"]):
            score += 0.2
        if any(term in text for term in ["need", "requirement", "pain", "problem", "use case"]):
            score += 0.2
        if any(term in text for term in ["timeline", "month", "quarter", "today", "urgent"]):
            score += 0.2
        if metadata.get("budget") or crm_context.get("budget"):
            score += 0.1
        return min(1.0, round(score, 4))

    @staticmethod
    def _score_buying_signal(text: str) -> float:
        strong_terms = [
            "need a demo",
            "ready to buy",
            "pricing",
            "this quarter",
            "this month",
            "next month",
            "can we start",
            "let us finalize",
            "schedule implementation",
            "i want this solution",
            "we need this",
            "today",
            "urgent",
            "procurement",
            "contract",
        ]
        matches = sum(1 for term in strong_terms if term in text)
        return min(1.0, round(0.15 + matches * 0.14, 4))

    @classmethod
    def _score_relationship(
        cls,
        text: str,
        voice_emotion: Optional[str],
        emotion_confidence: Optional[float],
        history: List[Dict[str, Any]],
        metadata: Dict[str, Any],
    ) -> float:
        score = 0.25
        if any(term in text for term in ["thank", "thanks", "appreciate", "great", "excellent", "love"]):
            score += 0.2
        if bool(metadata.get("returning_customer", False)):
            score += 0.2
        if len(history) >= 2:
            score += 0.15
        if voice_emotion in {"happy", "calm", "friendly"}:
            score += 0.15
        if emotion_confidence is not None:
            score += min(0.15, float(emotion_confidence) * 0.15)
        return min(1.0, round(score, 4))

    @staticmethod
    def _score_bant(text: str, history: List[Dict[str, Any]], metadata: Dict[str, Any], crm_context: Dict[str, Any]) -> Dict[str, Any]:
        history_text = " ".join(
            item.get("content", "")
            for item in history
            if isinstance(item, dict) and item.get("content")
        ).lower()

        def score_dimension(terms: List[str], strong_terms: List[str]) -> float:
            score = 0.15
            if any(term in text for term in terms):
                score += 0.35
            if any(term in text for term in strong_terms):
                score += 0.25
            if any(term in history_text for term in terms):
                score += 0.1
            if any(term in history_text for term in strong_terms):
                score += 0.1
            return min(1.0, max(0.0, round(score, 4)))

        budget = score_dimension(
            ["budget", "price", "pricing", "cost", "quote", "discount", "roi", "spend"],
            ["affordable", "approved budget", "budget approved", "how much", "costs", "pricing"],
        )
        authority = score_dimension(
            ["decision maker", "authority", "stakeholder", "cto", "ceo", "cfo", "vp", "approved"],
            ["i decide", "we decide", "final approval", "sign off", "signoff"],
        )
        need = score_dimension(
            ["need", "requirement", "pain", "problem", "challenge", "use case", "revenue", "sales", "profit", "profits", "growth"],
            ["increase profits", "increase revenue", "boost sales", "reduce churn", "improve conversion"],
        )
        timeline = score_dimension(
            ["timeline", "deadline", "urgent", "asap", "soon", "this month", "this quarter", "next week", "next month", "today"],
            ["ready now", "ready to start", "this week", "this month", "this quarter"],
        )

        if any(term in text for term in ["when", "by when", "how soon", "what timeframe"]):
            timeline = min(1.0, round(timeline + 0.1, 4))

        if metadata.get("budget") or crm_context.get("budget"):
            budget = min(1.0, round(budget + 0.1, 4))
        if metadata.get("decision_maker_score") or crm_context.get("decision_maker_score"):
            authority = min(1.0, round(authority + 0.1, 4))

        bant_score = (budget + authority + need + timeline) / 4.0
        qualified_dimensions = sum(1 for value in (budget, authority, need, timeline) if value >= 0.6)
        if qualified_dimensions >= 4:
            bant_category = "Fully Qualified"
            bant_recommendation = "Move to proposal or demo scheduling and confirm the buying process."
        elif qualified_dimensions >= 2:
            bant_category = "Partially Qualified"
            bant_recommendation = "Ask targeted discovery questions to complete the missing BANT fields."
        else:
            bant_category = "Needs Discovery"
            bant_recommendation = "Continue qualifying budget, authority, need, and timeline before pushing the sale."

        return {
            "bant_score": bant_score,
            "bant_scores": {
                "budget": round(budget * 100.0, 2),
                "authority": round(authority * 100.0, 2),
                "need": round(need * 100.0, 2),
                "timeline": round(timeline * 100.0, 2),
            },
            "bant_category": bant_category,
            "bant_recommendation": bant_recommendation,
        }

    @staticmethod
    def _score_meddic(text: str, history: List[Dict[str, Any]], metadata: Dict[str, Any], crm_context: Dict[str, Any]) -> Dict[str, Any]:
        """Score MEDDIC discovery evidence from the current conversation and CRM context.

        Scores represent evidence quality, not a claim that a buying criterion is true.
        This keeps qualification explainable and makes missing discovery actionable.
        """
        history_text = " ".join(
            item.get("content", "") for item in history
            if isinstance(item, dict) and item.get("content")
        ).lower()
        context_text = " ".join(str(value) for value in {**metadata, **crm_context}.values()).lower()

        def score_dimension(terms: List[str], strong_terms: List[str], context_keys: List[str]) -> float:
            score = 0.10
            if any(term in text for term in terms):
                score += 0.35
            if any(term in text for term in strong_terms):
                score += 0.25
            if any(term in history_text for term in terms):
                score += 0.10
            if any(term in context_text for term in terms) or any(metadata.get(key) or crm_context.get(key) for key in context_keys):
                score += 0.15
            return min(1.0, round(score, 4))

        metrics = score_dimension(
            ["roi", "revenue", "cost", "savings", "reduce", "increase", "conversion", "churn", "hours", "time", "%", "kpi"],
            ["save $", "reduce cost", "increase revenue", "quantifiable", "business impact", "payback", "return on investment"],
            ["roi", "economic_value", "expected_savings", "revenue_impact"],
        )
        economic_buyer = score_dimension(
            ["economic buyer", "cfo", "ceo", "finance", "budget owner", "procurement", "executive sponsor"],
            ["final budget approval", "signs the contract", "owns the budget", "financial sponsor"],
            ["economic_buyer", "budget_owner", "decision_maker"],
        )
        decision_criteria = score_dimension(
            ["criteria", "requirement", "security", "compliance", "integration", "scalability", "pricing", "implementation", "sla"],
            ["must have", "evaluation criteria", "success criteria", "vendor selection", "technical requirement"],
            ["decision_criteria", "requirements"],
        )
        decision_process = score_dimension(
            ["decision process", "approval", "procurement", "legal", "security review", "committee", "evaluation", "timeline", "next step"],
            ["approval process", "selection process", "contract review", "purchase order", "stakeholder review"],
            ["decision_process", "procurement_process", "next_step"],
        )
        identify_pain = score_dimension(
            ["pain", "problem", "challenge", "bottleneck", "manual", "inefficient", "risk", "delay", "frustrated", "unable"],
            ["current problem", "pain point", "business problem", "losing revenue", "wasting time", "urgent challenge"],
            ["pain_points", "current_problem", "challenges"],
        )
        champion = score_dimension(
            ["champion", "advocate", "sponsor", "internal supporter", "my team", "we will recommend"],
            ["I will champion", "I will advocate", "internal champion", "help us get buy-in", "recommend you internally"],
            ["champion", "internal_sponsor", "sales_advocate"],
        )

        dimensions = {
            "metrics": metrics,
            "economic_buyer": economic_buyer,
            "decision_criteria": decision_criteria,
            "decision_process": decision_process,
            "identify_pain": identify_pain,
            "champion": champion,
        }
        meddic_score = sum(dimensions.values()) / len(dimensions)
        qualified = sum(value >= 0.60 for value in dimensions.values())
        if qualified >= 4:
            category = "Well Qualified"
        elif qualified >= 2:
            category = "Discovery In Progress"
        else:
            category = "Needs MEDDIC Discovery"

        missing = [name for name, value in dimensions.items() if value < 0.60]
        followups = {
            "metrics": "What measurable revenue, cost, time, or risk outcome would make this project successful?",
            "economic_buyer": "Who owns the budget and gives final financial approval?",
            "decision_criteria": "Which must-have criteria will your team use to compare solutions?",
            "decision_process": "What are the approval steps, stakeholders, and target date after this conversation?",
            "identify_pain": "What is the current problem, its impact, and the cost of leaving it unresolved?",
            "champion": "Who will advocate for this initiative internally and help us navigate the buying team?",
        }
        recommendation = followups[missing[0]] if missing else "Confirm the documented MEDDIC evidence and align the proposal to the buying process."
        return {
            "meddic_score": meddic_score,
            "meddic_scores": {name: round(value * 100.0, 2) for name, value in dimensions.items()},
            "meddic_category": category,
            "meddic_recommendation": recommendation,
        }

    @classmethod
    def _build_lead_report(
        cls,
        text: str,
        scores: Dict[str, Any],
        lead_score: float,
        lead_category: str,
        recommendation: str,
        conversion_probability: float,
        history: List[Dict[str, Any]],
        metadata: Dict[str, Any],
        crm_context: Dict[str, Any],
        business_docs: List[str],
    ) -> Dict[str, Any]:
        combined_text = cls._normalize_text(
            " ".join(
                [
                    text or "",
                    " ".join(
                        item.get("content", "")
                        for item in history
                        if isinstance(item, dict) and item.get("content")
                    ),
                ]
            )
        )
        sentiment = scores.get("sentiment", {})
        bant_scores = scores.get("bant_scores", {})
        bant_category = scores.get("bant_category", "Needs Discovery")
        meddic_scores = scores.get("meddic_scores", {})
        meddic_category = scores.get("meddic_category", "Needs MEDDIC Discovery")
        topic_focus = cls._extract_topic_focus(combined_text)
        budget_summary = cls._summarize_budget(combined_text, scores)
        authority_summary = cls._summarize_authority(combined_text, scores)
        need_summary = cls._summarize_need(combined_text, scores)
        timeline_summary = cls._summarize_timeline(combined_text, scores)
        sentiment_label = sentiment.get("text", {}).get("sentiment", "neutral")
        tone_label = sentiment.get("tone", {}).get("label", "unknown")
        summary = (
            f"{lead_category} with {bant_category.lower()} BANT and {sentiment_label} sentiment. "
            f"The conversation is centered on {topic_focus}."
        )
        return {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "summary": summary,
            "status": bant_category,
            "topic_focus": topic_focus,
            "lead_score": round(float(lead_score), 2),
            "lead_category": lead_category,
            "bant_score": round(float(scores.get("bant_score", 0.0)), 2),
            "bant_category": bant_category,
            "meddic_score": round(float(scores.get("meddic_score", 0.0)), 2),
            "meddic_category": meddic_category,
            "conversion_probability_percent": round(float(conversion_probability), 2),
            "sentiment": sentiment,
            "tone": tone_label,
            "bant": {
                "budget": {
                    "score": bant_scores.get("budget", 0.0),
                    "summary": budget_summary,
                },
                "authority": {
                    "score": bant_scores.get("authority", 0.0),
                    "summary": authority_summary,
                },
                "need": {
                    "score": bant_scores.get("need", 0.0),
                    "summary": need_summary,
                },
                "timeline": {
                    "score": bant_scores.get("timeline", 0.0),
                    "summary": timeline_summary,
                },
            },
            "meddic": {
                "metrics": {"score": meddic_scores.get("metrics", 0.0), "summary": cls._meddic_summary(combined_text, "metrics", meddic_scores.get("metrics", 0.0))},
                "economic_buyer": {"score": meddic_scores.get("economic_buyer", 0.0), "summary": cls._meddic_summary(combined_text, "economic_buyer", meddic_scores.get("economic_buyer", 0.0))},
                "decision_criteria": {"score": meddic_scores.get("decision_criteria", 0.0), "summary": cls._meddic_summary(combined_text, "decision_criteria", meddic_scores.get("decision_criteria", 0.0))},
                "decision_process": {"score": meddic_scores.get("decision_process", 0.0), "summary": cls._meddic_summary(combined_text, "decision_process", meddic_scores.get("decision_process", 0.0))},
                "identify_pain": {"score": meddic_scores.get("identify_pain", 0.0), "summary": cls._meddic_summary(combined_text, "identify_pain", meddic_scores.get("identify_pain", 0.0))},
                "champion": {"score": meddic_scores.get("champion", 0.0), "summary": cls._meddic_summary(combined_text, "champion", meddic_scores.get("champion", 0.0))},
            },
            "metrics": {
                "icp_score": scores.get("icp_score", 0.0),
                "intent_score": scores.get("intent_score", 0.0),
                "engagement_score": scores.get("engagement_score", 0.0),
                "qualification_score": scores.get("qualification_score", 0.0),
                "buying_signal_score": scores.get("buying_signal_score", 0.0),
                "relationship_score": scores.get("relationship_score", 0.0),
            },
            "recommendation": recommendation,
            "budget": budget_summary,
            "authority_level": authority_summary,
            "need_summary": need_summary,
            "timeline_summary": timeline_summary,
            "meddic_recommendation": scores.get("meddic_recommendation", "Continue MEDDIC discovery."),
            "evidence": {
                "budget": cls._extract_phrase(combined_text, ["budget", "pricing", "price", "cost", "quote"]),
                "authority": cls._extract_phrase(combined_text, ["decision maker", "authority", "stakeholder", "cto", "ceo", "cfo", "vp"]),
                "need": cls._extract_phrase(combined_text, ["need", "requirement", "problem", "pain point", "use case"]),
                "timeline": cls._extract_phrase(combined_text, ["timeline", "deadline", "urgent", "asap", "this month", "this quarter", "next month"]),
            },
            "context": {
                "metadata_keys": sorted(list(metadata.keys())),
                "crm_context_keys": sorted(list(crm_context.keys())),
                "business_docs_count": len(business_docs),
            },
        }

    @staticmethod
    def _extract_phrase(text: str, keywords: List[str]) -> str:
        lowered = text.lower()
        for keyword in keywords:
            index = lowered.find(keyword)
            if index == -1:
                continue
            start = max(0, index - 28)
            end = min(len(text), index + 96)
            fragment = text[start:end].strip(" \n\r\t.,;:-")
            if fragment:
                return fragment[:1].upper() + fragment[1:]
        return "Not stated yet"

    @classmethod
    def _extract_topic_focus(cls, text: str) -> str:
        topic_terms = [
            "automation",
            "integration",
            "security",
            "compliance",
            "workflow",
            "pricing",
            "demo",
            "implementation",
            "support",
            "roi",
            "sales",
            "revenue",
            "growth",
            "product",
            "platform",
        ]
        matches = [term for term in topic_terms if term in text]
        if not matches:
            return "general product discussion"
        return ", ".join(dict.fromkeys(matches[:3])).replace("_", " ")

    @classmethod
    def _summarize_budget(cls, text: str, scores: Dict[str, Any]) -> str:
        if scores.get("bant_scores", {}).get("budget", 0.0) < 30:
            return "Budget is not clearly discussed yet."
        amount_match = re.search(r"(?:\$|usd\s*)?(\d{1,3}(?:,\d{3})+|\d+)(?:\s?(?:k|m|million|thousand))?", text, re.IGNORECASE)
        if amount_match:
            return f"Budget mention detected around {amount_match.group(0).strip()}."
        phrase = cls._extract_phrase(text, ["budget", "pricing", "price", "cost", "quote"])
        return f"Budget discussion detected: {phrase}."

    @classmethod
    def _summarize_authority(cls, text: str, scores: Dict[str, Any]) -> str:
        lower = text.lower()
        if any(term in lower for term in ["ceo", "cfo", "cto", "vp", "vice president", "director"]):
            return "Executive-level authority is mentioned."
        if any(term in lower for term in ["manager", "lead", "head of", "owner"]):
            return "Manager or team-lead level authority is mentioned."
        if scores.get("bant_scores", {}).get("authority", 0.0) >= 60:
            return "Authority signal is present, but the exact decision maker is not fully named."
        return "Authority is still unclear."

    @classmethod
    def _summarize_need(cls, text: str, scores: Dict[str, Any]) -> str:
        phrase = cls._extract_phrase(text, ["need", "requirement", "problem", "pain point", "use case", "challenge"])
        if scores.get("bant_scores", {}).get("need", 0.0) < 30:
            return "Need is still broad and needs discovery."
        return f"Need detected: {phrase}."

    @classmethod
    def _summarize_timeline(cls, text: str, scores: Dict[str, Any]) -> str:
        lower = text.lower()
        if any(term in lower for term in ["this week", "this month", "this quarter", "today", "asap", "urgent", "next week"]):
            return "Timeline looks near-term."
        if scores.get("bant_scores", {}).get("timeline", 0.0) < 30:
            return "Timeline has not been confirmed yet."
        return f"Timeline hint detected: {cls._extract_phrase(text, ['timeline', 'deadline', 'urgent', 'asap', 'soon'])}."

    @classmethod
    def _meddic_summary(cls, text: str, dimension: str, score: float) -> str:
        labels = {
            "metrics": ("Quantifiable outcome", ["roi", "revenue", "savings", "cost", "conversion", "hours"]),
            "economic_buyer": ("Economic buyer", ["economic buyer", "cfo", "ceo", "budget owner", "finance"]),
            "decision_criteria": ("Decision criteria", ["criteria", "requirement", "security", "compliance", "integration"]),
            "decision_process": ("Decision process", ["approval", "procurement", "legal", "committee", "next step"]),
            "identify_pain": ("Current lead problem", ["pain", "problem", "challenge", "bottleneck", "manual"]),
            "champion": ("Internal champion", ["champion", "advocate", "sponsor", "recommend", "buy-in"]),
        }
        label, keywords = labels[dimension]
        if score < 60:
            return f"{label} is not confirmed yet."
        return f"{label} evidence: {cls._extract_phrase(text, keywords)}."

    @classmethod
    def _analyze_text_sentiment(
        cls,
        text: str,
        voice_emotion: Optional[str] = None,
        emotion_confidence: Optional[float] = None,
    ) -> Dict[str, Any]:
        try:
            text_result = SentimentService.analyze(text or "")
            label = str(text_result.get("sentiment", "neutral"))
            confidence = float(text_result.get("confidence", 0.5))
            probabilities = dict(text_result.get("probabilities", {}))
            model = str(text_result.get("model", "text_sentiment"))
        except Exception as exc:  # pragma: no cover - fallback path
            logger.debug("Falling back to heuristic sentiment scoring: %s", exc)
            label = cls._heuristic_sentiment_label(text or "")
            confidence = cls._heuristic_sentiment_confidence(text or "", label)
            probabilities = cls._heuristic_sentiment_probabilities(label)
            model = "heuristic_fallback"

        blended_score = cls._sentiment_score(text or "", label, confidence, voice_emotion, emotion_confidence)
        tone_label = voice_emotion or "unknown"
        tone_confidence = float(emotion_confidence) if emotion_confidence is not None else 0.0
        return {
            "text": {
                "text": text,
                "sentiment": label,
                "confidence": round(confidence, 6),
                "probabilities": probabilities,
                "model": model,
            },
            "tone": {
                "label": tone_label,
                "confidence": round(tone_confidence, 6),
            },
            "blended_score": round(blended_score, 4),
        }

    @staticmethod
    def _heuristic_sentiment_label(text: str) -> str:
        positive_terms = ["great", "excellent", "love", "happy", "thanks", "appreciate", "good", "ready"]
        negative_terms = ["bad", "awful", "hate", "angry", "frustrated", "slow", "issue", "problem"]
        positive_hits = sum(1 for term in positive_terms if term in text)
        negative_hits = sum(1 for term in negative_terms if term in text)
        if negative_hits > positive_hits:
            return "negative"
        if positive_hits > negative_hits:
            return "positive"
        return "neutral"

    @staticmethod
    def _heuristic_sentiment_confidence(text: str, label: str) -> float:
        if label == "neutral":
            return 0.55 if text.strip() else 1.0
        return 0.66

    @staticmethod
    def _heuristic_sentiment_probabilities(label: str) -> Dict[str, float]:
        probabilities = {"negative": 0.2, "neutral": 0.2, "positive": 0.2}
        probabilities[label] = 0.6
        return probabilities

    @staticmethod
    def _sentiment_score(
        text: str,
        label: Optional[str] = None,
        confidence: Optional[float] = None,
        voice_emotion: Optional[str] = None,
        emotion_confidence: Optional[float] = None,
    ) -> float:
        label = (label or "neutral").lower()
        base_map = {"positive": 0.82, "neutral": 0.5, "negative": 0.18}
        score = base_map.get(label, 0.5)
        if confidence is not None:
            score = score * 0.75 + float(confidence) * 0.25
        if voice_emotion in {"happy", "excited", "calm", "friendly"}:
            score += 0.08
        if voice_emotion in {"angry", "frustrated", "sad"}:
            score -= 0.12
        if emotion_confidence is not None:
            score += min(0.06, float(emotion_confidence) * 0.06)
        if not text.strip():
            return 0.5
        return min(1.0, max(0.0, round(score, 4)))

    @staticmethod
    def _emotion_strength(voice_emotion: Optional[str], emotion_confidence: Optional[float]) -> float:
        if voice_emotion in {"happy", "excited", "calm"}:
            return 0.8 if emotion_confidence is None else min(1.0, 0.6 + float(emotion_confidence) * 0.2)
        if voice_emotion in {"angry", "frustrated", "sad"}:
            return 0.4 if emotion_confidence is None else min(1.0, 0.3 + float(emotion_confidence) * 0.2)
        return 0.5 if emotion_confidence is None else min(1.0, 0.4 + float(emotion_confidence) * 0.15)

import logging
from typing import Any, Dict, Optional

from app.services.lead_intelligence import LeadIntelligenceService
from app.services.advanced_metrics import AdvancedMetricsService

logger = logging.getLogger(__name__)


class LeadScoringService:
    @classmethod
    def calculate_lead_scores(
        cls,
        text: str,
        voice_emotion: Optional[str] = None,
        emotion_confidence: Optional[float] = None,
        history: Optional[list[dict[str, Any]]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        crm_context: Optional[Dict[str, Any]] = None,
        business_docs: Optional[list[str]] = None,
    ) -> Dict[str, Any]:
        return LeadIntelligenceService.calculate_lead_scores(
            text,
            voice_emotion=voice_emotion,
            emotion_confidence=emotion_confidence,
            history=history,
            metadata=metadata,
            crm_context=crm_context,
            business_docs=business_docs,
        )

    @classmethod
    def predict_conversion(
        cls,
        text: str,
        voice_emotion: Optional[str] = None,
        emotion_confidence: Optional[float] = None,
        history: Optional[list[dict[str, Any]]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        crm_context: Optional[Dict[str, Any]] = None,
        business_docs: Optional[list[str]] = None,
    ) -> Dict[str, Any]:
        return LeadIntelligenceService.predict_conversion(
            text,
            voice_emotion=voice_emotion,
            emotion_confidence=emotion_confidence,
            history=history,
            metadata=metadata,
            crm_context=crm_context,
            business_docs=business_docs,
        )

    @classmethod
    def categorize_lead(cls, lead_score: float) -> str:
        return LeadIntelligenceService.categorize_lead(lead_score)

    @classmethod
    def generate_next_action(
        cls,
        lead_score: float,
        scores: Optional[Dict[str, Any]] = None,
        text: Optional[str] = None,
    ) -> str:
        return LeadIntelligenceService.generate_next_action(lead_score, scores=scores, text=text)

    @classmethod
    def update_customer_profile(
        cls,
        session_id: str,
        profile: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        return LeadIntelligenceService.update_customer_profile(session_id, profile=profile, metadata=metadata)

    @classmethod
    def store_lead_history(
        cls,
        session_id: str,
        payload: Dict[str, Any],
        text: Optional[str] = None,
    ) -> Dict[str, Any]:
        return LeadIntelligenceService.store_lead_history(session_id, payload, text=text)

    @classmethod
    def score(
        cls,
        text: str,
        voice_emotion: Optional[str] = None,
        emotion_confidence: Optional[float] = None,
        history: Optional[list[dict[str, Any]]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        crm_context: Optional[Dict[str, Any]] = None,
        business_docs: Optional[list[str]] = None,
    ) -> Dict[str, Any]:
        return LeadIntelligenceService.score(
            text,
            voice_emotion=voice_emotion,
            emotion_confidence=emotion_confidence,
            history=history,
            metadata=metadata,
            crm_context=crm_context,
            business_docs=business_docs,
        )

    # ============ ADVANCED METRICS METHODS ============

    @classmethod
    def get_advanced_metrics(
        cls,
        text: str,
        history: Optional[list[dict[str, Any]]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        crm_context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Get comprehensive advanced metrics including economic value, buyer, criteria, process, and pain points.
        Production-ready report combining all advanced ML-powered metrics.
        """
        return AdvancedMetricsService.generate_comprehensive_report(
            text,
            history=history,
            metadata=metadata,
            crm_context=crm_context,
        )

    @classmethod
    def get_economic_value(
        cls,
        text: str,
        history: Optional[list[dict[str, Any]]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Get quantifiable economic value metrics."""
        return AdvancedMetricsService.calculate_economic_value(text, history, metadata)

    @classmethod
    def get_economic_buyer(
        cls,
        text: str,
        history: Optional[list[dict[str, Any]]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Identify economic buyer and decision authority."""
        return AdvancedMetricsService.identify_economic_buyer(text, history, metadata)

    @classmethod
    def get_decision_criteria(
        cls,
        text: str,
        history: Optional[list[dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """Extract and categorize decision criteria."""
        return AdvancedMetricsService.extract_decision_criteria(text, history)

    @classmethod
    def get_decision_process(
        cls,
        text: str,
        history: Optional[list[dict[str, Any]]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Analyze decision process stage and next steps."""
        return AdvancedMetricsService.analyze_decision_process(text, history, metadata)

    @classmethod
    def get_pain_points(
        cls,
        text: str,
        history: Optional[list[dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """Identify and analyze pain points with cosine similarity clustering."""
        return AdvancedMetricsService.identify_pain_points(text, history)

    @classmethod
    def compare_leads(
        cls,
        lead_text_1: str,
        lead_text_2: str,
    ) -> Dict[str, Any]:
        """Compare two leads using cosine similarity."""
        return AdvancedMetricsService.compare_leads_cosine_similarity(lead_text_1, lead_text_2)

    @classmethod
    def compare_pain_profiles(
        cls,
        pain_points_1: list[dict[str, Any]],
        pain_points_2: list[dict[str, Any]],
    ) -> Dict[str, Any]:
        """Compare pain point profiles between leads using cosine similarity."""
        return AdvancedMetricsService.compare_pain_points_similarity(pain_points_1, pain_points_2)

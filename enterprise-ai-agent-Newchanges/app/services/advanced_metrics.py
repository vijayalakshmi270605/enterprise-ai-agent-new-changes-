"""
Advanced Metrics Service - Production-Grade ML Scoring
Includes: Economic Value, Economic Buyer, Decision Criteria, Decision Process, Pain Points
Uses: Cosine Similarity for intelligent lead comparison and pain point matching
"""

import json
import logging
import re
from typing import Any, Dict, List, Optional, Tuple
from collections import Counter
from datetime import datetime, timezone

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

logger = logging.getLogger(__name__)


class AdvancedMetricsService:
    """Production-ready ML service for advanced lead metrics and pain point analysis."""

    # Economic value keywords with weights
    ECONOMIC_KEYWORDS = {
        "revenue": 1.0,
        "sales": 0.95,
        "profit": 1.0,
        "margin": 0.9,
        "growth": 0.85,
        "upsell": 0.8,
        "cross-sell": 0.8,
        "retention": 0.9,
        "churn": 0.85,
        "efficiency": 0.75,
        "productivity": 0.8,
        "cost": 0.9,
        "savings": 0.95,
        "reduction": 0.7,
        "roi": 1.0,
        "payback": 0.9,
        "investment": 0.8,
        "budget": 0.7,
        "spend": 0.7,
        "performance": 0.75,
        "metrics": 0.65,
        "kpi": 0.9,
        "benchmark": 0.7,
        "competitor": 0.75,
        "market": 0.7,
        "opportunity": 0.8,
        "deal": 0.9,
        "contract": 0.8,
        "quarter": 0.6,
        "year": 0.5,
        "millions": 1.0,
        "thousands": 0.9,
        "enterprise": 0.85,
        "expansion": 0.9,
        "scale": 0.8,
    }

    # Economic buyer keywords
    ECONOMIC_BUYER_KEYWORDS = {
        "cfo": 1.0,
        "chief financial officer": 1.0,
        "vp finance": 0.95,
        "director finance": 0.9,
        "finance manager": 0.85,
        "ceo": 0.95,
        "chief executive": 0.95,
        "coo": 0.9,
        "chief operating": 0.9,
        "vp operations": 0.85,
        "controller": 0.9,
        "treasurer": 0.9,
        "budget holder": 0.95,
        "approver": 0.8,
        "decision maker": 0.85,
        "stakeholder": 0.7,
        "sponsor": 0.85,
        "executive": 0.75,
        "board": 0.9,
        "owner": 0.8,
        "president": 0.85,
    }

    # Pain point keywords
    PAIN_POINT_KEYWORDS = {
        "problem": 0.9,
        "pain": 1.0,
        "challenge": 0.85,
        "issue": 0.8,
        "struggle": 0.9,
        "difficulty": 0.85,
        "bottleneck": 0.95,
        "inefficient": 0.9,
        "slow": 0.85,
        "manual": 0.9,
        "error": 0.9,
        "mistake": 0.85,
        "risk": 0.85,
        "compliance": 0.8,
        "security": 0.8,
        "scalability": 0.85,
        "downtime": 0.95,
        "outage": 0.95,
        "loss": 0.9,
        "churn": 0.9,
        "customer dissatisfaction": 0.95,
        "support burden": 0.85,
        "training": 0.75,
        "integration": 0.8,
        "compatibility": 0.8,
        "maintenance": 0.8,
        "complexity": 0.85,
        "frustration": 0.85,
        "waste": 0.9,
        "redundancy": 0.85,
        "silos": 0.9,
        "disconnect": 0.85,
    }

    # Decision criteria keywords
    DECISION_CRITERIA_KEYWORDS = {
        "must have": 1.0,
        "critical": 0.95,
        "required": 0.95,
        "essential": 0.9,
        "important": 0.85,
        "feature": 0.7,
        "functionality": 0.75,
        "integration": 0.8,
        "api": 0.75,
        "security": 0.9,
        "compliance": 0.9,
        "performance": 0.8,
        "scalability": 0.85,
        "support": 0.8,
        "training": 0.75,
        "documentation": 0.7,
        "budget": 0.85,
        "cost": 0.85,
        "pricing": 0.85,
        "sla": 0.9,
        "uptime": 0.9,
        "availability": 0.85,
        "maintenance": 0.75,
        "user friendly": 0.7,
        "customizable": 0.8,
        "flexible": 0.8,
        "roadmap": 0.75,
        "timeline": 0.8,
        "implementation": 0.8,
        "vendor": 0.75,
        "track record": 0.8,
        "reputation": 0.75,
    }

    # Decision process keywords
    DECISION_PROCESS_KEYWORDS = {
        "pilot": 0.95,
        "poc": 0.95,
        "proof of concept": 0.95,
        "trial": 0.9,
        "demo": 0.85,
        "evaluation": 0.9,
        "assessment": 0.85,
        "rfp": 0.95,
        "request for proposal": 0.95,
        "proposal": 0.85,
        "presentation": 0.75,
        "meeting": 0.6,
        "discussion": 0.65,
        "review": 0.8,
        "approval": 0.95,
        "sign off": 0.95,
        "signature": 0.95,
        "contract": 0.9,
        "negotiation": 0.9,
        "terms": 0.85,
        "implementation": 0.85,
        "kickoff": 0.9,
        "onboarding": 0.85,
        "training": 0.8,
        "go live": 0.95,
        "launch": 0.9,
        "deployment": 0.85,
        "vendor selection": 0.95,
        "shortlist": 0.9,
        "finalist": 0.9,
        "quote": 0.8,
    }

    @classmethod
    def calculate_economic_value(
        cls,
        text: str,
        history: Optional[List[Dict[str, Any]]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Calculate quantifiable economic value score and metrics.
        
        Returns:
            - economic_value_score: 0-100
            - economic_indicators: list of detected indicators
            - estimated_deal_size: category (small, medium, large, enterprise)
            - roi_potential: estimated ROI score
        """
        combined_text = cls._combine_text(text, history)
        
        # Extract numeric values (potential deal sizes, percentages, etc.)
        numeric_patterns = cls._extract_numeric_indicators(combined_text)
        
        # Score economic keywords
        economic_score = cls._score_keywords(combined_text, cls.ECONOMIC_KEYWORDS)
        
        # Detect deal size from context
        deal_size = cls._estimate_deal_size(numeric_patterns, combined_text, metadata)
        
        # Calculate ROI potential
        roi_score = cls._calculate_roi_potential(combined_text, economic_score)
        
        # Extract economic indicators with context
        indicators = cls._extract_economic_indicators(combined_text)
        
        return {
            "economic_value_score": round(economic_score * 100, 2),
            "economic_indicators": indicators,
            "estimated_deal_size": deal_size,
            "roi_potential": round(roi_score * 100, 2),
            "numeric_indicators": numeric_patterns,
            "confidence": round(min(1.0, len(indicators) / 3.0), 2),
        }

    @classmethod
    def identify_economic_buyer(
        cls,
        text: str,
        history: Optional[List[Dict[str, Any]]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Identify economic buyer (stakeholder with budget authority).
        
        Returns:
            - buyer_identified: bool
            - buyer_title: identified role
            - buyer_score: confidence 0-100
            - buyer_signals: contextual evidence
            - approval_authority: decision power level
        """
        combined_text = cls._combine_text(text, history)
        
        # Score buyer keywords
        buyer_score = cls._score_keywords(combined_text, cls.ECONOMIC_BUYER_KEYWORDS)
        
        # Detect specific titles
        detected_titles = cls._extract_matching_keywords(combined_text, cls.ECONOMIC_BUYER_KEYWORDS)
        
        # Determine approval authority level
        authority_level = cls._determine_authority_level(detected_titles, buyer_score)
        
        # Extract buyer signals
        buyer_signals = cls._extract_buyer_signals(combined_text, detected_titles)
        
        buyer_identified = buyer_score >= 0.5
        primary_title = detected_titles[0] if detected_titles else "Unknown"
        
        return {
            "buyer_identified": buyer_identified,
            "buyer_title": primary_title,
            "buyer_score": round(buyer_score * 100, 2),
            "buyer_signals": buyer_signals,
            "approval_authority": authority_level,
            "confidence": round(buyer_score, 2),
            "detected_titles": detected_titles[:3],  # Top 3 titles
        }

    @classmethod
    def extract_decision_criteria(
        cls,
        text: str,
        history: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """
        Extract and categorize decision criteria from conversation.
        
        Returns:
            - total_criteria: count
            - must_have_criteria: critical requirements
            - nice_to_have_criteria: secondary requirements
            - criteria_coverage: percentage of categories covered
            - risk_factors: missing critical criteria
        """
        combined_text = cls._combine_text(text, history)
        
        # Extract criteria
        criteria_list = cls._extract_criteria_list(combined_text)
        
        # Categorize by importance
        must_have = [c for c in criteria_list if c["weight"] >= 0.9]
        nice_to_have = [c for c in criteria_list if c["weight"] < 0.9]
        
        # Calculate coverage
        criteria_categories = {
            "functional": 0,
            "non_functional": 0,
            "commercial": 0,
            "support": 0,
            "integration": 0,
        }
        
        for criterion in criteria_list:
            category = cls._categorize_criterion(criterion["text"])
            if category in criteria_categories:
                criteria_categories[category] += 1
        
        coverage = len([v for v in criteria_categories.values() if v > 0]) / len(criteria_categories)
        
        # Identify missing critical criteria
        required_categories = {"functional", "commercial", "support"}
        missing = [cat for cat in required_categories if criteria_categories.get(cat, 0) == 0]
        
        return {
            "total_criteria": len(criteria_list),
            "must_have_criteria": must_have[:5],  # Top 5
            "nice_to_have_criteria": nice_to_have[:5],
            "criteria_categories": criteria_categories,
            "criteria_coverage": round(coverage * 100, 2),
            "missing_critical_criteria": missing,
            "risk_level": "high" if len(missing) >= 2 else "medium" if len(missing) == 1 else "low",
        }

    @classmethod
    def analyze_decision_process(
        cls,
        text: str,
        history: Optional[List[Dict[str, Any]]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Analyze decision process stage and followup requirements.
        
        Returns:
            - current_stage: identified decision stage
            - stage_score: confidence 0-100
            - process_steps: detected steps in process
            - next_steps: recommended followups
            - timeline_urgency: urgency level
        """
        combined_text = cls._combine_text(text, history)
        
        # Extract process keywords
        process_keywords = cls._extract_matching_keywords(combined_text, cls.DECISION_PROCESS_KEYWORDS)
        
        # Determine current stage
        current_stage = cls._determine_process_stage(process_keywords)
        stage_score = cls._score_keywords(combined_text, cls.DECISION_PROCESS_KEYWORDS)
        
        # Extract detected steps
        process_steps = sorted(
            process_keywords,
            key=lambda x: cls.DECISION_PROCESS_KEYWORDS.get(x, 0),
            reverse=True
        )[:5]
        
        # Recommend next steps based on stage
        next_steps = cls._recommend_next_steps(current_stage, process_steps, combined_text)
        
        # Analyze timeline urgency
        urgency = cls._analyze_urgency(combined_text)
        
        return {
            "current_stage": current_stage,
            "stage_score": round(stage_score * 100, 2),
            "process_steps": process_steps,
            "detected_stages": sorted(
                list(set(process_keywords)),
                key=lambda x: cls.DECISION_PROCESS_KEYWORDS.get(x, 0),
                reverse=True
            )[:10],
            "next_steps": next_steps,
            "timeline_urgency": urgency["urgency_level"],
            "urgency_score": round(urgency["score"], 2),
            "recommended_followup": urgency["recommended_action"],
        }

    @classmethod
    def identify_pain_points(
        cls,
        text: str,
        history: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """
        Identify and analyze pain points from conversation.
        Uses cosine similarity to match similar pain points.
        
        Returns:
            - pain_points: extracted pain points
            - primary_pain: most critical pain
            - pain_categories: categorized pain points
            - pain_severity: overall severity 0-100
            - pain_clusters: similar pain points grouped via cosine similarity
        """
        combined_text = cls._combine_text(text, history)
        
        # Extract pain points with context
        pain_points = cls._extract_pain_points_with_context(combined_text)
        
        if not pain_points:
            return {
                "pain_points": [],
                "primary_pain": None,
                "pain_categories": {},
                "pain_severity": 0.0,
                "pain_clusters": [],
                "has_pain": False,
            }
        
        # Cluster similar pain points using cosine similarity
        pain_clusters = cls._cluster_similar_pains(pain_points)
        
        # Categorize pain points
        pain_categories = cls._categorize_pain_points(pain_points)
        
        # Calculate severity
        severity = cls._calculate_pain_severity(pain_points)
        
        # Identify primary pain
        primary_pain = max(pain_points, key=lambda x: x["weight"]) if pain_points else None
        
        return {
            "pain_points": pain_points[:10],
            "primary_pain": primary_pain,
            "pain_categories": pain_categories,
            "pain_severity": round(severity * 100, 2),
            "pain_clusters": pain_clusters,
            "total_pain_indicators": len(pain_points),
            "has_pain": len(pain_points) > 0,
            "pain_frequency_distribution": cls._calculate_pain_frequency(pain_points),
        }

    @classmethod
    def compare_leads_cosine_similarity(
        cls,
        lead_text_1: str,
        lead_text_2: str,
    ) -> Dict[str, Any]:
        """
        Compare two leads using cosine similarity.
        
        Returns:
            - similarity_score: 0-100
            - matching_aspects: similar topics
            - diverging_aspects: different topics
            - recommendation: suggested grouping/handling
        """
        vectorizer = TfidfVectorizer(lowercase=True, stop_words="english", max_features=100)
        
        try:
            tfidf_matrix = vectorizer.fit_transform([lead_text_1, lead_text_2])
            similarity = float(cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0])
        except Exception as e:
            logger.error(f"Cosine similarity calculation failed: {e}")
            similarity = 0.0
        
        # Extract key terms from each lead
        terms_1 = set(vectorizer.get_feature_names_out()) if lead_text_1.strip() else set()
        terms_2 = set(vectorizer.get_feature_names_out()) if lead_text_2.strip() else set()
        
        matching = list(terms_1 & terms_2)[:5]
        diverging = list((terms_1 ^ terms_2))[:5]
        
        return {
            "similarity_score": round(similarity * 100, 2),
            "matching_aspects": matching,
            "diverging_aspects": diverging,
            "recommendation": cls._similarity_recommendation(similarity),
            "same_segment": similarity >= 0.7,
            "same_pain_profile": similarity >= 0.6,
        }

    @classmethod
    def compare_pain_points_similarity(
        cls,
        pain_points_1: List[Dict[str, Any]],
        pain_points_2: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Compare pain points between two leads using cosine similarity."""
        if not pain_points_1 or not pain_points_2:
            return {
                "similarity_score": 0.0,
                "matching_pains": [],
                "diverging_pains": [],
                "common_pain_percentage": 0.0,
            }
        
        text_1 = " ".join([p["text"] for p in pain_points_1])
        text_2 = " ".join([p["text"] for p in pain_points_2])
        
        vectorizer = TfidfVectorizer(lowercase=True, stop_words="english", max_features=50)
        
        try:
            tfidf_matrix = vectorizer.fit_transform([text_1, text_2])
            similarity = float(cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0])
        except Exception as e:
            logger.error(f"Pain point similarity calculation failed: {e}")
            similarity = 0.0
        
        matching_pains = [
            p for p in pain_points_1
            if any(word in text_2.lower() for word in p["text"].lower().split())
        ]
        
        common_percentage = (len(matching_pains) / max(len(pain_points_1), 1)) * 100
        
        return {
            "similarity_score": round(similarity * 100, 2),
            "matching_pains": matching_pains[:3],
            "total_matching": len(matching_pains),
            "common_pain_percentage": round(common_percentage, 2),
            "can_use_same_solution": similarity >= 0.65,
        }

    # ============ PRIVATE HELPER METHODS ============

    @staticmethod
    def _combine_text(text: str, history: Optional[List[Dict[str, Any]]] = None) -> str:
        """Combine text with conversation history."""
        parts = [text or ""]
        if history:
            for item in history:
                if isinstance(item, dict) and "content" in item:
                    parts.append(item.get("content", ""))
        return " ".join(parts).lower()

    @staticmethod
    def _extract_numeric_indicators(text: str) -> Dict[str, Any]:
        """Extract numeric values indicating deal size, percentages, etc."""
        indicators = {
            "large_numbers": [],
            "percentages": [],
            "time_references": [],
        }
        
        # Find large numbers (potential deal sizes)
        large_nums = re.findall(r"\b(\d{3,}(?:,\d{3})*|\d+[MK])\b", text)
        indicators["large_numbers"] = large_nums[:5]
        
        # Find percentages
        percentages = re.findall(r"(\d+\.?\d*)\s*%", text)
        indicators["percentages"] = percentages[:5]
        
        # Find time references
        time_refs = re.findall(r"\b(this\s+(?:week|month|quarter|year|q[1-4])|q[1-4]\s+\d{4})\b", text)
        indicators["time_references"] = list(set(time_refs))
        
        return indicators

    @staticmethod
    def _score_keywords(text: str, keyword_dict: Dict[str, float]) -> float:
        """Score text based on keyword presence and weights."""
        if not text or not keyword_dict:
            return 0.0
        
        score = 0.0
        max_weight = max(keyword_dict.values()) if keyword_dict else 1.0
        matches_count = 0
        
        for keyword, weight in keyword_dict.items():
            if keyword in text:
                score += weight
                matches_count += 1
        
        if matches_count == 0:
            return 0.0
        
        # Normalize: average weight of matched keywords divided by max possible
        normalized_score = (score / matches_count) / max_weight
        return min(1.0, max(0.0, normalized_score))

    @staticmethod
    def _extract_matching_keywords(text: str, keyword_dict: Dict[str, float]) -> List[str]:
        """Extract all matching keywords from text."""
        matched = []
        for keyword in keyword_dict.keys():
            if keyword in text:
                matched.append(keyword)
        return matched

    @staticmethod
    def _estimate_deal_size(
        numeric_patterns: Dict[str, Any],
        text: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Estimate deal size from context."""
        deal_indicators = {
            "enterprise": ["enterprise", "global", "international", "multi", "division"],
            "large": ["large", "thousands", "million", "significant"],
            "medium": ["mid", "growing", "expansion", "scaling"],
            "small": ["small", "startup", "single", "department"],
        }
        
        for size, keywords in deal_indicators.items():
            if any(kw in text for kw in keywords):
                return size
        
        # Check numeric patterns
        large_nums = numeric_patterns.get("large_numbers", [])
        if large_nums:
            try:
                max_num = max([int(n.replace(",", "").rstrip("MK")) for n in large_nums if n.replace(",", "").rstrip("MK").isdigit()])
                if max_num > 1000000:
                    return "enterprise"
                elif max_num > 100000:
                    return "large"
                elif max_num > 10000:
                    return "medium"
            except (ValueError, AttributeError):
                pass
        
        return "unknown"

    @staticmethod
    def _calculate_roi_potential(text: str, economic_score: float) -> float:
        """Calculate ROI potential score."""
        roi_keywords = ["roi", "payback", "return", "investment", "breakeven"]
        roi_boost = sum(1 for kw in roi_keywords if kw in text) * 0.15
        
        efficiency_keywords = ["efficiency", "productivity", "automation", "reduce"]
        efficiency_boost = sum(1 for kw in efficiency_keywords if kw in text) * 0.1
        
        return min(1.0, economic_score + roi_boost + efficiency_boost)

    @staticmethod
    def _extract_economic_indicators(text: str) -> List[str]:
        """Extract specific economic indicators mentioned."""
        indicators = []
        economic_phrases = {
            "revenue increase": r"(?:increase|grow|boost)\s+(?:revenue|sales)",
            "cost reduction": r"(?:reduce|cut|lower)\s+(?:cost|expense)",
            "efficiency": r"(?:improve|enhance)\s+(?:efficiency|productivity)",
            "roi": r"(?:roi|return\s+on\s+investment)",
            "margin": r"(?:margin|profit)",
            "churn": r"(?:reduce|lower|cut)\s+churn",
            "retention": r"(?:improve|increase)\s+retention",
        }
        
        for indicator, pattern in economic_phrases.items():
            if re.search(pattern, text):
                indicators.append(indicator)
        
        return indicators

    @staticmethod
    def _determine_authority_level(titles: List[str], score: float) -> str:
        """Determine approval authority level."""
        c_level = ["cfo", "ceo", "coo", "chief"]
        vp_level = ["vp", "director"]
        manager_level = ["manager", "coordinator"]
        
        for title in titles:
            if any(c in title for c in c_level):
                return "executive"
            elif any(v in title for v in vp_level):
                return "director"
            elif any(m in title for m in manager_level):
                return "manager"
        
        if score >= 0.8:
            return "executive"
        elif score >= 0.6:
            return "director"
        elif score >= 0.4:
            return "manager"
        return "contributor"

    @staticmethod
    def _extract_buyer_signals(text: str, titles: List[str]) -> List[str]:
        """Extract buyer-related signals."""
        signals = []
        
        if titles:
            signals.append(f"Title mentioned: {titles[0]}")
        
        if "approve" in text or "approval" in text:
            signals.append("Budget approval authority indicated")
        
        if "budget" in text:
            signals.append("Budget involvement confirmed")
        
        if "decision" in text:
            signals.append("Decision-making involvement mentioned")
        
        if "stakeholder" in text:
            signals.append("Key stakeholder identified")
        
        return signals

    @staticmethod
    def _extract_criteria_list(text: str) -> List[Dict[str, Any]]:
        """Extract decision criteria with weights."""
        criteria = []
        
        for keyword, weight in AdvancedMetricsService.DECISION_CRITERIA_KEYWORDS.items():
            if keyword in text:
                criteria.append({
                    "text": keyword,
                    "weight": weight,
                    "type": "explicit",
                })
        
        return sorted(criteria, key=lambda x: x["weight"], reverse=True)

    @staticmethod
    def _categorize_criterion(criterion_text: str) -> str:
        """Categorize criterion into type."""
        if any(word in criterion_text for word in ["feature", "functionality", "integration", "api"]):
            return "functional"
        elif any(word in criterion_text for word in ["performance", "scalability", "uptime", "sla"]):
            return "non_functional"
        elif any(word in criterion_text for word in ["budget", "cost", "pricing", "sla"]):
            return "commercial"
        elif any(word in criterion_text for word in ["support", "training", "documentation"]):
            return "support"
        elif any(word in criterion_text for word in ["integration", "api", "compatibility"]):
            return "integration"
        return "other"

    @staticmethod
    def _recommend_next_steps(
        current_stage: str,
        process_steps: List[str],
        text: str,
    ) -> List[str]:
        """Recommend next steps based on process stage."""
        stage_recommendations = {
            "discovery": ["Schedule detailed discovery call", "Request RFP", "Understand full requirements"],
            "evaluation": ["Provide product demo", "Share case studies", "Propose pilot/POC"],
            "pilot": ["Set success criteria", "Weekly check-ins", "Gather feedback"],
            "negotiation": ["Prepare proposal", "Discuss terms", "Address concerns"],
            "approval": ["Obtain signatures", "Schedule implementation kick-off", "Prepare SOW"],
            "implementation": ["Start onboarding", "Assign dedicated support", "Track KPIs"],
            "deployment": ["Go-live support", "Monitor performance", "Schedule post-implementation review"],
        }
        
        return stage_recommendations.get(current_stage, ["Continue nurturing", "Schedule follow-up"])

    @staticmethod
    def _determine_process_stage(keywords: List[str]) -> str:
        """Determine current stage in sales process."""
        stage_keywords = {
            "discovery": ["demo", "discussion", "review", "evaluation"],
            "evaluation": ["assessment", "evaluation", "poc", "trial"],
            "pilot": ["pilot", "poc", "proof of concept", "trial"],
            "negotiation": ["negotiation", "terms", "proposal", "quote"],
            "approval": ["approval", "sign off", "contract", "signature"],
            "implementation": ["implementation", "kickoff", "training", "onboarding"],
            "deployment": ["deployment", "go live", "launch"],
        }
        
        for stage, stage_kws in stage_keywords.items():
            if any(kw in keywords for kw in stage_kws):
                return stage
        
        return "initial_contact"

    @staticmethod
    def _analyze_urgency(text: str) -> Dict[str, Any]:
        """Analyze timeline urgency."""
        urgent_terms = {
            "immediate": ["asap", "today", "urgent", "emergency"],
            "short_term": ["this week", "next week", "this month"],
            "medium_term": ["this quarter", "next quarter", "6 months"],
            "long_term": ["next year", "future consideration"],
        }
        
        for urgency, terms in urgent_terms.items():
            if any(term in text for term in terms):
                return {
                    "urgency_level": urgency,
                    "score": {"immediate": 1.0, "short_term": 0.75, "medium_term": 0.5, "long_term": 0.25}.get(urgency, 0.5),
                    "recommended_action": {
                        "immediate": "High priority - immediate action required",
                        "short_term": "Schedule urgent demo and proposal",
                        "medium_term": "Nurture and schedule evaluation",
                        "long_term": "Add to nurture sequence",
                    }.get(urgency, "Follow up as scheduled"),
                }
        
        return {"urgency_level": "standard", "score": 0.5, "recommended_action": "Standard follow-up"}

    @staticmethod
    def _extract_pain_points_with_context(text: str) -> List[Dict[str, Any]]:
        """Extract pain points with their context and confidence."""
        pain_points = []
        
        for pain, weight in AdvancedMetricsService.PAIN_POINT_KEYWORDS.items():
            if pain in text:
                # Find context around the pain point
                pattern = f".{{0,50}}{pain}.{{0,50}}"
                matches = re.findall(pattern, text, re.IGNORECASE)
                
                for match in matches[:1]:  # Use first match for context
                    pain_points.append({
                        "text": pain,
                        "weight": weight,
                        "context": match.strip(),
                        "position": text.find(pain.lower()),
                    })
        
        return sorted(pain_points, key=lambda x: x["weight"], reverse=True)

    @staticmethod
    def _cluster_similar_pains(pain_points: List[Dict[str, Any]]) -> List[List[Dict[str, Any]]]:
        """Cluster similar pain points using cosine similarity."""
        if len(pain_points) <= 1:
            return [[pain_points[0]]] if pain_points else []
        
        vectorizer = TfidfVectorizer(lowercase=True, stop_words="english", max_features=20)
        
        try:
            texts = [p["text"] for p in pain_points]
            tfidf_matrix = vectorizer.fit_transform(texts)
            similarity_matrix = cosine_similarity(tfidf_matrix)
            
            clusters = []
            visited = set()
            
            for i in range(len(pain_points)):
                if i in visited:
                    continue
                
                cluster = [pain_points[i]]
                visited.add(i)
                
                for j in range(i + 1, len(pain_points)):
                    if j not in visited and similarity_matrix[i][j] >= 0.5:
                        cluster.append(pain_points[j])
                        visited.add(j)
                
                clusters.append(cluster)
            
            return clusters
        except Exception as e:
            logger.error(f"Pain clustering failed: {e}")
            return [[p] for p in pain_points]

    @staticmethod
    def _categorize_pain_points(pain_points: List[Dict[str, Any]]) -> Dict[str, List[str]]:
        """Categorize pain points by type."""
        categories = {
            "operational": [],
            "financial": [],
            "technical": [],
            "strategic": [],
            "compliance": [],
        }
        
        financial_keywords = ["cost", "budget", "roi", "savings", "profit", "revenue"]
        technical_keywords = ["integration", "api", "system", "performance", "scalability"]
        operational_keywords = ["manual", "inefficient", "process", "workflow", "bottleneck"]
        strategic_keywords = ["growth", "expansion", "competitor", "market"]
        compliance_keywords = ["security", "compliance", "audit", "regulation"]
        
        for pain in pain_points:
            text = pain["text"].lower()
            if any(kw in text for kw in financial_keywords):
                categories["financial"].append(pain["text"])
            elif any(kw in text for kw in technical_keywords):
                categories["technical"].append(pain["text"])
            elif any(kw in text for kw in operational_keywords):
                categories["operational"].append(pain["text"])
            elif any(kw in text for kw in strategic_keywords):
                categories["strategic"].append(pain["text"])
            elif any(kw in text for kw in compliance_keywords):
                categories["compliance"].append(pain["text"])
            else:
                categories["operational"].append(pain["text"])
        
        return {k: v for k, v in categories.items() if v}

    @staticmethod
    def _calculate_pain_severity(pain_points: List[Dict[str, Any]]) -> float:
        """Calculate overall pain severity."""
        if not pain_points:
            return 0.0
        
        high_severity_keywords = ["critical", "urgent", "critical", "outage", "loss"]
        
        weights = [p["weight"] for p in pain_points]
        base_severity = sum(weights) / max(len(weights), 1) / max(max(weights), 1.0)
        
        # Boost if critical terms present
        high_severity_count = sum(1 for p in pain_points if any(kw in p["text"] for kw in high_severity_keywords))
        severity_boost = (high_severity_count / max(len(pain_points), 1)) * 0.3
        
        return min(1.0, base_severity + severity_boost)

    @staticmethod
    def _calculate_pain_frequency(pain_points: List[Dict[str, Any]]) -> Dict[str, int]:
        """Calculate frequency distribution of pain points."""
        counter = Counter([p["text"] for p in pain_points])
        return dict(counter.most_common(10))

    @staticmethod
    def _similarity_recommendation(similarity_score: float) -> str:
        """Generate recommendation based on similarity score."""
        if similarity_score >= 0.85:
            return "Same segment - Use identical sales strategy"
        elif similarity_score >= 0.70:
            return "Similar segment - Adapt proven solution"
        elif similarity_score >= 0.50:
            return "Related segment - Customize approach"
        else:
            return "Different segment - Independent strategy required"

    @classmethod
    def generate_comprehensive_report(
        cls,
        text: str,
        history: Optional[List[Dict[str, Any]]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        crm_context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Generate comprehensive advanced metrics report.
        Production-ready report combining all advanced metrics.
        """
        try:
            economic_value = cls.calculate_economic_value(text, history, metadata)
            economic_buyer = cls.identify_economic_buyer(text, history, metadata)
            decision_criteria = cls.extract_decision_criteria(text, history)
            decision_process = cls.analyze_decision_process(text, history, metadata)
            pain_points = cls.identify_pain_points(text, history)
            
            return {
                "generated_at": datetime.now(timezone.utc).isoformat(),
                "economic_value": economic_value,
                "economic_buyer": economic_buyer,
                "decision_criteria": decision_criteria,
                "decision_process": decision_process,
                "pain_points": pain_points,
                "overall_readiness_score": round(
                    (
                        economic_value["economic_value_score"]
                        + (economic_buyer["buyer_score"] if economic_buyer["buyer_identified"] else 0)
                        + decision_criteria["criteria_coverage"]
                        + decision_process["stage_score"]
                    ) / 4.0,
                    2,
                ),
                "production_ready": True,
            }
        except Exception as e:
            logger.error(f"Comprehensive report generation failed: {e}", exc_info=True)
            raise

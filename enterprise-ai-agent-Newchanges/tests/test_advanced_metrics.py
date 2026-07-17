"""
Advanced Metrics Tests - Production-Ready Test Suite
Tests all advanced metrics including economic value, buyer, criteria, process, and pain points
"""

import pytest
from app.services.advanced_metrics import AdvancedMetricsService
from app.services.lead_scoring_service import LeadScoringService


# Sample test data
ENTERPRISE_LEAD_TEXT = """
We need to increase our annual revenue by 30% this quarter. 
We have a budget of 2 million dollars approved by our CFO. 
Our VP of Operations and I (the CTO) will be involved in the decision.
We need enterprise-grade security, compliance certifications, and scalability.
The main challenge is our current manual processes taking 40% of our team's time.
We're looking for a demo this week and want to finalize by end of Q2.
"""

STARTUP_LEAD_TEXT = """
We have a problem with our payment processing - we're losing customers due to slow integrations.
Our team is small, so we need something easy to implement.
Budget is tight but we can allocate 50k for this solution.
I'm the founder and only decision maker for now.
"""

PAIN_POINT_LEAD_TEXT = """
We're struggling with data silos across departments. 
Our compliance team is reporting critical security risks.
The integration bottleneck is causing customer dissatisfaction.
We need urgent help with our outage recovery process.
Our current vendor's support burden is overwhelming our team.
"""


class TestEconomicValueMetrics:
    """Test economic value scoring and deal size estimation."""

    def test_economic_value_calculation(self):
        """Test economic value score calculation."""
        result = LeadScoringService.get_economic_value(ENTERPRISE_LEAD_TEXT)
        
        assert "economic_value_score" in result
        assert 0 <= result["economic_value_score"] <= 100
        assert "economic_indicators" in result
        assert "estimated_deal_size" in result
        assert "roi_potential" in result
        assert result["estimated_deal_size"] in ["small", "medium", "large", "enterprise", "unknown"]

    def test_economic_indicators_extraction(self):
        """Test that economic indicators are properly extracted."""
        result = LeadScoringService.get_economic_value(ENTERPRISE_LEAD_TEXT)
        
        assert len(result["economic_indicators"]) > 0
        assert any("revenue" in indicator for indicator in result["economic_indicators"])

    def test_deal_size_estimation_enterprise(self):
        """Test deal size is correctly identified as enterprise."""
        result = LeadScoringService.get_economic_value(ENTERPRISE_LEAD_TEXT)
        
        assert result["estimated_deal_size"] in ["large", "enterprise"]
        assert result["economic_value_score"] > 50

    def test_deal_size_estimation_startup(self):
        """Test deal size identification for startup."""
        result = LeadScoringService.get_economic_value(STARTUP_LEAD_TEXT)
        
        assert result["estimated_deal_size"] in ["small", "medium"]

    def test_numeric_indicators(self):
        """Test that numeric indicators (budget, revenue) are extracted."""
        result = LeadScoringService.get_economic_value(ENTERPRISE_LEAD_TEXT)
        
        assert "numeric_indicators" in result
        numeric = result["numeric_indicators"]
        assert "large_numbers" in numeric
        assert len(numeric["large_numbers"]) > 0

    def test_roi_potential_calculation(self):
        """Test ROI potential scoring."""
        result = LeadScoringService.get_economic_value(ENTERPRISE_LEAD_TEXT)
        
        assert result["roi_potential"] > 0
        assert 0 <= result["roi_potential"] <= 100


class TestEconomicBuyerMetrics:
    """Test economic buyer identification and authority levels."""

    def test_economic_buyer_identification(self):
        """Test buyer identification."""
        result = LeadScoringService.get_economic_buyer(ENTERPRISE_LEAD_TEXT)
        
        assert "buyer_identified" in result
        assert "buyer_title" in result
        assert "buyer_score" in result
        assert "approval_authority" in result

    def test_buyer_score_high_for_cfo(self):
        """Test that CFO mentions result in high buyer score."""
        result = LeadScoringService.get_economic_buyer(ENTERPRISE_LEAD_TEXT)
        
        assert result["buyer_identified"] is True
        assert result["buyer_score"] > 50

    def test_authority_level_executive(self):
        """Test authority level identification for executive."""
        result = LeadScoringService.get_economic_buyer(ENTERPRISE_LEAD_TEXT)
        
        assert result["approval_authority"] in ["executive", "director", "manager", "contributor"]

    def test_buyer_signals_extraction(self):
        """Test buyer signals are properly extracted."""
        result = LeadScoringService.get_economic_buyer(ENTERPRISE_LEAD_TEXT)
        
        assert len(result["buyer_signals"]) > 0
        assert any("Title" in signal or "Budget" in signal for signal in result["buyer_signals"])

    def test_detected_titles(self):
        """Test that buyer titles are detected."""
        result = LeadScoringService.get_economic_buyer(ENTERPRISE_LEAD_TEXT)
        
        assert "detected_titles" in result
        assert len(result["detected_titles"]) > 0


class TestDecisionCriteriaMetrics:
    """Test decision criteria extraction and categorization."""

    def test_decision_criteria_extraction(self):
        """Test decision criteria are extracted."""
        result = LeadScoringService.get_decision_criteria(ENTERPRISE_LEAD_TEXT)
        
        assert "total_criteria" in result
        assert "must_have_criteria" in result
        assert "nice_to_have_criteria" in result

    def test_must_have_vs_nice_to_have(self):
        """Test categorization into must-have and nice-to-have."""
        result = LeadScoringService.get_decision_criteria(ENTERPRISE_LEAD_TEXT)
        
        assert len(result["must_have_criteria"]) > 0
        assert all(c["weight"] >= 0.9 for c in result["must_have_criteria"])

    def test_criteria_coverage(self):
        """Test criteria coverage scoring."""
        result = LeadScoringService.get_decision_criteria(ENTERPRISE_LEAD_TEXT)
        
        assert "criteria_coverage" in result
        assert 0 <= result["criteria_coverage"] <= 100

    def test_risk_assessment(self):
        """Test risk level assessment based on missing criteria."""
        result = LeadScoringService.get_decision_criteria(ENTERPRISE_LEAD_TEXT)
        
        assert "risk_level" in result
        assert result["risk_level"] in ["high", "medium", "low"]

    def test_missing_critical_criteria(self):
        """Test identification of missing critical criteria."""
        result = LeadScoringService.get_decision_criteria(STARTUP_LEAD_TEXT)
        
        assert "missing_critical_criteria" in result


class TestDecisionProcessMetrics:
    """Test decision process stage analysis and followup recommendations."""

    def test_process_stage_identification(self):
        """Test current process stage identification."""
        result = LeadScoringService.get_decision_process(ENTERPRISE_LEAD_TEXT)
        
        assert "current_stage" in result
        assert result["current_stage"] != ""

    def test_recommended_next_steps(self):
        """Test next steps recommendation."""
        result = LeadScoringService.get_decision_process(ENTERPRISE_LEAD_TEXT)
        
        assert "next_steps" in result
        assert len(result["next_steps"]) > 0

    def test_timeline_urgency(self):
        """Test timeline urgency detection."""
        result = LeadScoringService.get_decision_process(ENTERPRISE_LEAD_TEXT)
        
        assert "timeline_urgency" in result
        assert result["timeline_urgency"] in ["immediate", "short_term", "medium_term", "long_term", "standard"]

    def test_urgency_score(self):
        """Test urgency score is properly calculated."""
        result = LeadScoringService.get_decision_process(ENTERPRISE_LEAD_TEXT)
        
        assert 0 <= result["urgency_score"] <= 1.0

    def test_recommended_followup_action(self):
        """Test recommended followup action is provided."""
        result = LeadScoringService.get_decision_process(ENTERPRISE_LEAD_TEXT)
        
        assert "recommended_followup" in result
        assert result["recommended_followup"] != ""

    def test_high_urgency_detection(self):
        """Test detection of high-urgency leads."""
        high_urgency_text = "We need this ASAP, this week. Urgent."
        result = LeadScoringService.get_decision_process(high_urgency_text)
        
        assert result["timeline_urgency"] in ["immediate", "short_term"]


class TestPainPointMetrics:
    """Test pain point identification and cosine similarity clustering."""

    def test_pain_points_identification(self):
        """Test pain points are identified."""
        result = LeadScoringService.get_pain_points(PAIN_POINT_LEAD_TEXT)
        
        assert "pain_points" in result
        assert len(result["pain_points"]) > 0
        assert result["has_pain"] is True

    def test_pain_point_severity(self):
        """Test pain severity scoring."""
        result = LeadScoringService.get_pain_points(PAIN_POINT_LEAD_TEXT)
        
        assert "pain_severity" in result
        assert 0 <= result["pain_severity"] <= 100

    def test_primary_pain_identification(self):
        """Test primary pain point is identified."""
        result = LeadScoringService.get_pain_points(PAIN_POINT_LEAD_TEXT)
        
        assert "primary_pain" in result
        if result["has_pain"]:
            assert result["primary_pain"] is not None
            assert "text" in result["primary_pain"]
            assert "weight" in result["primary_pain"]

    def test_pain_point_clustering_cosine_similarity(self):
        """Test pain points are clustered using cosine similarity."""
        result = LeadScoringService.get_pain_points(PAIN_POINT_LEAD_TEXT)
        
        assert "pain_clusters" in result
        # Should have at least 1 cluster
        assert len(result["pain_clusters"]) >= 1

    def test_pain_categories(self):
        """Test pain points are categorized."""
        result = LeadScoringService.get_pain_points(PAIN_POINT_LEAD_TEXT)
        
        assert "pain_categories" in result
        categories = result["pain_categories"]
        valid_categories = {"operational", "financial", "technical", "strategic", "compliance"}
        assert any(cat in valid_categories for cat in categories.keys())

    def test_pain_frequency_distribution(self):
        """Test pain frequency distribution is calculated."""
        result = LeadScoringService.get_pain_points(PAIN_POINT_LEAD_TEXT)
        
        assert "pain_frequency_distribution" in result

    def test_no_pain_detected(self):
        """Test proper handling when no pain is detected."""
        result = LeadScoringService.get_pain_points("This is a great product and we're very happy.")
        
        assert result["has_pain"] is False
        assert result["pain_severity"] == 0.0


class TestCosineSimilarityComparison:
    """Test cosine similarity-based lead and pain point comparison."""

    def test_compare_leads_similarity(self):
        """Test comparing two leads using cosine similarity."""
        result = LeadScoringService.compare_leads(ENTERPRISE_LEAD_TEXT, STARTUP_LEAD_TEXT)
        
        assert "similarity_score" in result
        assert 0 <= result["similarity_score"] <= 100
        assert "matching_aspects" in result
        assert "diverging_aspects" in result
        assert "recommendation" in result

    def test_same_segment_leads(self):
        """Test identification of leads in same segment."""
        similar_text_1 = "We need enterprise security and scalability with high performance."
        similar_text_2 = "Enterprise-grade security and high performance are critical requirements."
        
        result = LeadScoringService.compare_leads(similar_text_1, similar_text_2)
        
        assert result["similarity_score"] > 60
        assert result["same_segment"] is True

    def test_different_segment_leads(self):
        """Test identification of leads in different segments."""
        result = LeadScoringService.compare_leads(ENTERPRISE_LEAD_TEXT, STARTUP_LEAD_TEXT)
        
        # These leads should have lower similarity
        assert result["similarity_score"] < 85

    def test_compare_pain_profiles(self):
        """Test comparing pain point profiles between leads."""
        pain_1 = LeadScoringService.get_pain_points(ENTERPRISE_LEAD_TEXT)["pain_points"]
        pain_2 = LeadScoringService.get_pain_points(PAIN_POINT_LEAD_TEXT)["pain_points"]
        
        if pain_1 and pain_2:
            result = LeadScoringService.compare_pain_profiles(pain_1, pain_2)
            
            assert "similarity_score" in result
            assert "matching_pains" in result
            assert "common_pain_percentage" in result
            assert "can_use_same_solution" in result


class TestAdvancedMetricsComprehensiveReport:
    """Test comprehensive advanced metrics report generation."""

    def test_comprehensive_report_generation(self):
        """Test full comprehensive report is generated."""
        result = LeadScoringService.get_advanced_metrics(ENTERPRISE_LEAD_TEXT)
        
        assert "generated_at" in result
        assert "economic_value" in result
        assert "economic_buyer" in result
        assert "decision_criteria" in result
        assert "decision_process" in result
        assert "pain_points" in result
        assert "overall_readiness_score" in result
        assert result["production_ready"] is True

    def test_overall_readiness_score(self):
        """Test overall readiness score is calculated."""
        result = LeadScoringService.get_advanced_metrics(ENTERPRISE_LEAD_TEXT)
        
        assert "overall_readiness_score" in result
        assert 0 <= result["overall_readiness_score"] <= 100

    def test_production_ready_flag(self):
        """Test production_ready flag is set."""
        result = LeadScoringService.get_advanced_metrics(ENTERPRISE_LEAD_TEXT)
        
        assert result["production_ready"] is True

    def test_report_timestamps(self):
        """Test report includes proper timestamps."""
        result = LeadScoringService.get_advanced_metrics(ENTERPRISE_LEAD_TEXT)
        
        assert "generated_at" in result
        # Should be ISO format timestamp
        assert "T" in result["generated_at"]


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_empty_text_handling(self):
        """Test handling of empty text."""
        result = LeadScoringService.get_economic_value("")
        
        assert result["economic_value_score"] == 0.0

    def test_minimal_text_handling(self):
        """Test handling of minimal text."""
        result = LeadScoringService.get_pain_points("problem")
        
        assert "pain_points" in result

    def test_none_history_handling(self):
        """Test handling of None history."""
        result = LeadScoringService.get_economic_buyer(ENTERPRISE_LEAD_TEXT, history=None)
        
        assert "buyer_identified" in result

    def test_empty_history_handling(self):
        """Test handling of empty history."""
        result = LeadScoringService.get_economic_buyer(ENTERPRISE_LEAD_TEXT, history=[])
        
        assert "buyer_identified" in result

    def test_special_characters_handling(self):
        """Test handling of special characters."""
        text_with_special = "We need $2M budget, 50% ROI, & enterprise scale."
        result = LeadScoringService.get_economic_value(text_with_special)
        
        assert result["economic_value_score"] > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

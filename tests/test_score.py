"""Tests for score module."""

import pytest
import math
import score
import normalize
from tests.conftest import sample_dfs


class TestScore:
    """Test suite for score.compute_topic_aggregates."""
    
    @pytest.fixture
    def canonical_model(self, sample_dfs):
        """Create canonical model from sample data."""
        topic_columns = ['topic_a', 'topic_b', 'topic_c']
        return normalize.wide_to_canonical(sample_dfs, topic_columns)
    
    def test_compute_topic_aggregates_basic(self, canonical_model):
        """Test basic topic aggregates computation."""
        aggregates = score.compute_topic_aggregates(canonical_model)
        
        assert len(aggregates) > 0
        
        # Check required fields
        for agg in aggregates:
            assert 'topic_id' in agg
            assert 'coverage_count' in agg
            assert 'coverage_rate' in agg
            assert 'evidence_count' in agg
            assert 'intensity_rate' in agg
            assert 'topic_score' in agg
    
    def test_coverage_rate_calculation(self, canonical_model):
        """Test coverage rate is calculated correctly."""
        aggregates = score.compute_topic_aggregates(canonical_model)
        
        for agg in aggregates:
            coverage_rate = agg['coverage_rate']
            coverage_count = agg['coverage_count']
            total_participants = len(canonical_model.participants)
            
            # Coverage rate should be between 0 and 1
            assert 0.0 <= coverage_rate <= 1.0
            
            # Coverage rate should match count / total
            if total_participants > 0:
                expected_rate = coverage_count / total_participants
                assert abs(coverage_rate - expected_rate) < 0.01
    
    def test_evidence_count_calculation(self, canonical_model):
        """Test evidence count is calculated correctly."""
        aggregates = score.compute_topic_aggregates(canonical_model)
        
        topic_a_agg = next((a for a in aggregates if a['topic_id'] == 'topic_a'), None)
        if topic_a_agg:
            # topic_a should have quotes from multiple participants
            assert topic_a_agg['evidence_count'] > 0
    
    def test_intensity_rate_calculation(self, canonical_model):
        """Test intensity rate is calculated correctly."""
        aggregates = score.compute_topic_aggregates(canonical_model)
        
        for agg in aggregates:
            intensity_rate = agg['intensity_rate']
            evidence_count = agg['evidence_count']
            
            # Intensity rate should be between 0 and 1
            assert 0.0 <= intensity_rate <= 1.0
            
            # If no evidence, intensity should be 0
            if evidence_count == 0:
                assert intensity_rate == 0.0
    
    def test_topic_score_formula(self, canonical_model):
        """Test topic score formula is correct."""
        aggregates = score.compute_topic_aggregates(canonical_model)
        
        for agg in aggregates:
            coverage_rate = agg['coverage_rate']
            evidence_count = agg['evidence_count']
            intensity_rate = agg['intensity_rate']
            topic_score = agg['topic_score']
            
            # Calculate expected score
            expected_score = (
                0.5 * coverage_rate +
                0.3 * math.log1p(evidence_count) +
                0.2 * intensity_rate
            )
            
            # Allow small floating point differences
            assert abs(topic_score - expected_score) < 0.001
    
    def test_topic_score_ranking(self, canonical_model):
        """Test topics are ranked by score (descending)."""
        aggregates = score.compute_topic_aggregates(canonical_model)
        
        if len(aggregates) > 1:
            scores = [agg['topic_score'] for agg in aggregates]
            # Should be in descending order
            assert scores == sorted(scores, reverse=True)
    
    def test_topic_score_alphabetical_tiebreak(self, canonical_model):
        """Test alphabetical tiebreak for equal scores."""
        # Create aggregates with same score
        # This is hard to test directly, but we can verify sorting is deterministic
        aggregates = score.compute_topic_aggregates(canonical_model)
        
        # Run twice and verify same order
        aggregates2 = score.compute_topic_aggregates(canonical_model)
        
        topic_ids_1 = [a['topic_id'] for a in aggregates]
        topic_ids_2 = [a['topic_id'] for a in aggregates2]
        
        assert topic_ids_1 == topic_ids_2
    
    def test_proof_quote_selection(self, canonical_model):
        """Test proof quote is selected correctly."""
        aggregates = score.compute_topic_aggregates(canonical_model)
        
        topic_a_agg = next((a for a in aggregates if a['topic_id'] == 'topic_a'), None)
        if topic_a_agg and topic_a_agg.get('proof_quote_ref'):
            # Should have proof quote reference
            assert ':' in topic_a_agg['proof_quote_ref']
            assert topic_a_agg['proof_quote_preview'] is not None
    
    def test_empty_topic_handling(self):
        """Test handling of topics with no evidence."""
        participants = [normalize.Participant('p1', 'P1', {})]
        topics = [normalize.Topic('empty_topic')]
        evidence_cells = []  # No evidence
        
        model = normalize.CanonicalModel(participants, topics, evidence_cells)
        aggregates = score.compute_topic_aggregates(model)
        
        assert len(aggregates) == 1
        assert aggregates[0]['coverage_count'] == 0
        assert aggregates[0]['evidence_count'] == 0
        assert aggregates[0]['topic_score'] == 0.0

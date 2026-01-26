"""Tests for digest module."""

import pytest
import digest
import normalize
import score
from tests.conftest import sample_dfs


class TestDigest:
    """Test suite for digest.build_digest."""
    
    @pytest.fixture
    def canonical_model(self, sample_dfs):
        """Create canonical model from sample data."""
        topic_columns = ['topic_a', 'topic_b', 'topic_c']
        return normalize.wide_to_canonical(sample_dfs, topic_columns)
    
    @pytest.fixture
    def topic_aggregates(self, canonical_model):
        """Create topic aggregates."""
        return score.compute_topic_aggregates(canonical_model)
    
    def test_build_digest_structure(self, canonical_model, topic_aggregates):
        """Test digest structure is correct."""
        digest_artifact = digest.build_digest(canonical_model, topic_aggregates, n_takeaways=3)
        
        assert 'takeaways' in digest_artifact
        assert 'topic_cards' in digest_artifact
        assert 'metadata' in digest_artifact
    
    def test_build_digest_takeaways(self, canonical_model, topic_aggregates):
        """Test takeaways are generated correctly."""
        digest_artifact = digest.build_digest(canonical_model, topic_aggregates, n_takeaways=3)
        
        takeaways = digest_artifact['takeaways']
        assert len(takeaways) <= 3
        
        for takeaway in takeaways:
            assert 'takeaway_index' in takeaway
            assert 'takeaway_text' in takeaway
            assert 'source_topic_id' in takeaway
    
    def test_build_digest_topic_cards(self, canonical_model, topic_aggregates):
        """Test topic cards are created correctly."""
        digest_artifact = digest.build_digest(canonical_model, topic_aggregates)
        
        topic_cards = digest_artifact['topic_cards']
        assert len(topic_cards) > 0
        
        for card in topic_cards:
            assert 'topic_id' in card
            assert 'topic_one_liner' in card
            assert 'coverage_rate' in card
            assert 'evidence_count' in card
            assert 'sentiment_mix' in card
            assert 'proof_quote_preview' in card
            assert 'receipt_links' in card
    
    def test_build_digest_sentiment_mix(self, canonical_model, topic_aggregates):
        """Test sentiment mix is computed correctly."""
        digest_artifact = digest.build_digest(canonical_model, topic_aggregates)
        
        topic_cards = digest_artifact['topic_cards']
        for card in topic_cards:
            sentiment_mix = card['sentiment_mix']
            assert isinstance(sentiment_mix, dict)
            assert 'positive' in sentiment_mix
            assert 'negative' in sentiment_mix
            assert 'neutral' in sentiment_mix
            assert 'mixed' in sentiment_mix
            assert 'unknown' in sentiment_mix
    
    def test_build_digest_receipt_links(self, canonical_model, topic_aggregates):
        """Test receipt links are generated correctly."""
        digest_artifact = digest.build_digest(canonical_model, topic_aggregates)
        
        topic_cards = digest_artifact['topic_cards']
        for card in topic_cards:
            receipt_links = card['receipt_links']
            assert isinstance(receipt_links, list)
            # Receipt links should be in format "participant_id:quote_index"
            for link in receipt_links:
                assert ':' in link
    
    def test_build_digest_metadata(self, canonical_model, topic_aggregates):
        """Test metadata is included."""
        digest_artifact = digest.build_digest(canonical_model, topic_aggregates)
        
        metadata = digest_artifact['metadata']
        assert 'n_takeaways' in metadata
        assert 'n_topics' in metadata
        assert 'total_participants' in metadata
        assert 'total_evidence_cells' in metadata
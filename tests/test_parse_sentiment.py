"""Tests for parse_sentiment module."""

import pytest
import parse_sentiment


class TestParseSentiment:
    """Test suite for parse_sentiment.parse_and_align_sentiments."""
    
    def test_parse_numbered_sentiments_aligned(self):
        """Test numbered sentiments aligned to numbered quotes."""
        quotes = [
            {'quote_index': 1, 'quote_text': 'First quote'},
            {'quote_index': 2, 'quote_text': 'Second quote'}
        ]
        sentiments = "1: positive; 2: negative"
        blocks = parse_sentiment.parse_and_align_sentiments(sentiments, quotes)
        
        assert len(blocks) == 2
        assert blocks[0]['quote_index'] == 1
        assert blocks[0]['tone_rollup'] == 'positive'
        assert blocks[1]['quote_index'] == 2
        assert blocks[1]['tone_rollup'] == 'negative'
        assert blocks[0]['alignment_confidence'] > 0.5
    
    def test_parse_flat_sentiments(self):
        """Test flat sentiment list parsing."""
        quotes = [
            {'quote_index': 0, 'quote_text': 'First'},
            {'quote_index': 1, 'quote_text': 'Second'}
        ]
        sentiments = "positive, negative"
        blocks = parse_sentiment.parse_and_align_sentiments(sentiments, quotes)
        
        assert len(blocks) >= 1
    
    def test_parse_sentiments_mismatched_counts(self):
        """Test handling of mismatched sentiment/quote counts."""
        quotes = [
            {'quote_index': 1, 'quote_text': 'First'},
            {'quote_index': 2, 'quote_text': 'Second'},
            {'quote_index': 3, 'quote_text': 'Third'}
        ]
        sentiments = "1: positive; 2: negative"  # Missing 3
        blocks = parse_sentiment.parse_and_align_sentiments(sentiments, quotes)
        
        assert len(blocks) == 3
        # Third should have unknown tone
        block_3 = next((b for b in blocks if b['quote_index'] == 3), None)
        if block_3:
            assert block_3['tone_rollup'] == 'unknown'
    
    def test_parse_sentiments_without_quotes(self):
        """Test sentiments without corresponding quotes."""
        quotes = []  # No quotes
        sentiments = "1: positive; 2: negative"
        blocks = parse_sentiment.parse_and_align_sentiments(sentiments, quotes)
        
        # Should handle gracefully
        assert isinstance(blocks, list)
    
    def test_tone_rollup_positive(self):
        """Test tone rollup for positive sentiment."""
        from parse_sentiment import _compute_tone_rollup
        assert _compute_tone_rollup(['positive', 'good']) == 'positive'
        assert _compute_tone_rollup(['positive']) == 'positive'
    
    def test_tone_rollup_negative(self):
        """Test tone rollup for negative sentiment."""
        from parse_sentiment import _compute_tone_rollup
        assert _compute_tone_rollup(['negative', 'bad']) == 'negative'
    
    def test_tone_rollup_mixed(self):
        """Test tone rollup for mixed sentiment."""
        from parse_sentiment import _compute_tone_rollup
        assert _compute_tone_rollup(['positive', 'negative']) == 'mixed'
    
    def test_tone_rollup_neutral(self):
        """Test tone rollup for neutral sentiment."""
        from parse_sentiment import _compute_tone_rollup
        assert _compute_tone_rollup(['neutral']) == 'neutral'
    
    def test_tone_rollup_unknown(self):
        """Test tone rollup for unknown sentiment."""
        from parse_sentiment import _compute_tone_rollup
        assert _compute_tone_rollup([]) == 'unknown'
        assert _compute_tone_rollup(['unknown_label']) == 'unknown'

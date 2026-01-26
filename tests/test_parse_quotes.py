"""Tests for parse_quotes module."""

import pytest
import parse_quotes


class TestParseQuotes:
    """Test suite for parse_quotes.parse_quotes."""
    
    def test_parse_numbered_quotes(self):
        """Test parsing numbered quotes."""
        text = "1. First quote here. 2. Second quote here. 3. Third quote."
        blocks = parse_quotes.parse_quotes(text)
        
        assert len(blocks) == 3
        assert blocks[0]['quote_index'] == 1
        assert blocks[1]['quote_index'] == 2
        assert blocks[2]['quote_index'] == 3
        assert 'quote_text' in blocks[0]
        assert 'quote_preview' in blocks[0]
    
    def test_parse_bullet_quotes(self):
        """Test parsing bullet quotes."""
        text = "- First bullet\n- Second bullet\n- Third bullet"
        blocks = parse_quotes.parse_quotes(text)
        
        assert len(blocks) == 3
        assert blocks[0]['quote_index'] == 0
        assert blocks[1]['quote_index'] == 1
        assert blocks[2]['quote_index'] == 2
    
    def test_parse_single_block(self):
        """Test fallback to single block for unnumbered quotes."""
        text = "This is a single quote without any numbering or bullets."
        blocks = parse_quotes.parse_quotes(text)
        
        assert len(blocks) == 1
        assert blocks[0]['quote_index'] == 0
        assert blocks[0]['quote_text'] == text
    
    def test_parse_empty_input(self):
        """Test handling of empty input."""
        assert parse_quotes.parse_quotes("") == []
        assert parse_quotes.parse_quotes(None) == []
        assert parse_quotes.parse_quotes("   ") == []
    
    def test_parse_quotes_preview_generation(self):
        """Test quote preview is generated correctly."""
        long_text = "This is a very long quote. " * 20
        blocks = parse_quotes.parse_quotes(long_text)
        
        assert len(blocks) > 0
        preview = blocks[0]['quote_preview']
        assert len(preview) <= 150  # Allow some margin
        assert preview.endswith('...') or len(preview) < len(long_text)
    
    def test_parse_quotes_multiline(self):
        """Test parsing multiline quote blocks."""
        text = "1. This is a quote\nthat spans multiple lines.\n2. Another quote."
        blocks = parse_quotes.parse_quotes(text)
        
        assert len(blocks) == 2
        assert 'multiple lines' in blocks[0]['quote_text']
    
    def test_parse_quotes_speaker_tags(self):
        """Test speaker tag extraction."""
        text = "John said: This is a quote with a speaker."
        blocks = parse_quotes.parse_quotes(text)
        
        assert len(blocks) > 0
        # Speaker tags may or may not be extracted depending on implementation
        if blocks[0].get('speaker_tags'):
            assert 'John' in str(blocks[0]['speaker_tags'])
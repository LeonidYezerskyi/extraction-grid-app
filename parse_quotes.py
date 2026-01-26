"""Module for parsing and extracting quotes from text data."""

import re
from typing import List, Dict, Optional, Any


def _generate_preview(text: str, max_length: int = 120) -> str:
    """
    Generate a preview of text, cutting at sentence boundary if possible.
    
    Args:
        text: Text to generate preview from
        max_length: Maximum length of preview (default 120)
    
    Returns:
        Preview string, cut at sentence boundary if possible, otherwise at max_length
    """
    if not text or len(text) <= max_length:
        return text
    
    # Try to find sentence boundary within reasonable distance
    # Look for sentence endings: . ! ? followed by space or end
    sentence_end_pattern = r'[.!?]\s+'
    matches = list(re.finditer(sentence_end_pattern, text[:max_length + 50]))
    
    if matches:
        # Find the last sentence boundary before or near max_length
        for match in reversed(matches):
            if match.end() <= max_length + 20:  # Allow some flexibility
                return text[:match.end()].strip()
    
    # Fallback: cut at word boundary near max_length
    if max_length < len(text):
        # Find last space before max_length
        last_space = text.rfind(' ', 0, max_length)
        if last_space > max_length * 0.7:  # Only use if not too short
            return text[:last_space].strip() + '...'
        else:
            return text[:max_length].strip() + '...'
    
    return text[:max_length].strip()


def _extract_speaker_tags(text: str) -> Optional[List[str]]:
    """
    Extract speaker tags from quote text (e.g., "John said:", "According to Mary:").
    
    Args:
        text: Quote text to analyze
    
    Returns:
        List of speaker tags if found, None otherwise
    """
    # Common patterns for speaker attribution
    patterns = [
        r'^([^:]+?)\s*(?:said|says|stated|notes|writes|explains|adds|continues|concludes)[:\-]',
        r'^According to\s+([^:]+?)[:\-]',
        r'^([^:]+?)\s*(?:according to|as quoted by)',
    ]
    
    speaker_tags = []
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            speaker = match.group(1).strip()
            if speaker and len(speaker) < 100:  # Reasonable speaker name length
                speaker_tags.append(speaker)
    
    return speaker_tags if speaker_tags else None


def _parse_numbered_blocks(text: str) -> Optional[List[Dict[str, Any]]]:
    """
    Parse text into numbered blocks (1., 2., etc.).
    
    Args:
        text: Text to parse
    
    Returns:
        List of quote blocks if numbered pattern found, None otherwise
    """
    if not text or not text.strip():
        return None
    
    # Pattern for numbered blocks: number followed by period/dot and space
    # Supports: 1. 2. 3. or (1) (2) (3) or 1) 2) 3)
    pattern_configs = [
        (r'\b(\d+)\.\s+', r'\b\d+\.\s+'),  # 1. 2. 3.
        (r'\((\d+)\)\s+', r'\(\d+\)\s+'),  # (1) (2) (3)
        (r'\b(\d+)\)\s+', r'\b\d+\)\s+'),  # 1) 2) 3)
    ]
    
    # Try each pattern to see which one matches
    for pattern, split_pattern in pattern_configs:
        # Find all matches in the text
        matches = list(re.finditer(pattern, text))
        
        if len(matches) >= 2:  # Need at least 2 matches to be considered numbered
            blocks = []
            
            # Extract blocks using the matches
            for i, match in enumerate(matches):
                quote_index = int(match.group(1))
                start_pos = match.end()
                
                # Find end position (start of next match or end of text)
                if i + 1 < len(matches):
                    end_pos = matches[i + 1].start()
                else:
                    end_pos = len(text)
                
                quote_text = text[start_pos:end_pos].strip()
                
                if quote_text:
                    blocks.append({
                        'quote_index': quote_index,
                        'quote_text': quote_text,
                        'quote_preview': _generate_preview(quote_text),
                        'speaker_tags': _extract_speaker_tags(quote_text)
                    })
            
            if len(blocks) >= 2:
                return blocks
    
    # Fallback: try line-by-line parsing for multiline content
    lines = text.split('\n')
    blocks = []
    current_block = None
    current_index = None
    
    for line in lines:
        line = line.strip()
        if not line:
            if current_block is not None:
                current_block += ' '
            continue
        
        # Try to match numbered pattern at start of line
        matched = False
        for pattern, _ in pattern_configs:
            # Remove \b anchor for line start matching
            line_pattern = pattern.replace(r'\b', '')
            match = re.match(r'^\s*' + line_pattern, line)
            if match:
                # Save previous block if exists
                if current_block is not None:
                    quote_text = current_block.strip()
                    blocks.append({
                        'quote_index': current_index,
                        'quote_text': quote_text,
                        'quote_preview': _generate_preview(quote_text),
                        'speaker_tags': _extract_speaker_tags(quote_text)
                    })
                
                # Start new block
                current_index = int(match.group(1))
                current_block = re.sub(r'^\s*' + line_pattern, '', line)
                matched = True
                break
        
        if not matched:
            # Continuation of current block
            if current_block is not None:
                current_block += ' ' + line
            else:
                # No numbered pattern found at start, return None
                return None
    
    # Add last block
    if current_block is not None:
        quote_text = current_block.strip()
        blocks.append({
            'quote_index': current_index,
            'quote_text': quote_text,
            'quote_preview': _generate_preview(quote_text),
            'speaker_tags': _extract_speaker_tags(quote_text)
        })
    
    # Only return if we found at least 2 blocks (to distinguish from single block)
    if len(blocks) >= 2:
        return blocks
    
    return None


def _parse_bullet_blocks(text: str) -> Optional[List[Dict[str, Any]]]:
    """
    Parse text into bullet blocks (-, •, *, etc.).
    
    Args:
        text: Text to parse
    
    Returns:
        List of quote blocks if bullet pattern found, None otherwise
    """
    if not text or not text.strip():
        return None
    
    # Pattern for bullet blocks: bullet character followed by space
    bullet_patterns = [
        r'^\s*[-•*]\s+',  # - • *
        r'^\s*[oO]\s+',  # o O (alternative bullet)
    ]
    
    blocks = []
    lines = text.split('\n')
    current_block = None
    quote_index = 0
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
        
        # Try to match bullet pattern
        matched = False
        for pattern in bullet_patterns:
            match = re.match(pattern, line)
            if match:
                # Save previous block if exists
                if current_block is not None:
                    quote_text = current_block.strip()
                    blocks.append({
                        'quote_index': quote_index,
                        'quote_text': quote_text,
                        'quote_preview': _generate_preview(quote_text),
                        'speaker_tags': _extract_speaker_tags(quote_text)
                    })
                    quote_index += 1
                
                # Start new block
                current_block = re.sub(pattern, '', line)
                matched = True
                break
        
        if not matched:
            # Continuation of current block
            if current_block is not None:
                current_block += ' ' + line
            else:
                # No bullet pattern found, return None to try other methods
                return None
    
    # Add last block
    if current_block is not None:
        quote_text = current_block.strip()
        blocks.append({
            'quote_index': quote_index,
            'quote_text': quote_text,
            'quote_preview': _generate_preview(quote_text),
            'speaker_tags': _extract_speaker_tags(quote_text)
        })
        quote_index += 1
    
    # Only return if we found at least 2 blocks (to distinguish from single block)
    if len(blocks) >= 2:
        return blocks
    
    return None


def parse_quotes(quotes_raw: str) -> List[Dict[str, Any]]:
    """
    Parse quotes from raw text into structured blocks.
    
    Recognizes numbered blocks (1., 2., etc.), bullet blocks (-, •), or falls back
    to a single block. Maintains stable ordering and generates sensible previews.
    
    Args:
        quotes_raw: Raw text containing quotes (may be None or empty)
    
    Returns:
        List of quote blocks, each containing:
        - 'quote_index': int - Index of the quote (0-based for bullets, number-based for numbered)
        - 'quote_text': str - Full text of the quote
        - 'quote_preview': str - Preview text (~120 chars, cut at sentence boundary if possible)
        - 'speaker_tags': Optional[List[str]] - Extracted speaker tags if found
    
    Examples:
        >>> quotes = "1. First quote. 2. Second quote."
        >>> blocks = parse_quotes(quotes)
        >>> len(blocks)
        2
        
        >>> quotes = "- First bullet\\n- Second bullet"
        >>> blocks = parse_quotes(quotes)
        >>> blocks[0]['quote_index']
        0
        
        >>> quotes = "Single quote without numbering"
        >>> blocks = parse_quotes(quotes)
        >>> len(blocks)
        1
    """
    if not quotes_raw or not quotes_raw.strip():
        return []
    
    text = quotes_raw.strip()
    
    # Try numbered blocks first
    blocks = _parse_numbered_blocks(text)
    if blocks is not None:
        return blocks
    
    # Try bullet blocks
    blocks = _parse_bullet_blocks(text)
    if blocks is not None:
        return blocks
    
    # Fallback to single block
    quote_text = text.strip()
    return [{
        'quote_index': 0,
        'quote_text': quote_text,
        'quote_preview': _generate_preview(quote_text),
        'speaker_tags': _extract_speaker_tags(quote_text)
    }]


# Unit tests
if __name__ == '__main__':
    import unittest
    
    class TestParseQuotes(unittest.TestCase):
        
        def test_numbered_blocks(self):
            """Test parsing numbered blocks."""
            text = "1. First quote here. 2. Second quote here. 3. Third quote."
            blocks = parse_quotes(text)
            self.assertEqual(len(blocks), 3)
            self.assertEqual(blocks[0]['quote_index'], 1)
            self.assertEqual(blocks[1]['quote_index'], 2)
            self.assertEqual(blocks[2]['quote_index'], 3)
            self.assertIn('quote_text', blocks[0])
            self.assertIn('quote_preview', blocks[0])
        
        def test_numbered_blocks_parentheses(self):
            """Test parsing numbered blocks with parentheses."""
            text = "(1) First quote. (2) Second quote."
            blocks = parse_quotes(text)
            self.assertEqual(len(blocks), 2)
            self.assertEqual(blocks[0]['quote_index'], 1)
        
        def test_bullet_blocks(self):
            """Test parsing bullet blocks."""
            text = "- First bullet point\n- Second bullet point\n- Third bullet"
            blocks = parse_quotes(text)
            self.assertEqual(len(blocks), 3)
            self.assertEqual(blocks[0]['quote_index'], 0)
            self.assertEqual(blocks[1]['quote_index'], 1)
            self.assertEqual(blocks[2]['quote_index'], 2)
        
        def test_bullet_blocks_unicode(self):
            """Test parsing unicode bullet blocks."""
            text = "• First point\n• Second point"
            blocks = parse_quotes(text)
            self.assertEqual(len(blocks), 2)
        
        def test_single_block(self):
            """Test fallback to single block."""
            text = "This is a single quote without any numbering or bullets."
            blocks = parse_quotes(text)
            self.assertEqual(len(blocks), 1)
            self.assertEqual(blocks[0]['quote_index'], 0)
            self.assertEqual(blocks[0]['quote_text'], text)
        
        def test_empty_input(self):
            """Test empty input."""
            self.assertEqual(parse_quotes(""), [])
            self.assertEqual(parse_quotes(None), [])
            self.assertEqual(parse_quotes("   "), [])
        
        def test_preview_generation(self):
            """Test preview generation."""
            long_text = "This is a very long quote. " * 10
            blocks = parse_quotes(long_text)
            self.assertLessEqual(len(blocks[0]['quote_preview']), 150)  # Allow some margin
        
        def test_preview_sentence_boundary(self):
            """Test preview cuts at sentence boundary."""
            text = "First sentence. Second sentence. " + "Third sentence. " * 20
            blocks = parse_quotes(text)
            preview = blocks[0]['quote_preview']
            # Should contain at least one sentence
            self.assertIn('.', preview)
        
        def test_speaker_tags(self):
            """Test speaker tag extraction."""
            text = "John said: This is a quote with a speaker."
            blocks = parse_quotes(text)
            self.assertIsNotNone(blocks[0]['speaker_tags'])
            self.assertIn('John', blocks[0]['speaker_tags'][0])
        
        def test_multiline_blocks(self):
            """Test multiline quote blocks."""
            text = "1. This is a quote\nthat spans multiple lines.\n2. Another quote."
            blocks = parse_quotes(text)
            self.assertEqual(len(blocks), 2)
            self.assertIn('multiple lines', blocks[0]['quote_text'])
        
        def test_mixed_content(self):
            """Test that numbered blocks take precedence."""
            text = "1. Numbered quote. - Bullet quote."
            blocks = parse_quotes(text)
            # Should parse as numbered, not bullet
            self.assertEqual(blocks[0]['quote_index'], 1)
    
    unittest.main()
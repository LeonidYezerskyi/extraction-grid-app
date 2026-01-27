"""Module for parsing and analyzing sentiment from text data."""

import re
from typing import List, Dict, Optional, Any, Tuple


# Sentiment label mappings - expanded to handle more variations
POSITIVE_LABELS = {
    'positive', 'pos', '+', 'good', 'favorable', 'optimistic', 'happy', 'satisfied',
    'p', 'pos.', 'positive.', 'good.', 'favorable.', 'optimistic.', 'happy.', 'satisfied.',
    'yes', 'y', 'agree', 'agreement', 'support', 'supportive', 'pro', 'liked', 'like',
    'excellent', 'great', 'wonderful', 'amazing', 'fantastic', 'love', 'loved'
}
NEGATIVE_LABELS = {
    'negative', 'neg', '-', 'bad', 'unfavorable', 'pessimistic', 'sad', 'dissatisfied',
    'n', 'neg.', 'negative.', 'bad.', 'unfavorable.', 'pessimistic.', 'sad.', 'dissatisfied.',
    'no', 'disagree', 'disagreement', 'against', 'oppose', 'opposed', 'con', 'disliked', 'dislike',
    'poor', 'terrible', 'awful', 'hate', 'hated', 'frustrated', 'angry', 'disappointed'
}
NEUTRAL_LABELS = {
    'neutral', 'neut', '0', 'none', 'indifferent', 'mixed-neutral',
    'n/a', 'na', 'n.a.', 'not applicable', 'not available', 'unknown', 'unclear',
    'mixed', 'both', 'ambivalent', 'uncertain', 'unsure', 'maybe', 'perhaps'
}


def _normalize_label(label: str) -> str:
    """
    Normalize a sentiment label to lowercase and strip whitespace.
    
    Args:
        label: Raw sentiment label
    
    Returns:
        Normalized label string
    """
    return label.lower().strip()


def _classify_label(label: str) -> Optional[str]:
    """
    Classify a sentiment label as positive, negative, or neutral.
    
    Args:
        label: Normalized sentiment label
    
    Returns:
        'positive', 'negative', 'neutral', or None if unknown
    """
    normalized = _normalize_label(label)
    
    if normalized in POSITIVE_LABELS:
        return 'positive'
    elif normalized in NEGATIVE_LABELS:
        return 'negative'
    elif normalized in NEUTRAL_LABELS:
        return 'neutral'
    else:
        # Check if label contains positive/negative keywords (more comprehensive)
        positive_keywords = ['positive', 'pos', 'good', 'favorable', 'optimistic', 'happy', 'satisfied', 
                            'yes', 'agree', 'support', 'pro', 'like', 'love', 'excellent', 'great', 'wonderful']
        negative_keywords = ['negative', 'neg', 'bad', 'unfavorable', 'pessimistic', 'sad', 'dissatisfied',
                            'no', 'disagree', 'against', 'oppose', 'con', 'dislike', 'hate', 'poor', 'terrible', 'awful']
        neutral_keywords = ['neutral', 'neut', 'none', 'n/a', 'na', 'unknown', 'unclear', 'mixed', 'both', 'ambivalent']
        
        # Check for positive keywords
        if any(keyword in normalized for keyword in positive_keywords):
            return 'positive'
        # Check for negative keywords
        elif any(keyword in normalized for keyword in negative_keywords):
            return 'negative'
        # Check for neutral keywords
        elif any(keyword in normalized for keyword in neutral_keywords):
            return 'neutral'
    
    return None


def _parse_numbered_sentiments(text: str) -> Optional[Dict[int, List[str]]]:
    """
    Parse numbered sentiments from text (e.g., "1: positive; 2: negative").
    
    Supports multiple formats:
    - 1: positive, 2: negative
    - 1. positive; 2. negative
    - (1) positive, (2) negative
    - 1) positive; 2) negative
    - Quote 1: positive; Quote 2: negative
    - Q1: positive; Q2: negative
    
    Args:
        text: Text containing numbered sentiments
    
    Returns:
        Dictionary mapping quote_index to list of labels, or None if no numbered pattern found
    """
    if not text or not text.strip():
        return None
    
    # Patterns for numbered sentiments (more flexible):
    # 1: positive, 2: negative
    # 1. positive, 2. negative
    # (1) positive, (2) negative
    # 1) positive, 2) negative
    # Quote 1: positive; Quote 2: negative
    # Q1: positive; Q2: negative
    patterns = [
        (r'(?:quote\s*|q\s*)?(\d+)[:\.]\s*([^;,\n]+)', r'(?:quote\s*|q\s*)?(\d+)[:\.]\s*'),  # 1: label, Q1: label, Quote 1: label
        (r'\((\d+)\)\s*([^;,\n]+)', r'\((\d+)\)\s*'),  # (1) label
        (r'(\d+)\)\s*([^;,\n]+)', r'(\d+)\)\s*'),  # 1) label
    ]
    
    numbered_sentiments = {}
    
    for pattern, split_pattern in patterns:
        matches = list(re.finditer(pattern, text, re.IGNORECASE))
        
        if len(matches) >= 1:
            for match in matches:
                quote_index = int(match.group(1))
                label_text = match.group(2).strip()
                
                # Parse labels (may be comma-separated, semicolon-separated, or space-separated)
                # First try splitting by semicolon, then comma, then newline
                if ';' in label_text:
                    labels = re.split(r'[;]', label_text)
                elif ',' in label_text:
                    labels = re.split(r'[,]', label_text)
                elif '\n' in label_text:
                    labels = re.split(r'[\n]', label_text)
                else:
                    # Single label, but check if it contains multiple words that might be separate labels
                    labels = [label_text]
                
                # Normalize and clean labels
                cleaned_labels = []
                for label in labels:
                    normalized = _normalize_label(label)
                    if normalized and normalized not in ['and', 'or', '&']:  # Skip common connectors
                        cleaned_labels.append(normalized)
                
                if cleaned_labels:
                    if quote_index not in numbered_sentiments:
                        numbered_sentiments[quote_index] = []
                    numbered_sentiments[quote_index].extend(cleaned_labels)
            
            if numbered_sentiments:
                return numbered_sentiments
    
    return None


def _parse_flat_sentiments(text: str) -> List[str]:
    """
    Parse flat list of sentiments (comma or semicolon separated, or space-separated).
    
    Also handles single words and simple lists without delimiters.
    
    Args:
        text: Text containing flat sentiment list
    
    Returns:
        List of normalized sentiment labels
    """
    if not text or not text.strip():
        return []
    
    text = text.strip()
    
    # First try splitting by common delimiters (semicolon, comma, newline)
    if ';' in text or ',' in text or '\n' in text:
        labels = re.split(r'[;,\n]', text)
        labels = [_normalize_label(l) for l in labels if l.strip()]
        return labels
    
    # If no delimiters, try splitting by spaces (but be careful with multi-word labels)
    # Check if it's a single word that might be a sentiment
    words = text.split()
    if len(words) == 1:
        # Single word - treat as one label
        normalized = _normalize_label(words[0])
        return [normalized] if normalized else []
    
    # Multiple words - check if they're all potential sentiment words
    # If so, treat each as a separate label
    potential_labels = []
    for word in words:
        normalized = _normalize_label(word)
        if normalized:
            # Check if it looks like a sentiment word
            if _classify_label(normalized) is not None:
                potential_labels.append(normalized)
            elif len(normalized) <= 15:  # Reasonable length for a sentiment label
                potential_labels.append(normalized)
    
    if potential_labels:
        return potential_labels
    
    # Fallback: treat entire text as one label
    normalized = _normalize_label(text)
    return [normalized] if normalized else []


def _compute_tone_rollup(labels: List[str]) -> str:
    """
    Compute tone rollup from a list of sentiment labels.
    
    Rules:
    - positive: if any positive and none negative
    - negative: if any negative and none positive
    - mixed: if both positive and negative
    - neutral: if all neutral
    - unknown: if no labels or all unknown
    
    Args:
        labels: List of sentiment labels
    
    Returns:
        Tone rollup: 'positive', 'negative', 'neutral', 'mixed', or 'unknown'
    """
    if not labels:
        return 'unknown'
    
    classifications = []
    for label in labels:
        classification = _classify_label(label)
        if classification:
            classifications.append(classification)
    
    if not classifications:
        return 'unknown'
    
    has_positive = 'positive' in classifications
    has_negative = 'negative' in classifications
    has_neutral = 'neutral' in classifications
    
    if has_positive and has_negative:
        return 'mixed'
    elif has_positive and not has_negative:
        return 'positive'
    elif has_negative and not has_positive:
        return 'negative'
    elif has_neutral and not has_positive and not has_negative:
        return 'neutral'
    else:
        return 'unknown'


def _align_sentiments_to_quotes(
    sentiments: Dict[int, List[str]],
    quote_blocks: List[Dict[str, Any]]
) -> Tuple[List[Dict[str, Any]], float]:
    """
    Align numbered sentiments to quote blocks by index.
    
    Args:
        sentiments: Dictionary mapping quote_index to labels
        quote_blocks: List of quote blocks with quote_index
    
    Returns:
        Tuple of (aligned sentiment_blocks, alignment_confidence)
    """
    sentiment_blocks = []
    matched_count = 0
    total_quotes = len(quote_blocks)
    
    if total_quotes == 0:
        return [], 0.0
    
    # Create a map of quote_index to quote blocks
    quote_map = {block['quote_index']: block for block in quote_blocks}
    
    # Check for count mismatch
    total_sentiments = len(sentiments)
    count_mismatch = total_sentiments != total_quotes
    
    # Align sentiments to quotes
    for quote_index, labels in sentiments.items():
        if quote_index in quote_map:
            tone_rollup = _compute_tone_rollup(labels)
            # Use lower confidence if counts mismatch
            confidence = 0.3 if count_mismatch else 1.0
            sentiment_blocks.append({
                'quote_index': quote_index,
                'labels': labels,
                'tone_rollup': tone_rollup,
                'alignment_confidence': confidence
            })
            matched_count += 1
        else:
            # Sentiment exists but no matching quote - skip or handle separately
            # This shouldn't happen in normal flow, but handle gracefully
            pass
    
    # Add sentiment blocks for quotes that don't have sentiments
    quote_indices_with_sentiments = {block['quote_index'] for block in sentiment_blocks}
    for quote_block in quote_blocks:
        if quote_block['quote_index'] not in quote_indices_with_sentiments:
            sentiment_blocks.append({
                'quote_index': quote_block['quote_index'],
                'labels': [],
                'tone_rollup': 'unknown',
                'alignment_confidence': 0.3 if count_mismatch else 0.5
            })
    
    # Calculate alignment confidence
    if count_mismatch:
        alignment_confidence = 0.3  # Low confidence for mismatched counts
    elif total_quotes > 0:
        alignment_confidence = matched_count / total_quotes
    else:
        alignment_confidence = 1.0 if matched_count > 0 else 0.0
    
    return sentiment_blocks, alignment_confidence


def parse_and_align_sentiments(
    sentiments_raw: str,
    quote_blocks: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """
    Parse sentiments from raw text and align them to quote blocks.
    
    Supports numbered sentiments (e.g., "1: positive; 2: negative") and flat lists.
    Alignment rules:
    - If both sentiment and quotes are numbered, align by index
    - Else attach the sentiment set to the first quote_block and mark alignment_confidence low
    
    Args:
        sentiments_raw: Raw text containing sentiments (may be None or empty)
        quote_blocks: List of quote blocks from parse_quotes, each with 'quote_index'
    
    Returns:
        List of sentiment blocks, each containing:
        - 'quote_index': int - Index of the quote this sentiment aligns to
        - 'labels': List[str] - List of sentiment labels
        - 'tone_rollup': str - Computed tone: 'positive', 'negative', 'neutral', 'mixed', or 'unknown'
        - 'alignment_confidence': float - Confidence in alignment (0.0 to 1.0)
    
    Examples:
        >>> quotes = [{'quote_index': 1, 'quote_text': 'First'}, {'quote_index': 2, 'quote_text': 'Second'}]
        >>> sentiments = "1: positive; 2: negative"
        >>> blocks = parse_and_align_sentiments(sentiments, quotes)
        >>> len(blocks)
        2
        >>> blocks[0]['tone_rollup']
        'positive'
    """
    if not sentiments_raw or not sentiments_raw.strip():
        # Return empty sentiment blocks for all quotes
        return [{
            'quote_index': block['quote_index'],
            'labels': [],
            'tone_rollup': 'unknown',
            'alignment_confidence': 0.0
        } for block in quote_blocks]
    
    if not quote_blocks:
        return []
    
    text = sentiments_raw.strip()
    
    # If text is a single word that looks like a sentiment, treat it as such
    # This handles cases where sentiments_raw is just "positive", "negative", etc.
    single_word_match = re.match(r'^([a-zA-Z]+)$', text)
    if single_word_match:
        single_word = _normalize_label(single_word_match.group(1))
        classification = _classify_label(single_word)
        if classification:
            # Apply to all quotes (or first quote if we want to be conservative)
            tone_rollup = classification if classification != 'neutral' else 'neutral'
            return [{
                'quote_index': block['quote_index'],
                'labels': [single_word],
                'tone_rollup': tone_rollup,
                'alignment_confidence': 0.5  # Moderate confidence for single word
            } for block in quote_blocks]
    
    # Try to parse as numbered sentiments
    numbered_sentiments = _parse_numbered_sentiments(text)
    
    if numbered_sentiments is not None:
        # Check if quote blocks are also numbered (have numeric indices)
        quote_indices = [block['quote_index'] for block in quote_blocks]
        quotes_are_numbered = all(isinstance(idx, int) and idx > 0 for idx in quote_indices)
        
        if quotes_are_numbered:
            # Both are numbered - align by index
            sentiment_blocks, alignment_confidence = _align_sentiments_to_quotes(
                numbered_sentiments, quote_blocks
            )
            
            # Apply overall alignment_confidence to all blocks (already set per-block, but ensure consistency)
            for block in sentiment_blocks:
                # Keep the per-block confidence if it's already lower (for mismatches)
                if block['alignment_confidence'] > alignment_confidence:
                    block['alignment_confidence'] = alignment_confidence
            
            # Sort by quote_index for stable ordering
            sentiment_blocks.sort(key=lambda x: x['quote_index'])
            return sentiment_blocks
        else:
            # Sentiments are numbered but quotes are not - attach to first quote with low confidence
            if quote_blocks:
                first_quote_index = quote_blocks[0]['quote_index']
                # Combine all numbered sentiments into one
                all_labels = []
                for labels in numbered_sentiments.values():
                    all_labels.extend(labels)
                
                tone_rollup = _compute_tone_rollup(all_labels)
                return [{
                    'quote_index': first_quote_index,
                    'labels': all_labels,
                    'tone_rollup': tone_rollup,
                    'alignment_confidence': 0.3  # Low confidence for mismatch
                }]
    
    # Parse as flat list
    flat_labels = _parse_flat_sentiments(text)
    
    if flat_labels:
        # Check if quote blocks are numbered
        quote_indices = [block['quote_index'] for block in quote_blocks]
        quotes_are_numbered = all(isinstance(idx, int) and idx > 0 for idx in quote_indices)
        
        # Try to align flat list to quotes (one-to-one if counts match)
        if quotes_are_numbered and len(flat_labels) == len(quote_blocks):
            # Perfect match - align one-to-one
            sentiment_blocks = []
            for i, quote_block in enumerate(quote_blocks):
                if i < len(flat_labels):
                    labels = [flat_labels[i]]
                else:
                    labels = []
                
                tone_rollup = _compute_tone_rollup(labels)
                sentiment_blocks.append({
                    'quote_index': quote_block['quote_index'],
                    'labels': labels,
                    'tone_rollup': tone_rollup,
                    'alignment_confidence': 0.7  # Moderate confidence for positional alignment
                })
            return sentiment_blocks
        elif len(flat_labels) == len(quote_blocks):
            # Counts match but quotes not numbered - still try positional alignment
            sentiment_blocks = []
            for i, quote_block in enumerate(quote_blocks):
                if i < len(flat_labels):
                    labels = [flat_labels[i]]
                else:
                    labels = []
                
                tone_rollup = _compute_tone_rollup(labels)
                sentiment_blocks.append({
                    'quote_index': quote_block['quote_index'],
                    'labels': labels,
                    'tone_rollup': tone_rollup,
                    'alignment_confidence': 0.6  # Slightly lower confidence for non-numbered
                })
            return sentiment_blocks
        elif len(flat_labels) > 0:
            # Counts don't match - distribute labels across quotes
            # Try to assign one label per quote, cycling if needed
            sentiment_blocks = []
            for i, quote_block in enumerate(quote_blocks):
                # Cycle through labels if we have fewer labels than quotes
                label_idx = i % len(flat_labels)
                labels = [flat_labels[label_idx]]
                
                tone_rollup = _compute_tone_rollup(labels)
                sentiment_blocks.append({
                    'quote_index': quote_block['quote_index'],
                    'labels': labels,
                    'tone_rollup': tone_rollup,
                    'alignment_confidence': 0.4  # Lower confidence for mismatched counts
                })
            return sentiment_blocks
        else:
            # No labels found - return unknown for all quotes
            return [{
                'quote_index': block['quote_index'],
                'labels': [],
                'tone_rollup': 'unknown',
                'alignment_confidence': 0.0
            } for block in quote_blocks]
    
    # No sentiments found - return unknown for all quotes
    return [{
        'quote_index': block['quote_index'],
        'labels': [],
        'tone_rollup': 'unknown',
        'alignment_confidence': 0.0
    } for block in quote_blocks]


# Unit tests
if __name__ == '__main__':
    import unittest
    
    class TestParseSentiment(unittest.TestCase):
        
        def test_numbered_sentiments_aligned(self):
            """Test numbered sentiments aligned to numbered quotes."""
            quotes = [
                {'quote_index': 1, 'quote_text': 'First quote'},
                {'quote_index': 2, 'quote_text': 'Second quote'}
            ]
            sentiments = "1: positive; 2: negative"
            blocks = parse_and_align_sentiments(sentiments, quotes)
            
            self.assertEqual(len(blocks), 2)
            self.assertEqual(blocks[0]['quote_index'], 1)
            self.assertEqual(blocks[0]['tone_rollup'], 'positive')
            self.assertEqual(blocks[1]['quote_index'], 2)
            self.assertEqual(blocks[1]['tone_rollup'], 'negative')
            self.assertGreater(blocks[0]['alignment_confidence'], 0.5)
        
        def test_numbered_sentiments_mismatch(self):
            """Test numbered sentiments with mismatched counts."""
            quotes = [
                {'quote_index': 1, 'quote_text': 'First'},
                {'quote_index': 2, 'quote_text': 'Second'},
                {'quote_index': 3, 'quote_text': 'Third'}
            ]
            sentiments = "1: positive; 2: negative"  # Missing 3
            blocks = parse_and_align_sentiments(sentiments, quotes)
            
            self.assertEqual(len(blocks), 3)
            # Should still align what it can
            self.assertEqual(blocks[0]['tone_rollup'], 'positive')
            self.assertEqual(blocks[1]['tone_rollup'], 'negative')
            self.assertEqual(blocks[2]['tone_rollup'], 'unknown')
        
        def test_flat_sentiments_to_numbered_quotes(self):
            """Test flat sentiment list aligned to numbered quotes."""
            quotes = [
                {'quote_index': 1, 'quote_text': 'First'},
                {'quote_index': 2, 'quote_text': 'Second'}
            ]
            sentiments = "positive, negative"
            blocks = parse_and_align_sentiments(sentiments, quotes)
            
            self.assertEqual(len(blocks), 2)
            # Should attempt positional alignment
            self.assertGreater(blocks[0]['alignment_confidence'], 0.0)
        
        def test_flat_sentiments_to_bullet_quotes(self):
            """Test flat sentiments attached to bullet quotes (non-numbered)."""
            quotes = [
                {'quote_index': 0, 'quote_text': 'First'},
                {'quote_index': 1, 'quote_text': 'Second'}
            ]
            sentiments = "positive, negative"
            blocks = parse_and_align_sentiments(sentiments, quotes)
            
            # Should attach to first quote with low confidence
            self.assertEqual(len(blocks), 1)
            self.assertEqual(blocks[0]['quote_index'], 0)
            self.assertLess(blocks[0]['alignment_confidence'], 0.5)
        
        def test_tone_rollup_positive(self):
            """Test tone rollup for positive sentiment."""
            self.assertEqual(_compute_tone_rollup(['positive', 'good']), 'positive')
            self.assertEqual(_compute_tone_rollup(['positive']), 'positive')
        
        def test_tone_rollup_negative(self):
            """Test tone rollup for negative sentiment."""
            self.assertEqual(_compute_tone_rollup(['negative', 'bad']), 'negative')
            self.assertEqual(_compute_tone_rollup(['negative']), 'negative')
        
        def test_tone_rollup_mixed(self):
            """Test tone rollup for mixed sentiment."""
            self.assertEqual(_compute_tone_rollup(['positive', 'negative']), 'mixed')
            self.assertEqual(_compute_tone_rollup(['positive', 'good', 'negative']), 'mixed')
        
        def test_tone_rollup_neutral(self):
            """Test tone rollup for neutral sentiment."""
            self.assertEqual(_compute_tone_rollup(['neutral']), 'neutral')
            self.assertEqual(_compute_tone_rollup(['neutral', 'none']), 'neutral')
        
        def test_tone_rollup_unknown(self):
            """Test tone rollup for unknown sentiment."""
            self.assertEqual(_compute_tone_rollup([]), 'unknown')
            self.assertEqual(_compute_tone_rollup(['unknown_label']), 'unknown')
        
        def test_empty_sentiments(self):
            """Test empty sentiment input."""
            quotes = [{'quote_index': 1, 'quote_text': 'First'}]
            blocks = parse_and_align_sentiments("", quotes)
            
            self.assertEqual(len(blocks), 1)
            self.assertEqual(blocks[0]['tone_rollup'], 'unknown')
            self.assertEqual(blocks[0]['alignment_confidence'], 0.0)
        
        def test_no_quotes(self):
            """Test with no quote blocks."""
            blocks = parse_and_align_sentiments("positive", [])
            self.assertEqual(len(blocks), 0)
        
        def test_multiple_labels_per_sentiment(self):
            """Test multiple labels per sentiment entry."""
            quotes = [{'quote_index': 1, 'quote_text': 'First'}]
            sentiments = "1: positive, good, favorable"
            blocks = parse_and_align_sentiments(sentiments, quotes)
            
            self.assertEqual(len(blocks), 1)
            self.assertEqual(blocks[0]['tone_rollup'], 'positive')
            self.assertGreater(len(blocks[0]['labels']), 1)
        
        def test_different_numbered_formats(self):
            """Test different numbered sentiment formats."""
            quotes = [
                {'quote_index': 1, 'quote_text': 'First'},
                {'quote_index': 2, 'quote_text': 'Second'}
            ]
            
            # Test 1. format
            blocks1 = parse_and_align_sentiments("1. positive; 2. negative", quotes)
            self.assertEqual(len(blocks1), 2)
            
            # Test (1) format
            blocks2 = parse_and_align_sentiments("(1) positive; (2) negative", quotes)
            self.assertEqual(len(blocks2), 2)
            
            # Test 1) format
            blocks3 = parse_and_align_sentiments("1) positive; 2) negative", quotes)
            self.assertEqual(len(blocks3), 2)
    
    unittest.main()
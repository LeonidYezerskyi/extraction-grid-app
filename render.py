"""Module for rendering data visualizations and UI components."""

from typing import Dict, List, Optional, Any, Tuple


# Truncation budgets
TAKEAWAY_MAX = 180
TOPIC_ONELINER_MAX = 240
QUOTE_PREVIEW_MAX = 320

# Receipt display limits
RECEIPT_DISPLAY_DEFAULT = 5  # Default number of receipts to show (small subset for readability)
RECEIPT_DISPLAY_INCREMENT = 10  # Number of additional receipts per "Show more" click
RECEIPT_CONTAINER_MAX_HEIGHT = "400px"  # Max height for scrollable receipts container


def truncate(text: Optional[str], max_chars: int) -> str:
    """
    Truncate text to max_chars, cutting at word boundary and appending ellipsis when needed.
    
    This is a pure, deterministic function that ensures exact budget enforcement.
    
    Args:
        text: Text to truncate (may be None)
        max_chars: Maximum number of characters (must be >= 3 for ellipsis)
    
    Returns:
        Truncated text with ellipsis if truncated, original text if within limit
    
    Examples:
        >>> truncate("This is a long text that needs truncation", 20)
        'This is a long text...'
        >>> truncate("Short", 20)
        'Short'
        >>> truncate(None, 20)
        ''
    """
    if text is None:
        return ''
    
    if max_chars < 3:
        # Too small for ellipsis, just truncate
        return text[:max_chars] if len(text) > max_chars else text
    
    if len(text) <= max_chars:
        return text
    
    # Truncate to max_chars - 3 (for ellipsis)
    truncated = text[:max_chars - 3]
    
    # Find last space to cut at word boundary
    last_space = truncated.rfind(' ')
    
    # If we found a space and it's not too close to the start (at least 70% of truncated length)
    if last_space > 0 and last_space > (max_chars - 3) * 0.7:
        truncated = truncated[:last_space]
    
    return truncated + '...'


def format_sentiment_mix_plain(sentiment_mix: Dict[str, int]) -> Dict[str, Any]:
    """
    Format sentiment mix as a plain dictionary for programmatic use.
    
    Args:
        sentiment_mix: Dictionary with sentiment counts {'positive': int, 'negative': int, ...}
    
    Returns:
        Dictionary with formatted sentiment data:
        - counts: Original counts
        - total: Total count
        - percentages: Dictionary with percentages for each sentiment
        - dominant: Dominant sentiment type (or None if all zero)
    """
    if not sentiment_mix:
        return {
            'counts': {},
            'total': 0,
            'percentages': {},
            'dominant': None
        }
    
    total = sum(sentiment_mix.values())
    
    percentages = {}
    if total > 0:
        for sentiment, count in sentiment_mix.items():
            percentages[sentiment] = round((count / total) * 100, 1)
    else:
        percentages = {sentiment: 0.0 for sentiment in sentiment_mix.keys()}
    
    # Find dominant sentiment
    dominant = None
    if total > 0:
        dominant = max(sentiment_mix.items(), key=lambda x: x[1])[0]
        # If all are zero or tied, return None
        if sentiment_mix[dominant] == 0:
            dominant = None
    
    return {
        'counts': sentiment_mix.copy(),
        'total': total,
        'percentages': percentages,
        'dominant': dominant
    }


def format_sentiment_mix_html(sentiment_mix: Dict[str, int]) -> str:
    """
    Format sentiment mix as a simple HTML snippet with inline styles.
    
    Creates small chips/badges for each sentiment type with counts.
    No external CSS dependencies - uses inline styles only.
    
    Args:
        sentiment_mix: Dictionary with sentiment counts {'positive': int, 'negative': int, ...}
    
    Returns:
        HTML string with sentiment chips
    
    Example:
        >>> mix = {'positive': 5, 'negative': 2, 'neutral': 1}
        >>> html = format_sentiment_mix_html(mix)
        >>> 'positive' in html.lower()
        True
    """
    if not sentiment_mix or sum(sentiment_mix.values()) == 0:
        return '<span style="color: #666;">No sentiment data</span>'
    
    # Color mapping for sentiment types
    color_map = {
        'positive': '#22c55e',  # green
        'negative': '#ef4444',  # red
        'neutral': '#6b7280',  # gray
        'mixed': '#f59e0b',  # amber
        'unknown': '#9ca3af'  # light gray
    }
    
    # Label mapping
    label_map = {
        'positive': 'Positive',
        'negative': 'Negative',
        'neutral': 'Neutral',
        'mixed': 'Mixed',
        'unknown': 'Unknown'
    }
    
    chips = []
    for sentiment in ['positive', 'negative', 'neutral', 'mixed', 'unknown']:
        count = sentiment_mix.get(sentiment, 0)
        if count > 0:
            color = color_map.get(sentiment, '#9ca3af')
            label = label_map.get(sentiment, sentiment.capitalize())
            # Add both class and data attribute for maximum CSS targeting
            chip_html = (
                f'<span class="sentiment-chip sentiment-chip-{sentiment}" data-sentiment="{sentiment}" '
                f'style="display: inline-block; padding: 2px 8px; margin: 2px; '
                f'background-color: {color} !important; color: white !important; border-radius: 12px; '
                f'font-size: 11px; font-weight: 500; border: none !important;">{label}: {count}</span>'
            )
            chips.append(chip_html)
    
    return ' '.join(chips) if chips else '<span style="color: #666;">No sentiment data</span>'


def format_coverage_bar_plain(coverage_rate: float) -> Dict[str, Any]:
    """
    Format coverage rate as a plain dictionary for programmatic use.
    
    Args:
        coverage_rate: Coverage rate (0.0 to 1.0)
    
    Returns:
        Dictionary with:
        - rate: Original rate (0.0 to 1.0)
        - percentage: Percentage (0 to 100)
        - width: Width value for bar (0 to 100)
    """
    # Clamp to valid range
    rate = max(0.0, min(1.0, coverage_rate))
    percentage = round(rate * 100, 1)
    
    return {
        'rate': rate,
        'percentage': percentage,
        'width': percentage
    }


def format_coverage_bar_html(coverage_rate: float, max_width: int = 200) -> str:
    """
    Format coverage rate as a simple HTML bar snippet with inline styles.
    
    Creates a progress bar representation.
    No external CSS dependencies - uses inline styles only.
    
    Args:
        coverage_rate: Coverage rate (0.0 to 1.0)
        max_width: Maximum width in pixels for the bar (default 200)
    
    Returns:
        HTML string with coverage bar
    
    Example:
        >>> html = format_coverage_bar_html(0.75)
        >>> 'width' in html.lower()
        True
    """
    # Clamp to valid range
    rate = max(0.0, min(1.0, coverage_rate))
    percentage = round(rate * 100, 1)
    width_px = int(rate * max_width)
    
    # Color based on coverage level
    if rate >= 0.7:
        color = '#22c55e'  # green
    elif rate >= 0.4:
        color = '#f59e0b'  # amber
    else:
        color = '#ef4444'  # red
    
    bar_html = (
        f'<div style="display: inline-block; width: {max_width}px; height: 8px; '
        f'background-color: #e5e7eb; border-radius: 4px; overflow: hidden; vertical-align: middle;">'
        f'<div style="width: {width_px}px; height: 100%; background-color: {color};"></div>'
        f'</div>'
        f'<span style="margin-left: 8px; font-size: 12px; color: #374151;">{percentage}%</span>'
    )
    
    return bar_html


def format_takeaway_text(text: Optional[str]) -> str:
    """
    Truncate takeaway text to TAKEAWAY_MAX characters.
    
    Args:
        text: Takeaway text to format
    
    Returns:
        Truncated takeaway text
    """
    return truncate(text, TAKEAWAY_MAX)


def format_topic_oneliner(text: Optional[str]) -> str:
    """
    Truncate topic one-liner to TOPIC_ONELINER_MAX characters.
    
    Args:
        text: Topic one-liner text to format
    
    Returns:
        Truncated one-liner text
    """
    return truncate(text, TOPIC_ONELINER_MAX)


def format_quote_preview(text: Optional[str]) -> str:
    """
    Truncate quote preview to QUOTE_PREVIEW_MAX characters.
    
    Args:
        text: Quote preview text to format
    
    Returns:
        Truncated quote preview text
    """
    return truncate(text, QUOTE_PREVIEW_MAX)


def build_receipt_display(
    receipt_ref: str,
    canonical_model,
    topic_id: Optional[str] = None,
    max_excerpt_length: int = 120
) -> Dict[str, Any]:
    """
    Convert a receipt reference (participant_id:quote_index) to a human-readable receipt display object.
    
    Args:
        receipt_ref: Receipt reference in format "participant_id:quote_index"
        canonical_model: CanonicalModel object with evidence_cells and participants
        topic_id: Optional topic_id to filter evidence cells (for efficiency)
        max_excerpt_length: Maximum length for quote excerpt (default 120)
    
    Returns:
        Dictionary with:
        - participant_id: str (internal ID, kept for reference)
        - participant_label: str (human-readable participant name/label)
        - quote_excerpt: str (truncated quote text)
        - quote_full: str (full quote text)
        - quote_index: int (internal quote index)
        - source_context: str (optional context like "Interview" or "Response")
        - receipt_ref: str (original reference, kept for internal use)
    """
    if not receipt_ref or ':' not in receipt_ref:
        return {
            'participant_id': '',
            'participant_label': 'Unknown',
            'quote_excerpt': 'Invalid receipt reference',
            'quote_full': '',
            'quote_index': -1,
            'source_context': None,
            'receipt_ref': receipt_ref
        }
    
    participant_id, quote_index_str = receipt_ref.split(':', 1)
    
    # Try to parse quote_index as int
    try:
        quote_index = int(quote_index_str)
    except ValueError:
        quote_index = -1
    
    # Find participant label
    participant_label = participant_id  # Default to ID
    if canonical_model and canonical_model.participants:
        participant = next(
            (p for p in canonical_model.participants if p.participant_id == participant_id),
            None
        )
        if participant:
            participant_label = participant.participant_label or participant.participant_id
    
    # Find the quote text from evidence cells
    quote_text = ''
    evidence_cell = None
    
    if canonical_model and canonical_model.evidence_cells:
        # Filter by topic_id if provided for efficiency
        cells_to_check = [
            ec for ec in canonical_model.evidence_cells
            if ec.participant_id == participant_id
            and (topic_id is None or ec.topic_id == topic_id)
        ]
        
        for cell in cells_to_check:
            if not cell.quotes_raw:
                continue
            
            import parse_quotes
            quote_blocks = parse_quotes.parse_quotes(cell.quotes_raw)
            
            for quote_block in quote_blocks:
                block_index = quote_block.get('quote_index', -1)
                # Match both string and int representations
                if block_index == quote_index or str(block_index) == quote_index_str:
                    quote_text = quote_block.get('quote_text', '').strip()
                    evidence_cell = cell
                    break
            
            if quote_text:
                break
    
    # Generate excerpt
    quote_excerpt = truncate(quote_text, max_excerpt_length) if quote_text else 'No quote text available'
    
    # Determine source context
    source_context = None
    if evidence_cell and evidence_cell.participant_meta:
        # Try to infer context from metadata
        meta = evidence_cell.participant_meta
        if 'source' in meta or 'context' in meta or 'type' in meta:
            source_context = meta.get('source') or meta.get('context') or meta.get('type')
    
    # Calculate relevance score for ranking
    # Factors: quote length (prefer medium-length quotes), has text, participant diversity
    relevance_score = 0.0
    if quote_text:
        quote_len = len(quote_text)
        # Prefer quotes between 50-300 characters (readable, substantial)
        if 50 <= quote_len <= 300:
            relevance_score += 10.0
        elif 20 <= quote_len < 50:
            relevance_score += 5.0
        elif quote_len > 300:
            relevance_score += 3.0
        else:
            relevance_score += 1.0
    
    return {
        'participant_id': participant_id,
        'participant_label': participant_label,
        'quote_excerpt': quote_excerpt,
        'quote_full': quote_text,
        'quote_index': quote_index,
        'source_context': source_context,
        'receipt_ref': receipt_ref,  # Keep original for internal reference
        'relevance_score': relevance_score  # For ranking
    }


def rank_and_limit_receipts(
    receipt_displays: List[Dict[str, Any]],
    max_display: int = RECEIPT_DISPLAY_DEFAULT,
    prioritize_diversity: bool = True
) -> Tuple[List[Dict[str, Any]], int]:
    """
    Rank receipts by relevance and limit display, ensuring participant diversity.
    
    Args:
        receipt_displays: List of receipt display dictionaries
        max_display: Maximum number of receipts to return (default RECEIPT_DISPLAY_DEFAULT)
        prioritize_diversity: If True, ensure at least one receipt per participant when possible
    
    Returns:
        Tuple of (ranked_receipts, total_count)
        - ranked_receipts: Top N receipts, ranked by relevance and diversity
        - total_count: Total number of receipts available
    """
    if not receipt_displays:
        return [], 0
    
    total_count = len(receipt_displays)
    
    if total_count <= max_display:
        # No need to limit, just sort by relevance
        sorted_receipts = sorted(receipt_displays, key=lambda r: r.get('relevance_score', 0), reverse=True)
        return sorted_receipts, total_count
    
    # Sort by relevance score (descending)
    sorted_by_relevance = sorted(receipt_displays, key=lambda r: r.get('relevance_score', 0), reverse=True)
    
    if not prioritize_diversity:
        # Simple case: just return top N by relevance
        return sorted_by_relevance[:max_display], total_count
    
    # Diversity-aware selection: ensure we get receipts from different participants
    selected = []
    participant_used = set()
    
    # First pass: select one best receipt per participant (up to max_display)
    for receipt in sorted_by_relevance:
        if len(selected) >= max_display:
            break
        participant_id = receipt.get('participant_id', '')
        if participant_id not in participant_used:
            selected.append(receipt)
            participant_used.add(participant_id)
    
    # Second pass: fill remaining slots with highest relevance receipts
    remaining_slots = max_display - len(selected)
    if remaining_slots > 0:
        for receipt in sorted_by_relevance:
            if len(selected) >= max_display:
                break
            if receipt not in selected:
                selected.append(receipt)
    
    return selected, total_count


# Unit tests
if __name__ == '__main__':
    import unittest
    
    class TestRender(unittest.TestCase):
        
        def test_truncate_exact_length(self):
            """Test truncate with text exactly at max length."""
            text = "A" * 20
            result = truncate(text, 20)
            self.assertEqual(len(result), 20)
            self.assertEqual(result, text)
        
        def test_truncate_over_length(self):
            """Test truncate with text over max length."""
            text = "This is a long text that needs truncation"
            result = truncate(text, 20)
            self.assertLessEqual(len(result), 20)
            self.assertTrue(result.endswith('...'))
        
        def test_truncate_word_boundary(self):
            """Test truncate cuts at word boundary."""
            text = "This is a test sentence for truncation"
            result = truncate(text, 20)
            # Should not cut in the middle of a word
            self.assertNotIn('trunca', result)
            self.assertTrue(result.endswith('...'))
        
        def test_truncate_none(self):
            """Test truncate with None input."""
            result = truncate(None, 20)
            self.assertEqual(result, '')
        
        def test_truncate_short_max(self):
            """Test truncate with very short max_chars."""
            text = "This is a test"
            result = truncate(text, 2)
            self.assertEqual(len(result), 2)
        
        def test_takeaway_max_constant(self):
            """Test TAKEAWAY_MAX constant."""
            self.assertEqual(TAKEAWAY_MAX, 180)
        
        def test_topic_oneliner_max_constant(self):
            """Test TOPIC_ONELINER_MAX constant."""
            self.assertEqual(TOPIC_ONELINER_MAX, 240)
        
        def test_quote_preview_max_constant(self):
            """Test QUOTE_PREVIEW_MAX constant."""
            self.assertEqual(QUOTE_PREVIEW_MAX, 320)
        
        def test_format_takeaway_text(self):
            """Test format_takeaway_text respects budget."""
            long_text = "A" * 200
            result = format_takeaway_text(long_text)
            self.assertLessEqual(len(result), TAKEAWAY_MAX)
        
        def test_format_topic_oneliner(self):
            """Test format_topic_oneliner respects budget."""
            long_text = "A" * 300
            result = format_topic_oneliner(long_text)
            self.assertLessEqual(len(result), TOPIC_ONELINER_MAX)
        
        def test_format_quote_preview(self):
            """Test format_quote_preview respects budget."""
            long_text = "A" * 400
            result = format_quote_preview(long_text)
            self.assertLessEqual(len(result), QUOTE_PREVIEW_MAX)
        
        def test_format_sentiment_mix_plain(self):
            """Test format_sentiment_mix_plain."""
            mix = {'positive': 5, 'negative': 2, 'neutral': 1, 'mixed': 0, 'unknown': 0}
            result = format_sentiment_mix_plain(mix)
            
            self.assertIn('counts', result)
            self.assertIn('total', result)
            self.assertIn('percentages', result)
            self.assertIn('dominant', result)
            self.assertEqual(result['total'], 8)
            self.assertEqual(result['dominant'], 'positive')
        
        def test_format_sentiment_mix_plain_empty(self):
            """Test format_sentiment_mix_plain with empty mix."""
            mix = {'positive': 0, 'negative': 0, 'neutral': 0, 'mixed': 0, 'unknown': 0}
            result = format_sentiment_mix_plain(mix)
            
            self.assertEqual(result['total'], 0)
            self.assertIsNone(result['dominant'])
        
        def test_format_sentiment_mix_html(self):
            """Test format_sentiment_mix_html."""
            mix = {'positive': 5, 'negative': 2, 'neutral': 1, 'mixed': 0, 'unknown': 0}
            result = format_sentiment_mix_html(mix)
            
            self.assertIn('positive', result.lower())
            self.assertIn('5', result)
            self.assertIn('style', result)
        
        def test_format_sentiment_mix_html_empty(self):
            """Test format_sentiment_mix_html with empty mix."""
            mix = {'positive': 0, 'negative': 0, 'neutral': 0, 'mixed': 0, 'unknown': 0}
            result = format_sentiment_mix_html(mix)
            
            self.assertIn('No sentiment data', result)
        
        def test_format_coverage_bar_plain(self):
            """Test format_coverage_bar_plain."""
            result = format_coverage_bar_plain(0.75)
            
            self.assertIn('rate', result)
            self.assertIn('percentage', result)
            self.assertIn('width', result)
            self.assertEqual(result['rate'], 0.75)
            self.assertEqual(result['percentage'], 75.0)
            self.assertEqual(result['width'], 75.0)
        
        def test_format_coverage_bar_plain_bounds(self):
            """Test format_coverage_bar_plain with out-of-bounds values."""
            result_high = format_coverage_bar_plain(1.5)
            self.assertEqual(result_high['rate'], 1.0)
            
            result_low = format_coverage_bar_plain(-0.5)
            self.assertEqual(result_low['rate'], 0.0)
        
        def test_format_coverage_bar_html(self):
            """Test format_coverage_bar_html."""
            result = format_coverage_bar_html(0.75)
            
            self.assertIn('width', result.lower())
            self.assertIn('75', result)
            self.assertIn('style', result)
        
        def test_format_coverage_bar_html_bounds(self):
            """Test format_coverage_bar_html with out-of-bounds values."""
            result_high = format_coverage_bar_html(1.5)
            self.assertIn('100', result)
            
            result_low = format_coverage_bar_html(-0.5)
            self.assertIn('0', result)
        
        def test_deterministic_truncate(self):
            """Test that truncate is deterministic."""
            text = "This is a test sentence for deterministic truncation testing"
            result1 = truncate(text, 30)
            result2 = truncate(text, 30)
            self.assertEqual(result1, result2)
        
        def test_deterministic_sentiment_format(self):
            """Test that sentiment formatting is deterministic."""
            mix = {'positive': 5, 'negative': 2, 'neutral': 1}
            result1 = format_sentiment_mix_plain(mix)
            result2 = format_sentiment_mix_plain(mix)
            self.assertEqual(result1, result2)
    
    unittest.main()
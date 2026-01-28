"""Data model for Explore table - canonical representation of topics for exploration."""

from dataclasses import dataclass, field
from typing import Literal, Optional, Tuple, Dict, List, Any
import math


@dataclass
class ExploreTopic:
    """
    Canonical representation of a topic for the Explore table.
    
    This model provides a standardized, type-safe representation that separates:
    - Identifiers (topic_id, topic_label)
    - Scores (importance_score, coverage_pct, sentiment_score)
    - Qualitative summaries (summary_text)
    - Provenance/metadata (source_document, speaker_id)
    
    All fields are explicitly typed and missing values are handled deterministically.
    """
    # Identifiers
    topic_id: str
    topic_label: str
    
    # Scores (all floats, 0.0 if missing)
    importance_score: float  # topic_score from aggregates
    coverage_pct: float  # coverage_rate * 100
    mentions_count: int  # evidence_count
    sentiment_score: float  # Computed from sentiment mix (0.0 to 1.0, where 1.0 = fully positive, -1.0 = fully negative)
    
    # Sentiment label (categorical)
    sentiment_label: Literal["positive", "neutral", "negative", "mixed", "unknown"]
    
    # Qualitative summary
    summary_text: str  # topic_one_liner, empty string if missing
    
    # Provenance / metadata
    source_document: Optional[str] = None  # For future use - document/file reference
    speaker_id: Optional[str] = None  # For future use - primary speaker/participant
    
    # Drill-down evidence (optional, populated via attach_evidence())
    evidence: Optional[Dict[str, Any]] = field(default=None)
    # Structure when populated:
    # {
    #     'quotes': List[str],  # Up to 5 representative quotes
    #     'sentiment_distribution': Dict[str, int],  # Counts by sentiment label
    #     'confidence_notes': Optional[str]  # Human-readable confidence indicators
    # }
    
    def __post_init__(self):
        """Validate and normalize values after initialization."""
        # Ensure numeric fields are within valid ranges
        self.importance_score = max(0.0, float(self.importance_score))
        self.coverage_pct = max(0.0, min(100.0, float(self.coverage_pct)))
        self.mentions_count = max(0, int(self.mentions_count))
        self.sentiment_score = max(-1.0, min(1.0, float(self.sentiment_score)))
        
        # Ensure sentiment_label is valid
        valid_labels = {"positive", "neutral", "negative", "mixed", "unknown"}
        if self.sentiment_label not in valid_labels:
            self.sentiment_label = "unknown"
        
        # Ensure strings are not None
        if self.topic_id is None:
            self.topic_id = ""
        if self.topic_label is None:
            self.topic_label = self.topic_id
        if self.summary_text is None:
            self.summary_text = ""
    
    def get_signal_bucket(self) -> Literal["high", "medium", "low"]:
        """
        Compute signal bucket based on importance_score and coverage_pct.
        
        Rules:
        - "high": importance_score >= 1.9 and coverage_pct >= 90
        - "medium": importance_score >= 1.7
        - "low": otherwise
        
        Returns:
            Signal bucket: "high", "medium", or "low"
        """
        if self.importance_score >= 1.9 and self.coverage_pct >= 90.0:
            return "high"
        elif self.importance_score >= 1.7:
            return "medium"
        else:
            return "low"
    
    def to_dict(self, include_signal_bucket: bool = False) -> dict:
        """
        Convert to dictionary for DataFrame construction.
        
        Args:
            include_signal_bucket: If True, include computed signal_bucket in output
        
        Returns:
            Dictionary with all fields
        """
        result = {
            'topic_id': self.topic_id,
            'topic_label': self.topic_label,
            'importance_score': self.importance_score,
            'coverage_pct': self.coverage_pct,
            'mentions_count': self.mentions_count,
            'sentiment_score': self.sentiment_score,
            'sentiment_label': self.sentiment_label,
            'summary_text': self.summary_text,
            'source_document': self.source_document,
            'speaker_id': self.speaker_id
        }
        
        if include_signal_bucket:
            result['signal_bucket'] = self.get_signal_bucket()
        
        # Include evidence if present (optional, as it can be large)
        if self.evidence is not None:
            result['evidence'] = self.evidence
        
        return result


def compute_sentiment_score(sentiment_mix: dict) -> Tuple[float, Literal["positive", "neutral", "negative", "mixed", "unknown"]]:
    """
    Compute sentiment score and label from sentiment mix.
    
    Args:
        sentiment_mix: Dictionary with counts {'positive': int, 'negative': int, 'neutral': int, 'mixed': int, 'unknown': int}
    
    Returns:
        Tuple of (sentiment_score, sentiment_label)
        - sentiment_score: float from -1.0 (fully negative) to 1.0 (fully positive)
        - sentiment_label: categorical label
    """
    positive = sentiment_mix.get('positive', 0)
    negative = sentiment_mix.get('negative', 0)
    neutral = sentiment_mix.get('neutral', 0)
    mixed = sentiment_mix.get('mixed', 0)
    unknown = sentiment_mix.get('unknown', 0)
    
    total = positive + negative + neutral + mixed + unknown
    
    if total == 0:
        return (0.0, "unknown")
    
    # Compute score: (positive - negative) / total, normalized to [-1, 1]
    # Mixed sentiments contribute 0 (neutral)
    score = (positive - negative) / total
    
    # Determine label
    if mixed > 0 and (positive > 0 or negative > 0):
        # Has mixed sentiments along with others
        if positive > negative:
            label = "mixed"
        elif negative > positive:
            label = "mixed"
        else:
            label = "mixed"
    elif positive > 0 and negative == 0:
        label = "positive"
    elif negative > 0 and positive == 0:
        label = "negative"
    elif neutral > 0 and positive == 0 and negative == 0:
        label = "neutral"
    elif unknown > 0 and positive == 0 and negative == 0 and neutral == 0:
        label = "unknown"
    else:
        # Default based on score
        if score > 0.1:
            label = "positive"
        elif score < -0.1:
            label = "negative"
        else:
            label = "neutral"
    
    return (score, label)


def from_topic_aggregate(
    topic_aggregate: dict,
    sentiment_mix: dict,
    topic_label: Optional[str] = None
) -> ExploreTopic:
    """
    Convert a topic aggregate dictionary to an ExploreTopic instance.
    
    This is the main constructor for creating ExploreTopic from the current data model.
    
    Args:
        topic_aggregate: Dictionary from score.compute_topic_aggregates with keys:
            - topic_id: str
            - topic_score: float (importance_score)
            - coverage_rate: float (0.0 to 1.0)
            - evidence_count: int (mentions_count)
            - topic_one_liner: Optional[str] (summary_text)
        sentiment_mix: Dictionary from digest._compute_sentiment_mix with sentiment counts
        topic_label: Optional label for display (defaults to topic_id)
    
    Returns:
        ExploreTopic instance with all fields populated
    
    Example:
        >>> topic_agg = {
        ...     'topic_id': 'topic1',
        ...     'topic_score': 0.85,
        ...     'coverage_rate': 0.7,
        ...     'evidence_count': 15,
        ...     'topic_one_liner': 'Sample summary'
        ... }
        >>> sentiment_mix = {'positive': 10, 'negative': 2, 'neutral': 3, 'mixed': 0, 'unknown': 0}
        >>> topic = from_topic_aggregate(topic_agg, sentiment_mix)
        >>> topic.importance_score
        0.85
    """
    # Extract identifiers
    topic_id = str(topic_aggregate.get('topic_id', ''))
    if not topic_label:
        topic_label = topic_id
    
    # Capitalize first letter of each word in topic_label (Title Case)
    if topic_label:
        # Split by spaces and capitalize each word
        words = topic_label.split()
        topic_label = ' '.join(word.capitalize() for word in words)
    
    # Extract scores with safe defaults
    importance_score = float(topic_aggregate.get('topic_score', 0.0))
    coverage_rate = float(topic_aggregate.get('coverage_rate', 0.0))
    coverage_pct = coverage_rate * 100.0
    mentions_count = int(topic_aggregate.get('evidence_count', 0))
    
    # Compute sentiment score and label
    sentiment_score, sentiment_label = compute_sentiment_score(sentiment_mix)
    
    # Extract summary text
    summary_text = str(topic_aggregate.get('topic_one_liner', '') or '')
    
    # Provenance fields (None for now, can be populated later)
    source_document = None
    speaker_id = None
    
    return ExploreTopic(
        topic_id=topic_id,
        topic_label=topic_label,
        importance_score=importance_score,
        coverage_pct=coverage_pct,
        mentions_count=mentions_count,
        sentiment_score=sentiment_score,
        sentiment_label=sentiment_label,
        summary_text=summary_text,
        source_document=source_document,
        speaker_id=speaker_id
    )


def from_dataframe_row(row: dict) -> ExploreTopic:
    """
    Convert a DataFrame row (dictionary) to an ExploreTopic instance.
    
    This is useful for reconstructing ExploreTopic from exported/serialized data.
    
    Args:
        row: Dictionary with keys matching ExploreTopic fields
    
    Returns:
        ExploreTopic instance
    
    Example:
        >>> row = {
        ...     'topic_id': 'topic1',
        ...     'topic_label': 'Topic 1',
        ...     'importance_score': 0.85,
        ...     'coverage_pct': 70.0,
        ...     'mentions_count': 15,
        ...     'sentiment_score': 0.5,
        ...     'sentiment_label': 'positive',
        ...     'summary_text': 'Sample summary'
        ... }
        >>> topic = from_dataframe_row(row)
    """
    # Safely extract all fields with defaults
    topic_id = str(row.get('topic_id', ''))
    topic_label = str(row.get('topic_label', topic_id))
    importance_score = float(row.get('importance_score', 0.0))
    coverage_pct = float(row.get('coverage_pct', 0.0))
    mentions_count = int(row.get('mentions_count', 0))
    sentiment_score = float(row.get('sentiment_score', 0.0))
    sentiment_label = str(row.get('sentiment_label', 'unknown'))
    summary_text = str(row.get('summary_text', ''))
    source_document = row.get('source_document')
    speaker_id = row.get('speaker_id')
    
    # Validate sentiment_label
    valid_labels = {"positive", "neutral", "negative", "mixed", "unknown"}
    if sentiment_label not in valid_labels:
        sentiment_label = "unknown"
    
    return ExploreTopic(
        topic_id=topic_id,
        topic_label=topic_label,
        importance_score=importance_score,
        coverage_pct=coverage_pct,
        mentions_count=mentions_count,
        sentiment_score=sentiment_score,
        sentiment_label=sentiment_label,
        summary_text=summary_text,
        source_document=source_document,
        speaker_id=speaker_id
    )


def rank_topics(topics: list[ExploreTopic]) -> list[ExploreTopic]:
    """
    Rank and sort topics with deterministic ordering and signal bucketing.
    
    This function provides explainable, stable topic ordering instead of 
    "magic-score-driven" ranking. Topics are sorted by:
    1. Primary: importance_score (descending)
    2. Secondary: coverage_pct (descending)
    3. Tertiary: mentions_count (descending)
    
    Each topic is also annotated with a signal_bucket computed via get_signal_bucket():
    - "high": importance_score >= 1.9 and coverage_pct >= 90
    - "medium": importance_score >= 1.7
    - "low": otherwise
    
    Args:
        topics: List of ExploreTopic instances to rank
    
    Returns:
        List of ExploreTopic instances, sorted and ready for use.
        Note: signal_bucket is computed via get_signal_bucket() method, not stored in dataclass.
    
    Example:
        >>> topics = [
        ...     ExploreTopic(topic_id='t1', topic_label='T1', importance_score=2.0, 
        ...                  coverage_pct=95.0, mentions_count=20, sentiment_score=0.5,
        ...                  sentiment_label='positive', summary_text='Summary 1'),
        ...     ExploreTopic(topic_id='t2', topic_label='T2', importance_score=1.8,
        ...                  coverage_pct=80.0, mentions_count=15, sentiment_score=0.3,
        ...                  sentiment_label='positive', summary_text='Summary 2')
        ... ]
        >>> ranked = rank_topics(topics)
        >>> ranked[0].topic_id  # Should be 't1' (higher importance_score)
        't1'
        >>> ranked[0].get_signal_bucket()  # Should be 'high'
        'high'
    """
    if not topics:
        return []
    
    # Sort topics using multi-level sorting
    # Primary: importance_score (descending)
    # Secondary: coverage_pct (descending)
    # Tertiary: mentions_count (descending)
    sorted_topics = sorted(
        topics,
        key=lambda t: (
            -t.importance_score,  # Negative for descending order
            -t.coverage_pct,      # Negative for descending order
            -t.mentions_count     # Negative for descending order
        )
    )
    
    return sorted_topics


def attach_evidence(
    topic: ExploreTopic,
    evidence_cells: List[Any],
    max_quotes: int = 5
) -> ExploreTopic:
    """
    Attach drill-down evidence to an ExploreTopic instance.
    
    This function aggregates supporting evidence for a topic:
    - Extracts up to max_quotes representative quotes
    - Computes sentiment distribution counts
    - Generates confidence indicators
    
    Uses immutable pattern - returns a new ExploreTopic instance with evidence populated.
    
    Args:
        topic: ExploreTopic instance to attach evidence to
        evidence_cells: List of EvidenceCell objects from canonical_model.evidence_cells
                       Filtered by topic_id before calling, or pass all cells and filter internally.
                       Each EvidenceCell should have:
                       - topic_id: str (to match)
                       - quotes_raw: Optional[str] (raw quotes text)
                       - sentiments_raw: Optional[str] (raw sentiments text)
        max_quotes: Maximum number of representative quotes to include (default: 5)
    
    Returns:
        New ExploreTopic instance with evidence field populated.
        Evidence dict contains:
        - 'quotes': List[str] - Up to max_quotes representative quotes
        - 'sentiment_distribution': Dict[str, int] - Counts by sentiment label
        - 'confidence_notes': Optional[str] - Human-readable confidence indicators
    
    Example:
        >>> topic = ExploreTopic(...)
        >>> evidence_cells = [ec for ec in canonical_model.evidence_cells if ec.topic_id == topic.topic_id]
        >>> topic_with_evidence = attach_evidence(topic, evidence_cells)
        >>> topic_with_evidence.evidence['quotes']
        ['Quote 1', 'Quote 2', ...]
        >>> topic_with_evidence.evidence['sentiment_distribution']
        {'positive': 5, 'negative': 2, 'neutral': 1, 'mixed': 0, 'unknown': 0}
    """
    import parse_quotes
    import parse_sentiment
    
    # Filter evidence cells by topic_id
    matching_cells = [ec for ec in evidence_cells if ec.topic_id == topic.topic_id]
    
    if not matching_cells:
        # No evidence found - return topic with empty evidence
        return ExploreTopic(
            topic_id=topic.topic_id,
            topic_label=topic.topic_label,
            importance_score=topic.importance_score,
            coverage_pct=topic.coverage_pct,
            mentions_count=topic.mentions_count,
            sentiment_score=topic.sentiment_score,
            sentiment_label=topic.sentiment_label,
            summary_text=topic.summary_text,
            source_document=topic.source_document,
            speaker_id=topic.speaker_id,
            evidence={
                'quotes': [],
                'sentiment_distribution': {},
                'confidence_notes': None
            }
        )
    
    # Collect all quotes
    all_quotes = []
    all_sentiment_blocks = []
    
    for cell in matching_cells:
        if not cell.quotes_raw:
            continue
        
        # Parse quotes
        quote_blocks = parse_quotes.parse_quotes(cell.quotes_raw)
        for quote_block in quote_blocks:
            quote_text = quote_block.get('quote_text', '').strip()
            if quote_text:
                all_quotes.append(quote_text)
        
        # Parse sentiments if available
        if cell.sentiments_raw and quote_blocks:
            sentiment_blocks = parse_sentiment.parse_and_align_sentiments(
                cell.sentiments_raw, quote_blocks
            )
            all_sentiment_blocks.extend(sentiment_blocks)
    
    # Select representative quotes (up to max_quotes)
    # Strategy: prioritize longer quotes, then take first N
    representative_quotes = []
    if all_quotes:
        # Sort by length (descending) to prioritize substantial quotes
        sorted_quotes = sorted(all_quotes, key=len, reverse=True)
        representative_quotes = sorted_quotes[:max_quotes]
    
    # Compute sentiment distribution
    sentiment_distribution = {
        'positive': 0,
        'negative': 0,
        'neutral': 0,
        'mixed': 0,
        'unknown': 0
    }
    
    for sentiment_block in all_sentiment_blocks:
        tone = sentiment_block.get('tone_rollup', 'unknown')
        if tone in sentiment_distribution:
            sentiment_distribution[tone] += 1
    
    # Generate confidence notes
    confidence_notes = _generate_confidence_notes(
        len(matching_cells),
        len(all_quotes),
        sentiment_distribution,
        topic.coverage_pct
    )
    
    # Create evidence dict
    evidence = {
        'quotes': representative_quotes,
        'sentiment_distribution': sentiment_distribution,
        'confidence_notes': confidence_notes
    }
    
    # Return new ExploreTopic instance with evidence (immutable pattern)
    return ExploreTopic(
        topic_id=topic.topic_id,
        topic_label=topic.topic_label,
        importance_score=topic.importance_score,
        coverage_pct=topic.coverage_pct,
        mentions_count=topic.mentions_count,
        sentiment_score=topic.sentiment_score,
        sentiment_label=topic.sentiment_label,
        summary_text=topic.summary_text,
        source_document=topic.source_document,
        speaker_id=topic.speaker_id,
        evidence=evidence
    )


def _generate_confidence_notes(
    evidence_cell_count: int,
    quote_count: int,
    sentiment_distribution: Dict[str, int],
    coverage_pct: float
) -> Optional[str]:
    """
    Generate human-readable confidence indicators.
    
    Args:
        evidence_cell_count: Number of evidence cells (participants) with data
        quote_count: Total number of quotes found
        sentiment_distribution: Dictionary with sentiment counts
        coverage_pct: Coverage percentage for the topic
    
    Returns:
        Confidence notes string or None if no meaningful notes
    """
    notes_parts = []
    
    # Coverage indicator
    if coverage_pct >= 90.0:
        notes_parts.append("High coverage (≥90%)")
    elif coverage_pct >= 70.0:
        notes_parts.append("Good coverage (≥70%)")
    elif coverage_pct >= 50.0:
        notes_parts.append("Moderate coverage (≥50%)")
    elif coverage_pct > 0:
        notes_parts.append("Low coverage (<50%)")
    
    # Quote count indicator
    if quote_count >= 10:
        notes_parts.append(f"Strong evidence ({quote_count} quotes)")
    elif quote_count >= 5:
        notes_parts.append(f"Moderate evidence ({quote_count} quotes)")
    elif quote_count > 0:
        notes_parts.append(f"Limited evidence ({quote_count} quotes)")
    
    # Sentiment consistency indicator
    total_sentiments = sum(sentiment_distribution.values())
    if total_sentiments > 0:
        max_sentiment = max(sentiment_distribution.items(), key=lambda x: x[1])
        max_label, max_count = max_sentiment
        if max_label != 'unknown':
            percentage = (max_count / total_sentiments) * 100
            if percentage >= 70:
                notes_parts.append(f"Consistent sentiment ({max_label}, {percentage:.0f}%)")
            elif percentage >= 50:
                notes_parts.append(f"Mixed sentiment ({max_label} dominant, {percentage:.0f}%)")
    
    if notes_parts:
        return "; ".join(notes_parts)
    
    return None


def attach_evidence_from_canonical_model(
    topic: ExploreTopic,
    canonical_model: Any,
    max_quotes: int = 5
) -> ExploreTopic:
    """
    Convenience wrapper to attach evidence from a CanonicalModel.
    
    This function extracts evidence_cells from canonical_model and calls attach_evidence().
    
    Args:
        topic: ExploreTopic instance to attach evidence to
        canonical_model: CanonicalModel object with evidence_cells
        max_quotes: Maximum number of representative quotes to include (default: 5)
    
    Returns:
        New ExploreTopic instance with evidence field populated
    
    Example:
        >>> topic = ExploreTopic(...)
        >>> topic_with_evidence = attach_evidence_from_canonical_model(topic, canonical_model)
    """
    evidence_cells = canonical_model.evidence_cells
    return attach_evidence(topic, evidence_cells, max_quotes)


def attach_evidence_from_dataframes(
    topic: ExploreTopic,
    quotes_df: Any,
    sentiments_df: Any,
    max_quotes: int = 5
) -> ExploreTopic:
    """
    Attach evidence from DataFrame format (alternative interface).
    
    This function matches rows by topic_id from quotes_df and sentiments_df,
    then aggregates evidence. This is an alternative to attach_evidence() for
    cases where data is in DataFrame format.
    
    Note: In the current architecture, evidence_cells from CanonicalModel is preferred.
    This function is provided for compatibility with DataFrame-based workflows.
    
    Args:
        topic: ExploreTopic instance to attach evidence to
        quotes_df: DataFrame with columns including 'topic_id' and quote data
        sentiments_df: DataFrame with columns including 'topic_id' and sentiment data
        max_quotes: Maximum number of representative quotes to include (default: 5)
    
    Returns:
        New ExploreTopic instance with evidence field populated
    
    Example:
        >>> topic = ExploreTopic(...)
        >>> topic_with_evidence = attach_evidence_from_dataframes(topic, quotes_df, sentiments_df)
    """
    import pandas as pd
    import parse_quotes
    import parse_sentiment
    
    # Filter rows by topic_id
    topic_id = topic.topic_id
    
    # Get matching quotes rows
    quotes_rows = quotes_df[quotes_df.get('topic_id', '') == topic_id] if not quotes_df.empty else pd.DataFrame()
    
    # Get matching sentiments rows
    sentiments_rows = sentiments_df[sentiments_df.get('topic_id', '') == topic_id] if not sentiments_df.empty else pd.DataFrame()
    
    # Collect quotes
    all_quotes = []
    for _, row in quotes_rows.iterrows():
        # Try to find quote column (could be 'quotes', 'quotes_raw', 'value', etc.)
        quote_text = None
        for col in ['quotes', 'quotes_raw', 'value', 'quote_text']:
            if col in row and pd.notna(row[col]):
                quote_text = str(row[col])
                break
        
        if quote_text:
            # Parse quotes if needed
            quote_blocks = parse_quotes.parse_quotes(quote_text)
            for quote_block in quote_blocks:
                q_text = quote_block.get('quote_text', '').strip()
                if q_text:
                    all_quotes.append(q_text)
    
    # Collect sentiment distribution
    sentiment_distribution = {
        'positive': 0,
        'negative': 0,
        'neutral': 0,
        'mixed': 0,
        'unknown': 0
    }
    
    for _, row in sentiments_rows.iterrows():
        # Try to find sentiment column
        sentiment_text = None
        for col in ['sentiments', 'sentiments_raw', 'value', 'sentiment']:
            if col in row and pd.notna(row[col]):
                sentiment_text = str(row[col])
                break
        
        if sentiment_text:
            # Parse flat sentiments
            flat_labels = parse_sentiment._parse_flat_sentiments(sentiment_text)
            for label in flat_labels:
                classification = parse_sentiment._classify_label(label)
                if classification and classification in sentiment_distribution:
                    sentiment_distribution[classification] += 1
    
    # Select representative quotes (up to max_quotes)
    representative_quotes = []
    if all_quotes:
        sorted_quotes = sorted(all_quotes, key=len, reverse=True)
        representative_quotes = sorted_quotes[:max_quotes]
    
    # Generate confidence notes
    confidence_notes = _generate_confidence_notes(
        len(quotes_rows),
        len(all_quotes),
        sentiment_distribution,
        topic.coverage_pct
    )
    
    # Create evidence dict
    evidence = {
        'quotes': representative_quotes,
        'sentiment_distribution': sentiment_distribution,
        'confidence_notes': confidence_notes
    }
    
    # Return new ExploreTopic instance with evidence (immutable pattern)
    return ExploreTopic(
        topic_id=topic.topic_id,
        topic_label=topic.topic_label,
        importance_score=topic.importance_score,
        coverage_pct=topic.coverage_pct,
        mentions_count=topic.mentions_count,
        sentiment_score=topic.sentiment_score,
        sentiment_label=topic.sentiment_label,
        summary_text=topic.summary_text,
        source_document=topic.source_document,
        speaker_id=topic.speaker_id,
        evidence=evidence
    )


# Unit tests
if __name__ == '__main__':
    import unittest
    
    class TestExploreTopic(unittest.TestCase):
        
        def test_from_topic_aggregate_basic(self):
            """Test basic conversion from topic aggregate."""
            topic_agg = {
                'topic_id': 'test_topic',
                'topic_score': 0.85,
                'coverage_rate': 0.7,
                'evidence_count': 15,
                'topic_one_liner': 'Test summary'
            }
            sentiment_mix = {'positive': 10, 'negative': 2, 'neutral': 3, 'mixed': 0, 'unknown': 0}
            
            topic = from_topic_aggregate(topic_agg, sentiment_mix)
            
            self.assertEqual(topic.topic_id, 'test_topic')
            self.assertEqual(topic.topic_label, 'test_topic')
            self.assertEqual(topic.importance_score, 0.85)
            self.assertEqual(topic.coverage_pct, 70.0)
            self.assertEqual(topic.mentions_count, 15)
            self.assertEqual(topic.summary_text, 'Test summary')
            self.assertEqual(topic.sentiment_label, 'positive')
            self.assertGreater(topic.sentiment_score, 0.0)
        
        def test_from_topic_aggregate_missing_fields(self):
            """Test conversion with missing fields."""
            topic_agg = {
                'topic_id': 'test_topic',
                'topic_score': 0.5
            }
            sentiment_mix = {'positive': 0, 'negative': 0, 'neutral': 0, 'mixed': 0, 'unknown': 5}
            
            topic = from_topic_aggregate(topic_agg, sentiment_mix)
            
            self.assertEqual(topic.coverage_pct, 0.0)
            self.assertEqual(topic.mentions_count, 0)
            self.assertEqual(topic.summary_text, '')
            self.assertEqual(topic.sentiment_label, 'unknown')
        
        def test_compute_sentiment_score_positive(self):
            """Test sentiment score computation for positive."""
            sentiment_mix = {'positive': 10, 'negative': 0, 'neutral': 0, 'mixed': 0, 'unknown': 0}
            score, label = compute_sentiment_score(sentiment_mix)
            
            self.assertEqual(label, 'positive')
            self.assertEqual(score, 1.0)
        
        def test_compute_sentiment_score_negative(self):
            """Test sentiment score computation for negative."""
            sentiment_mix = {'positive': 0, 'negative': 10, 'neutral': 0, 'mixed': 0, 'unknown': 0}
            score, label = compute_sentiment_score(sentiment_mix)
            
            self.assertEqual(label, 'negative')
            self.assertEqual(score, -1.0)
        
        def test_compute_sentiment_score_mixed(self):
            """Test sentiment score computation for mixed."""
            sentiment_mix = {'positive': 5, 'negative': 5, 'neutral': 0, 'mixed': 2, 'unknown': 0}
            score, label = compute_sentiment_score(sentiment_mix)
            
            self.assertEqual(label, 'mixed')
            self.assertEqual(score, 0.0)
        
        def test_from_dataframe_row(self):
            """Test conversion from DataFrame row."""
            row = {
                'topic_id': 'test_topic',
                'topic_label': 'Test Topic',
                'importance_score': 0.85,
                'coverage_pct': 70.0,
                'mentions_count': 15,
                'sentiment_score': 0.5,
                'sentiment_label': 'positive',
                'summary_text': 'Test summary'
            }
            
            topic = from_dataframe_row(row)
            
            self.assertEqual(topic.topic_id, 'test_topic')
            self.assertEqual(topic.topic_label, 'Test Topic')
            self.assertEqual(topic.importance_score, 0.85)
        
        def test_validation_bounds(self):
            """Test that values are clamped to valid ranges."""
            topic = ExploreTopic(
                topic_id='test',
                topic_label='Test',
                importance_score=-1.0,  # Should be clamped to 0.0
                coverage_pct=150.0,  # Should be clamped to 100.0
                mentions_count=-5,  # Should be clamped to 0
                sentiment_score=2.0,  # Should be clamped to 1.0
                sentiment_label='positive',
                summary_text='Test'
            )
            
            self.assertEqual(topic.importance_score, 0.0)
            self.assertEqual(topic.coverage_pct, 100.0)
            self.assertEqual(topic.mentions_count, 0)
            self.assertEqual(topic.sentiment_score, 1.0)
        
        def test_invalid_sentiment_label(self):
            """Test that invalid sentiment labels are converted to unknown."""
            topic = ExploreTopic(
                topic_id='test',
                topic_label='Test',
                importance_score=0.5,
                coverage_pct=50.0,
                mentions_count=10,
                sentiment_score=0.0,
                sentiment_label='invalid_label',  # Invalid
                summary_text='Test'
            )
            
            self.assertEqual(topic.sentiment_label, 'unknown')
        
        def test_get_signal_bucket_high(self):
            """Test signal bucket assignment for high signal."""
            topic = ExploreTopic(
                topic_id='test',
                topic_label='Test',
                importance_score=2.0,  # >= 1.9
                coverage_pct=95.0,  # >= 90
                mentions_count=20,
                sentiment_score=0.5,
                sentiment_label='positive',
                summary_text='Test'
            )
            
            self.assertEqual(topic.get_signal_bucket(), 'high')
        
        def test_get_signal_bucket_medium(self):
            """Test signal bucket assignment for medium signal."""
            # Test with importance_score >= 1.7 but not high
            topic1 = ExploreTopic(
                topic_id='test1',
                topic_label='Test1',
                importance_score=1.8,  # >= 1.7 but < 1.9
                coverage_pct=80.0,  # < 90
                mentions_count=15,
                sentiment_score=0.3,
                sentiment_label='positive',
                summary_text='Test'
            )
            
            self.assertEqual(topic1.get_signal_bucket(), 'medium')
            
            # Test with importance_score >= 1.9 but coverage < 90
            topic2 = ExploreTopic(
                topic_id='test2',
                topic_label='Test2',
                importance_score=2.0,  # >= 1.9
                coverage_pct=85.0,  # < 90
                mentions_count=15,
                sentiment_score=0.3,
                sentiment_label='positive',
                summary_text='Test'
            )
            
            self.assertEqual(topic2.get_signal_bucket(), 'medium')
        
        def test_get_signal_bucket_low(self):
            """Test signal bucket assignment for low signal."""
            topic = ExploreTopic(
                topic_id='test',
                topic_label='Test',
                importance_score=1.5,  # < 1.7
                coverage_pct=70.0,
                mentions_count=10,
                sentiment_score=0.0,
                sentiment_label='neutral',
                summary_text='Test'
            )
            
            self.assertEqual(topic.get_signal_bucket(), 'low')
        
        def test_rank_topics_primary_sort(self):
            """Test ranking by primary sort (importance_score)."""
            topics = [
                ExploreTopic(topic_id='t1', topic_label='T1', importance_score=1.5,
                            coverage_pct=80.0, mentions_count=10, sentiment_score=0.5,
                            sentiment_label='positive', summary_text='Summary 1'),
                ExploreTopic(topic_id='t2', topic_label='T2', importance_score=2.0,
                            coverage_pct=70.0, mentions_count=5, sentiment_score=0.3,
                            sentiment_label='positive', summary_text='Summary 2'),
                ExploreTopic(topic_id='t3', topic_label='T3', importance_score=1.8,
                            coverage_pct=90.0, mentions_count=15, sentiment_score=0.4,
                            sentiment_label='positive', summary_text='Summary 3')
            ]
            
            ranked = rank_topics(topics)
            
            # Should be sorted by importance_score descending
            self.assertEqual(ranked[0].topic_id, 't2')  # 2.0
            self.assertEqual(ranked[1].topic_id, 't3')  # 1.8
            self.assertEqual(ranked[2].topic_id, 't1')  # 1.5
        
        def test_rank_topics_secondary_sort(self):
            """Test ranking by secondary sort (coverage_pct) when importance_score is equal."""
            topics = [
                ExploreTopic(topic_id='t1', topic_label='T1', importance_score=2.0,
                            coverage_pct=80.0, mentions_count=10, sentiment_score=0.5,
                            sentiment_label='positive', summary_text='Summary 1'),
                ExploreTopic(topic_id='t2', topic_label='T2', importance_score=2.0,
                            coverage_pct=95.0, mentions_count=5, sentiment_score=0.3,
                            sentiment_label='positive', summary_text='Summary 2'),
                ExploreTopic(topic_id='t3', topic_label='T3', importance_score=2.0,
                            coverage_pct=70.0, mentions_count=15, sentiment_score=0.4,
                            sentiment_label='positive', summary_text='Summary 3')
            ]
            
            ranked = rank_topics(topics)
            
            # Should be sorted by coverage_pct descending (secondary sort)
            self.assertEqual(ranked[0].topic_id, 't2')  # 95.0
            self.assertEqual(ranked[1].topic_id, 't1')  # 80.0
            self.assertEqual(ranked[2].topic_id, 't3')  # 70.0
        
        def test_rank_topics_tertiary_sort(self):
            """Test ranking by tertiary sort (mentions_count) when importance and coverage are equal."""
            topics = [
                ExploreTopic(topic_id='t1', topic_label='T1', importance_score=2.0,
                            coverage_pct=90.0, mentions_count=10, sentiment_score=0.5,
                            sentiment_label='positive', summary_text='Summary 1'),
                ExploreTopic(topic_id='t2', topic_label='T2', importance_score=2.0,
                            coverage_pct=90.0, mentions_count=20, sentiment_score=0.3,
                            sentiment_label='positive', summary_text='Summary 2'),
                ExploreTopic(topic_id='t3', topic_label='T3', importance_score=2.0,
                            coverage_pct=90.0, mentions_count=5, sentiment_score=0.4,
                            sentiment_label='positive', summary_text='Summary 3')
            ]
            
            ranked = rank_topics(topics)
            
            # Should be sorted by mentions_count descending (tertiary sort)
            self.assertEqual(ranked[0].topic_id, 't2')  # 20
            self.assertEqual(ranked[1].topic_id, 't1')  # 10
            self.assertEqual(ranked[2].topic_id, 't3')  # 5
        
        def test_rank_topics_empty_list(self):
            """Test ranking with empty list."""
            ranked = rank_topics([])
            self.assertEqual(ranked, [])
        
        def test_rank_topics_single_topic(self):
            """Test ranking with single topic."""
            topics = [
                ExploreTopic(topic_id='t1', topic_label='T1', importance_score=2.0,
                            coverage_pct=95.0, mentions_count=20, sentiment_score=0.5,
                            sentiment_label='positive', summary_text='Summary 1')
            ]
            
            ranked = rank_topics(topics)
            self.assertEqual(len(ranked), 1)
            self.assertEqual(ranked[0].topic_id, 't1')
        
        def test_attach_evidence_basic(self):
            """Test attaching evidence to a topic."""
            from normalize import EvidenceCell
            
            topic = ExploreTopic(
                topic_id='test_topic',
                topic_label='Test Topic',
                importance_score=2.0,
                coverage_pct=90.0,
                mentions_count=10,
                sentiment_score=0.5,
                sentiment_label='positive',
                summary_text='Test summary'
            )
            
            # Create mock evidence cells
            evidence_cells = [
                EvidenceCell(
                    participant_id='p1',
                    topic_id='test_topic',
                    summary_text='Summary 1',
                    quotes_raw='1. First quote. 2. Second quote.',
                    sentiments_raw='1: positive; 2: positive',
                    participant_meta={}
                ),
                EvidenceCell(
                    participant_id='p2',
                    topic_id='test_topic',
                    summary_text='Summary 2',
                    quotes_raw='1. Third quote.',
                    sentiments_raw='1: neutral',
                    participant_meta={}
                )
            ]
            
            topic_with_evidence = attach_evidence(topic, evidence_cells, max_quotes=5)
            
            self.assertIsNotNone(topic_with_evidence.evidence)
            self.assertIn('quotes', topic_with_evidence.evidence)
            self.assertIn('sentiment_distribution', topic_with_evidence.evidence)
            self.assertIn('confidence_notes', topic_with_evidence.evidence)
            
            # Should have quotes
            self.assertGreater(len(topic_with_evidence.evidence['quotes']), 0)
            self.assertLessEqual(len(topic_with_evidence.evidence['quotes']), 5)
            
            # Should have sentiment distribution
            self.assertIn('positive', topic_with_evidence.evidence['sentiment_distribution'])
        
        def test_attach_evidence_no_quotes(self):
            """Test attaching evidence when no quotes are available."""
            from normalize import EvidenceCell
            
            topic = ExploreTopic(
                topic_id='test_topic',
                topic_label='Test Topic',
                importance_score=1.5,
                coverage_pct=50.0,
                mentions_count=0,
                sentiment_score=0.0,
                sentiment_label='unknown',
                summary_text='Test summary'
            )
            
            # Evidence cells without quotes
            evidence_cells = [
                EvidenceCell(
                    participant_id='p1',
                    topic_id='test_topic',
                    summary_text='Summary only',
                    quotes_raw=None,
                    sentiments_raw=None,
                    participant_meta={}
                )
            ]
            
            topic_with_evidence = attach_evidence(topic, evidence_cells)
            
            self.assertIsNotNone(topic_with_evidence.evidence)
            self.assertEqual(len(topic_with_evidence.evidence['quotes']), 0)
            self.assertEqual(sum(topic_with_evidence.evidence['sentiment_distribution'].values()), 0)
        
        def test_attach_evidence_empty_cells(self):
            """Test attaching evidence when no matching cells found."""
            from normalize import EvidenceCell
            
            topic = ExploreTopic(
                topic_id='test_topic',
                topic_label='Test Topic',
                importance_score=1.5,
                coverage_pct=50.0,
                mentions_count=0,
                sentiment_score=0.0,
                sentiment_label='unknown',
                summary_text='Test summary'
            )
            
            # Evidence cells for different topic
            evidence_cells = [
                EvidenceCell(
                    participant_id='p1',
                    topic_id='other_topic',
                    summary_text='Summary',
                    quotes_raw='1. Quote',
                    sentiments_raw='1: positive',
                    participant_meta={}
                )
            ]
            
            topic_with_evidence = attach_evidence(topic, evidence_cells)
            
            self.assertIsNotNone(topic_with_evidence.evidence)
            self.assertEqual(len(topic_with_evidence.evidence['quotes']), 0)
        
        def test_attach_evidence_max_quotes(self):
            """Test that attach_evidence respects max_quotes limit."""
            from normalize import EvidenceCell
            
            topic = ExploreTopic(
                topic_id='test_topic',
                topic_label='Test Topic',
                importance_score=2.0,
                coverage_pct=90.0,
                mentions_count=20,
                sentiment_score=0.5,
                sentiment_label='positive',
                summary_text='Test summary'
            )
            
            # Create many quotes
            quotes_text = ' '.join([f'{i}. Quote {i}.' for i in range(1, 11)])
            evidence_cells = [
                EvidenceCell(
                    participant_id='p1',
                    topic_id='test_topic',
                    summary_text='Summary',
                    quotes_raw=quotes_text,
                    sentiments_raw='; '.join([f'{i}: positive' for i in range(1, 11)]),
                    participant_meta={}
                )
            ]
            
            topic_with_evidence = attach_evidence(topic, evidence_cells, max_quotes=5)
            
            self.assertLessEqual(len(topic_with_evidence.evidence['quotes']), 5)
    
    unittest.main()


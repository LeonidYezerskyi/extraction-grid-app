"""Data model for Explore table - canonical representation of topics for exploration."""

from dataclasses import dataclass
from typing import Literal, Optional, Tuple
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
    
    # Ranking metadata (computed, not stored in dataclass by default)
    # Use signal_bucket property or add to to_dict() when needed
    
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
    
    unittest.main()


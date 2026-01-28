"""Recipient (Persona / Audience) Layer for tailoring insights."""

from dataclasses import dataclass
from typing import List, Optional
import explore_model


@dataclass
class RecipientProfile:
    """
    Profile for a recipient (persona/audience) that affects topic filtering and prioritization.
    
    Recipients are NOT users - they represent different audiences who will receive
    tailored insights. Each recipient has:
    - Priority topics that should be boosted
    - Deprioritized topics that should be hidden (unless high signal)
    
    This enables deterministic topic filtering without LLM calls.
    """
    recipient_id: str
    label: str  # e.g. "Product", "Marketing", "Leadership"
    priority_topics: List[str]  # List of topic_ids to boost
    deprioritized_topics: List[str]  # List of topic_ids to hide (unless high signal)
    
    def __post_init__(self):
        """Validate and normalize values after initialization."""
        # Ensure strings are not None
        if self.recipient_id is None:
            self.recipient_id = ""
        if self.label is None:
            self.label = ""
        
        # Ensure lists are not None
        if self.priority_topics is None:
            self.priority_topics = []
        if self.deprioritized_topics is None:
            self.deprioritized_topics = []
        
        # Normalize topic_ids (lowercase, strip)
        self.priority_topics = [str(tid).lower().strip() for tid in self.priority_topics if tid]
        self.deprioritized_topics = [str(tid).lower().strip() for tid in self.deprioritized_topics if tid]
    
    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            'recipient_id': self.recipient_id,
            'label': self.label,
            'priority_topics': self.priority_topics,
            'deprioritized_topics': self.deprioritized_topics
        }


def filter_topics_for_recipient(
    topics: List[explore_model.ExploreTopic],
    recipient: RecipientProfile
) -> List[explore_model.ExploreTopic]:
    """
    Filter and rank topics for a specific recipient.
    
    This function applies recipient-specific filtering rules:
    - Boosts priority_topics by +10% importance_score
    - Hides deprioritized_topics unless signal_bucket == "high"
    
    Args:
        topics: List of ExploreTopic instances to filter
        recipient: RecipientProfile with priority and deprioritized topics
    
    Returns:
        New list of ExploreTopic instances, filtered and re-ranked
    
    Example:
        >>> recipient = RecipientProfile(
        ...     recipient_id="product",
        ...     label="Product Team",
        ...     priority_topics=["feature_request", "usability"],
        ...     deprioritized_topics=["pricing"]
        ... )
        >>> filtered = filter_topics_for_recipient(topics, recipient)
    """
    if not topics:
        return []
    
    # Normalize recipient topic lists for matching
    priority_set = {tid.lower().strip() for tid in recipient.priority_topics}
    deprioritized_set = {tid.lower().strip() for tid in recipient.deprioritized_topics}
    
    filtered_topics = []
    
    for topic in topics:
        # Normalize topic_id for matching
        topic_id_normalized = topic.topic_id.lower().strip()
        
        # Check if topic should be hidden (deprioritized and not high signal)
        if topic_id_normalized in deprioritized_set:
            signal_bucket = topic.get_signal_bucket()
            if signal_bucket != "high":
                # Hide this topic (skip it)
                continue
        
        # Create a copy of the topic (immutable pattern)
        # Boost importance_score if it's a priority topic
        boosted_importance = topic.importance_score
        if topic_id_normalized in priority_set:
            # Boost by +10%
            boosted_importance = topic.importance_score * 1.10
        
        # Create new ExploreTopic with boosted importance (if applicable)
        if boosted_importance != topic.importance_score:
            # Need to create a new instance with boosted importance
            # This ensures immutability and proper signal_bucket recalculation
            filtered_topic = explore_model.ExploreTopic(
                topic_id=topic.topic_id,
                topic_label=topic.topic_label,
                importance_score=boosted_importance,
                coverage_pct=topic.coverage_pct,
                mentions_count=topic.mentions_count,
                sentiment_score=topic.sentiment_score,
                sentiment_label=topic.sentiment_label,
                summary_text=topic.summary_text,
                source_document=topic.source_document,
                speaker_id=topic.speaker_id,
                evidence=topic.evidence
            )
            filtered_topics.append(filtered_topic)
        else:
            # No boost needed, use original topic (immutability preserved)
            filtered_topics.append(topic)
    
    # Re-rank topics with new importance scores
    ranked_topics = explore_model.rank_topics(filtered_topics)
    
    return ranked_topics


def create_default_recipients() -> List[RecipientProfile]:
    """
    Create default recipient profiles for common personas.
    
    Returns:
        List of default RecipientProfile instances
    """
    return [
        RecipientProfile(
            recipient_id="product",
            label="Product Team",
            priority_topics=[],
            deprioritized_topics=[]
        ),
        RecipientProfile(
            recipient_id="marketing",
            label="Marketing Team",
            priority_topics=[],
            deprioritized_topics=[]
        ),
        RecipientProfile(
            recipient_id="leadership",
            label="Leadership",
            priority_topics=[],
            deprioritized_topics=[]
        ),
        RecipientProfile(
            recipient_id="general",
            label="General",
            priority_topics=[],
            deprioritized_topics=[]
        ),
        RecipientProfile(
            recipient_id="all",
            label="All Recipients",
            priority_topics=[],
            deprioritized_topics=[]
        )
    ]


# Unit tests
if __name__ == '__main__':
    import unittest
    
    class TestRecipientProfile(unittest.TestCase):
        
        def test_recipient_profile_initialization(self):
            """Test basic RecipientProfile initialization."""
            recipient = RecipientProfile(
                recipient_id="product",
                label="Product Team",
                priority_topics=["topic1", "topic2"],
                deprioritized_topics=["topic3"]
            )
            
            self.assertEqual(recipient.recipient_id, "product")
            self.assertEqual(recipient.label, "Product Team")
            self.assertEqual(len(recipient.priority_topics), 2)
            self.assertEqual(len(recipient.deprioritized_topics), 1)
        
        def test_recipient_profile_normalization(self):
            """Test topic_id normalization in RecipientProfile."""
            recipient = RecipientProfile(
                recipient_id="test",
                label="Test",
                priority_topics=["  TOPIC1  ", "Topic2"],
                deprioritized_topics=["  TOPIC3  "]
            )
            
            # Should be normalized to lowercase and stripped
            self.assertIn("topic1", recipient.priority_topics)
            self.assertIn("topic2", recipient.priority_topics)
            self.assertIn("topic3", recipient.deprioritized_topics)
        
        def test_filter_topics_priority_boost(self):
            """Test that priority topics get +10% importance boost."""
            # Create test topics
            topic1 = explore_model.ExploreTopic(
                topic_id="priority_topic",
                topic_label="Priority Topic",
                importance_score=1.0,
                coverage_pct=80.0,
                mentions_count=10,
                sentiment_score=0.5,
                sentiment_label="positive",
                summary_text="Test"
            )
            
            topic2 = explore_model.ExploreTopic(
                topic_id="normal_topic",
                topic_label="Normal Topic",
                importance_score=1.0,
                coverage_pct=80.0,
                mentions_count=10,
                sentiment_score=0.5,
                sentiment_label="positive",
                summary_text="Test"
            )
            
            topics = [topic1, topic2]
            
            # Create recipient with priority topic
            recipient = RecipientProfile(
                recipient_id="test",
                label="Test",
                priority_topics=["priority_topic"],
                deprioritized_topics=[]
            )
            
            filtered = filter_topics_for_recipient(topics, recipient)
            
            # Find priority topic in filtered list
            priority_filtered = next((t for t in filtered if t.topic_id == "priority_topic"), None)
            normal_filtered = next((t for t in filtered if t.topic_id == "normal_topic"), None)
            
            self.assertIsNotNone(priority_filtered)
            self.assertIsNotNone(normal_filtered)
            
            # Priority topic should have boosted importance (1.0 * 1.10 = 1.10)
            self.assertAlmostEqual(priority_filtered.importance_score, 1.10, places=2)
            # Normal topic should remain unchanged
            self.assertEqual(normal_filtered.importance_score, 1.0)
            
            # Priority topic should rank higher after boost
            self.assertEqual(filtered[0].topic_id, "priority_topic")
        
        def test_filter_topics_deprioritized_hide(self):
            """Test that deprioritized topics are hidden unless high signal."""
            # Create test topics
            topic_low = explore_model.ExploreTopic(
                topic_id="deprioritized_low",
                topic_label="Deprioritized Low",
                importance_score=1.5,  # Low signal (< 1.7)
                coverage_pct=70.0,
                mentions_count=5,
                sentiment_score=0.0,
                sentiment_label="neutral",
                summary_text="Test"
            )
            
            topic_high = explore_model.ExploreTopic(
                topic_id="deprioritized_high",
                topic_label="Deprioritized High",
                importance_score=2.0,  # High signal (>= 1.9)
                coverage_pct=95.0,  # >= 90
                mentions_count=20,
                sentiment_score=0.5,
                sentiment_label="positive",
                summary_text="Test"
            )
            
            topics = [topic_low, topic_high]
            
            # Create recipient that deprioritizes both
            recipient = RecipientProfile(
                recipient_id="test",
                label="Test",
                priority_topics=[],
                deprioritized_topics=["deprioritized_low", "deprioritized_high"]
            )
            
            filtered = filter_topics_for_recipient(topics, recipient)
            
            # Low signal topic should be hidden
            low_filtered = next((t for t in filtered if t.topic_id == "deprioritized_low"), None)
            self.assertIsNone(low_filtered)
            
            # High signal topic should still be visible
            high_filtered = next((t for t in filtered if t.topic_id == "deprioritized_high"), None)
            self.assertIsNotNone(high_filtered)
        
        def test_filter_topics_re_ranking(self):
            """Test that filtered topics are re-ranked after boosting."""
            # Create topics with different importance scores
            topic1 = explore_model.ExploreTopic(
                topic_id="topic1",
                topic_label="Topic 1",
                importance_score=2.0,  # Highest
                coverage_pct=90.0,
                mentions_count=20,
                sentiment_score=0.5,
                sentiment_label="positive",
                summary_text="Test"
            )
            
            topic2 = explore_model.ExploreTopic(
                topic_id="priority_topic",
                topic_label="Priority Topic",
                importance_score=1.8,  # Lower, but will be boosted
                coverage_pct=85.0,
                mentions_count=15,
                sentiment_score=0.5,
                sentiment_label="positive",
                summary_text="Test"
            )
            
            topics = [topic1, topic2]
            
            # Create recipient with priority topic
            recipient = RecipientProfile(
                recipient_id="test",
                label="Test",
                priority_topics=["priority_topic"],
                deprioritized_topics=[]
            )
            
            filtered = filter_topics_for_recipient(topics, recipient)
            
            # After boost, priority_topic should have 1.8 * 1.10 = 1.98
            # topic1 has 2.0, so topic1 should still rank first
            # But priority_topic should be boosted
            priority_filtered = next((t for t in filtered if t.topic_id == "priority_topic"), None)
            self.assertIsNotNone(priority_filtered)
            self.assertAlmostEqual(priority_filtered.importance_score, 1.98, places=2)
    
    unittest.main()


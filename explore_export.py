"""Export-ready architecture for Explore view with Recipients."""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from datetime import datetime
import explore_model
import recipient


@dataclass
class ExploreExportPayload:
    """
    Export-ready payload structure for Explore view with Recipient context.
    
    This structure provides a single, deterministic DTO for exporting Explore topics
    to various formats (PPT, PDF, CSV) while preserving recipient context and ordering.
    
    Fields:
    - recipient_profile: The recipient profile used for filtering
    - ordered_topics: List of topics with signal_bucket included
    - generated_at: Timestamp when export was generated
    - source_documents: List of source document identifiers (participant IDs)
    """
    recipient_profile: recipient.RecipientProfile
    ordered_topics: List[Dict[str, Any]]  # List of topic dicts with signal_bucket
    generated_at: datetime
    source_documents: List[str]  # List of participant IDs (source document identifiers)
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert to dictionary for serialization.
        
        Returns:
            Dictionary representation suitable for JSON/CSV export
        """
        return {
            'recipient_profile': self.recipient_profile.to_dict(),
            'ordered_topics': self.ordered_topics,
            'generated_at': self.generated_at.isoformat(),
            'source_documents': self.source_documents,
            'metadata': {
                'total_topics': len(self.ordered_topics),
                'total_source_documents': len(self.source_documents),
                'export_version': '1.0'
            }
        }


def build_export_payload(
    topics: List[explore_model.ExploreTopic],
    recipient_profile: recipient.RecipientProfile,
    canonical_model: Any,
    include_evidence: bool = False
) -> ExploreExportPayload:
    """
    Build an export-ready payload from topics and recipient profile.
    
    This function:
    - Converts topics to dictionaries with signal_bucket included
    - Extracts source documents (participant IDs) from canonical_model
    - Preserves deterministic ordering
    - Includes recipient context
    
    Args:
        topics: List of ExploreTopic instances (should already be filtered/ranked for recipient)
        recipient_profile: RecipientProfile used for filtering
        canonical_model: CanonicalModel object with participants and evidence_cells
        include_evidence: If True, include evidence data in topic dicts (default: False)
    
    Returns:
        ExploreExportPayload instance ready for export
    
    Example:
        >>> filtered_topics = recipient.filter_topics_for_recipient(ranked_topics, recipient_profile)
        >>> payload = build_export_payload(filtered_topics, recipient_profile, canonical_model)
        >>> payload.to_dict()
    """
    # Convert topics to dictionaries with signal_bucket
    ordered_topics = []
    for topic in topics:
        topic_dict = topic.to_dict(include_signal_bucket=True, include_evidence=include_evidence)
        ordered_topics.append(topic_dict)
    
    # Extract source documents (unique participant IDs from canonical_model)
    source_documents = []
    if hasattr(canonical_model, 'participants'):
        source_documents = [p.participant_id for p in canonical_model.participants if p.participant_id]
    else:
        # Fallback: extract from evidence_cells if participants not available
        if hasattr(canonical_model, 'evidence_cells'):
            unique_participants = set()
            for ec in canonical_model.evidence_cells:
                if ec.participant_id:
                    unique_participants.add(ec.participant_id)
            source_documents = sorted(list(unique_participants))
    
    # Generate timestamp
    generated_at = datetime.now()
    
    # Build payload
    payload = ExploreExportPayload(
        recipient_profile=recipient_profile,
        ordered_topics=ordered_topics,
        generated_at=generated_at,
        source_documents=source_documents
    )
    
    return payload


def get_export_summary(payload: ExploreExportPayload) -> Dict[str, Any]:
    """
    Get a summary of the export payload for preview/validation.
    
    Args:
        payload: ExploreExportPayload instance
    
    Returns:
        Dictionary with summary statistics
    """
    # Count topics by signal bucket
    signal_bucket_counts = {
        'high': 0,
        'medium': 0,
        'low': 0
    }
    
    for topic in payload.ordered_topics:
        bucket = topic.get('signal_bucket', 'low')
        if bucket in signal_bucket_counts:
            signal_bucket_counts[bucket] += 1
    
    # Count topics by sentiment
    sentiment_counts = {}
    for topic in payload.ordered_topics:
        sentiment = topic.get('sentiment_label', 'unknown')
        sentiment_counts[sentiment] = sentiment_counts.get(sentiment, 0) + 1
    
    return {
        'total_topics': len(payload.ordered_topics),
        'total_source_documents': len(payload.source_documents),
        'recipient_label': payload.recipient_profile.label,
        'signal_bucket_distribution': signal_bucket_counts,
        'sentiment_distribution': sentiment_counts,
        'generated_at': payload.generated_at.isoformat()
    }


# Unit tests
if __name__ == '__main__':
    import unittest
    
    class TestExploreExportPayload(unittest.TestCase):
        
        def test_payload_initialization(self):
            """Test basic payload initialization."""
            recipient_profile = recipient.RecipientProfile(
                recipient_id="test",
                label="Test Recipient",
                priority_topics=[],
                deprioritized_topics=[]
            )
            
            topics = [
                explore_model.ExploreTopic(
                    topic_id="topic1",
                    topic_label="Topic 1",
                    importance_score=2.0,
                    coverage_pct=90.0,
                    mentions_count=20,
                    sentiment_score=0.5,
                    sentiment_label="positive",
                    summary_text="Test summary"
                )
            ]
            
            # Mock canonical_model
            class MockCanonicalModel:
                participants = [
                    type('Participant', (), {'participant_id': 'p1'})(),
                    type('Participant', (), {'participant_id': 'p2'})()
                ]
            
            payload = build_export_payload(topics, recipient_profile, MockCanonicalModel())
            
            self.assertEqual(payload.recipient_profile.recipient_id, "test")
            self.assertEqual(len(payload.ordered_topics), 1)
            self.assertEqual(len(payload.source_documents), 2)
            self.assertIsNotNone(payload.generated_at)
        
        def test_payload_to_dict(self):
            """Test payload serialization to dictionary."""
            recipient_profile = recipient.RecipientProfile(
                recipient_id="test",
                label="Test",
                priority_topics=[],
                deprioritized_topics=[]
            )
            
            topics = []
            
            class MockCanonicalModel:
                participants = []
            
            payload = build_export_payload(topics, recipient_profile, MockCanonicalModel())
            payload_dict = payload.to_dict()
            
            self.assertIn('recipient_profile', payload_dict)
            self.assertIn('ordered_topics', payload_dict)
            self.assertIn('generated_at', payload_dict)
            self.assertIn('source_documents', payload_dict)
            self.assertIn('metadata', payload_dict)
        
        def test_payload_includes_signal_bucket(self):
            """Test that signal_bucket is included in topic dicts."""
            recipient_profile = recipient.RecipientProfile(
                recipient_id="test",
                label="Test",
                priority_topics=[],
                deprioritized_topics=[]
            )
            
            topic = explore_model.ExploreTopic(
                topic_id="topic1",
                topic_label="Topic 1",
                importance_score=2.0,
                coverage_pct=95.0,
                mentions_count=20,
                sentiment_score=0.5,
                sentiment_label="positive",
                summary_text="Test"
            )
            
            class MockCanonicalModel:
                participants = []
            
            payload = build_export_payload([topic], recipient_profile, MockCanonicalModel())
            
            self.assertEqual(len(payload.ordered_topics), 1)
            topic_dict = payload.ordered_topics[0]
            self.assertIn('signal_bucket', topic_dict)
            self.assertEqual(topic_dict['signal_bucket'], 'high')
        
        def test_get_export_summary(self):
            """Test export summary generation."""
            recipient_profile = recipient.RecipientProfile(
                recipient_id="test",
                label="Test",
                priority_topics=[],
                deprioritized_topics=[]
            )
            
            topics = [
                explore_model.ExploreTopic(
                    topic_id="topic1",
                    topic_label="Topic 1",
                    importance_score=2.0,
                    coverage_pct=95.0,
                    mentions_count=20,
                    sentiment_score=0.5,
                    sentiment_label="positive",
                    summary_text="Test"
                ),
                explore_model.ExploreTopic(
                    topic_id="topic2",
                    topic_label="Topic 2",
                    importance_score=1.5,
                    coverage_pct=70.0,
                    mentions_count=10,
                    sentiment_score=-0.3,
                    sentiment_label="negative",
                    summary_text="Test"
                )
            ]
            
            class MockCanonicalModel:
                participants = [
                    type('Participant', (), {'participant_id': 'p1'})()
                ]
            
            payload = build_export_payload(topics, recipient_profile, MockCanonicalModel())
            summary = get_export_summary(payload)
            
            self.assertEqual(summary['total_topics'], 2)
            self.assertEqual(summary['total_source_documents'], 1)
            self.assertIn('signal_bucket_distribution', summary)
            self.assertIn('sentiment_distribution', summary)
            self.assertEqual(summary['sentiment_distribution']['positive'], 1)
            self.assertEqual(summary['sentiment_distribution']['negative'], 1)
    
    unittest.main()


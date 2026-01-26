"""Module for digesting and summarizing processed data."""
"""Module for digesting and summarizing processed data."""

from typing import Dict, List, Optional, Any
from collections import defaultdict
import score


def _compute_sentiment_mix(evidence_cells, topic_id: str) -> Dict[str, int]:
    """
    Compute sentiment mix (counts of positive, negative, neutral, mixed, unknown) for a topic.
    
    Args:
        evidence_cells: List of evidence cells
        topic_id: Topic ID to filter by
    
    Returns:
        Dictionary with counts: {'positive': int, 'negative': int, 'neutral': int, 'mixed': int, 'unknown': int}
    """
    import parse_quotes
    import parse_sentiment
    
    sentiment_counts = {
        'positive': 0,
        'negative': 0,
        'neutral': 0,
        'mixed': 0,
        'unknown': 0
    }
    
    for evidence_cell in evidence_cells:
        if evidence_cell.topic_id != topic_id:
            continue
        
        if not evidence_cell.quotes_raw or not evidence_cell.sentiments_raw:
            continue
        
        quote_blocks = parse_quotes.parse_quotes(evidence_cell.quotes_raw)
        if not quote_blocks:
            continue
        
        sentiment_blocks = parse_sentiment.parse_and_align_sentiments(
            evidence_cell.sentiments_raw, quote_blocks
        )
        
        for sentiment_block in sentiment_blocks:
            tone = sentiment_block.get('tone_rollup', 'unknown')
            if tone in sentiment_counts:
                sentiment_counts[tone] += 1
            else:
                sentiment_counts['unknown'] += 1
    
    return sentiment_counts


def _build_topic_card(
    topic_aggregate: Dict[str, Any],
    canonical_model
) -> Dict[str, Any]:
    """
    Build a topic card for a selected topic.
    
    Args:
        topic_aggregate: Topic aggregate dictionary from compute_topic_aggregates
        canonical_model: CanonicalModel object
    
    Returns:
        Topic card dictionary with:
        - topic_id
        - topic_one_liner
        - coverage_rate
        - evidence_count
        - sentiment_mix (dict with counts)
        - proof_quote_preview
        - proof_quote_ref
        - receipt_links (list of participant_id:quote_index references)
    """
    topic_id = topic_aggregate['topic_id']
    
    # Get sentiment mix
    sentiment_mix = _compute_sentiment_mix(canonical_model.evidence_cells, topic_id)
    
    # Build receipt links (all quote references for this topic)
    receipt_links = []
    for evidence_cell in canonical_model.evidence_cells:
        if evidence_cell.topic_id != topic_id:
            continue
        
        if evidence_cell.quotes_raw:
            import parse_quotes
            quote_blocks = parse_quotes.parse_quotes(evidence_cell.quotes_raw)
            for quote_block in quote_blocks:
                receipt_ref = f"{evidence_cell.participant_id}:{quote_block['quote_index']}"
                receipt_links.append(receipt_ref)
    
    return {
        'topic_id': topic_id,
        'topic_one_liner': topic_aggregate.get('topic_one_liner'),
        'coverage_rate': topic_aggregate.get('coverage_rate', 0.0),
        'evidence_count': topic_aggregate.get('evidence_count', 0),
        'sentiment_mix': sentiment_mix,
        'proof_quote_preview': topic_aggregate.get('proof_quote_preview'),
        'proof_quote_ref': topic_aggregate.get('proof_quote_ref'),
        'receipt_links': receipt_links
    }


def _build_takeaways(selected_topics: List[Dict[str, Any]], n_takeaways: int) -> List[Dict[str, Any]]:
    """
    Build takeaways from top N selected topics.
    
    Deterministic MVP: convert topic_one_liner into takeaways.
    
    Args:
        selected_topics: List of topic aggregates, sorted by score (descending)
        n_takeaways: Number of takeaways to generate (3-5)
    
    Returns:
        List of takeaway dictionaries with:
        - takeaway_index: int (1-based)
        - takeaway_text: str (from topic_one_liner)
        - source_topic_id: str
    """
    takeaways = []
    
    # Take top N topics
    top_topics = selected_topics[:n_takeaways]
    
    for idx, topic_aggregate in enumerate(top_topics, start=1):
        topic_one_liner = topic_aggregate.get('topic_one_liner')
        
        # Use topic_one_liner as takeaway text, or fallback to topic_id
        if topic_one_liner and topic_one_liner.strip():
            takeaway_text = topic_one_liner.strip()
        else:
            # Fallback: use topic_id as takeaway
            takeaway_text = f"Topic: {topic_aggregate['topic_id']}"
        
        takeaways.append({
            'takeaway_index': idx,
            'takeaway_text': takeaway_text,
            'source_topic_id': topic_aggregate['topic_id']
        })
    
    return takeaways


def build_digest(
    canonical_model,
    selected_topics: Optional[List[Dict[str, Any]]] = None,
    n_takeaways: int = 5
) -> Dict[str, Any]:
    """
    Build a deterministic digest artifact from selected topics.
    
    Creates a "5-minute digest" with:
    - Takeaways (3-5): Top N selected topics converted to takeaways
    - Topic cards: For each selected topic, includes one-liners, coverage, sentiment mix, proof quotes
    
    Args:
        canonical_model: CanonicalModel object with participants, topics, evidence_cells
        selected_topics: Optional list of topic aggregates from compute_topic_aggregates.
                        If None, computes aggregates and uses top N by score.
        n_takeaways: Number of takeaways to generate (default 5, should be 3-5)
    
    Returns:
        Digest artifact dictionary with:
        - takeaways: List of takeaway dictionaries
        - topic_cards: List of topic card dictionaries
        - metadata: Dictionary with digest metadata (n_takeaways, n_topics, etc.)
    
    Example:
        >>> from normalize import CanonicalModel, Participant, Topic, EvidenceCell
        >>> model = CanonicalModel([], [], [])
        >>> aggregates = score.compute_topic_aggregates(model)
        >>> digest = build_digest(model, aggregates, n_takeaways=3)
        >>> 'takeaways' in digest
        True
        >>> 'topic_cards' in digest
        True
    """
    # Validate n_takeaways
    if n_takeaways < 3:
        n_takeaways = 3
    elif n_takeaways > 5:
        n_takeaways = 5
    
    # Compute topic aggregates if not provided
    if selected_topics is None:
        selected_topics = score.compute_topic_aggregates(canonical_model)
    
    if not selected_topics:
        # Empty digest
        return {
            'takeaways': [],
            'topic_cards': [],
            'metadata': {
                'n_takeaways': 0,
                'n_topics': 0,
                'total_participants': len(canonical_model.participants),
                'total_evidence_cells': len(canonical_model.evidence_cells)
            }
        }
    
    # Build takeaways from top N topics
    takeaways = _build_takeaways(selected_topics, n_takeaways)
    
    # Build topic cards for all selected topics (or top N if we want to limit)
    # For MVP, use all selected topics
    topic_cards = []
    for topic_aggregate in selected_topics:
        topic_card = _build_topic_card(topic_aggregate, canonical_model)
        topic_cards.append(topic_card)
    
    # Build metadata
    metadata = {
        'n_takeaways': len(takeaways),
        'n_topics': len(topic_cards),
        'total_participants': len(canonical_model.participants),
        'total_evidence_cells': len(canonical_model.evidence_cells),
        'n_takeaways_requested': n_takeaways
    }
    
    return {
        'takeaways': takeaways,
        'topic_cards': topic_cards,
        'metadata': metadata
    }


# Unit tests
if __name__ == '__main__':
    import unittest
    from normalize import CanonicalModel, Participant, Topic, EvidenceCell
    
    class TestDigest(unittest.TestCase):
        
        def test_build_digest_structure(self):
            """Test that digest contains expected fields."""
            participants = [Participant('p1', 'P1', {})]
            topics = [Topic('topic1')]
            evidence_cells = [
                EvidenceCell('p1', 'topic1', 'Summary text', '1. Quote.', '1: positive', {})
            ]
            
            model = CanonicalModel(participants, topics, evidence_cells)
            aggregates = score.compute_topic_aggregates(model)
            digest = build_digest(model, aggregates, n_takeaways=3)
            
            self.assertIn('takeaways', digest)
            self.assertIn('topic_cards', digest)
            self.assertIn('metadata', digest)
        
        def test_takeaways_count(self):
            """Test that correct number of takeaways are generated."""
            participants = [Participant('p1', 'P1', {})]
            topics = [Topic('topic1'), Topic('topic2'), Topic('topic3')]
            evidence_cells = [
                EvidenceCell('p1', 'topic1', 'Summary 1', '1. Quote.', None, {}),
                EvidenceCell('p1', 'topic2', 'Summary 2', '1. Quote.', None, {}),
                EvidenceCell('p1', 'topic3', 'Summary 3', '1. Quote.', None, {})
            ]
            
            model = CanonicalModel(participants, topics, evidence_cells)
            aggregates = score.compute_topic_aggregates(model)
            digest = build_digest(model, aggregates, n_takeaways=3)
            
            self.assertEqual(len(digest['takeaways']), 3)
            self.assertEqual(digest['takeaways'][0]['takeaway_index'], 1)
            self.assertEqual(digest['takeaways'][1]['takeaway_index'], 2)
            self.assertEqual(digest['takeaways'][2]['takeaway_index'], 3)
        
        def test_takeaways_content(self):
            """Test that takeaways contain expected fields."""
            participants = [Participant('p1', 'P1', {})]
            topics = [Topic('topic1')]
            evidence_cells = [
                EvidenceCell('p1', 'topic1', 'This is a summary', '1. Quote.', None, {})
            ]
            
            model = CanonicalModel(participants, topics, evidence_cells)
            aggregates = score.compute_topic_aggregates(model)
            digest = build_digest(model, aggregates, n_takeaways=1)
            
            self.assertGreater(len(digest['takeaways']), 0)
            takeaway = digest['takeaways'][0]
            self.assertIn('takeaway_index', takeaway)
            self.assertIn('takeaway_text', takeaway)
            self.assertIn('source_topic_id', takeaway)
            self.assertEqual(takeaway['takeaway_text'], 'This is a summary')
        
        def test_topic_card_structure(self):
            """Test that topic cards contain expected fields."""
            participants = [Participant('p1', 'P1', {})]
            topics = [Topic('topic1')]
            evidence_cells = [
                EvidenceCell('p1', 'topic1', 'Summary', '1. Quote.', '1: positive', {})
            ]
            
            model = CanonicalModel(participants, topics, evidence_cells)
            aggregates = score.compute_topic_aggregates(model)
            digest = build_digest(model, aggregates)
            
            self.assertGreater(len(digest['topic_cards']), 0)
            card = digest['topic_cards'][0]
            
            self.assertIn('topic_id', card)
            self.assertIn('topic_one_liner', card)
            self.assertIn('coverage_rate', card)
            self.assertIn('evidence_count', card)
            self.assertIn('sentiment_mix', card)
            self.assertIn('proof_quote_preview', card)
            self.assertIn('proof_quote_ref', card)
            self.assertIn('receipt_links', card)
        
        def test_sentiment_mix(self):
            """Test that sentiment mix is computed correctly."""
            participants = [Participant('p1', 'P1', {})]
            topics = [Topic('topic1')]
            evidence_cells = [
                EvidenceCell('p1', 'topic1', None, '1. Quote one. 2. Quote two.', '1: positive; 2: negative', {})
            ]
            
            model = CanonicalModel(participants, topics, evidence_cells)
            aggregates = score.compute_topic_aggregates(model)
            digest = build_digest(model, aggregates)
            
            card = digest['topic_cards'][0]
            sentiment_mix = card['sentiment_mix']
            
            self.assertIn('positive', sentiment_mix)
            self.assertIn('negative', sentiment_mix)
            self.assertIn('neutral', sentiment_mix)
            self.assertIn('mixed', sentiment_mix)
            self.assertIn('unknown', sentiment_mix)
            self.assertGreater(sentiment_mix['positive'], 0)
            self.assertGreater(sentiment_mix['negative'], 0)
        
        def test_receipt_links(self):
            """Test that receipt links are generated."""
            participants = [Participant('p1', 'P1', {})]
            topics = [Topic('topic1')]
            evidence_cells = [
                EvidenceCell('p1', 'topic1', None, '1. Quote one. 2. Quote two.', None, {})
            ]
            
            model = CanonicalModel(participants, topics, evidence_cells)
            aggregates = score.compute_topic_aggregates(model)
            digest = build_digest(model, aggregates)
            
            card = digest['topic_cards'][0]
            self.assertIn('receipt_links', card)
            self.assertEqual(len(card['receipt_links']), 2)
            self.assertIn('p1:1', card['receipt_links'])
            self.assertIn('p1:2', card['receipt_links'])
        
        def test_n_takeaways_bounds(self):
            """Test that n_takeaways is bounded to 3-5."""
            participants = [Participant('p1', 'P1', {})]
            topics = [Topic('topic1')]
            evidence_cells = [
                EvidenceCell('p1', 'topic1', 'Summary', '1. Quote.', None, {})
            ]
            
            model = CanonicalModel(participants, topics, evidence_cells)
            aggregates = score.compute_topic_aggregates(model)
            
            # Test lower bound
            digest = build_digest(model, aggregates, n_takeaways=1)
            self.assertGreaterEqual(len(digest['takeaways']), 1)
            
            # Test upper bound
            digest = build_digest(model, aggregates, n_takeaways=10)
            self.assertLessEqual(len(digest['takeaways']), 5)
        
        def test_empty_model(self):
            """Test digest with empty model."""
            model = CanonicalModel([], [], [])
            digest = build_digest(model)
            
            self.assertEqual(len(digest['takeaways']), 0)
            self.assertEqual(len(digest['topic_cards']), 0)
            self.assertIn('metadata', digest)
        
        def test_metadata(self):
            """Test that metadata is included."""
            participants = [Participant('p1', 'P1', {})]
            topics = [Topic('topic1')]
            evidence_cells = [
                EvidenceCell('p1', 'topic1', 'Summary', '1. Quote.', None, {})
            ]
            
            model = CanonicalModel(participants, topics, evidence_cells)
            aggregates = score.compute_topic_aggregates(model)
            digest = build_digest(model, aggregates, n_takeaways=3)
            
            metadata = digest['metadata']
            self.assertIn('n_takeaways', metadata)
            self.assertIn('n_topics', metadata)
            self.assertIn('total_participants', metadata)
            self.assertIn('total_evidence_cells', metadata)
            self.assertEqual(metadata['total_participants'], 1)
            self.assertEqual(metadata['total_evidence_cells'], 1)
    
    unittest.main()
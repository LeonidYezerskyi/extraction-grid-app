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
        
        # Parse and align sentiments
        sentiment_blocks = parse_sentiment.parse_and_align_sentiments(
            evidence_cell.sentiments_raw, quote_blocks
        )
        
        # Count sentiments from all blocks
        for sentiment_block in sentiment_blocks:
            tone = sentiment_block.get('tone_rollup', 'unknown')
            labels = sentiment_block.get('labels', [])
            
            # If tone is unknown but we have labels, try to reclassify them
            # This helps catch cases where labels weren't properly classified
            if tone == 'unknown' and labels:
                # Try to classify each label individually
                import parse_sentiment as ps
                classifications = []
                for label in labels:
                    # Try direct classification
                    classification = ps._classify_label(label)
                    if classification:
                        classifications.append(classification)
                    else:
                        # Try substring matching with comprehensive keyword lists
                        label_lower = label.lower()
                        # Positive keywords
                        if any(kw in label_lower for kw in ['positive', 'pos', 'good', 'favorable', 'optimistic', 
                                                           'happy', 'satisfied', 'yes', 'agree', 'support', 'pro', 
                                                           'like', 'love', 'excellent', 'great', 'wonderful', 'upbeat', 
                                                           'cheerful', 'pleased', 'enthusiastic', 'excited', 'hopeful',
                                                           'joy', 'joyful', 'delighted', 'thrilled', 'proud', 'grateful',
                                                           'admiration', 'appreciation', 'approval', 'optimism', 'trust',
                                                           'enjoyment', 'pleasure', 'delight', 'fun', 'content']):
                            classifications.append('positive')
                        # Negative keywords (including annoyance and other emotions)
                        elif any(kw in label_lower for kw in ['negative', 'neg', 'bad', 'unfavorable', 'pessimistic',
                                                             'sad', 'dissatisfied', 'no', 'disagree', 'against', 'oppose',
                                                             'con', 'dislike', 'hate', 'poor', 'terrible', 'awful',
                                                             'upset', 'worried', 'concerned', 'frustrated', 'annoyed',
                                                             'annoyance', 'irritated', 'irritation', 'aggravated', 'bothered',
                                                             'anger', 'angry', 'mad', 'furious', 'disgust', 'disgusted',
                                                             'fear', 'afraid', 'scared', 'sadness', 'depressed', 'shame',
                                                             'ashamed', 'contempt', 'worry', 'stress', 'stressed', 'confusion',
                                                             'confused', 'boredom', 'bored', 'loneliness', 'lonely', 'jealousy',
                                                             'jealous', 'regret', 'regretful', 'skeptical', 'doubtful', 'critical']):
                            classifications.append('negative')
                        # Neutral keywords
                        elif any(kw in label_lower for kw in ['neutral', 'neut', 'none', 'n/a', 'na', 'unclear',
                                                             'mixed', 'both', 'ok', 'okay', 'fine', 'average', 'moderate',
                                                             'calm', 'peaceful', 'serene', 'curious', 'curiosity', 'surprised',
                                                             'surprise', 'contemplative', 'thoughtful', 'accepting', 'acceptance',
                                                             'balanced', 'even', 'equal', 'similar', 'comparable']):
                            classifications.append('neutral')
                
                # Recompute tone from classifications
                if classifications:
                    has_positive = 'positive' in classifications
                    has_negative = 'negative' in classifications
                    has_neutral = 'neutral' in classifications
                    
                    if has_positive and has_negative:
                        tone = 'mixed'
                    elif has_positive and not has_negative:
                        tone = 'positive'
                    elif has_negative and not has_positive:
                        tone = 'negative'
                    elif has_neutral and not has_positive and not has_negative:
                        tone = 'neutral'
            
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
    # Use set to ensure uniqueness, then convert to list
    receipt_links_set = set()
    for evidence_cell in canonical_model.evidence_cells:
        if evidence_cell.topic_id != topic_id:
            continue
        
        if evidence_cell.quotes_raw:
            import parse_quotes
            quote_blocks = parse_quotes.parse_quotes(evidence_cell.quotes_raw)
            for quote_block in quote_blocks:
                receipt_ref = f"{evidence_cell.participant_id}:{quote_block['quote_index']}"
                receipt_links_set.add(receipt_ref)
    
    # Convert to list to maintain order (or use sorted if order matters)
    receipt_links = list(receipt_links_set)
    
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


def _transform_to_takeaway(
    topic_one_liner: str,
    topic_id: str,
    coverage_rate: float,
    evidence_count: int,
    index: int
) -> str:
    """
    Transform topic_one_liner into an implication-style takeaway.
    
    Uses rule-based framing to convert analytical topic descriptions into
    executive-focused insights. Ensures output is different from input.
    
    Args:
        topic_one_liner: Original topic description
        topic_id: Topic identifier
        coverage_rate: Coverage rate (0.0-1.0)
        evidence_count: Number of evidence excerpts
        index: Takeaway index (0-based) for deterministic variation when multiple strategies apply
    
    Returns:
        Implication-style takeaway text (max 180 chars)
    """
    if not topic_one_liner or not topic_one_liner.strip():
        # Fallback: create takeaway from topic_id
        topic_label = topic_id.replace('_', ' ').title()
        return f"Key finding: {topic_label} emerged as a significant theme."
    
    text = topic_one_liner.strip()
    text_lower = text.lower()
    
    # Determine framing strategy based on content and metrics
    # Use index for deterministic variation when multiple strategies apply
    strategies = []
    
    # High coverage suggests broad consensus
    if coverage_rate >= 0.5:
        strategies.append('consensus')
    
    # High evidence count suggests strong signal
    if evidence_count >= 10:
        strategies.append('strong_signal')
    
    # Check for value/benefit keywords
    value_keywords = ['value', 'benefit', 'advantage', 'strength', 'positive', 'good', 'effective', 'success']
    if any(kw in text_lower for kw in value_keywords):
        strategies.append('value')
    
    # Check for problem/friction keywords
    problem_keywords = ['problem', 'issue', 'challenge', 'difficulty', 'barrier', 'friction', 'negative', 'concern']
    if any(kw in text_lower for kw in problem_keywords):
        strategies.append('problem')
    
    # Check for behavior/pattern keywords
    behavior_keywords = ['participants', 'users', 'people', 'consistently', 'often', 'frequently', 'tend', 'typically']
    if any(kw in text_lower for kw in behavior_keywords):
        strategies.append('behavior')
    
    # Select strategy deterministically (use index to vary)
    if not strategies:
        # Default: use value framing
        strategy = 'value'
    else:
        strategy = strategies[index % len(strategies)]
    
    # Apply transformation based on strategy
    if strategy == 'consensus':
        # Consensus framing: emphasize broad agreement
        if text[0].isupper():
            takeaway = f"Participants widely agree: {text[0].lower() + text[1:]}"
        else:
            takeaway = f"Participants widely agree: {text}"
    elif strategy == 'strong_signal':
        # Strong signal framing: emphasize evidence strength
        if text[0].isupper():
            takeaway = f"Strong evidence indicates: {text[0].lower() + text[1:]}"
        else:
            takeaway = f"Strong evidence indicates: {text}"
    elif strategy == 'value':
        # Value framing: emphasize benefits
        if text[0].isupper():
            takeaway = f"The platform's value lies in: {text[0].lower() + text[1:]}"
        else:
            takeaway = f"The platform's value lies in: {text}"
    elif strategy == 'problem':
        # Problem framing: emphasize constraints
        if text[0].isupper():
            takeaway = f"Adoption is constrained by: {text[0].lower() + text[1:]}"
        else:
            takeaway = f"Adoption is constrained by: {text}"
    elif strategy == 'behavior':
        # Behavior framing: emphasize patterns
        if text[0].isupper():
            takeaway = f"Participants consistently emphasize: {text[0].lower() + text[1:]}"
        else:
            takeaway = f"Participants consistently emphasize: {text}"
    else:
        # Default: implication framing
        if text[0].isupper():
            takeaway = f"Key insight: {text[0].lower() + text[1:]}"
        else:
            takeaway = f"Key insight: {text}"
    
    # Ensure takeaway is different from original (add prefix if identical)
    if takeaway.strip() == text:
        takeaway = f"Key finding: {text}"
    
    # Enforce 180 character limit
    if len(takeaway) > 180:
        # Truncate at word boundary
        truncated = takeaway[:177]
        last_space = truncated.rfind(' ')
        if last_space > 140:  # Only truncate at word if we keep most of the text
            takeaway = truncated[:last_space] + '...'
        else:
            takeaway = truncated + '...'
    
    return takeaway


def _build_takeaways(selected_topics: List[Dict[str, Any]], n_takeaways: int) -> List[Dict[str, Any]]:
    """
    Build takeaways from top N selected topics.
    
    Converts topic_one_liner into implication-style takeaways using rule-based
    transformation. Ensures takeaways are distinct from topic_one_liner.
    
    Args:
        selected_topics: List of topic aggregates, sorted by score (descending)
        n_takeaways: Number of takeaways to generate (3-5)
    
    Returns:
        List of takeaway dictionaries with:
        - takeaway_index: int (1-based)
        - takeaway_text: str (implication-style, max 180 chars)
        - source_topic_id: str
        - evidence_count: int (from topic aggregate)
        - proof_quote_preview: str (from topic aggregate)
    """
    takeaways = []
    
    # Take top N topics
    top_topics = selected_topics[:n_takeaways]
    
    for idx, topic_aggregate in enumerate(top_topics, start=1):
        topic_one_liner = topic_aggregate.get('topic_one_liner', '')
        topic_id = topic_aggregate.get('topic_id', '')
        coverage_rate = topic_aggregate.get('coverage_rate', 0.0)
        evidence_count = topic_aggregate.get('evidence_count', 0)
        proof_quote_preview = topic_aggregate.get('proof_quote_preview', '')
        
        # Transform topic_one_liner into implication-style takeaway
        takeaway_text = _transform_to_takeaway(
            topic_one_liner,
            topic_id,
            coverage_rate,
            evidence_count,
            idx - 1  # 0-based index for strategy selection
        )
        
        # Ensure takeaway is different from topic_one_liner
        if takeaway_text.strip() == topic_one_liner.strip():
            # Force differentiation by adding prefix
            takeaway_text = f"Key insight: {topic_one_liner}"
            if len(takeaway_text) > 180:
                takeaway_text = takeaway_text[:177] + '...'
        
        takeaways.append({
            'takeaway_index': idx,
            'takeaway_text': takeaway_text,
            'source_topic_id': topic_id,
            'evidence_count': evidence_count,
            'proof_quote_preview': proof_quote_preview
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
            self.assertIn('evidence_count', takeaway)
            self.assertIn('proof_quote_preview', takeaway)
            # Takeaway text should be transformed, not identical to topic_one_liner
            self.assertNotEqual(takeaway['takeaway_text'], 'This is a summary')
            # But should contain the essence of the summary
            self.assertIn('summary', takeaway['takeaway_text'].lower())
        
        def test_takeaway_text_different_from_topic_one_liner(self):
            """Test that takeaway text is different from topic_one_liner."""
            participants = [Participant('p1', 'P1', {})]
            topics = [Topic('topic1'), Topic('topic2')]
            evidence_cells = [
                EvidenceCell('p1', 'topic1', 'Users value the platform', '1. Quote.', None, {}),
                EvidenceCell('p1', 'topic2', 'There are technical challenges', '1. Quote.', None, {})
            ]
            
            model = CanonicalModel(participants, topics, evidence_cells)
            aggregates = score.compute_topic_aggregates(model)
            digest = build_digest(model, aggregates, n_takeaways=2)
            
            self.assertEqual(len(digest['takeaways']), 2)
            
            # Check each takeaway
            for takeaway in digest['takeaways']:
                takeaway_text = takeaway['takeaway_text']
                source_topic_id = takeaway['source_topic_id']
                
                # Find corresponding topic card
                topic_card = next(
                    (tc for tc in digest['topic_cards'] if tc['topic_id'] == source_topic_id),
                    None
                )
                self.assertIsNotNone(topic_card, f"Topic card not found for {source_topic_id}")
                
                topic_one_liner = topic_card.get('topic_one_liner', '')
                if topic_one_liner:
                    # Takeaway text must be different from topic_one_liner
                    self.assertNotEqual(
                        takeaway_text.strip(),
                        topic_one_liner.strip(),
                        f"Takeaway text should differ from topic_one_liner: {takeaway_text}"
                    )
        
        def test_takeaway_text_length(self):
            """Test that takeaway text respects 180 character limit."""
            participants = [Participant('p1', 'P1', {})]
            topics = [Topic('topic1')]
            # Create a very long summary
            long_summary = 'This is a very long summary that goes on and on ' * 10
            evidence_cells = [
                EvidenceCell('p1', 'topic1', long_summary, '1. Quote.', None, {})
            ]
            
            model = CanonicalModel(participants, topics, evidence_cells)
            aggregates = score.compute_topic_aggregates(model)
            digest = build_digest(model, aggregates, n_takeaways=1)
            
            takeaway = digest['takeaways'][0]
            takeaway_text = takeaway['takeaway_text']
            # Takeaway should be truncated to max 180 chars
            self.assertLessEqual(len(takeaway_text), 180)
        
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
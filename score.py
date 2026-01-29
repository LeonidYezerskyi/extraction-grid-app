"""Module for scoring and rating data based on various criteria."""

import math
from typing import Dict, List, Optional, Any, Tuple
from collections import defaultdict
import parse_quotes
import parse_sentiment


# Moderator-related speaker tags that should be penalized
MODERATOR_TAGS = {
    'moderator', 'mod', 'facilitator', 'interviewer', 'researcher',
    'admin', 'administrator', 'coordinator', 'organizer'
}


def _is_moderator_tag(speaker_tags: Optional[List[str]]) -> bool:
    """
    Check if speaker tags indicate a moderator.
    
    Args:
        speaker_tags: List of speaker tags or None
    
    Returns:
        True if any tag suggests a moderator
    """
    if not speaker_tags:
        return False
    
    for tag in speaker_tags:
        tag_lower = tag.lower()
        if any(mod_tag in tag_lower for mod_tag in MODERATOR_TAGS):
            return True
    
    return False


def _is_readable_length(text: str, min_length: int = 20, max_length: int = 500) -> bool:
    """
    Check if text length is in a readable range.
    
    Args:
        text: Text to check
        min_length: Minimum readable length
        max_length: Maximum readable length
    
    Returns:
        True if text length is in readable range
    """
    if not text:
        return False
    length = len(text)
    return min_length <= length <= max_length


def _is_valid_quote_text(text: str, min_length: int = 10) -> bool:
    """
    Validate that quote text is meaningful and not just a placeholder/index.
    
    Checks:
    - Not empty or whitespace only
    - Minimum length requirement
    - Contains actual words (not just numbers/punctuation)
    - Not just a numeric index like "1." or "1)"
    
    Args:
        text: Quote text to validate
        min_length: Minimum character length (default 10)
    
    Returns:
        True if quote is valid and meaningful
    """
    if not text or not text.strip():
        return False
    
    text_stripped = text.strip()
    
    # Check minimum length
    if len(text_stripped) < min_length:
        return False
    
    # Check if it's just a numeric index pattern (e.g., "1.", "1)", "(1)", "1. ")
    import re
    numeric_patterns = [
        r'^\d+\.?\s*$',  # "1", "1.", "1. "
        r'^\d+\)\s*$',   # "1)"
        r'^\(\d+\)\s*$', # "(1)"
    ]
    for pattern in numeric_patterns:
        if re.match(pattern, text_stripped):
            return False
    
    # Check if it contains actual words (at least one letter)
    # Remove common punctuation and check for letters
    text_no_punct = re.sub(r'[^\w\s]', '', text_stripped)
    if not re.search(r'[a-zA-Z]', text_no_punct):
        # No letters found, likely just numbers/punctuation
        return False
    
    # Check if it's mostly punctuation/whitespace
    if len(text_no_punct.strip()) < min_length // 2:
        return False
    
    return True


def _get_quote_blocks_for_evidence(evidence_cell) -> List[Dict[str, Any]]:
    """
    Get parsed quote blocks for an evidence cell.
    
    Args:
        evidence_cell: EvidenceCell object with quotes_raw
    
    Returns:
        List of quote blocks
    """
    if not evidence_cell.quotes_raw:
        return []
    
    return parse_quotes.parse_quotes(evidence_cell.quotes_raw)


def _get_sentiment_blocks_for_evidence(evidence_cell, quote_blocks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Get parsed sentiment blocks for an evidence cell.
    
    Args:
        evidence_cell: EvidenceCell object with sentiments_raw
        quote_blocks: List of quote blocks to align to
    
    Returns:
        List of sentiment blocks
    """
    if not evidence_cell.sentiments_raw:
        return []
    
    return parse_sentiment.parse_and_align_sentiments(evidence_cell.sentiments_raw, quote_blocks)


def compute_topic_aggregates(canonical_model) -> List[Dict[str, Any]]:
    """
    Compute topic aggregates and scores for each topic.
    
    For each topic, computes:
    - coverage_count: Number of participants with any content for this topic
    - coverage_rate: coverage_count / total_participants
    - evidence_count: Total number of quote blocks across all participants
    - intensity_rate: Non-neutral quote blocks / total quote blocks
    - topic_score: 0.5*coverage_rate + 0.3*log1p(evidence_count) + 0.2*intensity_rate
    - topic_one_liner: Aggregated summary text (first non-empty summary, truncated elsewhere)
    - proof_quote_ref: Reference to selected proof quote (participant_id + quote_index)
    - proof_quote_preview: Preview text of proof quote
    
    Args:
        canonical_model: CanonicalModel object with participants, topics, evidence_cells
    
    Returns:
        List of topic aggregate dictionaries, sorted by topic_score (descending) with
        alphabetical tiebreak on topic_id
    """
    total_participants = len(canonical_model.participants)
    
    # Group evidence cells by topic
    topic_evidence = defaultdict(list)
    for evidence_cell in canonical_model.evidence_cells:
        topic_evidence[evidence_cell.topic_id].append(evidence_cell)
    
    topic_aggregates = []
    
    for topic in canonical_model.topics:
        topic_id = topic.topic_id
        evidence_cells = topic_evidence[topic_id]
        
        if not evidence_cells:
            # No evidence for this topic
            topic_aggregates.append({
                'topic_id': topic_id,
                'coverage_count': 0,
                'coverage_rate': 0.0,
                'evidence_count': 0,
                'intensity_rate': 0.0,
                'topic_score': 0.0,
                'topic_one_liner': None,
                'proof_quote_ref': None,
                'proof_quote_preview': None
            })
            continue
        
        # Compute coverage
        participants_with_content = set()
        all_quote_blocks = []  # List of (participant_id, quote_block) tuples
        all_sentiment_blocks = []  # List of (participant_id, sentiment_block) tuples
        summary_texts = []
        
        for evidence_cell in evidence_cells:
            # Check if participant has any content
            has_content = (
                evidence_cell.summary_text or
                evidence_cell.quotes_raw or
                evidence_cell.sentiments_raw
            )
            
            if has_content:
                participants_with_content.add(evidence_cell.participant_id)
            
            # Collect summary texts
            if evidence_cell.summary_text:
                summary_texts.append(evidence_cell.summary_text)
            
            # Parse quotes and sentiments
            quote_blocks = _get_quote_blocks_for_evidence(evidence_cell)
            if quote_blocks:
                # Store quotes with participant_id
                for quote_block in quote_blocks:
                    all_quote_blocks.append((evidence_cell.participant_id, quote_block))
                
                # Parse sentiments aligned to quotes
                sentiment_blocks = _get_sentiment_blocks_for_evidence(evidence_cell, quote_blocks)
                # Store sentiments with participant_id
                for sentiment_block in sentiment_blocks:
                    all_sentiment_blocks.append((evidence_cell.participant_id, sentiment_block))
        
        coverage_count = len(participants_with_content)
        coverage_rate = coverage_count / total_participants if total_participants > 0 else 0.0
        
        evidence_count = len(all_quote_blocks)
        
        # Compute intensity_rate (non-neutral / total)
        non_neutral_count = 0
        for participant_id, sentiment_block in all_sentiment_blocks:
            if sentiment_block.get('tone_rollup') not in ('neutral', 'unknown', None):
                non_neutral_count += 1
        
        intensity_rate = non_neutral_count / evidence_count if evidence_count > 0 else 0.0
        
        # Compute topic_score
        topic_score = (
            0.5 * coverage_rate +
            0.3 * math.log1p(evidence_count) +
            0.2 * intensity_rate
        )
        
        # Compute topic_one_liner (first non-empty summary, will be truncated elsewhere)
        topic_one_liner = None
        for summary in summary_texts:
            if summary and summary.strip():
                topic_one_liner = summary.strip()
                break
        
        # Select proof quote
        proof_quote_ref, proof_quote_preview = _select_proof_quote(
            topic_id, evidence_cells, 
            [qb for _, qb in all_quote_blocks], 
            [sb for _, sb in all_sentiment_blocks],
            all_quote_blocks, 
            all_sentiment_blocks
        )
        
        topic_aggregates.append({
            'topic_id': topic_id,
            'coverage_count': coverage_count,
            'coverage_rate': coverage_rate,
            'evidence_count': evidence_count,
            'intensity_rate': intensity_rate,
            'topic_score': topic_score,
            'topic_one_liner': topic_one_liner,
            'proof_quote_ref': proof_quote_ref,
            'proof_quote_preview': proof_quote_preview
        })
    
    # Sort by topic_score (descending), then alphabetically by topic_id for tiebreak
    topic_aggregates.sort(key=lambda x: (-x['topic_score'], x['topic_id']))
    
    return topic_aggregates


def _select_proof_quote(
    topic_id: str,
    evidence_cells: List,
    quote_blocks: List[Dict[str, Any]],
    sentiment_blocks: List[Dict[str, Any]],
    quote_blocks_with_participant: List[Tuple[str, Dict[str, Any]]],
    sentiment_blocks_with_participant: List[Tuple[str, Dict[str, Any]]]
) -> Tuple[Optional[str], Optional[str]]:
    """
    Select a proof quote for a topic.
    
    Selection criteria (in order of preference):
    1. Non-neutral tone_rollup
    2. Readable length (not too short/long)
    3. Penalize moderator-heavy speaker tags
    
    Args:
        topic_id: Topic ID
        evidence_cells: List of evidence cells for this topic
        quote_blocks: List of all quote blocks for this topic (without participant_id)
        sentiment_blocks: List of all sentiment blocks for this topic (without participant_id)
        quote_blocks_with_participant: List of (participant_id, quote_block) tuples
        sentiment_blocks_with_participant: List of (participant_id, sentiment_block) tuples
    
    Returns:
        Tuple of (proof_quote_ref, proof_quote_preview)
        proof_quote_ref format: "participant_id:quote_index"
        proof_quote_preview: Preview text of the quote
    """
    if not quote_blocks_with_participant:
        return None, None
    
    # Create a map of (participant_id, quote_index) -> quote_block
    quote_map = {}
    for participant_id, quote_block in quote_blocks_with_participant:
        key = (participant_id, quote_block['quote_index'])
        quote_map[key] = quote_block
    
    # Create a map of (participant_id, quote_index) -> sentiment_block
    sentiment_map = {}
    for participant_id, sentiment_block in sentiment_blocks_with_participant:
        key = (participant_id, sentiment_block['quote_index'])
        sentiment_map[key] = sentiment_block
    
    # Score each quote - FILTER OUT INVALID QUOTES
    scored_quotes = []
    
    for (participant_id, quote_index), quote_block in quote_map.items():
        # Get quote text from both quote_preview and quote_text
        quote_text = quote_block.get('quote_text', '')
        quote_preview = quote_block.get('quote_preview', '')
        
        # Use quote_text as primary, fallback to quote_preview
        primary_text = quote_text if quote_text else quote_preview
        
        # CRITICAL: Skip invalid quotes (placeholders, indices, too short, etc.)
        if not _is_valid_quote_text(primary_text):
            continue  # Skip this quote entirely
        
        score = 0.0
        sentiment_block = sentiment_map.get((participant_id, quote_index))
        
        # Preference 1: Non-neutral tone (higher score)
        if sentiment_block:
            tone = sentiment_block.get('tone_rollup', 'unknown')
            if tone == 'positive':
                score += 10.0
            elif tone == 'negative':
                score += 10.0
            elif tone == 'mixed':
                score += 8.0
            elif tone == 'neutral':
                score += 2.0
            else:  # unknown
                score += 1.0
        else:
            score += 1.0  # No sentiment info
        
        # Preference 2: Readable length (bonus)
        if _is_readable_length(primary_text):
            score += 5.0
        elif len(primary_text) < 20:
            score -= 2.0  # Penalize too short
        elif len(primary_text) > 500:
            score -= 1.0  # Slight penalty for too long
        
        # Preference 3: Penalize moderator tags
        speaker_tags = quote_block.get('speaker_tags')
        if _is_moderator_tag(speaker_tags):
            score -= 5.0
        
        scored_quotes.append({
            'participant_id': participant_id,
            'quote_index': quote_index,
            'quote_block': quote_block,
            'score': score,
            'quote_text': primary_text
        })
    
    # If no valid quotes found after filtering, return fallback message
    if not scored_quotes:
        return None, "No representative quote available"
    
    # Sort by score (descending), then by participant_id and quote_index for tiebreak
    scored_quotes.sort(key=lambda x: (-x['score'], x['participant_id'], x['quote_index']))
    
    # Select top quote and validate it one more time
    top_quote = scored_quotes[0]
    participant_id = top_quote['participant_id']
    quote_index = top_quote['quote_index']
    quote_block = top_quote['quote_block']
    final_quote_text = top_quote['quote_text']
    
    # Final validation - ensure the selected quote is still valid
    if not _is_valid_quote_text(final_quote_text):
        # If somehow an invalid quote got through, try next best
        for quote in scored_quotes[1:]:
            candidate_text = quote['quote_text']
            if _is_valid_quote_text(candidate_text):
                participant_id = quote['participant_id']
                quote_index = quote['quote_index']
                quote_block = quote['quote_block']
                final_quote_text = candidate_text
                break
        else:
            # No valid quote found even after checking all
            return None, "No representative quote available"
    
    proof_quote_ref = f"{participant_id}:{quote_index}"
    
    # Prefer quote_text, fallback to quote_preview, but ensure it's valid
    proof_quote_preview = quote_block.get('quote_text', '')
    if not proof_quote_preview or not _is_valid_quote_text(proof_quote_preview):
        proof_quote_preview = quote_block.get('quote_preview', '')
        if not proof_quote_preview or not _is_valid_quote_text(proof_quote_preview):
            # Use the validated text we already have
            proof_quote_preview = final_quote_text
    
    # Final safety check
    if not _is_valid_quote_text(proof_quote_preview):
        return None, "No representative quote available"
    
    return proof_quote_ref, proof_quote_preview


# Unit tests
if __name__ == '__main__':
    import unittest
    from normalize import CanonicalModel, Participant, Topic, EvidenceCell
    
    class TestScore(unittest.TestCase):
        
        def test_coverage_rate(self):
            """Test coverage rate calculation."""
            participants = [
                Participant('p1', 'Participant 1', {}),
                Participant('p2', 'Participant 2', {}),
                Participant('p3', 'Participant 3', {})
            ]
            topics = [Topic('topic1')]
            evidence_cells = [
                EvidenceCell('p1', 'topic1', 'Summary 1', None, None, {}),
                EvidenceCell('p2', 'topic1', 'Summary 2', None, None, {})
            ]
            
            model = CanonicalModel(participants, topics, evidence_cells)
            aggregates = compute_topic_aggregates(model)
            
            self.assertEqual(len(aggregates), 1)
            self.assertEqual(aggregates[0]['coverage_count'], 2)
            self.assertAlmostEqual(aggregates[0]['coverage_rate'], 2.0 / 3.0, places=5)
        
        def test_evidence_count(self):
            """Test evidence count calculation."""
            participants = [Participant('p1', 'P1', {})]
            topics = [Topic('topic1')]
            evidence_cells = [
                EvidenceCell('p1', 'topic1', None, '1. Quote one. 2. Quote two.', None, {})
            ]
            
            model = CanonicalModel(participants, topics, evidence_cells)
            aggregates = compute_topic_aggregates(model)
            
            self.assertEqual(aggregates[0]['evidence_count'], 2)
        
        def test_intensity_rate(self):
            """Test intensity rate calculation."""
            participants = [Participant('p1', 'P1', {})]
            topics = [Topic('topic1')]
            evidence_cells = [
                EvidenceCell('p1', 'topic1', None, '1. Quote one. 2. Quote two.', '1: positive; 2: neutral', {})
            ]
            
            model = CanonicalModel(participants, topics, evidence_cells)
            aggregates = compute_topic_aggregates(model)
            
            # One positive (non-neutral) out of two quotes
            self.assertAlmostEqual(aggregates[0]['intensity_rate'], 0.5, places=2)
        
        def test_topic_score_formula(self):
            """Test topic score formula."""
            participants = [Participant('p1', 'P1', {}), Participant('p2', 'P2', {})]
            topics = [Topic('topic1')]
            evidence_cells = [
                EvidenceCell('p1', 'topic1', 'Summary', '1. Quote.', '1: positive', {}),
                EvidenceCell('p2', 'topic1', 'Summary', '1. Quote.', '1: positive', {})
            ]
            
            model = CanonicalModel(participants, topics, evidence_cells)
            aggregates = compute_topic_aggregates(model)
            
            agg = aggregates[0]
            expected_score = (
                0.5 * agg['coverage_rate'] +
                0.3 * math.log1p(agg['evidence_count']) +
                0.2 * agg['intensity_rate']
            )
            
            self.assertAlmostEqual(agg['topic_score'], expected_score, places=5)
        
        def test_ranking_alphabetical_tiebreak(self):
            """Test ranking with alphabetical tiebreak."""
            participants = [Participant('p1', 'P1', {})]
            topics = [Topic('topic_a'), Topic('topic_b')]
            evidence_cells = [
                EvidenceCell('p1', 'topic_a', 'Summary', '1. Quote.', None, {}),
                EvidenceCell('p1', 'topic_b', 'Summary', '1. Quote.', None, {})
            ]
            
            model = CanonicalModel(participants, topics, evidence_cells)
            aggregates = compute_topic_aggregates(model)
            
            # Both should have same score, so topic_a should come first (alphabetical)
            self.assertEqual(aggregates[0]['topic_id'], 'topic_a')
            self.assertEqual(aggregates[1]['topic_id'], 'topic_b')
        
        def test_proof_quote_selection(self):
            """Test proof quote selection."""
            participants = [Participant('p1', 'P1', {})]
            topics = [Topic('topic1')]
            evidence_cells = [
                EvidenceCell('p1', 'topic1', None, '1. Short. 2. This is a longer quote with positive sentiment.', '1: neutral; 2: positive', {})
            ]
            
            model = CanonicalModel(participants, topics, evidence_cells)
            aggregates = compute_topic_aggregates(model)
            
            # Should prefer the positive quote (quote 2)
            self.assertIsNotNone(aggregates[0]['proof_quote_ref'])
            self.assertIn('2', aggregates[0]['proof_quote_ref'])
            self.assertIsNotNone(aggregates[0]['proof_quote_preview'])
        
        def test_empty_topic(self):
            """Test topic with no evidence."""
            participants = [Participant('p1', 'P1', {})]
            topics = [Topic('topic1')]
            evidence_cells = []
            
            model = CanonicalModel(participants, topics, evidence_cells)
            aggregates = compute_topic_aggregates(model)
            
            self.assertEqual(aggregates[0]['coverage_count'], 0)
            self.assertEqual(aggregates[0]['evidence_count'], 0)
            self.assertEqual(aggregates[0]['topic_score'], 0.0)
        
        def test_topic_one_liner(self):
            """Test topic_one_liner extraction."""
            participants = [Participant('p1', 'P1', {})]
            topics = [Topic('topic1')]
            evidence_cells = [
                EvidenceCell('p1', 'topic1', 'First summary', None, None, {}),
                EvidenceCell('p1', 'topic1', 'Second summary', None, None, {})
            ]
            
            model = CanonicalModel(participants, topics, evidence_cells)
            aggregates = compute_topic_aggregates(model)
            
            # Should use first non-empty summary
            self.assertEqual(aggregates[0]['topic_one_liner'], 'First summary')
    
    unittest.main()
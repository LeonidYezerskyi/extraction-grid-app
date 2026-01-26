"""Module for handling edge cases in the pipeline."""

import re
from typing import Dict, List, Set, Optional, Any
from collections import defaultdict


def validate_file_readable(validation_report: Dict[str, Any]) -> tuple[bool, Optional[str]]:
    """
    Validate that file is readable and core sheets are present.
    
    Args:
        validation_report: Validation report from ingest.read_workbook
    
    Returns:
        Tuple of (is_valid, error_message)
    """
    if not validation_report.get('is_readable', True):
        return False, validation_report.get('error', 'File is not readable')
    
    # Check for core required sheets
    core_sheets = ['summary']  # At minimum, summary is required
    missing_sheets = validation_report.get('missing_sheets', [])
    missing_core = [s for s in core_sheets if s in missing_sheets]
    
    if missing_core:
        error_msg = f"Core required sheets missing: {', '.join(missing_core)}"
        return False, error_msg
    
    return True, None


def identify_single_sheet_topics(
    canonical_model,
    topic_columns: Set[str]
) -> Set[str]:
    """
    Identify topics that are present in only one sheet.
    
    A topic is considered single-sheet if it only has data in one of:
    summary, quotes, or sentiments.
    
    Args:
        canonical_model: CanonicalModel object
        topic_columns: Set of topic column names
    
    Returns:
        Set of topic IDs that are present in only one sheet
    """
    single_sheet_topics = set()
    
    # Count sheets per topic
    topic_sheet_count = defaultdict(set)
    
    for evidence_cell in canonical_model.evidence_cells:
        topic_id = evidence_cell.topic_id
        if topic_id not in topic_columns:
            continue
        
        # Count which sheets have data for this topic
        if evidence_cell.summary_text:
            topic_sheet_count[topic_id].add('summary')
        if evidence_cell.quotes_raw:
            topic_sheet_count[topic_id].add('quotes')
        if evidence_cell.sentiments_raw:
            topic_sheet_count[topic_id].add('sentiments')
    
    # Topics with only one sheet
    for topic_id, sheets in topic_sheet_count.items():
        if len(sheets) == 1:
            single_sheet_topics.add(topic_id)
    
    return single_sheet_topics


def identify_sparse_topics(
    topic_aggregates: List[Dict[str, Any]],
    min_evidence_count: int = 2,
    min_coverage_rate: float = 0.1
) -> List[str]:
    """
    Identify sparse topics (low evidence or coverage).
    
    Args:
        topic_aggregates: List of topic aggregates
        min_evidence_count: Minimum evidence count threshold
        min_coverage_rate: Minimum coverage rate threshold
    
    Returns:
        List of topic IDs that are considered sparse
    """
    sparse_topics = []
    
    for topic_agg in topic_aggregates:
        evidence_count = topic_agg.get('evidence_count', 0)
        coverage_rate = topic_agg.get('coverage_rate', 0.0)
        
        # Consider sparse if low evidence AND low coverage
        if evidence_count < min_evidence_count and coverage_rate < min_coverage_rate:
            sparse_topics.append(topic_agg['topic_id'])
    
    return sparse_topics


def filter_participants_by_regex(
    canonical_model,
    denylist_patterns: List[str]
) -> tuple:
    """
    Filter out participants matching regex denylist patterns.
    
    Args:
        canonical_model: CanonicalModel object
        denylist_patterns: List of regex patterns to match against participant_id
    
    Returns:
        Tuple of (filtered_canonical_model, filtered_participant_ids)
    """
    if not denylist_patterns:
        return canonical_model, set()
    
    # Compile regex patterns
    compiled_patterns = []
    for pattern in denylist_patterns:
        try:
            compiled_patterns.append(re.compile(pattern, re.IGNORECASE))
        except re.error:
            # Invalid regex, skip
            continue
    
    if not compiled_patterns:
        return canonical_model, set()
    
    # Find participants to filter
    filtered_participant_ids = set()
    for participant in canonical_model.participants:
        participant_id = participant.participant_id
        for pattern in compiled_patterns:
            if pattern.search(participant_id):
                filtered_participant_ids.add(participant_id)
                break
    
    if not filtered_participant_ids:
        return canonical_model, set()
    
    # Create filtered model
    from normalize import CanonicalModel, Participant, Topic, EvidenceCell
    
    filtered_participants = [
        p for p in canonical_model.participants
        if p.participant_id not in filtered_participant_ids
    ]
    
    filtered_evidence_cells = [
        ec for ec in canonical_model.evidence_cells
        if ec.participant_id not in filtered_participant_ids
    ]
    
    filtered_model = CanonicalModel(
        participants=filtered_participants,
        topics=canonical_model.topics,
        evidence_cells=filtered_evidence_cells
    )
    
    return filtered_model, filtered_participant_ids


def handle_sentiment_without_quotes(
    evidence_cell,
    sentiment_blocks: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """
    Handle sentiments that exist without corresponding quotes.
    
    Creates placeholder quote blocks for sentiments without quotes.
    
    Args:
        evidence_cell: EvidenceCell object
        sentiment_blocks: List of sentiment blocks
    
    Returns:
        List of sentiment blocks with 'unknown' tone_rollup for unmatched sentiments
    """
    if not sentiment_blocks:
        return []
    
    # Check if we have quotes
    has_quotes = bool(evidence_cell.quotes_raw)
    
    if not has_quotes:
        # All sentiments are without quotes - mark as unknown
        for sentiment_block in sentiment_blocks:
            sentiment_block['tone_rollup'] = 'unknown'
            sentiment_block['has_quote'] = False
    
    return sentiment_blocks


# Unit tests
if __name__ == '__main__':
    import unittest
    
    class TestEdgeCases(unittest.TestCase):
        
        def test_validate_file_readable_valid(self):
            """Test validation with valid file."""
            report = {
                'is_readable': True,
                'missing_sheets': []
            }
            is_valid, error = validate_file_readable(report)
            self.assertTrue(is_valid)
            self.assertIsNone(error)
        
        def test_validate_file_readable_unreadable(self):
            """Test validation with unreadable file."""
            report = {
                'is_readable': False,
                'error': 'Corrupted file'
            }
            is_valid, error = validate_file_readable(report)
            self.assertFalse(is_valid)
            self.assertIsNotNone(error)
        
        def test_validate_file_readable_missing_core(self):
            """Test validation with missing core sheet."""
            report = {
                'is_readable': True,
                'missing_sheets': ['summary']
            }
            is_valid, error = validate_file_readable(report)
            self.assertFalse(is_valid)
            self.assertIn('summary', error)
        
        def test_identify_single_sheet_topics(self):
            """Test identification of single-sheet topics."""
            from normalize import CanonicalModel, Participant, Topic, EvidenceCell
            
            participants = [Participant('p1', 'P1', {})]
            topics = [Topic('topic1'), Topic('topic2')]
            evidence_cells = [
                # topic1: only summary
                EvidenceCell('p1', 'topic1', 'Summary', None, None, {}),
                # topic2: summary + quotes
                EvidenceCell('p1', 'topic2', 'Summary', 'Quote', None, {})
            ]
            
            model = CanonicalModel(participants, topics, evidence_cells)
            single_sheet = identify_single_sheet_topics(model, {'topic1', 'topic2'})
            
            self.assertIn('topic1', single_sheet)
            self.assertNotIn('topic2', single_sheet)
        
        def test_identify_sparse_topics(self):
            """Test identification of sparse topics."""
            aggregates = [
                {'topic_id': 'topic1', 'evidence_count': 1, 'coverage_rate': 0.05},
                {'topic_id': 'topic2', 'evidence_count': 10, 'coverage_rate': 0.8},
            ]
            
            sparse = identify_sparse_topics(aggregates)
            self.assertIn('topic1', sparse)
            self.assertNotIn('topic2', sparse)
        
        def test_filter_participants_by_regex(self):
            """Test participant filtering by regex."""
            from normalize import CanonicalModel, Participant, Topic, EvidenceCell
            
            participants = [
                Participant('moderator_1', 'Mod', {}),
                Participant('p1', 'P1', {}),
                Participant('admin_1', 'Admin', {})
            ]
            topics = [Topic('topic1')]
            evidence_cells = [
                EvidenceCell('moderator_1', 'topic1', 'Summary', None, None, {}),
                EvidenceCell('p1', 'topic1', 'Summary', None, None, {}),
                EvidenceCell('admin_1', 'topic1', 'Summary', None, None, {})
            ]
            
            model = CanonicalModel(participants, topics, evidence_cells)
            filtered_model, filtered_ids = filter_participants_by_regex(
                model, [r'moderator', r'admin']
            )
            
            self.assertIn('moderator_1', filtered_ids)
            self.assertIn('admin_1', filtered_ids)
            self.assertNotIn('p1', filtered_ids)
            self.assertEqual(len(filtered_model.participants), 1)
        
        def test_handle_sentiment_without_quotes(self):
            """Test handling sentiments without quotes."""
            from normalize import EvidenceCell
            
            evidence_cell = EvidenceCell('p1', 'topic1', None, None, '1: positive', {})
            sentiment_blocks = [
                {'quote_index': 1, 'labels': ['positive'], 'tone_rollup': 'positive'}
            ]
            
            result = handle_sentiment_without_quotes(evidence_cell, sentiment_blocks)
            self.assertEqual(result[0]['tone_rollup'], 'unknown')
            self.assertFalse(result[0]['has_quote'])
    
    unittest.main()

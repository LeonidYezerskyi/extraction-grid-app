"""Tests for edge case handling."""

import pytest
import pandas as pd
import edge_cases
import normalize
import ingest
from tests.conftest import (
    sparse_topic_dfs, single_sheet_topic_dfs, sentiment_without_quotes_dfs,
    duplicate_participant_dfs, sample_excel_bytes
)


class TestEdgeCases:
    """Test suite for edge case handling."""
    
    def test_validate_file_readable_valid(self):
        """Test validation with valid file."""
        report = {
            'is_readable': True,
            'missing_sheets': []
        }
        is_valid, error = edge_cases.validate_file_readable(report)
        assert is_valid is True
        assert error is None
    
    def test_validate_file_readable_unreadable(self):
        """Test validation with unreadable file."""
        report = {
            'is_readable': False,
            'error': 'Corrupted file'
        }
        is_valid, error = edge_cases.validate_file_readable(report)
        assert is_valid is False
        assert error is not None
    
    def test_validate_file_readable_missing_core(self):
        """Test validation with missing core sheet."""
        report = {
            'is_readable': True,
            'missing_sheets': ['summary']
        }
        is_valid, error = edge_cases.validate_file_readable(report)
        assert is_valid is False
        assert 'summary' in error
    
    def test_identify_single_sheet_topics(self, single_sheet_topic_dfs):
        """Test identification of single-sheet topics."""
        topic_columns = ['single_sheet_topic']
        canonical_model = normalize.wide_to_canonical(single_sheet_topic_dfs, topic_columns)
        
        single_sheet = edge_cases.identify_single_sheet_topics(
            canonical_model, set(topic_columns)
        )
        
        assert 'single_sheet_topic' in single_sheet
    
    def test_identify_sparse_topics(self):
        """Test identification of sparse topics."""
        aggregates = [
            {'topic_id': 'topic1', 'evidence_count': 1, 'coverage_rate': 0.05},
            {'topic_id': 'topic2', 'evidence_count': 10, 'coverage_rate': 0.8},
        ]
        
        sparse = edge_cases.identify_sparse_topics(aggregates)
        assert 'topic1' in sparse
        assert 'topic2' not in sparse
    
    def test_filter_participants_by_regex(self, duplicate_participant_dfs):
        """Test participant filtering by regex."""
        topic_columns = ['topic_a']
        canonical_model = normalize.wide_to_canonical(duplicate_participant_dfs, topic_columns)
        
        filtered_model, filtered_ids = edge_cases.filter_participants_by_regex(
            canonical_model, [r'moderator', r'admin']
        )
        
        assert 'moderator_1' in filtered_ids
        assert 'admin_1' in filtered_ids
        assert 'p1' not in filtered_ids
        assert len(filtered_model.participants) < len(canonical_model.participants)
    
    def test_filter_participants_no_match(self, duplicate_participant_dfs):
        """Test participant filtering with no matches."""
        topic_columns = ['topic_a']
        canonical_model = normalize.wide_to_canonical(duplicate_participant_dfs, topic_columns)
        
        filtered_model, filtered_ids = edge_cases.filter_participants_by_regex(
            canonical_model, [r'nonexistent']
        )
        
        assert len(filtered_ids) == 0
        assert len(filtered_model.participants) == len(canonical_model.participants)
    
    def test_handle_sentiment_without_quotes(self, sentiment_without_quotes_dfs):
        """Test handling sentiments without quotes."""
        topic_columns = ['topic_x']
        canonical_model = normalize.wide_to_canonical(sentiment_without_quotes_dfs, topic_columns)
        
        # Find evidence cell with sentiment but no quote
        evidence_cell = next(
            (ec for ec in canonical_model.evidence_cells if ec.topic_id == 'topic_x'),
            None
        )
        
        assert evidence_cell is not None
        assert evidence_cell.sentiments_raw is not None
        assert evidence_cell.quotes_raw is None or not evidence_cell.quotes_raw.strip()
        
        # Parse sentiments
        import parse_sentiment
        sentiment_blocks = parse_sentiment.parse_and_align_sentiments(
            evidence_cell.sentiments_raw, []
        )
        
        # Should handle gracefully
        assert isinstance(sentiment_blocks, list)
    
    def test_sparse_topic_handling(self, sparse_topic_dfs):
        """Test sparse topic is identified correctly."""
        topic_columns = ['sparse_topic']
        canonical_model = normalize.wide_to_canonical(sparse_topic_dfs, topic_columns)
        
        import score
        aggregates = score.compute_topic_aggregates(canonical_model)
        
        sparse_topics = edge_cases.identify_sparse_topics(aggregates)
        assert 'sparse_topic' in sparse_topics
    
    def test_missing_sheet_handling(self):
        """Test handling of missing sheet in workbook."""
        import io
        import pandas as pd
        
        # Create workbook with only summary sheet
        buffer = io.BytesIO()
        with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
            pd.DataFrame({
                'participant_id': ['p1'],
                'topic_a': ['Summary']
            }).to_excel(writer, sheet_name='summary', index=False)
        buffer.seek(0)
        bytes_data = buffer.getvalue()
        
        dict_of_dfs, validation_report = ingest.read_workbook(bytes_data)
        
        assert 'summary' in validation_report['matched_sheets']
        assert 'quotes' in validation_report['missing_sheets']
        assert 'sentiments' in validation_report['missing_sheets']
        
        # Should still be valid (summary is core)
        is_valid, error = edge_cases.validate_file_readable(validation_report)
        assert is_valid is True
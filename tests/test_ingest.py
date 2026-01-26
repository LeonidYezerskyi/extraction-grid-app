"""Tests for ingest module."""

import pytest
import pandas as pd
import ingest
from tests.conftest import sample_excel_bytes, sample_dfs


class TestIngest:
    """Test suite for ingest.read_workbook."""
    
    def test_read_workbook_success(self, sample_excel_bytes):
        """Test successful workbook reading."""
        dict_of_dfs, validation_report = ingest.read_workbook(sample_excel_bytes)
        
        assert 'summary' in dict_of_dfs
        assert 'quotes' in dict_of_dfs
        assert 'sentiments' in dict_of_dfs
        assert dict_of_dfs['summary'] is not None
        assert dict_of_dfs['quotes'] is not None
        assert dict_of_dfs['sentiments'] is not None
        
        assert validation_report['is_readable'] is True
        assert validation_report['is_valid'] is True
        assert 'summary' in validation_report['matched_sheets']
        assert 'quotes' in validation_report['matched_sheets']
        assert 'sentiments' in validation_report['matched_sheets']
    
    def test_read_workbook_sheet_detection(self, sample_excel_bytes):
        """Test sheet name fuzzy matching."""
        dict_of_dfs, validation_report = ingest.read_workbook(sample_excel_bytes)
        
        # Should match exact sheet names
        assert 'summary' in validation_report['matched_sheets']
        assert 'quotes' in validation_report['matched_sheets']
        assert 'sentiments' in validation_report['matched_sheets']
        
        # Should have topic columns
        assert len(validation_report['topic_columns']) > 0
        assert 'topic_a' in validation_report['topic_columns']
        assert 'topic_b' in validation_report['topic_columns']
    
    def test_read_workbook_topic_columns_intersection(self, sample_excel_bytes):
        """Test topic columns are intersection across sheets."""
        dict_of_dfs, validation_report = ingest.read_workbook(sample_excel_bytes)
        
        topic_columns = validation_report['topic_columns']
        
        # Topic columns should be present in all matched sheets
        for role in ['summary', 'quotes', 'sentiments']:
            if dict_of_dfs[role] is not None:
                df = dict_of_dfs[role]
                for topic_col in topic_columns:
                    assert topic_col in df.columns
    
    def test_read_workbook_metadata_detection(self, sample_excel_bytes):
        """Test metadata columns are identified correctly."""
        dict_of_dfs, validation_report = ingest.read_workbook(sample_excel_bytes)
        
        # participant_id should be identified as metadata
        summary_df = dict_of_dfs['summary']
        assert 'participant_id' in summary_df.columns
        
        # Topic columns should not include participant_id
        assert 'participant_id' not in validation_report['topic_columns']
    
    def test_read_workbook_empty_file(self):
        """Test handling of empty/invalid file."""
        invalid_bytes = b'invalid excel content'
        dict_of_dfs, validation_report = ingest.read_workbook(invalid_bytes)
        
        assert validation_report['is_readable'] is False
        assert 'error' in validation_report
        assert validation_report['is_valid'] is False or 'is_valid' not in validation_report
    
    def test_read_workbook_missing_sheet(self):
        """Test handling of missing required sheet."""
        import io
        buffer = io.BytesIO()
        with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
            # Only include quotes, missing summary
            pd.DataFrame({'participant_id': ['p1'], 'topic_a': ['Quote']}).to_excel(
                writer, sheet_name='quotes', index=False
            )
        buffer.seek(0)
        bytes_data = buffer.getvalue()
        
        dict_of_dfs, validation_report = ingest.read_workbook(bytes_data)
        
        assert 'summary' in validation_report['missing_sheets']
        assert validation_report['is_valid'] is False
        assert 'error' in validation_report
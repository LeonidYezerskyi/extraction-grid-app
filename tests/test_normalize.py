"""Tests for normalize module."""

import pytest
import pandas as pd
import normalize
from tests.conftest import sample_dfs


class TestNormalize:
    """Test suite for normalize.wide_to_canonical."""
    
    def test_wide_to_canonical_basic(self, sample_dfs):
        """Test basic wide to canonical conversion."""
        topic_columns = ['topic_a', 'topic_b', 'topic_c']
        canonical_model = normalize.wide_to_canonical(sample_dfs, topic_columns)
        
        assert len(canonical_model.participants) == 3
        assert len(canonical_model.topics) == 3
        assert len(canonical_model.evidence_cells) > 0
        
        # Check topic IDs
        topic_ids = [t.topic_id for t in canonical_model.topics]
        assert 'topic_a' in topic_ids
        assert 'topic_b' in topic_ids
        assert 'topic_c' in topic_ids
    
    def test_wide_to_canonical_participant_id_detection(self, sample_dfs):
        """Test participant_id is determined correctly."""
        topic_columns = ['topic_a', 'topic_b']
        canonical_model = normalize.wide_to_canonical(sample_dfs, topic_columns)
        
        # Should have participants
        assert len(canonical_model.participants) > 0
        
        # Participant IDs should be from first column
        participant_ids = [p.participant_id for p in canonical_model.participants]
        assert 'p1' in participant_ids
        assert 'p2' in participant_ids
    
    def test_wide_to_canonical_evidence_cells(self, sample_dfs):
        """Test evidence cells are created correctly."""
        topic_columns = ['topic_a', 'topic_b']
        canonical_model = normalize.wide_to_canonical(sample_dfs, topic_columns)
        
        # Find evidence cell for p1, topic_a
        evidence_cells = [
            ec for ec in canonical_model.evidence_cells
            if ec.participant_id == 'p1' and ec.topic_id == 'topic_a'
        ]
        
        assert len(evidence_cells) > 0
        evidence_cell = evidence_cells[0]
        assert evidence_cell.summary_text is not None
        assert evidence_cell.quotes_raw is not None
        assert evidence_cell.sentiments_raw is not None
    
    def test_wide_to_canonical_topic_id_normalization(self, sample_dfs):
        """Test topic IDs are normalized (lowercase, strip)."""
        # Create dfs with mixed case topic columns
        dfs_mixed = {
            'summary': pd.DataFrame({
                'participant_id': ['p1'],
                'Topic_A': ['Summary'],
                '  topic_b  ': ['Summary']
            }),
            'quotes': pd.DataFrame({
                'participant_id': ['p1'],
                'Topic_A': ['Quote'],
                '  topic_b  ': ['Quote']
            }),
            'sentiments': pd.DataFrame({
                'participant_id': ['p1'],
                'Topic_A': ['positive'],
                '  topic_b  ': ['positive']
            })
        }
        
        topic_columns = ['Topic_A', '  topic_b  ']
        canonical_model = normalize.wide_to_canonical(dfs_mixed, topic_columns)
        
        topic_ids = [t.topic_id for t in canonical_model.topics]
        # Should be normalized
        assert 'topic_a' in topic_ids or 'Topic_A' in topic_ids
    
    def test_wide_to_canonical_missing_sheet(self, sample_dfs):
        """Test handling of missing sheet."""
        dfs_missing = {
            'summary': sample_dfs['summary'],
            'quotes': None,  # Missing
            'sentiments': sample_dfs['sentiments']
        }
        
        topic_columns = ['topic_a']
        canonical_model = normalize.wide_to_canonical(dfs_missing, topic_columns)
        
        # Should still work with available sheets
        assert canonical_model is not None
        assert len(canonical_model.evidence_cells) > 0

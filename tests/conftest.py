"""Pytest fixtures for test suite."""

import pytest
import pandas as pd
import io
from typing import Dict, Optional
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


@pytest.fixture
def sample_summary_df():
    """Fixture for sample summary sheet DataFrame."""
    return pd.DataFrame({
        'participant_id': ['p1', 'p2', 'p3'],
        'topic_a': ['Summary A for p1', 'Summary A for p2', 'Summary A for p3'],
        'topic_b': ['Summary B for p1', 'Summary B for p2', None],
        'topic_c': ['Summary C for p1', None, None]
    })


@pytest.fixture
def sample_quotes_df():
    """Fixture for sample quotes sheet DataFrame."""
    return pd.DataFrame({
        'participant_id': ['p1', 'p2', 'p3'],
        'topic_a': [
            '1. First quote about topic A. 2. Second quote about topic A.',
            '1. Quote from p2 about topic A.',
            None
        ],
        'topic_b': [
            '1. Quote about topic B.',
            '- Bullet point one\n- Bullet point two',
            None
        ],
        'topic_c': [
            'Single unnumbered quote for topic C.',
            None,
            None
        ]
    })


@pytest.fixture
def sample_sentiments_df():
    """Fixture for sample sentiments sheet DataFrame."""
    return pd.DataFrame({
        'participant_id': ['p1', 'p2', 'p3'],
        'topic_a': [
            '1: positive; 2: negative',
            '1: positive',
            None
        ],
        'topic_b': [
            '1: neutral',
            'positive, negative',  # Flat list
            None
        ],
        'topic_c': [
            'positive',  # Sentiment without numbered quotes
            None,
            None
        ]
    })


@pytest.fixture
def sample_dfs(sample_summary_df, sample_quotes_df, sample_sentiments_df):
    """Fixture combining all three sheet DataFrames."""
    return {
        'summary': sample_summary_df,
        'quotes': sample_quotes_df,
        'sentiments': sample_sentiments_df
    }


@pytest.fixture
def sample_excel_bytes(sample_dfs):
    """Fixture creating Excel file bytes from DataFrames."""
    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
        for sheet_name, df in sample_dfs.items():
            df.to_excel(writer, sheet_name=sheet_name, index=False)
    buffer.seek(0)
    return buffer.getvalue()


@pytest.fixture
def sparse_topic_dfs():
    """Fixture for sparse topic scenario."""
    return {
        'summary': pd.DataFrame({
            'participant_id': ['p1'],
            'sparse_topic': ['Single summary']
        }),
        'quotes': pd.DataFrame({
            'participant_id': ['p1'],
            'sparse_topic': [None]  # No quotes
        }),
        'sentiments': pd.DataFrame({
            'participant_id': ['p1'],
            'sparse_topic': [None]  # No sentiments
        })
    }


@pytest.fixture
def single_sheet_topic_dfs():
    """Fixture for single-sheet topic (only in summary)."""
    return {
        'summary': pd.DataFrame({
            'participant_id': ['p1'],
            'single_sheet_topic': ['Only in summary']
        }),
        'quotes': pd.DataFrame({
            'participant_id': ['p1'],
            'single_sheet_topic': [None]
        }),
        'sentiments': pd.DataFrame({
            'participant_id': ['p1'],
            'single_sheet_topic': [None]
        })
    }


@pytest.fixture
def sentiment_without_quotes_dfs():
    """Fixture for sentiment without quotes scenario."""
    return {
        'summary': pd.DataFrame({
            'participant_id': ['p1'],
            'topic_x': ['Summary']
        }),
        'quotes': pd.DataFrame({
            'participant_id': ['p1'],
            'topic_x': [None]  # No quotes
        }),
        'sentiments': pd.DataFrame({
            'participant_id': ['p1'],
            'topic_x': ['1: positive']  # Sentiment but no quote
        })
    }


@pytest.fixture
def duplicate_participant_dfs():
    """Fixture with duplicate/non-participant rows."""
    return {
        'summary': pd.DataFrame({
            'participant_id': ['p1', 'moderator_1', 'p2', 'admin_1', 'p3'],
            'topic_a': ['Summary 1', 'Mod summary', 'Summary 2', 'Admin summary', 'Summary 3']
        }),
        'quotes': pd.DataFrame({
            'participant_id': ['p1', 'moderator_1', 'p2', 'admin_1', 'p3'],
            'topic_a': ['Quote 1', 'Mod quote', 'Quote 2', 'Admin quote', 'Quote 3']
        }),
        'sentiments': pd.DataFrame({
            'participant_id': ['p1', 'moderator_1', 'p2', 'admin_1', 'p3'],
            'topic_a': ['positive', 'neutral', 'negative', 'neutral', 'positive']
        })
    }

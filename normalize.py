"""Module for normalizing and cleaning ingested data."""

from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
import pandas as pd
import numpy as np


@dataclass
class Participant:
    """Represents a participant in the canonical model."""
    participant_id: str
    participant_label: str
    participant_meta: Dict[str, Any]


@dataclass
class Topic:
    """Represents a topic in the canonical model."""
    topic_id: str


@dataclass
class EvidenceCell:
    """Represents an evidence cell linking participant, topic, and evidence data."""
    participant_id: str
    topic_id: str
    summary_text: Optional[str]
    quotes_raw: Optional[str]
    sentiments_raw: Optional[str]
    participant_meta: Dict[str, Any]


@dataclass
class CanonicalModel:
    """Container for the canonical data model."""
    participants: List[Participant]
    topics: List[Topic]
    evidence_cells: List[EvidenceCell]


def _determine_participant_id(df: pd.DataFrame) -> Tuple[str, pd.Series, bool]:
    """
    Determine participant_id column: prefer explicit first column if mostly unique/non-null,
    otherwise use row index.
    
    Args:
        df: DataFrame to analyze
    
    Returns:
        Tuple of (participant_id_column_name, participant_id_series, use_index)
        where use_index is True if row index is used, False if first column is used
    """
    if df.empty or len(df.columns) == 0:
        # Fallback to index if no columns
        return 'participant_id', pd.Series([f"row_{i}" for i in range(len(df))], index=df.index), True
    
    first_col = df.columns[0]
    first_col_series = df[first_col]
    
    # Check if first column is mostly unique and non-null
    non_null_count = first_col_series.notna().sum()
    unique_count = first_col_series.nunique()
    total_count = len(df)
    
    # Consider it valid if:
    # - At least 80% non-null
    # - At least 80% unique (or all unique if small dataset)
    non_null_ratio = non_null_count / total_count if total_count > 0 else 0
    unique_ratio = unique_count / total_count if total_count > 0 else 0
    
    if non_null_ratio >= 0.8 and (unique_ratio >= 0.8 or unique_count == total_count):
        # Use first column
        participant_ids = first_col_series.astype(str)
        return first_col, participant_ids, False
    else:
        # Use row index
        participant_ids = pd.Series([f"row_{i}" for i in range(len(df))], index=df.index)
        return 'participant_id', participant_ids, True


def _identify_metadata_columns(df: pd.DataFrame, participant_id_col: str) -> List[str]:
    """
    Identify metadata columns (excluding the participant_id column and topic columns).
    
    Args:
        df: DataFrame to analyze
        participant_id_col: Name of the participant_id column
    
    Returns:
        List of metadata column names
    """
    # Metadata columns are all columns except participant_id
    # Topic columns will be identified separately during melting
    metadata_cols = [col for col in df.columns if col != participant_id_col]
    return metadata_cols


def _normalize_topic_id(topic_id: Any) -> str:
    """
    Normalize topic_id string: lowercase and strip whitespace.
    
    Args:
        topic_id: Raw topic_id value
    
    Returns:
        Normalized topic_id string
    """
    if pd.isna(topic_id):
        return ""
    return str(topic_id).lower().strip()


def _melt_sheet_to_long(
    df: pd.DataFrame,
    participant_id_col: str,
    participant_ids: pd.Series,
    use_index: bool,
    topic_columns: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    Melt a wide DataFrame into long format keyed by (participant_id, topic_id).
    
    Args:
        df: Wide DataFrame to melt
        participant_id_col: Name of the participant_id column (or 'participant_id' if using index)
        participant_ids: Series of participant IDs
        use_index: Whether participant_id is from row index
        topic_columns: Optional list of topic column names. If None, all columns except
                      participant_id are treated as topics.
    
    Returns:
        Long-format DataFrame with columns: participant_id, topic_id, value
    """
    if df.empty:
        return pd.DataFrame(columns=['participant_id', 'topic_id', 'value'])
    
    # Determine topic columns
    if topic_columns is None:
        if use_index:
            topic_columns = list(df.columns)
        else:
            topic_columns = [col for col in df.columns if col != participant_id_col]
    
    if not topic_columns:
        return pd.DataFrame(columns=['participant_id', 'topic_id', 'value'])
    
    # Prepare DataFrame for melting
    df_melt = df.copy()
    
    # Add participant_id column if using index
    if use_index:
        df_melt['participant_id'] = participant_ids
    else:
        df_melt[participant_id_col] = participant_ids
    
    # Melt the DataFrame
    id_vars = ['participant_id'] if use_index else [participant_id_col]
    melted = pd.melt(
        df_melt,
        id_vars=id_vars,
        value_vars=topic_columns,
        var_name='topic_id',
        value_name='value'
    )
    
    # Normalize topic_id
    melted['topic_id'] = melted['topic_id'].apply(_normalize_topic_id)
    
    # Rename participant_id column if needed
    if not use_index and participant_id_col != 'participant_id':
        melted = melted.rename(columns={participant_id_col: 'participant_id'})
    
    # Convert participant_id to string
    melted['participant_id'] = melted['participant_id'].astype(str)
    
    # Filter out empty topic_ids
    melted = melted[melted['topic_id'] != '']
    
    return melted[['participant_id', 'topic_id', 'value']]


def _extract_participant_meta(
    df: pd.DataFrame,
    participant_id_col: str,
    participant_ids: pd.Series,
    use_index: bool
) -> Dict[str, Dict[str, Any]]:
    """
    Extract participant metadata from metadata columns.
    
    Args:
        df: DataFrame with metadata columns
        participant_id_col: Name of the participant_id column
        participant_ids: Series of participant IDs
        use_index: Whether participant_id is from row index
    
    Returns:
        Dictionary mapping participant_id to metadata dict
    """
    participant_meta = {}
    
    if df.empty:
        return participant_meta
    
    # Identify metadata columns (all columns except participant_id)
    if use_index:
        metadata_cols = list(df.columns)
    else:
        metadata_cols = [col for col in df.columns if col != participant_id_col]
    
    if not metadata_cols:
        # No metadata columns, return empty dicts for each participant
        for pid in participant_ids:
            participant_meta[str(pid)] = {}
        return participant_meta
    
    # Create a DataFrame with participant_id and metadata
    meta_df = df[metadata_cols].copy()
    
    if use_index:
        meta_df['participant_id'] = participant_ids.values
    else:
        meta_df['participant_id'] = df[participant_id_col].astype(str)
    
    # Convert to dict
    for _, row in meta_df.iterrows():
        pid = str(row['participant_id'])
        meta_dict = {col: row[col] for col in metadata_cols}
        # Convert NaN to None for JSON-serializable output
        meta_dict = {k: (None if pd.isna(v) else v) for k, v in meta_dict.items()}
        participant_meta[pid] = meta_dict
    
    return participant_meta


def wide_to_canonical(
    dfs: Dict[str, Optional[pd.DataFrame]],
    topic_columns: Optional[List[str]] = None
) -> CanonicalModel:
    """
    Convert wide-format DataFrames into canonical long-format model.
    
    This function:
    - Determines participant_id from first column (if mostly unique/non-null) or row index
    - Melts each sheet (summary, quotes, sentiments) into long format keyed by (participant_id, topic_id)
    - Normalizes topic_id strings consistently (lowercase, strip)
    - Joins melted sheets into EvidenceCell records with fields: summary_text, quotes_raw, sentiments_raw, participant_meta
    - Returns structured Python objects representing Participant, Topic, and EvidenceCell lists
    
    Args:
        dfs: Dictionary mapping role names ('summary', 'quotes', 'sentiments') to DataFrames
             (None if sheet not found). Expected keys: 'summary', 'quotes', 'sentiments'
        topic_columns: Optional list of topic column names. If None, all non-metadata columns
                      are treated as topics. If provided, only these columns are used as topics.
    
    Returns:
        CanonicalModel containing:
        - participants: List of Participant objects with participant_id, participant_label, participant_meta
        - topics: List of Topic objects with topic_id
        - evidence_cells: List of EvidenceCell objects with participant_id, topic_id, summary_text,
                         quotes_raw, sentiments_raw, participant_meta
    
    Example:
        >>> dfs = {'summary': df1, 'quotes': df2, 'sentiments': df3}
        >>> model = wide_to_canonical(dfs, topic_columns=['topic1', 'topic2'])
        >>> print(len(model.participants))
        >>> print(len(model.evidence_cells))
    """
    # Determine participant_id strategy from the first available sheet
    # This strategy will be applied consistently to all sheets
    participant_id_col = 'participant_id'
    use_index = True
    reference_df = None
    reference_role = None
    
    for role in ['summary', 'quotes', 'sentiments']:
        if dfs.get(role) is not None and not dfs[role].empty:
            reference_df = dfs[role]
            reference_role = role
            participant_id_col, _, use_index = _determine_participant_id(reference_df)
            break
    
    if reference_df is None:
        # No data available, return empty model
        return CanonicalModel(participants=[], topics=[], evidence_cells=[])
    
    # Melt each sheet into long format
    melted_sheets = {}
    all_participant_meta = {}
    all_participant_ids_set = set()
    
    for role in ['summary', 'quotes', 'sentiments']:
        df = dfs.get(role)
        if df is None or df.empty:
            melted_sheets[role] = pd.DataFrame(columns=['participant_id', 'topic_id', 'value'])
            continue
        
        # Apply the same strategy: use first column if use_index is False, otherwise use row index
        if use_index:
            # Use row index for all sheets
            participant_ids = pd.Series([f"row_{i}" for i in range(len(df))], index=df.index)
            sheet_participant_id_col = 'participant_id'
        else:
            # Try to use the same column name as reference sheet, otherwise use first column
            if participant_id_col in df.columns:
                sheet_participant_id_col = participant_id_col
            else:
                # Fallback to first column if reference column doesn't exist
                sheet_participant_id_col = df.columns[0]
            participant_ids = df[sheet_participant_id_col].astype(str)
        
        all_participant_ids_set.update(participant_ids.astype(str))
        
        melted = _melt_sheet_to_long(df, sheet_participant_id_col, participant_ids, use_index, topic_columns)
        melted_sheets[role] = melted
        
        # Extract participant metadata from this sheet
        meta = _extract_participant_meta(df, sheet_participant_id_col, participant_ids, use_index)
        # Merge metadata (later sheets override earlier ones for same keys)
        for pid, meta_dict in meta.items():
            if pid not in all_participant_meta:
                all_participant_meta[pid] = {}
            all_participant_meta[pid].update(meta_dict)
    
    # Collect all unique participant_ids (already collected above, but also from melted data)
    all_participant_ids = all_participant_ids_set.copy()
    for melted_df in melted_sheets.values():
        if not melted_df.empty:
            all_participant_ids.update(melted_df['participant_id'].unique())
    
    # Collect all unique topic_ids
    all_topic_ids = set()
    for melted_df in melted_sheets.values():
        if not melted_df.empty:
            all_topic_ids.update(melted_df['topic_id'].unique())
    
    # Create participant_labels from reference sheet
    participant_labels = {}
    if reference_df is not None:
        if use_index:
            for i in range(len(reference_df)):
                pid = f"row_{i}"
                participant_labels[pid] = pid
        else:
            ref_col = reference_df.columns[0]
            for i, val in enumerate(reference_df[ref_col]):
                pid = str(val)
                participant_labels[pid] = str(val)
    
    # Create Participant objects
    participants = []
    for pid in sorted(all_participant_ids):
        label = participant_labels.get(pid, pid)
        meta = all_participant_meta.get(pid, {})
        participants.append(Participant(
            participant_id=pid,
            participant_label=label,
            participant_meta=meta
        ))
    
    # Create Topic objects
    topics = [Topic(topic_id=tid) for tid in sorted(all_topic_ids) if tid]
    
    # Join melted sheets into EvidenceCell records
    # Create a combined DataFrame with all participant_id, topic_id combinations
    evidence_cells = []
    
    # Get all combinations of participant_id and topic_id
    for pid in sorted(all_participant_ids):
        for tid in sorted(all_topic_ids):
            if not tid:
                continue
            
            # Get values from each melted sheet
            summary_text = None
            quotes_raw = None
            sentiments_raw = None
            
            summary_df = melted_sheets['summary']
            if not summary_df.empty:
                summary_rows = summary_df[(summary_df['participant_id'] == pid) & (summary_df['topic_id'] == tid)]
                if not summary_rows.empty:
                    val = summary_rows.iloc[0]['value']
                    summary_text = None if pd.isna(val) else str(val)
            
            quotes_df = melted_sheets['quotes']
            if not quotes_df.empty:
                quotes_rows = quotes_df[(quotes_df['participant_id'] == pid) & (quotes_df['topic_id'] == tid)]
                if not quotes_rows.empty:
                    val = quotes_rows.iloc[0]['value']
                    quotes_raw = None if pd.isna(val) else str(val)
            
            sentiments_df = melted_sheets['sentiments']
            if not sentiments_df.empty:
                sentiments_rows = sentiments_df[(sentiments_df['participant_id'] == pid) & (sentiments_df['topic_id'] == tid)]
                if not sentiments_rows.empty:
                    val = sentiments_rows.iloc[0]['value']
                    sentiments_raw = None if pd.isna(val) else str(val)
            
            # Only create EvidenceCell if at least one field has data
            if summary_text or quotes_raw or sentiments_raw:
                meta = all_participant_meta.get(pid, {})
                evidence_cells.append(EvidenceCell(
                    participant_id=pid,
                    topic_id=tid,
                    summary_text=summary_text,
                    quotes_raw=quotes_raw,
                    sentiments_raw=sentiments_raw,
                    participant_meta=meta
                ))
    
    return CanonicalModel(
        participants=participants,
        topics=topics,
        evidence_cells=evidence_cells
    )
"""Main Streamlit application entry point."""

import streamlit as st
import pandas as pd
from typing import Dict, List, Optional, Any, Tuple
import hashlib
import ingest
import normalize
import parse_quotes
import parse_sentiment
import score
import digest
import re
import render
import export
import edge_cases
import explore_model
import recipient


# Pipeline version for cache invalidation
PIPELINE_VERSION = "1.0.0"

# Session state keys
SESSION_KEYS = {
    'selected_topics': 'selected_topics',
    'top_n': 'top_n',
    'auto_select_top_n': 'auto_select_top_n',
    'filters': 'filters',
    'search_query': 'search_query',
    'uploaded_file': 'uploaded_file',
    'file_hash': 'file_hash',
    'file_bytes': 'file_bytes',
    'canonical_model': 'canonical_model',
    'topic_aggregates': 'topic_aggregates',
    'validation_report': 'validation_report'
}


def compute_file_hash(file_bytes: bytes) -> str:
    """
    Compute MD5 hash of file bytes for cache key.
    
    Args:
        file_bytes: File bytes
    
    Returns:
        MD5 hash as hex string
    """
    return hashlib.md5(file_bytes).hexdigest()


def get_config_knobs_hash(config: Dict[str, Any]) -> str:
    """
    Get hash of configuration knobs for cache key.
    
    Args:
        config: Configuration dictionary
    
    Returns:
        Hash string of sorted config items
    """
    # Sort items for deterministic hash
    sorted_items = sorted(config.items())
    config_str = str(sorted_items)
    return hashlib.md5(config_str.encode()).hexdigest()


def initialize_session_state():
    """Initialize session state with default values."""
    if SESSION_KEYS['top_n'] not in st.session_state:
        st.session_state[SESSION_KEYS['top_n']] = 4
    if SESSION_KEYS['auto_select_top_n'] not in st.session_state:
        st.session_state[SESSION_KEYS['auto_select_top_n']] = True
    if SESSION_KEYS['selected_topics'] not in st.session_state:
        st.session_state[SESSION_KEYS['selected_topics']] = []
    if SESSION_KEYS['filters'] not in st.session_state:
        st.session_state[SESSION_KEYS['filters']] = {
            'coverage_tier': None,
            'tone_rollup': None,
            'high_emotion': False,
        }
    if SESSION_KEYS['search_query'] not in st.session_state:
        st.session_state[SESSION_KEYS['search_query']] = ''
    if SESSION_KEYS['file_hash'] not in st.session_state:
        st.session_state[SESSION_KEYS['file_hash']] = None
    if 'auto_add_on_change' not in st.session_state:
        st.session_state['auto_add_on_change'] = False
    if 'previous_top_n' not in st.session_state:
        st.session_state['previous_top_n'] = None
    if 'participant_filter_patterns' not in st.session_state:
        st.session_state['participant_filter_patterns'] = []
    if 'single_sheet_topics' not in st.session_state:
        st.session_state['single_sheet_topics'] = set()
    if 'sparse_topics' not in st.session_state:
        st.session_state['sparse_topics'] = []


@st.cache_data
def cached_read_workbook(
    file_hash: str,
    pipeline_version: str,
    config_hash: str,
    file_bytes: bytes
) -> Tuple[Dict[str, Optional[pd.DataFrame]], Dict[str, Any]]:
    """
    Cached wrapper for ingest.read_workbook.
    
    Cache key includes file_hash, pipeline_version, and config_hash.
    This ensures cache invalidation on file change or pipeline update.
    
    Args:
        file_hash: MD5 hash of file bytes (for cache key)
        pipeline_version: Pipeline version string (for cache key)
        config_hash: Hash of configuration knobs (for cache key)
        file_bytes: Actual file bytes (used for processing)
    
    Returns:
        Tuple of (dict_of_dfs, validation_report)
    """
    return ingest.read_workbook(file_bytes)


@st.cache_data
def cached_normalize(
    file_hash: str,
    pipeline_version: str,
    config_hash: str,
    topic_columns: Tuple[str, ...],
    file_bytes: bytes
) -> normalize.CanonicalModel:
    """
    Cached wrapper for normalize.wide_to_canonical.
    
    Cache key includes file_hash, pipeline_version, config_hash, and topic_columns.
    
    Args:
        file_hash: MD5 hash of file bytes (for cache key)
        pipeline_version: Pipeline version string (for cache key)
        config_hash: Hash of configuration knobs (for cache key)
        topic_columns: Tuple of topic column names (must be hashable)
        file_bytes: File bytes (passed through for read_workbook)
    
    Returns:
        CanonicalModel object
    """
    # Get dict_of_dfs from cached read_workbook
    dict_of_dfs, _ = cached_read_workbook(file_hash, pipeline_version, config_hash, file_bytes)
    
    return normalize.wide_to_canonical(dict_of_dfs, list(topic_columns))


@st.cache_data
def cached_compute_scoring(
    file_hash: str,
    pipeline_version: str,
    config_hash: str,
    topic_columns: Tuple[str, ...],
    file_bytes: bytes
) -> List[Dict[str, Any]]:
    """
    Cached wrapper for score.compute_topic_aggregates.
    
    Cache key includes file_hash, pipeline_version, config_hash, and topic_columns.
    This ensures scoring is only recomputed when input data or config changes.
    
    Args:
        file_hash: MD5 hash of file bytes (for cache key)
        pipeline_version: Pipeline version string (for cache key)
        config_hash: Hash of configuration knobs (for cache key)
        topic_columns: Tuple of topic column names
        file_bytes: File bytes (passed through for pipeline)
    
    Returns:
        List of topic aggregates
    """
    canonical_model = cached_normalize(file_hash, pipeline_version, config_hash, topic_columns, file_bytes)
    return score.compute_topic_aggregates(canonical_model)


def process_uploaded_file(uploaded_file_bytes: bytes, config: Optional[Dict[str, Any]] = None) -> Tuple[Dict[str, Optional[pd.DataFrame]], Dict[str, Any], normalize.CanonicalModel, List[str]]:
    """
    Process uploaded file: ingest and normalize with proper caching.
    
    This function orchestrates the cached pipeline steps.
    
    Args:
        uploaded_file_bytes: Bytes of uploaded Excel file
        config: Optional configuration dictionary for cache key
    
    Returns:
        Tuple of (dict_of_dfs, validation_report, canonical_model, topic_columns)
    """
    # Compute hashes for cache keys
    file_hash = compute_file_hash(uploaded_file_bytes)
    config = config or {}
    config_hash = get_config_knobs_hash(config)
    
    # Ingest (cached)
    dict_of_dfs, validation_report = cached_read_workbook(file_hash, PIPELINE_VERSION, config_hash, uploaded_file_bytes)
    
    # Get topic columns from validation report
    topic_columns = list(validation_report.get('topic_columns', []))
    topic_columns_tuple = tuple(topic_columns)  # Make hashable for cache
    
    # Normalize (cached)
    canonical_model = cached_normalize(file_hash, PIPELINE_VERSION, config_hash, topic_columns_tuple, uploaded_file_bytes)
    
    return dict_of_dfs, validation_report, canonical_model, topic_columns


def compute_scoring_with_cache(
    file_hash: str,
    file_bytes: bytes,
    config: Optional[Dict[str, Any]] = None
) -> List[Dict[str, Any]]:
    """
    Compute topic aggregates and scores with proper caching.
    
    Args:
        file_hash: MD5 hash of file bytes
        file_bytes: File bytes (needed for cache chain)
        config: Optional configuration dictionary for cache key
    
    Returns:
        List of topic aggregates
    """
    config = config or {}
    config_hash = get_config_knobs_hash(config)
    
    # Get topic columns from session state
    topic_columns = st.session_state.get('topic_columns', [])
    topic_columns_tuple = tuple(topic_columns)
    
    return cached_compute_scoring(file_hash, PIPELINE_VERSION, config_hash, topic_columns_tuple, file_bytes)


def filter_topics(
    topic_aggregates: List[Dict[str, Any]],
    filters: Dict[str, Any],
    search_query: str,
    canonical_model
) -> List[Dict[str, Any]]:
    """
    Filter topics based on filters and search query.
    
    Args:
        topic_aggregates: List of topic aggregates
        filters: Filter dictionary
        search_query: Search query string
        canonical_model: CanonicalModel for search
    
    Returns:
        Filtered list of topic aggregates
    """
    filtered = topic_aggregates.copy()
    
    # Coverage tier filter
    coverage_tier = filters.get('coverage_tier')
    if coverage_tier:
        if coverage_tier == 'High':
            filtered = [t for t in filtered if t.get('coverage_rate', 0) >= 0.7]
        elif coverage_tier == 'Medium':
            filtered = [t for t in filtered if 0.4 <= t.get('coverage_rate', 0) < 0.7]
        elif coverage_tier == 'Low':
            filtered = [t for t in filtered if t.get('coverage_rate', 0) < 0.4]
    
    # Tone rollup filter
    tone_rollup = filters.get('tone_rollup')
    if tone_rollup:
        # Need to compute sentiment mix for each topic
        # Note: tone_rollup is already lowercase from render_sidebar, but we normalize it here for safety
        tone_rollup_normalized = tone_rollup.lower() if isinstance(tone_rollup, str) else tone_rollup
        filtered_by_tone = []
        for topic_agg in filtered:
            topic_id = topic_agg['topic_id']
            sentiment_mix = digest._compute_sentiment_mix(canonical_model.evidence_cells, topic_id)
            total = sum(sentiment_mix.values())
            if total > 0:
                dominant = max(sentiment_mix.items(), key=lambda x: x[1])[0]
                if dominant == tone_rollup_normalized:
                    filtered_by_tone.append(topic_agg)
        filtered = filtered_by_tone
    
    # High-emotion filter
    if filters.get('high_emotion', False):
        filtered_by_emotion = []
        for topic_agg in filtered:
            if topic_agg.get('intensity_rate', 0) > 0.5:  # More than 50% non-neutral
                filtered_by_emotion.append(topic_agg)
        filtered = filtered_by_emotion
    
    # Search query filter
    if search_query:
        query_lower = search_query.lower()
        filtered_by_search = []
        for topic_agg in filtered:
            topic_id = topic_agg['topic_id']
            # Search in topic_id, one-liner, and evidence cells
            matches = False
            
            if query_lower in topic_id.lower():
                matches = True
            
            one_liner = topic_agg.get('topic_one_liner', '')
            if one_liner and query_lower in one_liner.lower():
                matches = True
            
            # Search in evidence cells
            if not matches:
                for evidence_cell in canonical_model.evidence_cells:
                    if evidence_cell.topic_id == topic_id:
                        if (evidence_cell.summary_text and query_lower in evidence_cell.summary_text.lower()) or \
                           (evidence_cell.quotes_raw and query_lower in evidence_cell.quotes_raw.lower()):
                            matches = True
                            break
            
            if matches:
                filtered_by_search.append(topic_agg)
        filtered = filtered_by_search
    
    return filtered


def compute_top_topics(topic_aggregates: List[Dict[str, Any]], exclude_single_sheet: bool = True) -> List[str]:
    """
    Compute top topics as a ranked list of topic IDs.
    
    Topics are already sorted by score (descending) with alphabetical tiebreak
    from score.compute_topic_aggregates.
    
    Excludes single-sheet topics from Top N by default.
    
    Args:
        topic_aggregates: List of topic aggregates (already sorted by score)
        exclude_single_sheet: If True, exclude single-sheet topics from Top N
    
    Returns:
        List of topic IDs in ranked order (excluding single-sheet if requested)
    """
    single_sheet_topics = st.session_state.get('single_sheet_topics', set())
    
    if exclude_single_sheet and single_sheet_topics:
        # Filter out single-sheet topics
        filtered = [t['topic_id'] for t in topic_aggregates if t['topic_id'] not in single_sheet_topics]
        return filtered
    
    return [t['topic_id'] for t in topic_aggregates]


def compute_selected_topics(
    top_topics: List[str],
    current_selected: List[str],
    top_n: int,
    auto_select: bool,
    auto_add_on_change: bool,
    previous_top_n: Optional[int] = None
) -> List[str]:
    """
    Compute selected topics based on selection behavior rules.
    
    Rules:
    - If auto_select is True and no current selection, default to top_topics[:N]
    - When N changes:
      - If N increases and auto_add_on_change is True, append newly included topics
      - If N decreases, do not auto-remove manually added topics
    - Maintains stability: preserves user's manual additions
    
    Args:
        top_topics: Ranked list of topic IDs
        current_selected: Currently selected topic IDs
        top_n: Current N value
        auto_select: Whether auto-select is enabled
        auto_add_on_change: Whether to auto-add when N increases
        previous_top_n: Previous N value (for detecting changes)
    
    Returns:
        Updated list of selected topic IDs
    """
    if not top_topics:
        return []
    
    # If auto_select is False, return current selection as-is
    if not auto_select:
        return current_selected.copy()
    
    # Determine top N topics
    top_n_topics = top_topics[:top_n]
    
    # If no current selection, initialize with top N
    if not current_selected:
        return top_n_topics.copy()
    
    # Check if N changed
    n_increased = previous_top_n is not None and top_n > previous_top_n
    n_decreased = previous_top_n is not None and top_n < previous_top_n
    
    # Build new selection
    new_selected = []
    
    # Always include topics that are in top N
    for topic_id in top_n_topics:
        if topic_id not in new_selected:
            new_selected.append(topic_id)
    
    # If N increased and auto_add_on_change, add newly included topics
    if n_increased and auto_add_on_change:
        # Find topics that are now in top N but weren't before
        previous_top_n_topics = top_topics[:previous_top_n] if previous_top_n else []
        newly_included = [t for t in top_n_topics if t not in previous_top_n_topics]
        for topic_id in newly_included:
            if topic_id not in new_selected:
                new_selected.append(topic_id)
    
    # Preserve manually added topics (those not in top N)
    # These are topics the user added that are outside the current top N
    manually_added = [t for t in current_selected if t not in top_n_topics]
    
    # If N decreased, preserve all manually added topics
    if n_decreased:
        for topic_id in manually_added:
            if topic_id not in new_selected:
                new_selected.append(topic_id)
    else:
        # If N didn't decrease, still preserve manually added topics
        # (user intent: they explicitly added these)
        for topic_id in manually_added:
            if topic_id not in new_selected:
                new_selected.append(topic_id)
    
    # Maintain order: top N first, then manually added
    ordered_selected = []
    # Add top N topics in order
    for topic_id in top_topics:
        if topic_id in new_selected:
            ordered_selected.append(topic_id)
    # Add manually added topics at the end
    for topic_id in manually_added:
        if topic_id in new_selected and topic_id not in ordered_selected:
            ordered_selected.append(topic_id)
    
    return ordered_selected


def reset_to_top_n(top_topics: List[str], top_n: int) -> List[str]:
    """
    Reset selection to top N topics.
    
    Args:
        top_topics: Ranked list of topic IDs
        top_n: Number of topics to select
    
    Returns:
        List of top N topic IDs
    """
    return top_topics[:top_n].copy()


def clear_selection() -> List[str]:
    """
    Clear all selected topics.
    
    Returns:
        Empty list
    """
    return []


def render_sidebar(canonical_model, topic_aggregates: List[Dict[str, Any]]):
    """Render sidebar with all controls."""
    st.sidebar.markdown("### Controls")
    
    # File uploader
    uploaded_file = st.sidebar.file_uploader(
        "Upload Excel File",
        type=['xlsx', 'xls'],
        key='file_uploader'
    )
    
    validation_report = {}
    
    if uploaded_file is not None:
        bytes_data = uploaded_file.read()
        file_hash = compute_file_hash(bytes_data)
        current_file_hash = st.session_state.get(SESSION_KEYS['file_hash'])
        
        # Check if file changed (by hash)
        if current_file_hash != file_hash:
            # New file uploaded - update hash and process
            st.session_state[SESSION_KEYS['file_hash']] = file_hash
            st.session_state[SESSION_KEYS['file_bytes']] = bytes_data
            st.session_state[SESSION_KEYS['uploaded_file']] = uploaded_file
            
            # Show loading indicator with progress
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            try:
                status_text.text("📊 Reading file...")
                progress_bar.progress(10)
                
                # Process file (includes validation)
                config = {}
                dict_of_dfs, validation_report, canonical_model, topic_columns = process_uploaded_file(bytes_data, config)
                
                # Validate file is readable
                is_valid, error_msg = edge_cases.validate_file_readable(validation_report)
                
                if not is_valid:
                    st.session_state['validation_error'] = error_msg
                    st.session_state[SESSION_KEYS['canonical_model']] = None
                    st.session_state[SESSION_KEYS['topic_aggregates']] = []
                    progress_bar.empty()
                    status_text.empty()
                    return None, [], validation_report
                
                status_text.text("🔄 Normalizing data...")
                progress_bar.progress(40)
                
                # Apply participant filtering if patterns exist
                filter_patterns = st.session_state.get('participant_filter_patterns', [])
                if filter_patterns:
                    canonical_model, filtered_ids = edge_cases.filter_participants_by_regex(
                        canonical_model, filter_patterns
                    )
                    if filtered_ids:
                        st.session_state['filtered_participants'] = list(filtered_ids)
                
                status_text.text("📈 Computing scores...")
                progress_bar.progress(70)
                
                # Identify single-sheet and sparse topics
                single_sheet_topics = edge_cases.identify_single_sheet_topics(
                    canonical_model, set(topic_columns)
                )
                st.session_state['single_sheet_topics'] = single_sheet_topics
                
                # Store in session state
                st.session_state[SESSION_KEYS['canonical_model']] = canonical_model
                st.session_state[SESSION_KEYS['validation_report']] = validation_report
                st.session_state['topic_columns'] = topic_columns
                
                # Compute scoring with cache
                topic_aggregates = compute_scoring_with_cache(file_hash, bytes_data, config)
                
                status_text.text("✅ Finalizing...")
                progress_bar.progress(90)
                
                # Identify sparse topics
                sparse_topics = edge_cases.identify_sparse_topics(topic_aggregates)
                st.session_state['sparse_topics'] = sparse_topics
                
                st.session_state[SESSION_KEYS['topic_aggregates']] = topic_aggregates
                
                # Reset selection and tracking
                st.session_state[SESSION_KEYS['selected_topics']] = []
                st.session_state['previous_top_n'] = None
                
                progress_bar.progress(100)
                status_text.text("✅ File processed successfully!")
                
                # Clear progress indicators after brief delay
                import time
                time.sleep(0.2)
                progress_bar.empty()
                status_text.empty()
                
                # Mark that processing is complete - this will trigger auto-refresh
                st.session_state['_processing_complete'] = True
                
            except Exception as e:
                progress_bar.empty()
                status_text.empty()
                st.error(f"❌ Error processing file: {str(e)}")
                st.session_state[SESSION_KEYS['canonical_model']] = None
                st.session_state[SESSION_KEYS['topic_aggregates']] = []
                return None, [], {}
        else:
            # Same file - use cached data from session state
            canonical_model = st.session_state.get(SESSION_KEYS['canonical_model'])
            topic_aggregates = st.session_state.get(SESSION_KEYS['topic_aggregates'], [])
            validation_report = st.session_state.get(SESSION_KEYS['validation_report'], {})
            
            # Get bytes_data from session state (file already read)
            bytes_data = st.session_state.get(SESSION_KEYS['file_bytes'])
            if bytes_data is None:
                # Fallback: read file again if bytes not in session state
                bytes_data = uploaded_file.read()
                st.session_state[SESSION_KEYS['file_bytes']] = bytes_data
            
            # If missing, recompute from cache
            if canonical_model is None or not topic_aggregates:
                config = {}
                dict_of_dfs, validation_report, canonical_model, topic_columns = process_uploaded_file(bytes_data, config)
                st.session_state[SESSION_KEYS['canonical_model']] = canonical_model
                st.session_state[SESSION_KEYS['validation_report']] = validation_report
                st.session_state['topic_columns'] = topic_columns
                topic_aggregates = compute_scoring_with_cache(file_hash, bytes_data, config)
                st.session_state[SESSION_KEYS['topic_aggregates']] = topic_aggregates
    else:
        # No file uploaded, use session state if available
        canonical_model = st.session_state.get(SESSION_KEYS['canonical_model'])
        topic_aggregates = st.session_state.get(SESSION_KEYS['topic_aggregates'], [])
        validation_report = st.session_state.get('validation_report', {})
    
    st.sidebar.divider()
    
    # Top N slider
    previous_top_n = st.session_state.get('previous_top_n')
    top_n = st.sidebar.slider(
        "Top N Topics",
        min_value=4,
        max_value=20,
        value=st.session_state[SESSION_KEYS['top_n']],
        key='top_n_slider'
    )
    
    # Track if N changed
    n_changed = previous_top_n is not None and top_n != previous_top_n
    st.session_state['previous_top_n'] = top_n
    st.session_state[SESSION_KEYS['top_n']] = top_n
    
    # Auto-select Top N checkbox
    auto_select = st.sidebar.checkbox(
        "Auto-select Top N",
        value=st.session_state[SESSION_KEYS['auto_select_top_n']],
        key='auto_select_checkbox'
    )
    st.session_state[SESSION_KEYS['auto_select_top_n']] = auto_select
    
    # Auto-add when N changes checkbox
    auto_add_on_change = st.sidebar.checkbox(
        "Auto-add when N changes",
        value=st.session_state.get('auto_add_on_change', False),
        key='auto_add_on_change_checkbox'
    )
    st.session_state['auto_add_on_change'] = auto_add_on_change
    
    if not topic_aggregates:
        st.sidebar.info("Upload a file to begin")
        return canonical_model, topic_aggregates, validation_report
    
    # Compute top topics
    top_topics = compute_top_topics(topic_aggregates)
    
    # Get available topic IDs
    available_topic_ids = [t['topic_id'] for t in topic_aggregates]
    
    # Limit options for performance (max 10 items in dropdown)
    MAX_DROPDOWN_ITEMS = 10
    
    # Compute selected topics based on behavior rules
    current_selected = st.session_state.get(SESSION_KEYS['selected_topics'], [])
    
    # Update selection if auto_select is enabled or N changed
    if auto_select or n_changed:
        new_selected = compute_selected_topics(
            top_topics,
            current_selected,
            top_n,
            auto_select,
            auto_add_on_change,
            previous_top_n
        )
        if new_selected != current_selected:
            st.session_state[SESSION_KEYS['selected_topics']] = new_selected
            current_selected = new_selected
    
    # Filter topics for multiselect - prioritize selected and top N
    top_n_ids = set([t['topic_id'] for t in topic_aggregates[:top_n]])
    selected_ids = set(current_selected)
    
    # Build prioritized list: selected first, then top N, then others
    # CRITICAL: Always include ALL selected topics first (even if they exceed MAX_DROPDOWN_ITEMS)
    prioritized_topics = []
    # First: all selected topics (must be included to avoid errors)
    selected_topics_list = [tid for tid in current_selected if tid in available_topic_ids]
    prioritized_topics.extend(selected_topics_list)
    # Second: top N topics that aren't selected
    prioritized_topics.extend([tid for tid in available_topic_ids if tid in top_n_ids and tid not in selected_ids])
    # Third: other topics
    prioritized_topics.extend([tid for tid in available_topic_ids if tid not in top_n_ids and tid not in selected_ids])
    
    # Limit to max items for performance, but ALWAYS keep all selected topics
    # If selected topics exceed MAX_DROPDOWN_ITEMS, include all of them anyway
    if len(prioritized_topics) > MAX_DROPDOWN_ITEMS:
        selected_count = len(selected_topics_list)
        if selected_count >= MAX_DROPDOWN_ITEMS:
            # All selected topics exceed limit - include only them (no choice)
            prioritized_topics = selected_topics_list
        else:
            # Keep all selected topics + fill remaining slots with others
            remaining_topics = prioritized_topics[selected_count:]
            remaining_slots = MAX_DROPDOWN_ITEMS - selected_count
            if remaining_slots > 0 and remaining_topics:
                prioritized_topics = selected_topics_list + remaining_topics[:remaining_slots]
            else:
                prioritized_topics = selected_topics_list
    
    # Filter default to only include topics that are in options (safety check)
    valid_default = [tid for tid in current_selected if tid in prioritized_topics]
    
    # Multi-select for selected topics (with limited options for performance)
    selected = st.sidebar.multiselect(
        "Selected Topics",
        options=prioritized_topics,
        default=valid_default,
        key='topic_multiselect',
        max_selections=50  # Limit selections for performance
    )
    st.session_state[SESSION_KEYS['selected_topics']] = selected
    
    # Search-add dropdown for topics outside Top N (with search functionality)
    other_topics = [tid for tid in available_topic_ids if tid not in top_n_ids]
    
    if other_topics:
        st.sidebar.markdown("**Add Topic**")
        
        # Add search box for filtering
        search_query = st.sidebar.text_input(
            "🔍 Search topic...",
            value="",
            key='topic_search_input',
            help="Type to filter topics"
        )
        
        # Filter topics by search query
        if search_query:
            filtered_topics = [tid for tid in other_topics if search_query.lower() in tid.lower()]
        else:
            # Limit to first 10 for performance when no search
            filtered_topics = other_topics[:10]
        
        if filtered_topics:
            add_topic = st.sidebar.selectbox(
                "Select topic to add",
                options=[''] + filtered_topics,
                key='add_topic_selectbox'
            )
            if add_topic and add_topic not in st.session_state[SESSION_KEYS['selected_topics']]:
                st.session_state[SESSION_KEYS['selected_topics']].append(add_topic)
                st.rerun()
        else:
            st.sidebar.caption("No topics found matching your search")
    
    # Reset and Clear buttons
    col1, col2 = st.sidebar.columns(2)
    with col1:
        if st.button("Reset to Top N", key='reset_button'):
            new_selection = reset_to_top_n(top_topics, top_n)
            st.session_state[SESSION_KEYS['selected_topics']] = new_selection
            st.rerun()
    
    with col2:
        if st.button("Clear Selection", key='clear_button'):
            st.session_state[SESSION_KEYS['selected_topics']] = clear_selection()
            st.rerun()
    
    st.sidebar.divider()
    
    # Filters
    st.sidebar.markdown("**Filters**")
    
    # Coverage tier
    coverage_tier = st.sidebar.selectbox(
        "Coverage Tier",
        options=['', 'High', 'Medium', 'Low'],
        key='coverage_tier_filter'
    )
    st.session_state[SESSION_KEYS['filters']]['coverage_tier'] = coverage_tier if coverage_tier else None
    
    # Tone rollup
    tone_rollup = st.sidebar.selectbox(
        "Tone Roll-up",
        options=['', 'Positive', 'Negative', 'Neutral', 'Mixed'],
        key='tone_rollup_filter'
    )
    st.session_state[SESSION_KEYS['filters']]['tone_rollup'] = tone_rollup.lower() if tone_rollup else None
    
    # High-emotion toggle
    high_emotion = st.sidebar.checkbox(
        "High Emotion Only",
        value=st.session_state[SESSION_KEYS['filters']]['high_emotion'],
        key='high_emotion_filter'
    )
    st.session_state[SESSION_KEYS['filters']]['high_emotion'] = high_emotion
    
    # Search box
    search_query = st.sidebar.text_input(
        "Search (summaries + quotes)",
        value=st.session_state[SESSION_KEYS['search_query']],
        key='search_input'
    )
    st.session_state[SESSION_KEYS['search_query']] = search_query
    
    # Reset filters button (in Filters section)
    if st.sidebar.button("🔄 Reset Filters", use_container_width=True, key='reset_filters_button', help="Clear all filters and search query"):
        # Reset all filters to default values
        st.session_state[SESSION_KEYS['filters']] = {
            'coverage_tier': None,
            'tone_rollup': None,
            'high_emotion': False,
        }
        st.session_state[SESSION_KEYS['search_query']] = ''
        st.session_state['participant_filter_patterns'] = []
        # Reset widget values by clearing their keys
        if 'coverage_tier_filter' in st.session_state:
            del st.session_state['coverage_tier_filter']
        if 'tone_rollup_filter' in st.session_state:
            del st.session_state['tone_rollup_filter']
        if 'high_emotion_filter' in st.session_state:
            del st.session_state['high_emotion_filter']
        if 'search_input' in st.session_state:
            del st.session_state['search_input']
        if 'participant_filter_input' in st.session_state:
            del st.session_state['participant_filter_input']
        st.rerun()
    
    st.sidebar.divider()
    
    # Participant filter (regex denylist)
    st.sidebar.markdown("**Participant Filter**")
    st.sidebar.caption("Filter out participants matching regex patterns (e.g., 'moderator|admin')")
    filter_pattern_input = st.sidebar.text_input(
        "Regex patterns (pipe-separated)",
        value='|'.join(st.session_state.get('participant_filter_patterns', [])),
        key='participant_filter_input',
        help="Enter regex patterns separated by | to exclude matching participant IDs"
    )
    
    if filter_pattern_input:
        patterns = [p.strip() for p in filter_pattern_input.split('|') if p.strip()]
        st.session_state['participant_filter_patterns'] = patterns
    else:
        st.session_state['participant_filter_patterns'] = []
    
    st.sidebar.divider()
    
    # Export buttons
    st.sidebar.markdown("**Export**")
    
    # Build digest for export (use filtered aggregates if available)
    selected_topic_ids = st.session_state.get(SESSION_KEYS['selected_topics'], [])
    if canonical_model:
        # Use filtered aggregates if available (after filters are applied), otherwise use all topic_aggregates
        aggregates_for_export = st.session_state.get('filtered_aggregates', topic_aggregates)
        if aggregates_for_export:
            selected_aggregates = [t for t in aggregates_for_export if t['topic_id'] in selected_topic_ids]
            digest_artifact = digest.build_digest(canonical_model, selected_aggregates, n_takeaways=5)
        
        html_content = export.export_to_html(digest_artifact, canonical_model)
        st.sidebar.download_button(
            label="📥 Export HTML",
            data=html_content,
            file_name="digest.html",
            mime="text/html",
            width='stretch'
        )
        
        md_content = export.export_to_markdown(digest_artifact, canonical_model)
        st.sidebar.download_button(
            label="📄 Export Markdown",
            data=md_content,
            file_name="digest.md",
            mime="text/markdown",
            width='stretch'
        )
    
    return canonical_model, topic_aggregates, validation_report


@st.cache_data(show_spinner=False)
def _format_matched_sheets(matched_sheets: Dict) -> str:
    """Format matched sheets as simple text for performance."""
    if not matched_sheets:
        return ""
    lines = []
    for role, sheet_name in matched_sheets.items():
        lines.append(f"✓ {role}: {sheet_name}")
    return "\n".join(lines)

@st.cache_data(show_spinner=False)
def _format_unmatched_sheets(unmatched_sheets: List) -> str:
    """Format unmatched sheets as simple text for performance."""
    if not unmatched_sheets:
        return ""
    return "\n".join([f"  - {name}" for name in unmatched_sheets])

@st.cache_data(show_spinner=False)
def _format_warnings(warnings: List) -> str:
    """Format warnings as simple text for performance."""
    if not warnings:
        return ""
    return "\n".join([f"⚠️ {w}" for w in warnings])

def render_validation_stats(validation_report: Dict, canonical_model):
    """Render validation report and dataset stats."""
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Participants", len(canonical_model.participants))
    
    with col2:
        st.metric("Topics", len(canonical_model.topics))
    
    with col3:
        st.metric("Evidence Cells", len(canonical_model.evidence_cells))
    
    with col4:
        matched = len(validation_report.get('matched_sheets', {}))
        st.metric("Matched Sheets", f"{matched}/3")
    
    # Show matched sheets info (simplified for performance)
    matched_sheets = validation_report.get('matched_sheets', {})
    if matched_sheets:
        with st.expander("📋 Matched Sheets Details", expanded=False):
            # Use simple text instead of multiple st.success calls
            matched_text = _format_matched_sheets(matched_sheets)
            st.text(matched_text)
    
    # Show unmatched sheets info (simplified for performance)
    unmatched_sheets = validation_report.get('unmatched_sheets', [])
    if unmatched_sheets:
        with st.expander("📄 Unmatched Sheets (not recognized)", expanded=False):
            st.info("These sheets were found but couldn't be matched to expected roles (summary, quotes, sentiments):")
            # Use simple text instead of multiple st.write calls
            unmatched_text = _format_unmatched_sheets(unmatched_sheets)
            st.text(unmatched_text)
            st.caption("💡 Tip: Rename sheets to include 'summary', 'quotes', or 'sentiments' in their names")
    
    # Validation warnings (simplified for performance)
    warnings = validation_report.get('warnings', [])
    if warnings:
        with st.expander("⚠️ Validation Warnings", expanded=False):
            # Use simple text instead of multiple st.warning calls
            warnings_text = _format_warnings(warnings)
            st.text(warnings_text)


def render_takeaways(digest_artifact: Dict[str, Any], canonical_model):
    """Render takeaways row with truncation budgets enforced."""
    st.subheader("Key Takeaways")
    takeaways = digest_artifact.get('takeaways', [])
    
    if not takeaways:
        st.info("No takeaways available. Select topics to generate takeaways.")
        return
    
    for takeaway in takeaways:
        takeaway_index = takeaway.get('takeaway_index', 0)
        takeaway_text_full = takeaway.get('takeaway_text', '')
        takeaway_text_truncated = render.format_takeaway_text(takeaway_text_full)
        source_topic_id = takeaway.get('source_topic_id', '')
        
        # Capitalize first letter of each word in source_topic_id (Title Case)
        source_topic_label = source_topic_id
        if source_topic_label:
            words = source_topic_label.split()
            source_topic_label = ' '.join(word.capitalize() for word in words)
        
        # Find corresponding topic card
        topic_cards = digest_artifact.get('topic_cards', [])
        topic_card = next((tc for tc in topic_cards if tc['topic_id'] == source_topic_id), None)
        
        # Check if text was truncated
        is_truncated = len(takeaway_text_full) > render.TAKEAWAY_MAX
        
        with st.container():
            col1, col2 = st.columns([3, 1])
            
            with col1:
                # Show truncated text (max 180 chars)
                st.write(f"**{takeaway_index}.** {takeaway_text_truncated}")
                
                # Always show expander for full takeaway text
                with st.expander("📖 Show full takeaway"):
                    st.write(takeaway_text_full)
                
                if source_topic_id:
                    st.caption(f"From: {source_topic_label}")
            
            with col2:
                if topic_card:
                    # Show evidence count as credibility signal (no extra labels)
                    evidence_count = topic_card.get('evidence_count', 0)
                    st.metric("Evidence", evidence_count)
            
            st.divider()


def render_topic_cards(digest_artifact: Dict[str, Any], canonical_model):
    """Render topic cards row with truncation budgets enforced."""
    st.subheader("Topic Cards")
    topic_cards = digest_artifact.get('topic_cards', [])
    
    if not topic_cards:
        st.info("No topic cards available. Select topics to view details.")
        return
    
    for card in topic_cards:
        topic_id = card.get('topic_id', '')
        # Capitalize first letter of each word in topic_id (Title Case)
        topic_label = topic_id
        if topic_label:
            words = topic_label.split()
            topic_label = ' '.join(word.capitalize() for word in words)
        
        topic_oneliner_full = card.get('topic_one_liner') or ''
        topic_oneliner_truncated = render.format_topic_oneliner(topic_oneliner_full)
        topic_oneliner_is_truncated = len(topic_oneliner_full) > render.TOPIC_ONELINER_MAX
        
        coverage_rate = card.get('coverage_rate', 0.0)
        evidence_count = card.get('evidence_count', 0)  # Always show this
        sentiment_mix = card.get('sentiment_mix', {})
        proof_quote_preview_full = card.get('proof_quote_preview', '')
        
        # Check if this is a fallback message or invalid quote
        is_fallback = proof_quote_preview_full == "No representative quote available"
        is_valid_quote = proof_quote_preview_full and proof_quote_preview_full.strip() and not is_fallback
        
        # Additional validation: check if it's not just a numeric placeholder
        if is_valid_quote:
            text_stripped = proof_quote_preview_full.strip()
            # Check if it's just a numeric index pattern
            numeric_patterns = [
                r'^\d+\.?\s*$',  # "1", "1.", "1. "
                r'^\d+\)\s*$',   # "1)"
                r'^\(\d+\)\s*$', # "(1)"
            ]
            for pattern in numeric_patterns:
                if re.match(pattern, text_stripped):
                    is_valid_quote = False
                    break
            # Check if it has actual words (letters)
            if is_valid_quote:
                text_no_punct = re.sub(r'[^\w\s]', '', text_stripped)
                if not re.search(r'[a-zA-Z]', text_no_punct):
                    is_valid_quote = False
        
        receipt_links = card.get('receipt_links', [])
        
        with st.container():
            st.markdown(f"### {topic_label}")
            
            # Topic one-liner (truncated to 240 chars)
            if topic_oneliner_truncated:
                st.write(topic_oneliner_truncated)
                # Show full text in expander if truncated
                if topic_oneliner_is_truncated:
                    with st.expander("📖 Show full one-liner"):
                        st.write(topic_oneliner_full)
            
            # Coverage and evidence (always shown)
            col1, col2 = st.columns(2)
            with col1:
                st.markdown(render.format_coverage_bar_html(coverage_rate), unsafe_allow_html=True)
            with col2:
                # Clarify what evidence_count means: total supporting excerpts
                st.metric("Total Evidence", evidence_count)
                st.caption("Number of source excerpts supporting this topic")
            
            # Sentiment mix
            st.markdown(render.format_sentiment_mix_html(sentiment_mix), unsafe_allow_html=True)
            
            # Proof quote preview (static, 2-3 lines, ~150 chars, with sentiment chip)
            if is_valid_quote and proof_quote_preview_full:
                # Get sentiment for proof quote
                proof_quote_ref = card.get('proof_quote_ref', '')
                proof_quote_sentiment = None
                if proof_quote_ref and ':' in proof_quote_ref:
                    # Get sentiment from receipt display
                    proof_receipt = render.build_receipt_display(
                        proof_quote_ref, canonical_model, topic_id=topic_id
                    )
                    proof_quote_sentiment = proof_receipt.get('sentiment')
                
                # Truncate to ~150 chars for 2-3 lines preview
                proof_preview_short = render.truncate(proof_quote_preview_full, 150)
                is_quote_truncated = len(proof_quote_preview_full) > 150
                
                # Display static preview with sentiment chip
                col1, col2 = st.columns([4, 1])
                with col1:
                    st.caption(f"*\"{proof_preview_short}\"*")
                with col2:
                    if proof_quote_sentiment:
                        st.markdown(render.format_sentiment_chip(proof_quote_sentiment), unsafe_allow_html=True)
                
                # Always show expander for full proof quote
                with st.expander("💬 Proof Quote"):
                    st.write(proof_quote_preview_full)
            
            # Receipts expander with pagination
            if receipt_links:
                total_receipts = len(receipt_links)
                with st.expander(f"📋 Show receipts ({total_receipts})"):
                    # Convert receipt references to display objects
                    receipt_displays = []
                    for receipt_ref in receipt_links:
                        receipt_display = render.build_receipt_display(
                            receipt_ref, canonical_model, topic_id=topic_id
                        )
                        receipt_displays.append(receipt_display)
                    
                    # Rank receipts for consistent ordering
                    ranked_receipts, _ = render.rank_and_limit_receipts(
                        receipt_displays,
                        max_display=len(receipt_displays),  # Rank all, don't limit yet
                        prioritize_diversity=True
                    )
                    
                    # Pagination setup
                    receipt_page_key = f'receipts_page_topic_{topic_id}'
                    if receipt_page_key not in st.session_state:
                        st.session_state[receipt_page_key] = 0
                    
                    current_page = st.session_state[receipt_page_key]
                    page_size = 8  # 5-10 receipts per page
                    total_pages = (len(ranked_receipts) + page_size - 1) // page_size if ranked_receipts else 1
                    start_idx = current_page * page_size
                    end_idx = min(start_idx + page_size, len(ranked_receipts))
                    
                    # Get receipts for current page
                    displayed_receipts = ranked_receipts[start_idx:end_idx]
                    
                    # Show progress indicator
                    if total_pages > 1:
                        st.caption(f"Page {current_page + 1} of {total_pages} — Showing receipts {start_idx + 1}-{end_idx} of {len(ranked_receipts)}")
                    
                    # Display receipts
                    for receipt_idx, receipt in enumerate(displayed_receipts):
                        participant_label = receipt.get('participant_label', 'Unknown')
                        quote_full = receipt.get('quote_full', '')
                        sentiment = receipt.get('sentiment')
                        
                        if quote_full and quote_full != 'No quote text available':
                            # Check if quote is longer than 400 chars
                            is_long_quote = len(quote_full) > 400
                            
                            # Display receipt with participant label, quote, and sentiment
                            col1, col2 = st.columns([4, 1])
                            with col1:
                                if is_long_quote:
                                    # For long quotes: use toggle to switch between truncated and full
                                    quote_preview = render.truncate(quote_full, 400)
                                    
                                    # Create unique key for this receipt's toggle
                                    toggle_key = f"show_full_quote_{topic_id}_{receipt_idx}_{current_page}"
                                    show_full = st.toggle("📖 Show full quote", key=toggle_key, value=False)
                                    
                                    # Show either preview or full quote based on toggle
                                    if show_full:
                                        st.write(f"**{participant_label}**: \"{quote_full}\"")
                                    else:
                                        st.write(f"**{participant_label}**: \"{quote_preview}\"")
                                else:
                                    # Short quote - show full text as one block
                                    st.write(f"**{participant_label}**: \"{quote_full}\"")
                            with col2:
                                if sentiment:
                                    st.markdown(render.format_sentiment_chip(sentiment), unsafe_allow_html=True)
                        else:
                            st.caption(f"**{participant_label}**: (Quote text not available)")
                        
                        st.markdown("---")
                    
                    # Pagination controls
                    if total_pages > 1:
                        col1, col2, col3 = st.columns([1, 1, 1])
                        
                        with col1:
                            if st.button("◀ Previous", key=f"receipts_prev_topic_{topic_id}", disabled=(current_page == 0), use_container_width=True):
                                st.session_state[receipt_page_key] = max(0, current_page - 1)
                                st.rerun()
                        
                        with col2:
                            st.markdown(f"<div style='text-align: center; padding-top: 8px;'><strong>{current_page + 1} / {total_pages}</strong></div>", unsafe_allow_html=True)
                        
                        with col3:
                            if st.button("Next ▶", key=f"receipts_next_topic_{topic_id}", disabled=(current_page >= total_pages - 1), use_container_width=True):
                                st.session_state[receipt_page_key] = min(total_pages - 1, current_page + 1)
                                st.rerun()
            else:
                st.caption("No evidence available for this topic.")
            
            st.divider()


def _format_sentiment_with_icon(sentiment_label: str) -> str:
    """Format sentiment label with icon for visual scanning."""
    icons = {
        'positive': '✅',
        'negative': '❌',
        'neutral': '⚪',
        'mixed': '🔄',
        'unknown': '❓'
    }
    colors = {
        'positive': '#22c55e',  # green
        'negative': '#ef4444',  # red
        'neutral': '#6b7280',  # gray
        'mixed': '#f59e0b',     # amber
        'unknown': '#9ca3af'     # light gray
    }
    icon = icons.get(sentiment_label, '❓')
    color = colors.get(sentiment_label, '#9ca3af')
    label_text = sentiment_label.capitalize()
    return f'<span style="color: {color};">{icon} {label_text}</span>'


def _format_importance_with_rank(importance_score: float, rank: int) -> str:
    """Format importance score with rank index."""
    return f"{importance_score:.2f} (#{rank})"


def _format_coverage_bar_html(coverage_pct: float, max_width: int = 100) -> str:
    """Format coverage as HTML progress bar."""
    width_px = int((coverage_pct / 100.0) * max_width)
    return f'''
    <div style="display: flex; align-items: center; gap: 8px;">
        <div style="width: {max_width}px; height: 8px; background-color: #e5e7eb; border-radius: 4px; overflow: hidden;">
            <div style="width: {width_px}px; height: 100%; background-color: #655CFE;"></div>
        </div>
        <span style="font-size: 12px; color: #374151;">{coverage_pct:.1f}%</span>
    </div>
    '''


def _format_sentiment_distribution_bar(sentiment_dist: Dict[str, int]) -> str:
    """Format sentiment distribution as HTML bar chart."""
    total = sum(sentiment_dist.values())
    if total == 0:
        return '<div style="color: #9ca3af; font-size: 12px;">No sentiment data</div>'
    
    # Color mapping for sentiments
    colors = {
        'positive': '#22c55e',  # green
        'negative': '#ef4444',  # red
        'neutral': '#94a3b8',   # gray
        'mixed': '#f59e0b',     # amber
        'unknown': '#9ca3af'     # gray
    }
    
    bars = []
    for sentiment, count in sentiment_dist.items():
        if count > 0:
            percentage = (count / total) * 100
            color = colors.get(sentiment, '#9ca3af')
            bars.append(f'''
                <div style="display: flex; align-items: center; margin-bottom: 4px;">
                    <div style="width: 60px; font-size: 11px; color: #374151;">{sentiment.capitalize()}</div>
                    <div style="flex: 1; height: 16px; background-color: #e5e7eb; border-radius: 2px; overflow: hidden; margin: 0 8px;">
                        <div style="width: {percentage}%; height: 100%; background-color: {color};"></div>
                    </div>
                    <div style="width: 30px; font-size: 11px; color: #6b7280; text-align: right;">{count}</div>
                </div>
            ''')
    
    return '<div>' + ''.join(bars) + '</div>'


def _get_topic_confidence_data(topic: explore_model.ExploreTopic, canonical_model) -> Dict[str, Any]:
    """
    Compute confidence signals for a topic.
    
    Returns:
        Dictionary with:
        - mentions_count: int
        - source_documents_count: int (unique participants)
        - sentiment_distribution: Dict[str, int]
    """
    topic_id = topic.topic_id
    
    # Get evidence cells for this topic
    evidence_cells = [ec for ec in canonical_model.evidence_cells if ec.topic_id == topic_id]
    
    # Count unique participants (source documents)
    unique_participants = set(ec.participant_id for ec in evidence_cells if ec.participant_id)
    source_documents_count = len(unique_participants)
    
    # Get sentiment distribution
    sentiment_distribution = digest._compute_sentiment_mix(evidence_cells, topic_id)
    
    return {
        'mentions_count': topic.mentions_count,
        'source_documents_count': source_documents_count,
        'sentiment_distribution': sentiment_distribution
    }


def render_explore_tab(topic_aggregates: List[Dict[str, Any]], canonical_model):
    """
    Render Explore tab with improved UX table view and recipient filtering.
    
    Features:
    - Recipient selector for persona-based filtering
    - Importance score with rank index
    - Coverage as progress bar
    - Sentiment with icons
    - Truncated summaries with expand toggle
    - Pagination (15 per page)
    """
    if not topic_aggregates:
        st.info("Upload a file and compute scores to view data.")
        return
    
    # Cache key for base ranked topics (to avoid recomputation)
    cache_key = f"explore_base_ranked_topics_{hash(str(topic_aggregates))}"
    
    # Convert to ExploreTopic model and rank (only once, cached)
    if cache_key not in st.session_state:
        explore_topics = []
        for topic_agg in topic_aggregates:
            topic_id = topic_agg['topic_id']
            sentiment_mix = digest._compute_sentiment_mix(canonical_model.evidence_cells, topic_id)
            explore_topic = explore_model.from_topic_aggregate(topic_agg, sentiment_mix, topic_id)
            explore_topics.append(explore_topic)
        
        # Rank topics (base ranking, before recipient filtering)
        ranked_topics = explore_model.rank_topics(explore_topics)
        st.session_state[cache_key] = ranked_topics
    else:
        ranked_topics = st.session_state[cache_key]
    
    # Get available recipients
    default_recipients = recipient.create_default_recipients()
    
    # Ensure "General" is the first option
    recipient_options = {r.recipient_id: r for r in default_recipients}
    if "general" not in recipient_options:
        recipient_options["general"] = recipient.RecipientProfile(
            recipient_id="general",
            label="General",
            priority_topics=[],
            deprioritized_topics=[]
        )
    
    # Recipient selector - ensure "general" is first
    recipient_ids = list(recipient_options.keys())
    # Sort to put "general" first, then others
    recipient_ids = sorted(recipient_ids, key=lambda x: (x != "general", x))
    recipient_labels = [recipient_options[rid].label for rid in recipient_ids]
    
    # Initialize selected recipient (default to "general")
    if 'explore_selected_recipient' not in st.session_state:
        st.session_state['explore_selected_recipient'] = "general"
    
    # Track previous recipient to detect changes
    previous_recipient = st.session_state.get('explore_previous_recipient', None)
    
    # Create selectbox for recipient selection
    selected_recipient_idx = recipient_ids.index(st.session_state['explore_selected_recipient']) if st.session_state['explore_selected_recipient'] in recipient_ids else 0
    selected_label = st.selectbox(
        "View for:",
        options=recipient_labels,
        index=selected_recipient_idx,
        key='explore_recipient_selector',
        help="Select a recipient persona to filter topics"
    )
    
    # Update session state with selected recipient ID
    selected_recipient_id = recipient_ids[recipient_labels.index(selected_label)]
    
    # Detect recipient change and reset page to avoid out-of-bounds
    if previous_recipient != selected_recipient_id:
        st.session_state['explore_page'] = 1
        st.session_state['explore_previous_recipient'] = selected_recipient_id
    
    st.session_state['explore_selected_recipient'] = selected_recipient_id
    
    # Get selected recipient profile
    selected_recipient = recipient_options[selected_recipient_id]
    
    # Apply recipient filtering (instant, no recomputation)
    filtered_topics = recipient.filter_topics_for_recipient(ranked_topics, selected_recipient)
    
    # Show active recipient label
    st.markdown(f"### 📊 Explore Topics — *{selected_recipient.label}*")
    
    # Empty state: No topics after recipient filtering
    if not filtered_topics:
        if len(ranked_topics) == 0:
            # No topics at all in the data
            st.warning("⚠️ **No topics found in the uploaded data.**\n\nPlease check your file and ensure it contains valid topic data.")
        else:
            # Topics exist but were filtered out
            deprioritized_topics = [t for t in ranked_topics 
                                   if t.topic_id.lower().strip() in [tid.lower().strip() 
                                                                     for tid in selected_recipient.deprioritized_topics]]
            has_deprioritized = len(deprioritized_topics) > 0
            has_priority = len(selected_recipient.priority_topics) > 0
            
            explanation_parts = []
            if has_deprioritized:
                explanation_parts.append(f"- {len(deprioritized_topics)} topic(s) are deprioritized for this recipient")
                explanation_parts.append("- Deprioritized topics are hidden unless they have high signal (importance ≥ 1.9 and coverage ≥ 90%)")
            if has_priority:
                explanation_parts.append(f"- Priority topics are boosted but may still be filtered if they don't meet signal thresholds")
            
            if not explanation_parts:
                explanation_parts.append("- The recipient filter may be too restrictive")
            
            st.warning(
                f"⚠️ **No topics match the current filter for *{selected_recipient.label}*.**\n\n"
                f"**Why this happened:**\n" + "\n".join(explanation_parts) + "\n\n"
                f"**What to try:**\n"
                f"- Select 'General' recipient to see all {len(ranked_topics)} topics\n"
                f"- Check the recipient's priority and deprioritized topic settings\n"
                f"- Verify that topics meet the signal thresholds"
            )
        return
    
    # Show topic count caption
    if selected_recipient.priority_topics or selected_recipient.deprioritized_topics:
        filtered_count = len(filtered_topics)
        total_count = len(ranked_topics)
        if filtered_count < total_count:
            st.info(f"ℹ️ Showing {filtered_count} of {total_count} topics (filtered for {selected_recipient.label})")
        else:
            st.caption(f"Showing all {filtered_count} topics")
    else:
        st.caption(f"Showing all {len(filtered_topics)} topics")
    
    # Pagination (on filtered topics)
    items_per_page = 15
    total_pages = (len(filtered_topics) + items_per_page - 1) // items_per_page
    
    # Initialize page in session state if not exists
    if 'explore_page' not in st.session_state:
        st.session_state['explore_page'] = 1
    
    # Reset page if out of bounds
    if st.session_state['explore_page'] > total_pages:
        st.session_state['explore_page'] = 1
    
    if total_pages > 1:
        # Use session state value and update it
        current_page = st.session_state.get('explore_page', 1)
        page = st.number_input(
            "Page",
            min_value=1,
            max_value=total_pages,
            value=current_page,
            key='explore_page_number_input',
            help=f"Showing {items_per_page} topics per page"
        )
        # Update session state
        st.session_state['explore_page'] = page
        start_idx = (page - 1) * items_per_page
        end_idx = start_idx + items_per_page
        page_topics = filtered_topics[start_idx:end_idx]
        st.caption(f"Page {page} of {total_pages} ({len(filtered_topics)} total topics)")
    else:
        page_topics = filtered_topics
        page = 1
        start_idx = 0
        st.session_state['explore_page'] = 1
    
    # Summary truncation length
    SUMMARY_TRUNCATE_LENGTH = 100
    
    # Add CSS for frozen first column effect, sticky header, and full-width expander
    st.markdown("""
    <style>
        /* Ensure sentiment chips have correct colors - override any brand color CSS */
        span[style*="background-color: #ef4444"] {
            background-color: #ef4444 !important;
            color: white !important;
        }
        
        /* Style for Explore table - frozen first column effect */
        .explore-table-row {
            border-bottom: 1px solid #e5e7eb;
            padding: 8px 0;
        }
        /* Make topic column stand out (frozen effect) */
        div[data-testid="column"]:first-child {
            position: sticky;
            left: 0;
            background-color: white;
            z-index: 1;
            padding-right: 16px;
        }
        
        /* Sticky table header - make header row stick to top when scrolling */
        .explore-table-header,
        [data-testid="stHorizontalBlock"].explore-table-header {
            position: sticky !important;
            top: 0 !important;
            z-index: 100 !important;
            background-color: white !important;
            padding: 12px 0 !important;
            margin-bottom: 8px !important;
            border-bottom: 2px solid #e5e7eb !important;
            box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05) !important;
        }
        
        /* Ensure header columns have proper styling */
        .explore-table-header [data-testid="column"],
        [data-testid="stHorizontalBlock"].explore-table-header [data-testid="column"] {
            background-color: white !important;
            padding: 8px 12px !important;
        }
        
        /* Make header text bold and slightly larger */
        .explore-table-header [data-testid="stMarkdownContainer"],
        [data-testid="stHorizontalBlock"].explore-table-header [data-testid="stMarkdownContainer"] {
            font-weight: 600 !important;
            font-size: 0.95rem !important;
            color: #1f2937 !important;
        }
        
        /* Alternative selector for header row */
        [data-testid="stHorizontalBlock"]:has([data-testid="column"]:has-text("Topic")):has([data-testid="column"]:has-text("Importance")) {
            position: sticky !important;
            top: 0 !important;
            z-index: 100 !important;
            background-color: white !important;
            padding: 12px 0 !important;
            border-bottom: 2px solid #e5e7eb !important;
            box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05) !important;
        }
        
        /* Full-width expander when open - AGGRESSIVE approach */
        [data-testid="stExpander"].full-width-expander {
            position: fixed !important;
            left: 0 !important;
            right: 0 !important;
            width: 100vw !important;
            max-width: 100vw !important;
            margin-left: 0 !important;
            margin-right: 0 !important;
            padding-left: 2rem !important;
            padding-right: 2rem !important;
            z-index: 9999 !important;
            background-color: white !important;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1) !important;
        }
        
        /* Break out of ALL parent containers */
        [data-testid="column"]:has([data-testid="stExpander"].full-width-expander),
        [data-testid="stHorizontalBlock"]:has([data-testid="stExpander"].full-width-expander),
        [data-testid="stVerticalBlock"]:has([data-testid="stExpander"].full-width-expander) {
            width: 100vw !important;
            max-width: 100vw !important;
            padding-left: 0 !important;
            padding-right: 0 !important;
        }
        
        /* Ensure expander content is also full width */
        [data-testid="stExpander"].full-width-expander > div {
            max-width: 100% !important;
            width: 100% !important;
            padding-left: 0 !important;
            padding-right: 0 !important;
        }
        
        [data-testid="stExpander"].full-width-expander [data-testid="stExpanderContent"] {
            max-width: 100% !important;
            width: 100% !important;
            padding-left: 0 !important;
            padding-right: 0 !important;
        }
        
        /* Make all inner content full width */
        [data-testid="stExpander"].full-width-expander * {
            max-width: 100% !important;
        }
    </style>
    <script>
        // Make "Show full summary" expander full-width when open
        function makeExpanderFullWidth() {
            // Find all expanders
            document.querySelectorAll('[data-testid="stExpander"]').forEach(expander => {
                const button = expander.querySelector('button');
                if (!button) return;
                
                // Check if this is a "Show full summary" expander (with or without emoji)
                const buttonText = button.textContent || button.innerText || '';
                const isSummaryExpander = buttonText.includes('Show full summary') || 
                                         buttonText.includes('full summary') ||
                                         buttonText.includes('📖 Show full summary');
                
                if (!isSummaryExpander) {
                    // Reset any styles if not a summary expander
                    expander.classList.remove('full-width-expander');
                    expander.style.width = '';
                    expander.style.maxWidth = '';
                    expander.style.position = '';
                    expander.style.left = '';
                    expander.style.right = '';
                    expander.style.marginLeft = '';
                    expander.style.marginRight = '';
                    expander.style.zIndex = '';
                    return;
                }
                
                // Check if expanded
                const isExpanded = button.getAttribute('aria-expanded') === 'true' || 
                                   expander.classList.contains('streamlit-expanderHeader--is-open') ||
                                   button.classList.contains('streamlit-expanderHeader--is-open');
                
                if (isExpanded) {
                    // Make full width when expanded - use fixed positioning
                    expander.classList.add('full-width-expander');
                    
                    // Get current position to maintain vertical position
                    const rect = expander.getBoundingClientRect();
                    const scrollY = window.scrollY || window.pageYOffset;
                    
                    // Apply full-width fixed styles
                    expander.style.position = 'fixed';
                    expander.style.top = `${rect.top + scrollY}px`;
                    expander.style.left = '0';
                    expander.style.right = '0';
                    expander.style.width = '100vw';
                    expander.style.maxWidth = '100vw';
                    expander.style.marginLeft = '0';
                    expander.style.marginRight = '0';
                    expander.style.paddingLeft = '2rem';
                    expander.style.paddingRight = '2rem';
                    expander.style.zIndex = '9999';
                    expander.style.backgroundColor = 'white';
                    expander.style.boxShadow = '0 4px 6px rgba(0, 0, 0, 0.1)';
                    
                    // Make content full width
                    const content = expander.querySelector('[data-testid="stExpanderContent"]');
                    if (content) {
                        content.style.maxWidth = '100%';
                        content.style.width = '100%';
                        content.style.paddingLeft = '0';
                        content.style.paddingRight = '0';
                    }
                    
                    // Also expand parent containers
                    let parent = expander.parentElement;
                    while (parent && parent !== document.body) {
                        if (parent.hasAttribute('data-testid')) {
                            parent.style.width = '100vw';
                            parent.style.maxWidth = '100vw';
                        }
                        parent = parent.parentElement;
                    }
                } else {
                    // Reset to normal width when collapsed
                    expander.classList.remove('full-width-expander');
                    expander.style.width = '';
                    expander.style.maxWidth = '';
                    expander.style.position = '';
                    expander.style.left = '';
                    expander.style.right = '';
                    expander.style.marginLeft = '';
                    expander.style.marginRight = '';
                    expander.style.paddingLeft = '';
                    expander.style.paddingRight = '';
                    expander.style.zIndex = '';
                    
                    // Reset content
                    const content = expander.querySelector('[data-testid="stExpanderContent"]');
                    if (content) {
                        content.style.maxWidth = '';
                        content.style.width = '';
                        content.style.paddingLeft = '';
                        content.style.paddingRight = '';
                    }
                }
            });
        }
        
        // Run on load
        if (document.readyState === 'loading') {
            document.addEventListener('DOMContentLoaded', makeExpanderFullWidth);
        } else {
            makeExpanderFullWidth();
        }
        
        // Watch for changes (when expander is clicked or aria-expanded changes)
        const expanderObserver = new MutationObserver((mutations) => {
            let shouldUpdate = false;
            mutations.forEach(mutation => {
                if (mutation.type === 'attributes' && mutation.attributeName === 'aria-expanded') {
                    shouldUpdate = true;
                }
                if (mutation.type === 'childList') {
                    shouldUpdate = true;
                }
            });
            if (shouldUpdate) {
                setTimeout(makeExpanderFullWidth, 50);
            }
        });
        
        expanderObserver.observe(document.body, { 
            childList: true, 
            subtree: true, 
            attributes: true, 
            attributeFilter: ['aria-expanded', 'class'] 
        });
        
        // Listen for click events on expander buttons
        document.addEventListener('click', function(e) {
            const expanderButton = e.target.closest('[data-testid="stExpander"] button');
            if (expanderButton) {
                setTimeout(makeExpanderFullWidth, 150);
            }
        });
        
        // Also run on window resize
        let resizeTimeout;
        window.addEventListener('resize', function() {
            clearTimeout(resizeTimeout);
            resizeTimeout = setTimeout(makeExpanderFullWidth, 100);
        });
        
        // Make table header sticky
        function makeHeaderSticky() {
            // Find all horizontal blocks (columns containers)
            const horizontalBlocks = document.querySelectorAll('[data-testid="stHorizontalBlock"]');
            
            for (let block of horizontalBlocks) {
                const columns = block.querySelectorAll('[data-testid="column"]');
                if (columns.length < 6) continue;
                
                // Check if this block contains header text
                let hasTopic = false;
                let hasImportance = false;
                let hasCoverage = false;
                let hasMentions = false;
                let hasSentiment = false;
                let hasSummary = false;
                
                for (let col of columns) {
                    const text = (col.textContent || col.innerText || '').trim().toLowerCase();
                    if (text.includes('topic') && !text.includes('topic_id')) hasTopic = true;
                    if (text.includes('importance')) hasImportance = true;
                    if (text.includes('coverage')) hasCoverage = true;
                    if (text.includes('mentions')) hasMentions = true;
                    if (text.includes('sentiment')) hasSentiment = true;
                    if (text.includes('summary')) hasSummary = true;
                }
                
                // If it has all header columns, make it sticky
                if (hasTopic && hasImportance && hasCoverage && hasMentions && hasSentiment && hasSummary) {
                    block.classList.add('explore-table-header');
                    // Also ensure all child columns have white background
                    columns.forEach(col => {
                        col.style.backgroundColor = 'white';
                    });
                    break; // Found the header, stop searching
                }
            }
        }
        
        // Run header sticky on load
        if (document.readyState === 'loading') {
            document.addEventListener('DOMContentLoaded', makeHeaderSticky);
        } else {
            setTimeout(makeHeaderSticky, 100);
        }
        
        // Reapply on Streamlit updates
        const headerObserver = new MutationObserver(() => {
            setTimeout(makeHeaderSticky, 200);
        });
        headerObserver.observe(document.body, { childList: true, subtree: true });
    </script>
    """, unsafe_allow_html=True)
    
    # Table header
    header_cols = st.columns([2, 1.5, 1.5, 1.5, 1.5, 3])
    with header_cols[0]:
        st.markdown("**Topic**")
    with header_cols[1]:
        st.markdown("**Importance**")
    with header_cols[2]:
        st.markdown("**Coverage**")
    with header_cols[3]:
        st.markdown("**Mentions**")
    with header_cols[4]:
        st.markdown("**Sentiment**")
    with header_cols[5]:
        st.markdown("**Summary**")
    
    st.divider()
    
    # Render rows
    for idx, topic in enumerate(page_topics):
        rank = start_idx + idx + 1 if total_pages > 1 else idx + 1
        
        row_cols = st.columns([2, 1.5, 1.5, 1.5, 1.5, 3])
        
        with row_cols[0]:
            # Topic label (frozen column effect via styling)
            # topic_label is already capitalized in explore_model.from_topic_aggregate
            st.markdown(f"**{topic.topic_label}**")
            st.caption(topic.topic_id)
        
        with row_cols[1]:
            # Importance score with rank
            importance_text = _format_importance_with_rank(topic.importance_score, rank)
            st.markdown(importance_text)
            # Show signal bucket as subtle indicator
            bucket = topic.get_signal_bucket()
            st.caption(f"Signal: {bucket}")
        
        with row_cols[2]:
            # Coverage as progress bar
            coverage_html = _format_coverage_bar_html(topic.coverage_pct)
            st.markdown(coverage_html, unsafe_allow_html=True)
        
        with row_cols[3]:
            # Mentions count
            st.markdown(f"**{topic.mentions_count}**")
        
        with row_cols[4]:
            # Sentiment with icon
            sentiment_text = _format_sentiment_with_icon(topic.sentiment_label)
            
            # Check for sparse sentiment data (low confidence)
            confidence_data = _get_topic_confidence_data(topic, canonical_model)
            sentiment_dist = confidence_data.get('sentiment_distribution', {})
            total_sentiments = sum(sentiment_dist.values())
            unknown_count = sentiment_dist.get('unknown', 0)
            
            # Show "Low confidence" warning if sentiment data is sparse
            is_sparse = (
                total_sentiments == 0 or  # No sentiment data
                (total_sentiments > 0 and unknown_count / total_sentiments > 0.7) or  # >70% unknown
                (total_sentiments < 3 and topic.sentiment_label == 'unknown')  # Very few sentiments and all unknown
            )
            
            # Use st.html for proper HTML rendering
            st.html(sentiment_text)
            if is_sparse:
                st.caption("*<span style='color: #f59e0b;'>⚠️ Low confidence</span>*", unsafe_allow_html=True)
        
        with row_cols[5]:
            # Summary with truncation and expand toggle
            summary_full = topic.summary_text
            # Handle missing summaries with placeholder
            if not summary_full or summary_full.strip() == "":
                st.markdown("*<span style='color: #9ca3af; font-style: italic;'>No summary available</span>*", unsafe_allow_html=True)
            else:
                summary_truncated = render.truncate(summary_full, SUMMARY_TRUNCATE_LENGTH)
                is_truncated = len(summary_full) > SUMMARY_TRUNCATE_LENGTH
                
                if is_truncated:
                    # Show truncated text
                    st.markdown(summary_truncated)
                    # Expand/collapse using expander (local state only)
                    with st.expander("📖 Show full summary", expanded=False):
                        st.markdown(summary_full)
                else:
                    # Show full text if not truncated
                    st.markdown(summary_full)
        
        # Confidence section (collapsible)
        confidence_data = _get_topic_confidence_data(topic, canonical_model)
        with st.expander(f"🔍 Confidence Signals — {topic.topic_label}", expanded=False):
            # Mentions count
            st.markdown(f"**Mentions:** {confidence_data['mentions_count']}")
            
            # Source documents count
            st.markdown(f"**Source Documents:** {confidence_data['source_documents_count']}")
            
            # Sentiment distribution as chips/badges
            st.markdown("**Sentiment Distribution:**")
            sentiment_chips_html = render.format_sentiment_mix_html(confidence_data['sentiment_distribution'])
            st.html(sentiment_chips_html)
        
        st.divider()


def apply_brand_styles():
    """Apply brand color (#655CFE) to Streamlit UI elements."""
    brand_color = "#655CFE"
    brand_color_hover = "#5548E8"  # Slightly darker for hover states
    
    st.markdown(f"""
    <style>
        /* Brand color: #655CFE */
        :root {{
            --brand-color: {brand_color};
            --brand-color-hover: {brand_color_hover};
        }}
        
        /* Global override for Streamlit red colors */
        * {{
            --primary-color: {brand_color} !important;
        }}
        
        /* Override Streamlit's default red primary color - AGGRESSIVE */
        [data-baseweb="slider"] > div > div {{
            background-color: {brand_color} !important;
        }}
        
        [data-baseweb="slider-track"] {{
            background-color: {brand_color} !important;
        }}
        
        [data-baseweb="slider-handle"] {{
            background-color: {brand_color} !important;
            border-color: {brand_color} !important;
        }}
        
        /* Slider value display */
        .stSlider [data-testid="stMarkdownContainer"] {{
            color: {brand_color} !important;
        }}
        
        .stSlider label + div {{
            color: {brand_color} !important;
        }}
        
        /* Checkbox - override all red */
        [data-baseweb="checkbox"] input[type="checkbox"]:checked {{
            background-color: {brand_color} !important;
            border-color: {brand_color} !important;
        }}
        
        [data-baseweb="checkbox"] input[type="checkbox"]:checked + span {{
            background-color: {brand_color} !important;
            border-color: {brand_color} !important;
        }}
        
        [data-baseweb="checkbox"] svg {{
            color: {brand_color} !important;
        }}
        
        /* CRITICAL: Ensure sentiment chips keep their colors - MAXIMUM PRIORITY */
        .sentiment-chip-positive {{
            background-color: #22c55e !important;
            color: white !important;
            border: none !important;
        }}
        .sentiment-chip-negative {{
            background-color: #ef4444 !important;
            color: white !important;
            border: none !important;
        }}
        .sentiment-chip-neutral {{
            background-color: #6b7280 !important;
            color: white !important;
            border: none !important;
        }}
        .sentiment-chip-mixed {{
            background-color: #f59e0b !important;
            color: white !important;
            border: none !important;
        }}
        .sentiment-chip-unknown {{
            background-color: #9ca3af !important;
            color: white !important;
            border: none !important;
        }}
        
        /* Also target by data attribute */
        [data-sentiment="positive"] {{
            background-color: #22c55e !important;
            color: white !important;
        }}
        [data-sentiment="negative"] {{
            background-color: #ef4444 !important;
            color: white !important;
        }}
        [data-sentiment="neutral"] {{
            background-color: #6b7280 !important;
            color: white !important;
        }}
        [data-sentiment="mixed"] {{
            background-color: #f59e0b !important;
            color: white !important;
        }}
        [data-sentiment="unknown"] {{
            background-color: #9ca3af !important;
            color: white !important;
        }}
        
        [data-baseweb="tag"] {{
            background-color: {brand_color} !important;
            color: white !important;
        }}
        
        [data-baseweb="tag"] svg {{
            color: white !important;
        }}
        
        /* Ultra Compact Sidebar */
        .stSidebar {{
            font-size: 0.8rem;
        }}
        
        .stSidebar [data-testid="stHeader"] {{
            font-size: 1rem;
            margin-bottom: 0.25rem;
            padding-bottom: 0.15rem;
            margin-top: 0.25rem;
        }}
        
        .stSidebar [data-testid="stSubheader"] {{
            font-size: 0.9rem;
            margin-top: 0.4rem;
            margin-bottom: 0.3rem;
            padding-bottom: 0.15rem;
        }}
        
        .stSidebar [data-testid="stMarkdownContainer"] {{
            margin-bottom: 0.15rem;
            margin-top: 0.15rem;
        }}
        
        .stSidebar .stDivider {{
            margin: 0.3rem 0;
        }}
        
        /* Reduce spacing in all sidebar elements */
        .stSidebar > div {{
            padding-top: 0.3rem;
            padding-bottom: 0.3rem;
        }}
        
        .stSidebar [data-testid="stVerticalBlock"] {{
            gap: 0.3rem;
        }}
        
        .stSidebar [data-testid="stHorizontalBlock"] {{
            gap: 0.3rem;
        }}
        
        /* Prevent text wrapping in pagination buttons */
        .stButton > button {{
            white-space: nowrap !important;
        }}
        .stButton > button > div {{
            white-space: nowrap !important;
        }}
        
        /* Ultra compact buttons */
        .stSidebar .stButton > button {{
            background-color: {brand_color};
            color: white;
            border: none;
            border-radius: 0.2rem;
            padding: 0.25rem 0.6rem;
            font-size: 0.8rem;
            font-weight: 500;
            transition: all 0.3s;
            height: auto;
            min-height: 1.75rem;
            margin: 0.15rem 0;
        }}
        
        .stSidebar .stButton > button:hover {{
            background-color: {brand_color_hover};
            box-shadow: 0 2px 8px rgba(101, 92, 254, 0.3);
        }}
        
        /* Ultra compact download buttons */
        .stSidebar .stDownloadButton > button {{
            background-color: {brand_color};
            color: white;
            padding: 0.25rem 0.6rem;
            font-size: 0.8rem;
            height: auto;
            min-height: 1.75rem;
            margin: 0.15rem 0;
        }}
        
        .stSidebar .stDownloadButton > button:hover {{
            background-color: {brand_color_hover};
        }}
        
        /* Ultra compact inputs */
        .stSidebar input, .stSidebar select {{
            font-size: 0.8rem;
            padding: 0.25rem 0.4rem;
            margin: 0.15rem 0;
        }}
        
        .stSidebar [data-baseweb="select"] {{
            font-size: 0.8rem;
        }}
        
        /* Ultra compact multiselect */
        .stSidebar [data-baseweb="select"] > div {{
            padding: 0.25rem 0.4rem;
            font-size: 0.8rem;
            margin: 0.15rem 0;
        }}
        
        .stSidebar [data-baseweb="select"] [data-baseweb="input"] {{
            padding: 0.25rem 0.4rem;
            min-height: 1.75rem;
        }}
        
        /* Brand color checkboxes - MAXIMUM OVERRIDE */
        .stSidebar [data-baseweb="checkbox"] input[type="checkbox"]:checked,
        .stSidebar [data-baseweb="checkbox"] input[type="checkbox"][checked] {{
            background-color: {brand_color} !important;
            border-color: {brand_color} !important;
            accent-color: {brand_color} !important;
        }}
        
        .stSidebar [data-baseweb="checkbox"] input[type="checkbox"]:checked + span,
        .stSidebar [data-baseweb="checkbox"] input[type="checkbox"][checked] + span {{
            background-color: {brand_color} !important;
            border-color: {brand_color} !important;
        }}
        
        .stSidebar [data-baseweb="checkbox"] input[type="checkbox"]:focus {{
            box-shadow: 0 0 0 0.2rem rgba(101, 92, 254, 0.25) !important;
        }}
        
        .stSidebar [data-baseweb="checkbox"] svg {{
            color: {brand_color} !important;
            fill: {brand_color} !important;
        }}
        
        /* Override ANY red colors in checkbox - AGGRESSIVE */
        .stSidebar [data-baseweb="checkbox"] *[style*="rgb(255, 75, 75)"],
        .stSidebar [data-baseweb="checkbox"] *[style*="rgb(239, 68, 68)"],
        .stSidebar [data-baseweb="checkbox"] *[style*="rgb(255, 107, 107)"],
        .stSidebar [data-baseweb="checkbox"] *[style*="#ff4b4b"],
        .stSidebar [data-baseweb="checkbox"] *[style*="#ef4444"] {{
            background-color: {brand_color} !important;
            border-color: {brand_color} !important;
            color: {brand_color} !important;
            fill: {brand_color} !important;
        }}
        
        /* Override checkbox container */
        .stSidebar [data-baseweb="checkbox"] {{
            color: {brand_color} !important;
        }}
        
        /* Ultra compact slider with brand color */
        .stSidebar .stSlider {{
            margin: 0.25rem 0;
            padding: 0.15rem 0;
        }}
        
        .stSidebar .stSlider label {{
            font-size: 0.8rem;
            margin-bottom: 0.15rem;
        }}
        
        .stSidebar .stSlider > div > div > div {{
            background-color: {brand_color} !important;
        }}
        
        .stSidebar .stSlider [data-baseweb="slider"] > div > div {{
            background-color: {brand_color} !important;
        }}
        
        .stSidebar .stSlider [data-baseweb="slider"] > div > div > div {{
            background-color: {brand_color} !important;
        }}
        
        .stSidebar .stSlider [data-baseweb="slider-track"] {{
            background-color: {brand_color} !important;
        }}
        
        .stSidebar .stSlider [data-baseweb="slider-handle"] {{
            background-color: {brand_color} !important;
            border-color: {brand_color} !important;
        }}
        
        /* Multiselect tags - brand color */
        .stSidebar [data-baseweb="tag"] {{
            background-color: {brand_color} !important;
            color: white !important;
        }}
        
        .stSidebar [data-baseweb="tag"] svg {{
            color: white !important;
        }}
        
        .stSidebar [data-baseweb="select"] [data-baseweb="tag"] {{
            background-color: {brand_color} !important;
            color: white !important;
        }}
        
        .stSidebar [data-baseweb="select"] [data-baseweb="tag"] svg {{
            color: white !important;
        }}
        
        /* Ultra compact file uploader - fixed spacing */
        .stSidebar .stFileUploader {{
            font-size: 0.75rem;
            margin: 0.1rem 0;
        }}
        
        .stSidebar .stFileUploader > div {{
            min-height: auto !important;
            height: auto !important;
            display: flex !important;
            flex-direction: column !important;
            gap: 0.2rem !important;
        }}
        
        .stSidebar .stFileUploader > div > div {{
            border-color: {brand_color};
            padding: 0.3rem 0.4rem !important;
            min-height: 2.5rem !important;
            height: auto !important;
        }}
        
        .stSidebar .stFileUploader label {{
            font-size: 0.75rem;
            margin: 0.1rem 0 0.2rem 0;
            padding: 0;
        }}
        
        .stSidebar .stFileUploader [data-testid="stFileUploaderDropzone"] {{
            min-height: 2.5rem !important;
            height: auto !important;
            padding: 0.3rem 0.4rem !important;
            margin-bottom: 0.2rem !important;
        }}
        
        .stSidebar .stFileUploader [data-testid="stFileUploaderDropzoneInstructions"] {{
            font-size: 0.7rem;
            margin: 0.1rem 0;
            padding: 0;
        }}
        
        .stSidebar .stFileUploader [data-testid="stFileUploaderDropzoneInstructions"] p {{
            margin: 0.1rem 0;
            font-size: 0.7rem;
            line-height: 1.2;
        }}
        
        .stSidebar .stFileUploader button {{
            padding: 0.2rem 0.5rem !important;
            font-size: 0.7rem !important;
            min-height: 1.6rem !important;
            height: 1.6rem !important;
            margin: 0.1rem 0 !important;
        }}
        
        .stSidebar .stFileUploader [data-testid="stFileUploaderFileName"] {{
            font-size: 0.7rem;
            margin: 0.15rem 0;
            padding: 0.1rem 0;
            display: block;
        }}
        
        .stSidebar .stFileUploader [data-testid="stFileUploaderFileSize"] {{
            font-size: 0.65rem;
            margin: 0.05rem 0;
            padding: 0;
            display: block;
        }}
        
        /* Fix file uploader file info spacing */
        .stSidebar .stFileUploader [data-testid="stFileUploaderFileStatus"] {{
            margin-top: 0.2rem !important;
            margin-bottom: 0.1rem !important;
        }}
        
        /* Ultra compact captions */
        .stSidebar [data-testid="stCaption"] {{
            font-size: 0.7rem;
            margin-top: 0.15rem;
            margin-bottom: 0.15rem;
        }}
        
        /* Compact checkbox labels */
        .stSidebar [data-baseweb="checkbox"] label {{
            font-size: 0.8rem;
            margin: 0.15rem 0;
            padding: 0.15rem 0;
        }}
        
        /* Reduce spacing in columns */
        .stSidebar [data-testid="column"] {{
            padding: 0.15rem;
        }}
        
        /* Reduce info box spacing */
        .stSidebar .stAlert {{
            padding: 0.4rem 0.6rem;
            margin: 0.25rem 0;
            font-size: 0.8rem;
        }}
        
        /* Additional compact spacing */
        .stSidebar [data-baseweb="base-input"] {{
            padding: 0.25rem 0.4rem;
            min-height: 1.75rem;
        }}
        
        .stSidebar [data-baseweb="select"] [data-baseweb="popover"] {{
            max-height: 200px;
        }}
        
        /* Reduce line height */
        .stSidebar * {{
            line-height: 1.3;
        }}
        
        /* Compact multiselect container */
        .stSidebar [data-baseweb="select"] {{
            margin: 0.15rem 0;
        }}
        
        /* Remove extra padding from text inputs */
        .stSidebar input[type="text"] {{
            padding: 0.25rem 0.4rem !important;
            min-height: 1.75rem !important;
        }}
        
        /* Remove red error/warning colors */
        .stSidebar .stAlert {{
            border-left-color: {brand_color} !important;
        }}
        
        .stSidebar [data-baseweb="notification"] {{
            border-left-color: {brand_color} !important;
        }}
        
        /* Override Streamlit default red colors */
        .stSidebar [style*="rgb(255, 75, 75)"],
        .stSidebar [style*="rgb(239, 68, 68)"],
        .stSidebar [style*="#ef4444"],
        .stSidebar [style*="#ff4b4b"] {{
            background-color: {brand_color} !important;
            border-color: {brand_color} !important;
            color: white !important;
        }}
        
        /* Slider track and handle - override red */
        .stSidebar [data-baseweb="slider"] [style*="rgb(255, 75, 75)"],
        .stSidebar [data-baseweb="slider"] [style*="#ff4b4b"] {{
            background-color: {brand_color} !important;
        }}
        
        /* Checkbox - override red */
        .stSidebar [data-baseweb="checkbox"] [style*="rgb(255, 75, 75)"],
        .stSidebar [data-baseweb="checkbox"] [style*="#ff4b4b"] {{
            background-color: {brand_color} !important;
            border-color: {brand_color} !important;
        }}
        
        /* Tag - override red */
        .stSidebar [data-baseweb="tag"] [style*="rgb(255, 75, 75)"],
        .stSidebar [data-baseweb="tag"] [style*="#ff4b4b"] {{
            background-color: {brand_color} !important;
        }}
        
        /* Primary buttons (main area) */
        .stButton > button {{
            background-color: {brand_color};
            color: white;
            border: none;
            border-radius: 0.25rem;
            padding: 0.5rem 1rem;
            font-weight: 500;
            transition: all 0.3s;
        }}
        
        .stButton > button:hover {{
            background-color: {brand_color_hover};
            box-shadow: 0 2px 8px rgba(101, 92, 254, 0.3);
        }}
        
        /* Download buttons (main area) */
        .stDownloadButton > button {{
            background-color: {brand_color};
            color: white;
        }}
        
        .stDownloadButton > button:hover {{
            background-color: {brand_color_hover};
        }}
        
        /* Progress bar */
        .stProgress > div > div > div {{
            background-color: {brand_color};
        }}
        
        /* Tabs - ONLY ONE brand color underline, NO red/pink */
        .stTabs [data-baseweb="tab-list"] {{
            gap: 8px;
            border-bottom: 1px solid #e5e7eb;
        }}
        
        /* Remove ALL borders from all tabs */
        .stTabs [data-baseweb="tab"] {{
            color: #666;
            border: none !important;
            border-top: none !important;
            border-left: none !important;
            border-right: none !important;
            border-bottom: none !important;
            background: none !important;
        }}
        
        /* Remove ALL pseudo-elements */
        .stTabs [data-baseweb="tab"]::after,
        .stTabs [data-baseweb="tab"]::before,
        .stTabs [data-baseweb="tab"] *::after,
        .stTabs [data-baseweb="tab"] *::before {{
            display: none !important;
            content: none !important;
        }}
        
        /* Remove borders from inner divs */
        .stTabs [data-baseweb="tab"] > div,
        .stTabs [data-baseweb="tab"] > div > div {{
            border: none !important;
            border-top: none !important;
            border-left: none !important;
            border-right: none !important;
            border-bottom: none !important;
        }}
        
        /* Remove borders from buttons */
        .stTabs [data-baseweb="tab"] button {{
            border: none !important;
            border-top: none !important;
            border-left: none !important;
            border-right: none !important;
            border-bottom: none !important;
        }}
        
        /* Active tab - use Streamlit's default red color */
        .stTabs [aria-selected="true"] {{
            color: #ff4b4b !important;
        }}
        
        /* Remove ONLY brand color borders from tabs - keep Streamlit's default red */
        .stTabs [data-baseweb="tab"] [style*="{brand_color}"],
        .stTabs [data-baseweb="tab"] [style*="#655CFE"] {{
            border-bottom: none !important;
            background-color: transparent !important;
        }}
        
        /* Ensure only ONE underline - remove duplicate borders from children */
        .stTabs [aria-selected="true"] > * {{
            border-bottom: none !important;
        }}
        
        .stTabs [aria-selected="true"] > * > * {{
            border-bottom: none !important;
        }}
        
        /* Sidebar headers */
        .stSidebar [data-testid="stHeader"] {{
            color: {brand_color};
        }}
        
        /* Links */
        a {{
            color: {brand_color};
        }}
        
        a:hover {{
            color: {brand_color_hover};
        }}
        
        /* Selectbox/Multiselect focus */
        .stSelectbox > div > div {{
            border-color: {brand_color};
        }}
        
        /* Slider */
        .stSlider > div > div > div {{
            background-color: {brand_color};
        }}
        
        /* File uploader */
        .stFileUploader > div > div {{
            border-color: {brand_color};
        }}
        
        /* Metrics */
        [data-testid="stMetricValue"] {{
            color: {brand_color};
        }}
        
        /* Expander */
        .streamlit-expanderHeader {{
            color: {brand_color};
        }}
        
        /* Info boxes */
        .stInfo {{
            border-left: 4px solid {brand_color};
        }}
        
        /* Success messages */
        .stSuccess {{
            border-left: 4px solid {brand_color};
        }}
        
        /* Remove red from warnings/errors */
        .stWarning {{
            border-left-color: #f59e0b !important;
            background-color: rgba(245, 158, 11, 0.1) !important;
        }}
        
        .stError {{
            border-left-color: {brand_color} !important;
            background-color: rgba(101, 92, 254, 0.1) !important;
        }}
    </style>
    <script>
        // Force brand color on slider and checkbox - run after page load
        function applyBrandColors() {{
            const brandColor = '{brand_color}';
            
            // Override slider colors
            document.querySelectorAll('[data-baseweb="slider-handle"]').forEach(el => {{
                el.style.backgroundColor = brandColor;
                el.style.borderColor = brandColor;
            }});
            
            document.querySelectorAll('[data-baseweb="slider-track"]').forEach(el => {{
                el.style.backgroundColor = brandColor;
            }});
            
            // Override checkbox colors - AGGRESSIVE
            document.querySelectorAll('[data-baseweb="checkbox"]').forEach(container => {{
                const checkbox = container.querySelector('input[type="checkbox"]');
                if (checkbox && checkbox.checked) {{
                    checkbox.style.backgroundColor = brandColor;
                    checkbox.style.borderColor = brandColor;
                    checkbox.style.accentColor = brandColor;
                    
                    // Fix span wrapper
                    const span = checkbox.nextElementSibling;
                    if (span && span.tagName === 'SPAN') {{
                        span.style.backgroundColor = brandColor;
                        span.style.borderColor = brandColor;
                    }}
                }}
                
                // Fix SVG
                const svg = container.querySelector('svg');
                if (svg) {{
                    svg.style.color = brandColor;
                    svg.style.fill = brandColor;
                    svg.querySelectorAll('path, circle, rect').forEach(path => {{
                        path.style.fill = brandColor;
                        path.style.stroke = brandColor;
                    }});
                }}
            }});
            
            // Override slider value number color
            document.querySelectorAll('.stSlider [data-testid="stMarkdownContainer"] p').forEach(el => {{
                el.style.color = brandColor;
            }});
        }}
        
        // Run on load and after Streamlit reruns
        if (document.readyState === 'loading') {{
            document.addEventListener('DOMContentLoaded', applyBrandColors);
        }} else {{
            applyBrandColors();
        }}
        
        // Reapply after Streamlit updates
        const observer = new MutationObserver(applyBrandColors);
        observer.observe(document.body, {{ childList: true, subtree: true }});
        
        // Fix tabs - remove brand color, keep Streamlit's default red underline (only ONE)
        function fixTabs() {{
            const brandColor = '{brand_color}';
            const redColor = '#ff4b4b'; // Streamlit default red
            
            // Only remove brand color, don't touch Streamlit's default red borders
            document.querySelectorAll('[data-baseweb="tab"]').forEach(tab => {{
                // Set color for active tab
                if (tab.getAttribute('aria-selected') === 'true') {{
                    tab.style.color = redColor;
                }}
                
                // Remove ONLY brand color borders from tab and all children
                const removeBrandBorders = (element) => {{
                    const computed = window.getComputedStyle(element);
                    // Remove brand color borders only (keep red)
                    if (computed.borderBottom && (
                        computed.borderBottom.includes(brandColor) ||
                        computed.borderBottom.includes('#655CFE')
                    )) {{
                        element.style.borderBottom = 'none';
                        element.style.setProperty('border-bottom', 'none', 'important');
                    }}
                    
                    // Remove inline brand color
                    if (element.style.borderBottom && (
                        element.style.borderBottom.includes(brandColor) ||
                        element.style.borderBottom.includes('#655CFE')
                    )) {{
                        element.style.borderBottom = 'none';
                        element.style.setProperty('border-bottom', 'none', 'important');
                    }}
                    
                    Array.from(element.children).forEach(child => removeBrandBorders(child));
                }};
                
                removeBrandBorders(tab);
            }});
        }}
        
        // Fix sentiment chips - FORCE correct colors using classes and data attributes
        function fixSentimentChips() {{
            const colorMap = {{
                'positive': '#22c55e',
                'negative': '#ef4444',
                'neutral': '#6b7280',
                'mixed': '#f59e0b',
                'unknown': '#9ca3af'
            }};
            
            // Fix by data attribute
            document.querySelectorAll('[data-sentiment]').forEach(chip => {{
                const sentiment = chip.getAttribute('data-sentiment');
                if (sentiment && colorMap[sentiment]) {{
                    chip.style.setProperty('background-color', colorMap[sentiment], 'important');
                    chip.style.setProperty('color', 'white', 'important');
                    chip.style.setProperty('border', 'none', 'important');
                }}
            }});
            
            // Fix by class
            document.querySelectorAll('.sentiment-chip-positive').forEach(chip => {{
                chip.style.setProperty('background-color', '#22c55e', 'important');
                chip.style.setProperty('color', 'white', 'important');
            }});
            document.querySelectorAll('.sentiment-chip-negative').forEach(chip => {{
                chip.style.setProperty('background-color', '#ef4444', 'important');
                chip.style.setProperty('color', 'white', 'important');
            }});
            document.querySelectorAll('.sentiment-chip-neutral').forEach(chip => {{
                chip.style.setProperty('background-color', '#6b7280', 'important');
                chip.style.setProperty('color', 'white', 'important');
            }});
            document.querySelectorAll('.sentiment-chip-mixed').forEach(chip => {{
                chip.style.setProperty('background-color', '#f59e0b', 'important');
                chip.style.setProperty('color', 'white', 'important');
            }});
            document.querySelectorAll('.sentiment-chip-unknown').forEach(chip => {{
                chip.style.setProperty('background-color', '#9ca3af', 'important');
                chip.style.setProperty('color', 'white', 'important');
            }});
        }}
        
        // Run immediately and watch for changes
        if (document.readyState === 'loading') {{
            document.addEventListener('DOMContentLoaded', fixSentimentChips);
        }} else {{
            fixSentimentChips();
        }}
        
        const sentimentObserver = new MutationObserver(() => {{
            fixSentimentChips();
        }});
        sentimentObserver.observe(document.body, {{ childList: true, subtree: true }});
        
        // Fix checkboxes - force brand color
        function fixCheckboxes() {{
            const brandColor = '{brand_color}';
            
            document.querySelectorAll('[data-baseweb="checkbox"]').forEach(checkboxContainer => {{
                const checkbox = checkboxContainer.querySelector('input[type="checkbox"]');
                if (checkbox && checkbox.checked) {{
                    checkbox.style.backgroundColor = brandColor;
                    checkbox.style.borderColor = brandColor;
                    checkbox.style.accentColor = brandColor;
                    
                    // Fix the span wrapper
                    const span = checkbox.nextElementSibling;
                    if (span && span.tagName === 'SPAN') {{
                        span.style.backgroundColor = brandColor;
                        span.style.borderColor = brandColor;
                    }}
                    
                    // Fix SVG icon
                    const svg = checkboxContainer.querySelector('svg');
                    if (svg) {{
                        svg.style.color = brandColor;
                        svg.style.fill = brandColor;
                        // Also set stroke if exists
                        svg.querySelectorAll('path, circle, rect').forEach(path => {{
                            path.style.fill = brandColor;
                            path.style.stroke = brandColor;
                        }});
                    }}
                }}
            }});
        }}
        
        // Run tab fix
        if (document.readyState === 'loading') {{
            document.addEventListener('DOMContentLoaded', fixTabs);
        }} else {{
            fixTabs();
        }}
        
        // Reapply tab fix after Streamlit updates
        const tabObserver = new MutationObserver(() => {{
            fixTabs();
            fixCheckboxes();
            fixSentimentChips();
        }});
        tabObserver.observe(document.body, {{ childList: true, subtree: true }});
        
        // Also run fixes on load
        if (document.readyState === 'loading') {{
            document.addEventListener('DOMContentLoaded', () => {{
                fixCheckboxes();
                fixSentimentChips();
            }});
        }} else {{
            fixCheckboxes();
            fixSentimentChips();
        }}
    </script>
    """, unsafe_allow_html=True)


def main():
    """Main Streamlit application."""
    st.set_page_config(
        page_title="5-Minute Digest",
        page_icon="📊",
        layout="wide"
    )
    
    # Apply brand styles
    apply_brand_styles()
    
    st.title("📊 5-Minute Digest")
    
    # Initialize session state
    initialize_session_state()
    
    # Check if file processing just completed and trigger refresh
    if st.session_state.get('_processing_complete', False):
        st.session_state['_processing_complete'] = False
        # Use a small delay to ensure state is saved, then refresh
        st.rerun()
    
    # Get canonical model and topic aggregates from session state or process file
    canonical_model = st.session_state.get(SESSION_KEYS['canonical_model'])
    topic_aggregates = st.session_state.get(SESSION_KEYS['topic_aggregates'], [])
    validation_report = {}
    
    # Render sidebar
    canonical_model, topic_aggregates, validation_report = render_sidebar(canonical_model, topic_aggregates)
    
    if canonical_model is None:
        st.info("👈 Please upload an Excel file to begin")
        return
    
    # Apply filters
    filtered_aggregates = filter_topics(
        topic_aggregates,
        st.session_state[SESSION_KEYS['filters']],
        st.session_state[SESSION_KEYS['search_query']],
        canonical_model
    )
    
    # Store filtered aggregates in session state for export
    st.session_state['filtered_aggregates'] = filtered_aggregates
    
    # Get selected topic aggregates
    selected_topic_ids = st.session_state[SESSION_KEYS['selected_topics']]
    selected_aggregates = [t for t in filtered_aggregates if t['topic_id'] in selected_topic_ids]
    
    # Build digest
    digest_artifact = digest.build_digest(canonical_model, selected_aggregates, n_takeaways=5)
    
    # Main tabs
    tab1, tab2 = st.tabs(["Digest", "Explore"])
    
    with tab1:
        # Row 1: Validation + Stats
        if hasattr(canonical_model, 'participants'):
            render_validation_stats(validation_report, canonical_model)
        
        st.divider()
        
        # Row 2: Takeaways
        render_takeaways(digest_artifact, canonical_model)
        
        st.divider()
        
        # Row 3: Topic Cards
        render_topic_cards(digest_artifact, canonical_model)
    
    with tab2:
        render_explore_tab(topic_aggregates, canonical_model)


# Unit tests for topic selection behavior
if __name__ == "__main__" and False:  # Set to True to run tests
    import unittest
    
    class TestTopicSelection(unittest.TestCase):
        
        def test_compute_top_topics(self):
            """Test computing top topics from aggregates."""
            aggregates = [
                {'topic_id': 'topic1', 'topic_score': 0.9},
                {'topic_id': 'topic2', 'topic_score': 0.8},
                {'topic_id': 'topic3', 'topic_score': 0.7},
            ]
            top_topics = compute_top_topics(aggregates)
            self.assertEqual(top_topics, ['topic1', 'topic2', 'topic3'])
        
        def test_compute_selected_topics_initial(self):
            """Test initial selection when auto_select is True."""
            top_topics = ['t1', 't2', 't3', 't4', 't5']
            selected = compute_selected_topics(top_topics, [], 3, True, False, None)
            self.assertEqual(selected, ['t1', 't2', 't3'])
        
        def test_compute_selected_topics_n_increase_with_auto_add(self):
            """Test N increase with auto_add_on_change enabled."""
            top_topics = ['t1', 't2', 't3', 't4', 't5']
            current = ['t1', 't2', 't3']  # Top 3
            # Increase to 5 with auto_add enabled
            selected = compute_selected_topics(top_topics, current, 5, True, True, 3)
            # Should include t4 and t5
            self.assertIn('t4', selected)
            self.assertIn('t5', selected)
            self.assertEqual(len(selected), 5)
        
        def test_compute_selected_topics_n_increase_without_auto_add(self):
            """Test N increase without auto_add_on_change."""
            top_topics = ['t1', 't2', 't3', 't4', 't5']
            current = ['t1', 't2', 't3']  # Top 3
            # Increase to 5 without auto_add
            selected = compute_selected_topics(top_topics, current, 5, True, False, 3)
            # Should still only have top 3 (no auto-add)
            self.assertEqual(set(selected), {'t1', 't2', 't3'})
        
        def test_compute_selected_topics_n_decrease_preserves_manual(self):
            """Test N decrease preserves manually added topics."""
            top_topics = ['t1', 't2', 't3', 't4', 't5']
            # User has top 5 selected, plus manually added t10
            current = ['t1', 't2', 't3', 't4', 't5', 't10']
            # Decrease to 3
            selected = compute_selected_topics(top_topics, current, 3, True, False, 5)
            # Should preserve t10 (manually added)
            self.assertIn('t10', selected)
            # Should have top 3
            self.assertIn('t1', selected)
            self.assertIn('t2', selected)
            self.assertIn('t3', selected)
        
        def test_compute_selected_topics_manual_add_preserved(self):
            """Test that manually added topics are preserved."""
            top_topics = ['t1', 't2', 't3', 't4', 't5']
            # User manually added t10 (outside top N=3)
            current = ['t1', 't2', 't3', 't10']
            selected = compute_selected_topics(top_topics, current, 3, True, False, 3)
            # Should preserve t10
            self.assertIn('t10', selected)
            # Should have top 3
            self.assertEqual(set(selected[:3]), {'t1', 't2', 't3'})
        """Main Streamlit application entry point."""

import streamlit as st
import pandas as pd
from typing import Dict, List, Optional, Any, Tuple
import hashlib
import ingest
import normalize
import parse_quotes
import parse_sentiment
import score
import digest
import re
import render
import export
import edge_cases
import explore_model
import recipient


# Pipeline version for cache invalidation
PIPELINE_VERSION = "1.0.0"

# Session state keys
SESSION_KEYS = {
    'selected_topics': 'selected_topics',
    'top_n': 'top_n',
    'auto_select_top_n': 'auto_select_top_n',
    'filters': 'filters',
    'search_query': 'search_query',
    'uploaded_file': 'uploaded_file',
    'file_hash': 'file_hash',
    'file_bytes': 'file_bytes',
    'canonical_model': 'canonical_model',
    'topic_aggregates': 'topic_aggregates',
    'validation_report': 'validation_report'
}


def compute_file_hash(file_bytes: bytes) -> str:
    """
    Compute MD5 hash of file bytes for cache key.
    
    Args:
        file_bytes: File bytes
    
    Returns:
        MD5 hash as hex string
    """
    return hashlib.md5(file_bytes).hexdigest()


def get_config_knobs_hash(config: Dict[str, Any]) -> str:
    """
    Get hash of configuration knobs for cache key.
    
    Args:
        config: Configuration dictionary
    
    Returns:
        Hash string of sorted config items
    """
    # Sort items for deterministic hash
    sorted_items = sorted(config.items())
    config_str = str(sorted_items)
    return hashlib.md5(config_str.encode()).hexdigest()


def initialize_session_state():
    """Initialize session state with default values."""
    if SESSION_KEYS['top_n'] not in st.session_state:
        st.session_state[SESSION_KEYS['top_n']] = 4
    if SESSION_KEYS['auto_select_top_n'] not in st.session_state:
        st.session_state[SESSION_KEYS['auto_select_top_n']] = True
    if SESSION_KEYS['selected_topics'] not in st.session_state:
        st.session_state[SESSION_KEYS['selected_topics']] = []
    if SESSION_KEYS['filters'] not in st.session_state:
        st.session_state[SESSION_KEYS['filters']] = {
            'coverage_tier': None,
            'tone_rollup': None,
            'high_emotion': False,
        }
    if SESSION_KEYS['search_query'] not in st.session_state:
        st.session_state[SESSION_KEYS['search_query']] = ''
    if SESSION_KEYS['file_hash'] not in st.session_state:
        st.session_state[SESSION_KEYS['file_hash']] = None
    if 'auto_add_on_change' not in st.session_state:
        st.session_state['auto_add_on_change'] = False
    if 'previous_top_n' not in st.session_state:
        st.session_state['previous_top_n'] = None
    if 'participant_filter_patterns' not in st.session_state:
        st.session_state['participant_filter_patterns'] = []
    if 'single_sheet_topics' not in st.session_state:
        st.session_state['single_sheet_topics'] = set()
    if 'sparse_topics' not in st.session_state:
        st.session_state['sparse_topics'] = []


@st.cache_data
def cached_read_workbook(
    file_hash: str,
    pipeline_version: str,
    config_hash: str,
    file_bytes: bytes
) -> Tuple[Dict[str, Optional[pd.DataFrame]], Dict[str, Any]]:
    """
    Cached wrapper for ingest.read_workbook.
    
    Cache key includes file_hash, pipeline_version, and config_hash.
    This ensures cache invalidation on file change or pipeline update.
    
    Args:
        file_hash: MD5 hash of file bytes (for cache key)
        pipeline_version: Pipeline version string (for cache key)
        config_hash: Hash of configuration knobs (for cache key)
        file_bytes: Actual file bytes (used for processing)
    
    Returns:
        Tuple of (dict_of_dfs, validation_report)
    """
    return ingest.read_workbook(file_bytes)


@st.cache_data
def cached_normalize(
    file_hash: str,
    pipeline_version: str,
    config_hash: str,
    topic_columns: Tuple[str, ...],
    file_bytes: bytes
) -> normalize.CanonicalModel:
    """
    Cached wrapper for normalize.wide_to_canonical.
    
    Cache key includes file_hash, pipeline_version, config_hash, and topic_columns.
    
    Args:
        file_hash: MD5 hash of file bytes (for cache key)
        pipeline_version: Pipeline version string (for cache key)
        config_hash: Hash of configuration knobs (for cache key)
        topic_columns: Tuple of topic column names (must be hashable)
        file_bytes: File bytes (passed through for read_workbook)
    
    Returns:
        CanonicalModel object
    """
    # Get dict_of_dfs from cached read_workbook
    dict_of_dfs, _ = cached_read_workbook(file_hash, pipeline_version, config_hash, file_bytes)
    
    return normalize.wide_to_canonical(dict_of_dfs, list(topic_columns))


@st.cache_data
def cached_compute_scoring(
    file_hash: str,
    pipeline_version: str,
    config_hash: str,
    topic_columns: Tuple[str, ...],
    file_bytes: bytes
) -> List[Dict[str, Any]]:
    """
    Cached wrapper for score.compute_topic_aggregates.
    
    Cache key includes file_hash, pipeline_version, config_hash, and topic_columns.
    This ensures scoring is only recomputed when input data or config changes.
    
    Args:
        file_hash: MD5 hash of file bytes (for cache key)
        pipeline_version: Pipeline version string (for cache key)
        config_hash: Hash of configuration knobs (for cache key)
        topic_columns: Tuple of topic column names
        file_bytes: File bytes (passed through for pipeline)
    
    Returns:
        List of topic aggregates
    """
    canonical_model = cached_normalize(file_hash, pipeline_version, config_hash, topic_columns, file_bytes)
    return score.compute_topic_aggregates(canonical_model)


def process_uploaded_file(uploaded_file_bytes: bytes, config: Optional[Dict[str, Any]] = None) -> Tuple[Dict[str, Optional[pd.DataFrame]], Dict[str, Any], normalize.CanonicalModel, List[str]]:
    """
    Process uploaded file: ingest and normalize with proper caching.
    
    This function orchestrates the cached pipeline steps.
    
    Args:
        uploaded_file_bytes: Bytes of uploaded Excel file
        config: Optional configuration dictionary for cache key
    
    Returns:
        Tuple of (dict_of_dfs, validation_report, canonical_model, topic_columns)
    """
    # Compute hashes for cache keys
    file_hash = compute_file_hash(uploaded_file_bytes)
    config = config or {}
    config_hash = get_config_knobs_hash(config)
    
    # Ingest (cached)
    dict_of_dfs, validation_report = cached_read_workbook(file_hash, PIPELINE_VERSION, config_hash, uploaded_file_bytes)
    
    # Get topic columns from validation report
    topic_columns = list(validation_report.get('topic_columns', []))
    topic_columns_tuple = tuple(topic_columns)  # Make hashable for cache
    
    # Normalize (cached)
    canonical_model = cached_normalize(file_hash, PIPELINE_VERSION, config_hash, topic_columns_tuple, uploaded_file_bytes)
    
    return dict_of_dfs, validation_report, canonical_model, topic_columns


def compute_scoring_with_cache(
    file_hash: str,
    file_bytes: bytes,
    config: Optional[Dict[str, Any]] = None
) -> List[Dict[str, Any]]:
    """
    Compute topic aggregates and scores with proper caching.
    
    Args:
        file_hash: MD5 hash of file bytes
        file_bytes: File bytes (needed for cache chain)
        config: Optional configuration dictionary for cache key
    
    Returns:
        List of topic aggregates
    """
    config = config or {}
    config_hash = get_config_knobs_hash(config)
    
    # Get topic columns from session state
    topic_columns = st.session_state.get('topic_columns', [])
    topic_columns_tuple = tuple(topic_columns)
    
    return cached_compute_scoring(file_hash, PIPELINE_VERSION, config_hash, topic_columns_tuple, file_bytes)


def filter_topics(
    topic_aggregates: List[Dict[str, Any]],
    filters: Dict[str, Any],
    search_query: str,
    canonical_model
) -> List[Dict[str, Any]]:
    """
    Filter topics based on filters and search query.
    
    Args:
        topic_aggregates: List of topic aggregates
        filters: Filter dictionary
        search_query: Search query string
        canonical_model: CanonicalModel for search
    
    Returns:
        Filtered list of topic aggregates
    """
    filtered = topic_aggregates.copy()
    
    # Coverage tier filter
    coverage_tier = filters.get('coverage_tier')
    if coverage_tier:
        if coverage_tier == 'High':
            filtered = [t for t in filtered if t.get('coverage_rate', 0) >= 0.7]
        elif coverage_tier == 'Medium':
            filtered = [t for t in filtered if 0.4 <= t.get('coverage_rate', 0) < 0.7]
        elif coverage_tier == 'Low':
            filtered = [t for t in filtered if t.get('coverage_rate', 0) < 0.4]
    
    # Tone rollup filter
    tone_rollup = filters.get('tone_rollup')
    if tone_rollup:
        # Need to compute sentiment mix for each topic
        # Note: tone_rollup is already lowercase from render_sidebar, but we normalize it here for safety
        tone_rollup_normalized = tone_rollup.lower() if isinstance(tone_rollup, str) else tone_rollup
        filtered_by_tone = []
        for topic_agg in filtered:
            topic_id = topic_agg['topic_id']
            sentiment_mix = digest._compute_sentiment_mix(canonical_model.evidence_cells, topic_id)
            total = sum(sentiment_mix.values())
            if total > 0:
                dominant = max(sentiment_mix.items(), key=lambda x: x[1])[0]
                if dominant == tone_rollup_normalized:
                    filtered_by_tone.append(topic_agg)
        filtered = filtered_by_tone
    
    # High-emotion filter
    if filters.get('high_emotion', False):
        filtered_by_emotion = []
        for topic_agg in filtered:
            if topic_agg.get('intensity_rate', 0) > 0.5:  # More than 50% non-neutral
                filtered_by_emotion.append(topic_agg)
        filtered = filtered_by_emotion
    
    # Search query filter
    if search_query:
        query_lower = search_query.lower()
        filtered_by_search = []
        for topic_agg in filtered:
            topic_id = topic_agg['topic_id']
            # Search in topic_id, one-liner, and evidence cells
            matches = False
            
            if query_lower in topic_id.lower():
                matches = True
            
            one_liner = topic_agg.get('topic_one_liner', '')
            if one_liner and query_lower in one_liner.lower():
                matches = True
            
            # Search in evidence cells
            if not matches:
                for evidence_cell in canonical_model.evidence_cells:
                    if evidence_cell.topic_id == topic_id:
                        if (evidence_cell.summary_text and query_lower in evidence_cell.summary_text.lower()) or \
                           (evidence_cell.quotes_raw and query_lower in evidence_cell.quotes_raw.lower()):
                            matches = True
                            break
            
            if matches:
                filtered_by_search.append(topic_agg)
        filtered = filtered_by_search
    
    return filtered


def compute_top_topics(topic_aggregates: List[Dict[str, Any]], exclude_single_sheet: bool = True) -> List[str]:
    """
    Compute top topics as a ranked list of topic IDs.
    
    Topics are already sorted by score (descending) with alphabetical tiebreak
    from score.compute_topic_aggregates.
    
    Excludes single-sheet topics from Top N by default.
    
    Args:
        topic_aggregates: List of topic aggregates (already sorted by score)
        exclude_single_sheet: If True, exclude single-sheet topics from Top N
    
    Returns:
        List of topic IDs in ranked order (excluding single-sheet if requested)
    """
    single_sheet_topics = st.session_state.get('single_sheet_topics', set())
    
    if exclude_single_sheet and single_sheet_topics:
        # Filter out single-sheet topics
        filtered = [t['topic_id'] for t in topic_aggregates if t['topic_id'] not in single_sheet_topics]
        return filtered
    
    return [t['topic_id'] for t in topic_aggregates]


def compute_selected_topics(
    top_topics: List[str],
    current_selected: List[str],
    top_n: int,
    auto_select: bool,
    auto_add_on_change: bool,
    previous_top_n: Optional[int] = None
) -> List[str]:
    """
    Compute selected topics based on selection behavior rules.
    
    Rules:
    - If auto_select is True and no current selection, default to top_topics[:N]
    - When N changes:
      - If N increases and auto_add_on_change is True, append newly included topics
      - If N decreases, do not auto-remove manually added topics
    - Maintains stability: preserves user's manual additions
    
    Args:
        top_topics: Ranked list of topic IDs
        current_selected: Currently selected topic IDs
        top_n: Current N value
        auto_select: Whether auto-select is enabled
        auto_add_on_change: Whether to auto-add when N increases
        previous_top_n: Previous N value (for detecting changes)
    
    Returns:
        Updated list of selected topic IDs
    """
    if not top_topics:
        return []
    
    # If auto_select is False, return current selection as-is
    if not auto_select:
        return current_selected.copy()
    
    # Determine top N topics
    top_n_topics = top_topics[:top_n]
    
    # If no current selection, initialize with top N
    if not current_selected:
        return top_n_topics.copy()
    
    # Check if N changed
    n_increased = previous_top_n is not None and top_n > previous_top_n
    n_decreased = previous_top_n is not None and top_n < previous_top_n
    
    # Build new selection
    new_selected = []
    
    # Always include topics that are in top N
    for topic_id in top_n_topics:
        if topic_id not in new_selected:
            new_selected.append(topic_id)
    
    # If N increased and auto_add_on_change, add newly included topics
    if n_increased and auto_add_on_change:
        # Find topics that are now in top N but weren't before
        previous_top_n_topics = top_topics[:previous_top_n] if previous_top_n else []
        newly_included = [t for t in top_n_topics if t not in previous_top_n_topics]
        for topic_id in newly_included:
            if topic_id not in new_selected:
                new_selected.append(topic_id)
    
    # Preserve manually added topics (those not in top N)
    # These are topics the user added that are outside the current top N
    manually_added = [t for t in current_selected if t not in top_n_topics]
    
    # If N decreased, preserve all manually added topics
    if n_decreased:
        for topic_id in manually_added:
            if topic_id not in new_selected:
                new_selected.append(topic_id)
    else:
        # If N didn't decrease, still preserve manually added topics
        # (user intent: they explicitly added these)
        for topic_id in manually_added:
            if topic_id not in new_selected:
                new_selected.append(topic_id)
    
    # Maintain order: top N first, then manually added
    ordered_selected = []
    # Add top N topics in order
    for topic_id in top_topics:
        if topic_id in new_selected:
            ordered_selected.append(topic_id)
    # Add manually added topics at the end
    for topic_id in manually_added:
        if topic_id in new_selected and topic_id not in ordered_selected:
            ordered_selected.append(topic_id)
    
    return ordered_selected


def reset_to_top_n(top_topics: List[str], top_n: int) -> List[str]:
    """
    Reset selection to top N topics.
    
    Args:
        top_topics: Ranked list of topic IDs
        top_n: Number of topics to select
    
    Returns:
        List of top N topic IDs
    """
    return top_topics[:top_n].copy()


def clear_selection() -> List[str]:
    """
    Clear all selected topics.
    
    Returns:
        Empty list
    """
    return []


def render_sidebar(canonical_model, topic_aggregates: List[Dict[str, Any]]):
    """Render sidebar with all controls."""
    st.sidebar.markdown("### Controls")
    
    # File uploader
    uploaded_file = st.sidebar.file_uploader(
        "Upload Excel File",
        type=['xlsx', 'xls'],
        key='file_uploader'
    )
    
    validation_report = {}
    
    if uploaded_file is not None:
        bytes_data = uploaded_file.read()
        file_hash = compute_file_hash(bytes_data)
        current_file_hash = st.session_state.get(SESSION_KEYS['file_hash'])
        
        # Check if file changed (by hash)
        if current_file_hash != file_hash:
            # New file uploaded - update hash and process
            st.session_state[SESSION_KEYS['file_hash']] = file_hash
            st.session_state[SESSION_KEYS['file_bytes']] = bytes_data
            st.session_state[SESSION_KEYS['uploaded_file']] = uploaded_file
            
            # Show loading indicator with progress
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            try:
                status_text.text("📊 Reading file...")
                progress_bar.progress(10)
                
                # Process file (includes validation)
                config = {}
                dict_of_dfs, validation_report, canonical_model, topic_columns = process_uploaded_file(bytes_data, config)
                
                # Validate file is readable
                is_valid, error_msg = edge_cases.validate_file_readable(validation_report)
                
                if not is_valid:
                    st.session_state['validation_error'] = error_msg
                    st.session_state[SESSION_KEYS['canonical_model']] = None
                    st.session_state[SESSION_KEYS['topic_aggregates']] = []
                    progress_bar.empty()
                    status_text.empty()
                    return None, [], validation_report
                
                status_text.text("🔄 Normalizing data...")
                progress_bar.progress(40)
                
                # Apply participant filtering if patterns exist
                filter_patterns = st.session_state.get('participant_filter_patterns', [])
                if filter_patterns:
                    canonical_model, filtered_ids = edge_cases.filter_participants_by_regex(
                        canonical_model, filter_patterns
                    )
                    if filtered_ids:
                        st.session_state['filtered_participants'] = list(filtered_ids)
                
                status_text.text("📈 Computing scores...")
                progress_bar.progress(70)
                
                # Identify single-sheet and sparse topics
                single_sheet_topics = edge_cases.identify_single_sheet_topics(
                    canonical_model, set(topic_columns)
                )
                st.session_state['single_sheet_topics'] = single_sheet_topics
                
                # Store in session state
                st.session_state[SESSION_KEYS['canonical_model']] = canonical_model
                st.session_state[SESSION_KEYS['validation_report']] = validation_report
                st.session_state['topic_columns'] = topic_columns
                
                # Compute scoring with cache
                topic_aggregates = compute_scoring_with_cache(file_hash, bytes_data, config)
                
                status_text.text("✅ Finalizing...")
                progress_bar.progress(90)
                
                # Identify sparse topics
                sparse_topics = edge_cases.identify_sparse_topics(topic_aggregates)
                st.session_state['sparse_topics'] = sparse_topics
                
                st.session_state[SESSION_KEYS['topic_aggregates']] = topic_aggregates
                
                # Reset selection and tracking
                st.session_state[SESSION_KEYS['selected_topics']] = []
                st.session_state['previous_top_n'] = None
                
                progress_bar.progress(100)
                status_text.text("✅ File processed successfully!")
                
                # Clear progress indicators after brief delay
                import time
                time.sleep(0.2)
                progress_bar.empty()
                status_text.empty()
                
                # Mark that processing is complete - this will trigger auto-refresh
                st.session_state['_processing_complete'] = True
                
            except Exception as e:
                progress_bar.empty()
                status_text.empty()
                st.error(f"❌ Error processing file: {str(e)}")
                st.session_state[SESSION_KEYS['canonical_model']] = None
                st.session_state[SESSION_KEYS['topic_aggregates']] = []
                return None, [], {}
        else:
            # Same file - use cached data from session state
            canonical_model = st.session_state.get(SESSION_KEYS['canonical_model'])
            topic_aggregates = st.session_state.get(SESSION_KEYS['topic_aggregates'], [])
            validation_report = st.session_state.get(SESSION_KEYS['validation_report'], {})
            
            # Get bytes_data from session state (file already read)
            bytes_data = st.session_state.get(SESSION_KEYS['file_bytes'])
            if bytes_data is None:
                # Fallback: read file again if bytes not in session state
                bytes_data = uploaded_file.read()
                st.session_state[SESSION_KEYS['file_bytes']] = bytes_data
            
            # If missing, recompute from cache
            if canonical_model is None or not topic_aggregates:
                config = {}
                dict_of_dfs, validation_report, canonical_model, topic_columns = process_uploaded_file(bytes_data, config)
                st.session_state[SESSION_KEYS['canonical_model']] = canonical_model
                st.session_state[SESSION_KEYS['validation_report']] = validation_report
                st.session_state['topic_columns'] = topic_columns
                topic_aggregates = compute_scoring_with_cache(file_hash, bytes_data, config)
                st.session_state[SESSION_KEYS['topic_aggregates']] = topic_aggregates
    else:
        # No file uploaded, use session state if available
        canonical_model = st.session_state.get(SESSION_KEYS['canonical_model'])
        topic_aggregates = st.session_state.get(SESSION_KEYS['topic_aggregates'], [])
        validation_report = st.session_state.get('validation_report', {})
    
    st.sidebar.divider()
    
    # Top N slider
    previous_top_n = st.session_state.get('previous_top_n')
    top_n = st.sidebar.slider(
        "Top N Topics",
        min_value=4,
        max_value=20,
        value=st.session_state[SESSION_KEYS['top_n']],
        key='top_n_slider'
    )
    
    # Track if N changed
    n_changed = previous_top_n is not None and top_n != previous_top_n
    st.session_state['previous_top_n'] = top_n
    st.session_state[SESSION_KEYS['top_n']] = top_n
    
    # Auto-select Top N checkbox
    auto_select = st.sidebar.checkbox(
        "Auto-select Top N",
        value=st.session_state[SESSION_KEYS['auto_select_top_n']],
        key='auto_select_checkbox'
    )
    st.session_state[SESSION_KEYS['auto_select_top_n']] = auto_select
    
    # Auto-add when N changes checkbox
    auto_add_on_change = st.sidebar.checkbox(
        "Auto-add when N changes",
        value=st.session_state.get('auto_add_on_change', False),
        key='auto_add_on_change_checkbox'
    )
    st.session_state['auto_add_on_change'] = auto_add_on_change
    
    if not topic_aggregates:
        st.sidebar.info("Upload a file to begin")
        return canonical_model, topic_aggregates, validation_report
    
    # Compute top topics
    top_topics = compute_top_topics(topic_aggregates)
    
    # Get available topic IDs
    available_topic_ids = [t['topic_id'] for t in topic_aggregates]
    
    # Limit options for performance (max 10 items in dropdown)
    MAX_DROPDOWN_ITEMS = 10
    
    # Compute selected topics based on behavior rules
    current_selected = st.session_state.get(SESSION_KEYS['selected_topics'], [])
    
    # Update selection if auto_select is enabled or N changed
    if auto_select or n_changed:
        new_selected = compute_selected_topics(
            top_topics,
            current_selected,
            top_n,
            auto_select,
            auto_add_on_change,
            previous_top_n
        )
        if new_selected != current_selected:
            st.session_state[SESSION_KEYS['selected_topics']] = new_selected
            current_selected = new_selected
    
    # Filter topics for multiselect - prioritize selected and top N
    top_n_ids = set([t['topic_id'] for t in topic_aggregates[:top_n]])
    selected_ids = set(current_selected)
    
    # Build prioritized list: selected first, then top N, then others
    # CRITICAL: Always include ALL selected topics first (even if they exceed MAX_DROPDOWN_ITEMS)
    prioritized_topics = []
    # First: all selected topics (must be included to avoid errors)
    selected_topics_list = [tid for tid in current_selected if tid in available_topic_ids]
    prioritized_topics.extend(selected_topics_list)
    # Second: top N topics that aren't selected
    prioritized_topics.extend([tid for tid in available_topic_ids if tid in top_n_ids and tid not in selected_ids])
    # Third: other topics
    prioritized_topics.extend([tid for tid in available_topic_ids if tid not in top_n_ids and tid not in selected_ids])
    
    # Limit to max items for performance, but ALWAYS keep all selected topics
    # If selected topics exceed MAX_DROPDOWN_ITEMS, include all of them anyway
    if len(prioritized_topics) > MAX_DROPDOWN_ITEMS:
        selected_count = len(selected_topics_list)
        if selected_count >= MAX_DROPDOWN_ITEMS:
            # All selected topics exceed limit - include only them (no choice)
            prioritized_topics = selected_topics_list
        else:
            # Keep all selected topics + fill remaining slots with others
            remaining_topics = prioritized_topics[selected_count:]
            remaining_slots = MAX_DROPDOWN_ITEMS - selected_count
            if remaining_slots > 0 and remaining_topics:
                prioritized_topics = selected_topics_list + remaining_topics[:remaining_slots]
            else:
                prioritized_topics = selected_topics_list
    
    # Filter default to only include topics that are in options (safety check)
    valid_default = [tid for tid in current_selected if tid in prioritized_topics]
    
    # Multi-select for selected topics (with limited options for performance)
    selected = st.sidebar.multiselect(
        "Selected Topics",
        options=prioritized_topics,
        default=valid_default,
        key='topic_multiselect',
        max_selections=50  # Limit selections for performance
    )
    st.session_state[SESSION_KEYS['selected_topics']] = selected
    
    # Search-add dropdown for topics outside Top N (with search functionality)
    other_topics = [tid for tid in available_topic_ids if tid not in top_n_ids]
    
    if other_topics:
        st.sidebar.markdown("**Add Topic**")
        
        # Add search box for filtering
        search_query = st.sidebar.text_input(
            "🔍 Search topic...",
            value="",
            key='topic_search_input',
            help="Type to filter topics"
        )
        
        # Filter topics by search query
        if search_query:
            filtered_topics = [tid for tid in other_topics if search_query.lower() in tid.lower()]
        else:
            # Limit to first 10 for performance when no search
            filtered_topics = other_topics[:10]
        
        if filtered_topics:
            add_topic = st.sidebar.selectbox(
                "Select topic to add",
                options=[''] + filtered_topics,
                key='add_topic_selectbox'
            )
            if add_topic and add_topic not in st.session_state[SESSION_KEYS['selected_topics']]:
                st.session_state[SESSION_KEYS['selected_topics']].append(add_topic)
                st.rerun()
        else:
            st.sidebar.caption("No topics found matching your search")
    
    # Reset and Clear buttons
    col1, col2 = st.sidebar.columns(2)
    with col1:
        if st.button("Reset to Top N", key='reset_button'):
            new_selection = reset_to_top_n(top_topics, top_n)
            st.session_state[SESSION_KEYS['selected_topics']] = new_selection
            st.rerun()
    
    with col2:
        if st.button("Clear Selection", key='clear_button'):
            st.session_state[SESSION_KEYS['selected_topics']] = clear_selection()
            st.rerun()
    
    st.sidebar.divider()
    
    # Filters
    st.sidebar.markdown("**Filters**")
    
    # Coverage tier
    coverage_tier = st.sidebar.selectbox(
        "Coverage Tier",
        options=['', 'High', 'Medium', 'Low'],
        key='coverage_tier_filter'
    )
    st.session_state[SESSION_KEYS['filters']]['coverage_tier'] = coverage_tier if coverage_tier else None
    
    # Tone rollup
    tone_rollup = st.sidebar.selectbox(
        "Tone Roll-up",
        options=['', 'Positive', 'Negative', 'Neutral', 'Mixed'],
        key='tone_rollup_filter'
    )
    st.session_state[SESSION_KEYS['filters']]['tone_rollup'] = tone_rollup.lower() if tone_rollup else None
    
    # High-emotion toggle
    high_emotion = st.sidebar.checkbox(
        "High Emotion Only",
        value=st.session_state[SESSION_KEYS['filters']]['high_emotion'],
        key='high_emotion_filter'
    )
    st.session_state[SESSION_KEYS['filters']]['high_emotion'] = high_emotion
    
    # Search box
    search_query = st.sidebar.text_input(
        "Search (summaries + quotes)",
        value=st.session_state[SESSION_KEYS['search_query']],
        key='search_input'
    )
    st.session_state[SESSION_KEYS['search_query']] = search_query
    
    # Reset filters button (in Filters section)
    if st.sidebar.button("🔄 Reset Filters", use_container_width=True, key='reset_filters_button', help="Clear all filters and search query"):
        # Reset all filters to default values
        st.session_state[SESSION_KEYS['filters']] = {
            'coverage_tier': None,
            'tone_rollup': None,
            'high_emotion': False,
        }
        st.session_state[SESSION_KEYS['search_query']] = ''
        st.session_state['participant_filter_patterns'] = []
        # Reset widget values by clearing their keys
        if 'coverage_tier_filter' in st.session_state:
            del st.session_state['coverage_tier_filter']
        if 'tone_rollup_filter' in st.session_state:
            del st.session_state['tone_rollup_filter']
        if 'high_emotion_filter' in st.session_state:
            del st.session_state['high_emotion_filter']
        if 'search_input' in st.session_state:
            del st.session_state['search_input']
        if 'participant_filter_input' in st.session_state:
            del st.session_state['participant_filter_input']
        st.rerun()
    
    st.sidebar.divider()
    
    # Participant filter (regex denylist)
    st.sidebar.markdown("**Participant Filter**")
    st.sidebar.caption("Filter out participants matching regex patterns (e.g., 'moderator|admin')")
    filter_pattern_input = st.sidebar.text_input(
        "Regex patterns (pipe-separated)",
        value='|'.join(st.session_state.get('participant_filter_patterns', [])),
        key='participant_filter_input',
        help="Enter regex patterns separated by | to exclude matching participant IDs"
    )
    
    if filter_pattern_input:
        patterns = [p.strip() for p in filter_pattern_input.split('|') if p.strip()]
        st.session_state['participant_filter_patterns'] = patterns
    else:
        st.session_state['participant_filter_patterns'] = []
    
    st.sidebar.divider()
    
    # Export buttons
    st.sidebar.markdown("**Export**")
    
    # Build digest for export (use filtered aggregates if available)
    selected_topic_ids = st.session_state.get(SESSION_KEYS['selected_topics'], [])
    if canonical_model:
        # Use filtered aggregates if available (after filters are applied), otherwise use all topic_aggregates
        aggregates_for_export = st.session_state.get('filtered_aggregates', topic_aggregates)
        if aggregates_for_export:
            selected_aggregates = [t for t in aggregates_for_export if t['topic_id'] in selected_topic_ids]
            digest_artifact = digest.build_digest(canonical_model, selected_aggregates, n_takeaways=5)
        
        html_content = export.export_to_html(digest_artifact, canonical_model)
        st.sidebar.download_button(
            label="📥 Export HTML",
            data=html_content,
            file_name="digest.html",
            mime="text/html",
            width='stretch'
        )
        
        md_content = export.export_to_markdown(digest_artifact, canonical_model)
        st.sidebar.download_button(
            label="📄 Export Markdown",
            data=md_content,
            file_name="digest.md",
            mime="text/markdown",
            width='stretch'
        )
    
    return canonical_model, topic_aggregates, validation_report


@st.cache_data(show_spinner=False)
def _format_matched_sheets(matched_sheets: Dict) -> str:
    """Format matched sheets as simple text for performance."""
    if not matched_sheets:
        return ""
    lines = []
    for role, sheet_name in matched_sheets.items():
        lines.append(f"✓ {role}: {sheet_name}")
    return "\n".join(lines)

@st.cache_data(show_spinner=False)
def _format_unmatched_sheets(unmatched_sheets: List) -> str:
    """Format unmatched sheets as simple text for performance."""
    if not unmatched_sheets:
        return ""
    return "\n".join([f"  - {name}" for name in unmatched_sheets])

@st.cache_data(show_spinner=False)
def _format_warnings(warnings: List) -> str:
    """Format warnings as simple text for performance."""
    if not warnings:
        return ""
    return "\n".join([f"⚠️ {w}" for w in warnings])

def render_validation_stats(validation_report: Dict, canonical_model):
    """Render validation report and dataset stats."""
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Participants", len(canonical_model.participants))
    
    with col2:
        st.metric("Topics", len(canonical_model.topics))
    
    with col3:
        st.metric("Evidence Cells", len(canonical_model.evidence_cells))
    
    with col4:
        matched = len(validation_report.get('matched_sheets', {}))
        st.metric("Matched Sheets", f"{matched}/3")
    
    # Show matched sheets info (simplified for performance)
    matched_sheets = validation_report.get('matched_sheets', {})
    if matched_sheets:
        with st.expander("📋 Matched Sheets Details", expanded=False):
            # Use simple text instead of multiple st.success calls
            matched_text = _format_matched_sheets(matched_sheets)
            st.text(matched_text)
    
    # Show unmatched sheets info (simplified for performance)
    unmatched_sheets = validation_report.get('unmatched_sheets', [])
    if unmatched_sheets:
        with st.expander("📄 Unmatched Sheets (not recognized)", expanded=False):
            st.info("These sheets were found but couldn't be matched to expected roles (summary, quotes, sentiments):")
            # Use simple text instead of multiple st.write calls
            unmatched_text = _format_unmatched_sheets(unmatched_sheets)
            st.text(unmatched_text)
            st.caption("💡 Tip: Rename sheets to include 'summary', 'quotes', or 'sentiments' in their names")
    
    # Validation warnings (simplified for performance)
    warnings = validation_report.get('warnings', [])
    if warnings:
        with st.expander("⚠️ Validation Warnings", expanded=False):
            # Use simple text instead of multiple st.warning calls
            warnings_text = _format_warnings(warnings)
            st.text(warnings_text)


def render_takeaways(digest_artifact: Dict[str, Any], canonical_model):
    """Render takeaways row with truncation budgets enforced."""
    st.subheader("Key Takeaways")
    takeaways = digest_artifact.get('takeaways', [])
    
    if not takeaways:
        st.info("No takeaways available. Select topics to generate takeaways.")
        return
    
    for takeaway in takeaways:
        takeaway_index = takeaway.get('takeaway_index', 0)
        takeaway_text_full = takeaway.get('takeaway_text', '')
        takeaway_text_truncated = render.format_takeaway_text(takeaway_text_full)
        source_topic_id = takeaway.get('source_topic_id', '')
        
        # Capitalize first letter of each word in source_topic_id (Title Case)
        source_topic_label = source_topic_id
        if source_topic_label:
            words = source_topic_label.split()
            source_topic_label = ' '.join(word.capitalize() for word in words)
        
        # Find corresponding topic card
        topic_cards = digest_artifact.get('topic_cards', [])
        topic_card = next((tc for tc in topic_cards if tc['topic_id'] == source_topic_id), None)
        
        # Check if text was truncated
        is_truncated = len(takeaway_text_full) > render.TAKEAWAY_MAX
        
        with st.container():
            col1, col2 = st.columns([3, 1])
            
            with col1:
                # Show truncated text (max 180 chars)
                st.write(f"**{takeaway_index}.** {takeaway_text_truncated}")
                
                # Always show expander for full takeaway text
                with st.expander("📖 Show full takeaway"):
                    st.write(takeaway_text_full)
                
                if source_topic_id:
                    st.caption(f"From: {source_topic_label}")
            
            with col2:
                if topic_card:
                    # Show evidence count as credibility signal (no extra labels)
                    evidence_count = topic_card.get('evidence_count', 0)
                    st.metric("Evidence", evidence_count)
            
            st.divider()


def render_topic_cards(digest_artifact: Dict[str, Any], canonical_model):
    """Render topic cards row with truncation budgets enforced."""
    st.subheader("Topic Cards")
    topic_cards = digest_artifact.get('topic_cards', [])
    
    if not topic_cards:
        st.info("No topic cards available. Select topics to view details.")
        return
    
    for card in topic_cards:
        topic_id = card.get('topic_id', '')
        # Capitalize first letter of each word in topic_id (Title Case)
        topic_label = topic_id
        if topic_label:
            words = topic_label.split()
            topic_label = ' '.join(word.capitalize() for word in words)
        
        topic_oneliner_full = card.get('topic_one_liner') or ''
        topic_oneliner_truncated = render.format_topic_oneliner(topic_oneliner_full)
        topic_oneliner_is_truncated = len(topic_oneliner_full) > render.TOPIC_ONELINER_MAX
        
        coverage_rate = card.get('coverage_rate', 0.0)
        evidence_count = card.get('evidence_count', 0)  # Always show this
        sentiment_mix = card.get('sentiment_mix', {})
        proof_quote_preview_full = card.get('proof_quote_preview', '')
        
        # Check if this is a fallback message or invalid quote
        is_fallback = proof_quote_preview_full == "No representative quote available"
        is_valid_quote = proof_quote_preview_full and proof_quote_preview_full.strip() and not is_fallback
        
        # Additional validation: check if it's not just a numeric placeholder
        if is_valid_quote:
            text_stripped = proof_quote_preview_full.strip()
            # Check if it's just a numeric index pattern
            numeric_patterns = [
                r'^\d+\.?\s*$',  # "1", "1.", "1. "
                r'^\d+\)\s*$',   # "1)"
                r'^\(\d+\)\s*$', # "(1)"
            ]
            for pattern in numeric_patterns:
                if re.match(pattern, text_stripped):
                    is_valid_quote = False
                    break
            # Check if it has actual words (letters)
            if is_valid_quote:
                text_no_punct = re.sub(r'[^\w\s]', '', text_stripped)
                if not re.search(r'[a-zA-Z]', text_no_punct):
                    is_valid_quote = False
        
        receipt_links = card.get('receipt_links', [])
        
        with st.container():
            st.markdown(f"### {topic_label}")
            
            # Topic one-liner (truncated to 240 chars)
            if topic_oneliner_truncated:
                st.write(topic_oneliner_truncated)
                # Show full text in expander if truncated
                if topic_oneliner_is_truncated:
                    with st.expander("📖 Show full one-liner"):
                        st.write(topic_oneliner_full)
            
            # Coverage and evidence (always shown)
            col1, col2 = st.columns(2)
            with col1:
                st.markdown(render.format_coverage_bar_html(coverage_rate), unsafe_allow_html=True)
            with col2:
                # Clarify what evidence_count means: total supporting excerpts
                st.metric("Total Evidence", evidence_count)
                st.caption("Number of source excerpts supporting this topic")
            
            # Sentiment mix
            st.markdown(render.format_sentiment_mix_html(sentiment_mix), unsafe_allow_html=True)
            
            # Proof quote preview (static, 2-3 lines, ~150 chars, with sentiment chip)
            if is_valid_quote and proof_quote_preview_full:
                # Get sentiment for proof quote
                proof_quote_ref = card.get('proof_quote_ref', '')
                proof_quote_sentiment = None
                if proof_quote_ref and ':' in proof_quote_ref:
                    # Get sentiment from receipt display
                    proof_receipt = render.build_receipt_display(
                        proof_quote_ref, canonical_model, topic_id=topic_id
                    )
                    proof_quote_sentiment = proof_receipt.get('sentiment')
                
                # Truncate to ~150 chars for 2-3 lines preview
                proof_preview_short = render.truncate(proof_quote_preview_full, 150)
                is_quote_truncated = len(proof_quote_preview_full) > 150
                
                # Display static preview with sentiment chip
                col1, col2 = st.columns([4, 1])
                with col1:
                    st.caption(f"*\"{proof_preview_short}\"*")
                with col2:
                    if proof_quote_sentiment:
                        st.markdown(render.format_sentiment_chip(proof_quote_sentiment), unsafe_allow_html=True)
                
                # Always show expander for full proof quote
                with st.expander("💬 Proof Quote"):
                    st.write(proof_quote_preview_full)
            
            # Receipts expander with pagination
            if receipt_links:
                total_receipts = len(receipt_links)
                with st.expander(f"📋 Show receipts ({total_receipts})"):
                    # Convert receipt references to display objects
                    receipt_displays = []
                    for receipt_ref in receipt_links:
                        receipt_display = render.build_receipt_display(
                            receipt_ref, canonical_model, topic_id=topic_id
                        )
                        receipt_displays.append(receipt_display)
                    
                    # Rank receipts for consistent ordering
                    ranked_receipts, _ = render.rank_and_limit_receipts(
                        receipt_displays,
                        max_display=len(receipt_displays),  # Rank all, don't limit yet
                        prioritize_diversity=True
                    )
                    
                    # Pagination setup
                    receipt_page_key = f'receipts_page_topic_{topic_id}'
                    if receipt_page_key not in st.session_state:
                        st.session_state[receipt_page_key] = 0
                    
                    current_page = st.session_state[receipt_page_key]
                    page_size = 8  # 5-10 receipts per page
                    total_pages = (len(ranked_receipts) + page_size - 1) // page_size if ranked_receipts else 1
                    start_idx = current_page * page_size
                    end_idx = min(start_idx + page_size, len(ranked_receipts))
                    
                    # Get receipts for current page
                    displayed_receipts = ranked_receipts[start_idx:end_idx]
                    
                    # Show progress indicator
                    if total_pages > 1:
                        st.caption(f"Page {current_page + 1} of {total_pages} — Showing receipts {start_idx + 1}-{end_idx} of {len(ranked_receipts)}")
                    
                    # Display receipts
                    for receipt_idx, receipt in enumerate(displayed_receipts):
                        participant_label = receipt.get('participant_label', 'Unknown')
                        quote_full = receipt.get('quote_full', '')
                        sentiment = receipt.get('sentiment')
                        
                        if quote_full and quote_full != 'No quote text available':
                            # Check if quote is longer than 400 chars
                            is_long_quote = len(quote_full) > 400
                            
                            # Display receipt with participant label, quote, and sentiment
                            col1, col2 = st.columns([4, 1])
                            with col1:
                                if is_long_quote:
                                    # For long quotes: use toggle to switch between truncated and full
                                    quote_preview = render.truncate(quote_full, 400)
                                    
                                    # Create unique key for this receipt's toggle
                                    toggle_key = f"show_full_quote_{topic_id}_{receipt_idx}_{current_page}"
                                    show_full = st.toggle("📖 Show full quote", key=toggle_key, value=False)
                                    
                                    # Show either preview or full quote based on toggle
                                    if show_full:
                                        st.write(f"**{participant_label}**: \"{quote_full}\"")
                                    else:
                                        st.write(f"**{participant_label}**: \"{quote_preview}\"")
                                else:
                                    # Short quote - show full text as one block
                                    st.write(f"**{participant_label}**: \"{quote_full}\"")
                            with col2:
                                if sentiment:
                                    st.markdown(render.format_sentiment_chip(sentiment), unsafe_allow_html=True)
                        else:
                            st.caption(f"**{participant_label}**: (Quote text not available)")
                        
                        st.markdown("---")
                    
                    # Pagination controls
                    if total_pages > 1:
                        col1, col2, col3 = st.columns([1, 1, 1])
                        
                        with col1:
                            if st.button("◀ Previous", key=f"receipts_prev_topic_{topic_id}", disabled=(current_page == 0), use_container_width=True):
                                st.session_state[receipt_page_key] = max(0, current_page - 1)
                                st.rerun()
                        
                        with col2:
                            st.markdown(f"<div style='text-align: center; padding-top: 8px;'><strong>{current_page + 1} / {total_pages}</strong></div>", unsafe_allow_html=True)
                        
                        with col3:
                            if st.button("Next ▶", key=f"receipts_next_topic_{topic_id}", disabled=(current_page >= total_pages - 1), use_container_width=True):
                                st.session_state[receipt_page_key] = min(total_pages - 1, current_page + 1)
                                st.rerun()
            else:
                st.caption("No evidence available for this topic.")
            
            st.divider()


def _format_sentiment_with_icon(sentiment_label: str) -> str:
    """Format sentiment label with icon for visual scanning."""
    icons = {
        'positive': '✅',
        'negative': '❌',
        'neutral': '⚪',
        'mixed': '🔄',
        'unknown': '❓'
    }
    colors = {
        'positive': '#22c55e',  # green
        'negative': '#ef4444',  # red
        'neutral': '#6b7280',  # gray
        'mixed': '#f59e0b',     # amber
        'unknown': '#9ca3af'     # light gray
    }
    icon = icons.get(sentiment_label, '❓')
    color = colors.get(sentiment_label, '#9ca3af')
    label_text = sentiment_label.capitalize()
    return f'<span style="color: {color};">{icon} {label_text}</span>'


def _format_importance_with_rank(importance_score: float, rank: int) -> str:
    """Format importance score with rank index."""
    return f"{importance_score:.2f} (#{rank})"


def _format_coverage_bar_html(coverage_pct: float, max_width: int = 100) -> str:
    """Format coverage as HTML progress bar."""
    width_px = int((coverage_pct / 100.0) * max_width)
    return f'''
    <div style="display: flex; align-items: center; gap: 8px;">
        <div style="width: {max_width}px; height: 8px; background-color: #e5e7eb; border-radius: 4px; overflow: hidden;">
            <div style="width: {width_px}px; height: 100%; background-color: #655CFE;"></div>
        </div>
        <span style="font-size: 12px; color: #374151;">{coverage_pct:.1f}%</span>
    </div>
    '''


def _format_sentiment_distribution_bar(sentiment_dist: Dict[str, int]) -> str:
    """Format sentiment distribution as HTML bar chart."""
    total = sum(sentiment_dist.values())
    if total == 0:
        return '<div style="color: #9ca3af; font-size: 12px;">No sentiment data</div>'
    
    # Color mapping for sentiments
    colors = {
        'positive': '#22c55e',  # green
        'negative': '#ef4444',  # red
        'neutral': '#94a3b8',   # gray
        'mixed': '#f59e0b',     # amber
        'unknown': '#9ca3af'     # gray
    }
    
    bars = []
    for sentiment, count in sentiment_dist.items():
        if count > 0:
            percentage = (count / total) * 100
            color = colors.get(sentiment, '#9ca3af')
            bars.append(f'''
                <div style="display: flex; align-items: center; margin-bottom: 4px;">
                    <div style="width: 60px; font-size: 11px; color: #374151;">{sentiment.capitalize()}</div>
                    <div style="flex: 1; height: 16px; background-color: #e5e7eb; border-radius: 2px; overflow: hidden; margin: 0 8px;">
                        <div style="width: {percentage}%; height: 100%; background-color: {color};"></div>
                    </div>
                    <div style="width: 30px; font-size: 11px; color: #6b7280; text-align: right;">{count}</div>
                </div>
            ''')
    
    return '<div>' + ''.join(bars) + '</div>'


def _get_topic_confidence_data(topic: explore_model.ExploreTopic, canonical_model) -> Dict[str, Any]:
    """
    Compute confidence signals for a topic.
    
    Returns:
        Dictionary with:
        - mentions_count: int
        - source_documents_count: int (unique participants)
        - sentiment_distribution: Dict[str, int]
    """
    topic_id = topic.topic_id
    
    # Get evidence cells for this topic
    evidence_cells = [ec for ec in canonical_model.evidence_cells if ec.topic_id == topic_id]
    
    # Count unique participants (source documents)
    unique_participants = set(ec.participant_id for ec in evidence_cells if ec.participant_id)
    source_documents_count = len(unique_participants)
    
    # Get sentiment distribution
    sentiment_distribution = digest._compute_sentiment_mix(evidence_cells, topic_id)
    
    return {
        'mentions_count': topic.mentions_count,
        'source_documents_count': source_documents_count,
        'sentiment_distribution': sentiment_distribution
    }


def render_explore_tab(topic_aggregates: List[Dict[str, Any]], canonical_model):
    """
    Render Explore tab with improved UX table view and recipient filtering.
    
    Features:
    - Recipient selector for persona-based filtering
    - Importance score with rank index
    - Coverage as progress bar
    - Sentiment with icons
    - Truncated summaries with expand toggle
    - Pagination (15 per page)
    """
    if not topic_aggregates:
        st.info("Upload a file and compute scores to view data.")
        return
    
    # Cache key for base ranked topics (to avoid recomputation)
    cache_key = f"explore_base_ranked_topics_{hash(str(topic_aggregates))}"
    
    # Convert to ExploreTopic model and rank (only once, cached)
    if cache_key not in st.session_state:
        explore_topics = []
        for topic_agg in topic_aggregates:
            topic_id = topic_agg['topic_id']
            sentiment_mix = digest._compute_sentiment_mix(canonical_model.evidence_cells, topic_id)
            explore_topic = explore_model.from_topic_aggregate(topic_agg, sentiment_mix, topic_id)
            explore_topics.append(explore_topic)
        
        # Rank topics (base ranking, before recipient filtering)
        ranked_topics = explore_model.rank_topics(explore_topics)
        st.session_state[cache_key] = ranked_topics
    else:
        ranked_topics = st.session_state[cache_key]
    
    # Get available recipients
    default_recipients = recipient.create_default_recipients()
    
    # Ensure "General" is the first option
    recipient_options = {r.recipient_id: r for r in default_recipients}
    if "general" not in recipient_options:
        recipient_options["general"] = recipient.RecipientProfile(
            recipient_id="general",
            label="General",
            priority_topics=[],
            deprioritized_topics=[]
        )
    
    # Recipient selector - ensure "general" is first
    recipient_ids = list(recipient_options.keys())
    # Sort to put "general" first, then others
    recipient_ids = sorted(recipient_ids, key=lambda x: (x != "general", x))
    recipient_labels = [recipient_options[rid].label for rid in recipient_ids]
    
    # Initialize selected recipient (default to "general")
    if 'explore_selected_recipient' not in st.session_state:
        st.session_state['explore_selected_recipient'] = "general"
    
    # Track previous recipient to detect changes
    previous_recipient = st.session_state.get('explore_previous_recipient', None)
    
    # Create selectbox for recipient selection
    selected_recipient_idx = recipient_ids.index(st.session_state['explore_selected_recipient']) if st.session_state['explore_selected_recipient'] in recipient_ids else 0
    selected_label = st.selectbox(
        "View for:",
        options=recipient_labels,
        index=selected_recipient_idx,
        key='explore_recipient_selector',
        help="Select a recipient persona to filter topics"
    )
    
    # Update session state with selected recipient ID
    selected_recipient_id = recipient_ids[recipient_labels.index(selected_label)]
    
    # Detect recipient change and reset page to avoid out-of-bounds
    if previous_recipient != selected_recipient_id:
        st.session_state['explore_page'] = 1
        st.session_state['explore_previous_recipient'] = selected_recipient_id
    
    st.session_state['explore_selected_recipient'] = selected_recipient_id
    
    # Get selected recipient profile
    selected_recipient = recipient_options[selected_recipient_id]
    
    # Apply recipient filtering (instant, no recomputation)
    filtered_topics = recipient.filter_topics_for_recipient(ranked_topics, selected_recipient)
    
    # Show active recipient label
    st.markdown(f"### 📊 Explore Topics — *{selected_recipient.label}*")
    
    # Empty state: No topics after recipient filtering
    if not filtered_topics:
        if len(ranked_topics) == 0:
            # No topics at all in the data
            st.warning("⚠️ **No topics found in the uploaded data.**\n\nPlease check your file and ensure it contains valid topic data.")
        else:
            # Topics exist but were filtered out
            deprioritized_topics = [t for t in ranked_topics 
                                   if t.topic_id.lower().strip() in [tid.lower().strip() 
                                                                     for tid in selected_recipient.deprioritized_topics]]
            has_deprioritized = len(deprioritized_topics) > 0
            has_priority = len(selected_recipient.priority_topics) > 0
            
            explanation_parts = []
            if has_deprioritized:
                explanation_parts.append(f"- {len(deprioritized_topics)} topic(s) are deprioritized for this recipient")
                explanation_parts.append("- Deprioritized topics are hidden unless they have high signal (importance ≥ 1.9 and coverage ≥ 90%)")
            if has_priority:
                explanation_parts.append(f"- Priority topics are boosted but may still be filtered if they don't meet signal thresholds")
            
            if not explanation_parts:
                explanation_parts.append("- The recipient filter may be too restrictive")
            
            st.warning(
                f"⚠️ **No topics match the current filter for *{selected_recipient.label}*.**\n\n"
                f"**Why this happened:**\n" + "\n".join(explanation_parts) + "\n\n"
                f"**What to try:**\n"
                f"- Select 'General' recipient to see all {len(ranked_topics)} topics\n"
                f"- Check the recipient's priority and deprioritized topic settings\n"
                f"- Verify that topics meet the signal thresholds"
            )
        return
    
    # Show topic count caption
    if selected_recipient.priority_topics or selected_recipient.deprioritized_topics:
        filtered_count = len(filtered_topics)
        total_count = len(ranked_topics)
        if filtered_count < total_count:
            st.info(f"ℹ️ Showing {filtered_count} of {total_count} topics (filtered for {selected_recipient.label})")
        else:
            st.caption(f"Showing all {filtered_count} topics")
    else:
        st.caption(f"Showing all {len(filtered_topics)} topics")
    
    # Pagination (on filtered topics)
    items_per_page = 15
    total_pages = (len(filtered_topics) + items_per_page - 1) // items_per_page
    
    # Initialize page in session state if not exists
    if 'explore_page' not in st.session_state:
        st.session_state['explore_page'] = 1
    
    # Reset page if out of bounds
    if st.session_state['explore_page'] > total_pages:
        st.session_state['explore_page'] = 1
    
    if total_pages > 1:
        # Use session state value and update it
        current_page = st.session_state.get('explore_page', 1)
        page = st.number_input(
            "Page",
            min_value=1,
            max_value=total_pages,
            value=current_page,
            key='explore_page_number_input',
            help=f"Showing {items_per_page} topics per page"
        )
        # Update session state
        st.session_state['explore_page'] = page
        start_idx = (page - 1) * items_per_page
        end_idx = start_idx + items_per_page
        page_topics = filtered_topics[start_idx:end_idx]
        st.caption(f"Page {page} of {total_pages} ({len(filtered_topics)} total topics)")
    else:
        page_topics = filtered_topics
        page = 1
        start_idx = 0
        st.session_state['explore_page'] = 1
    
    # Summary truncation length
    SUMMARY_TRUNCATE_LENGTH = 100
    
    # Add CSS for frozen first column effect, sticky header, and full-width expander
    st.markdown("""
    <style>
        /* Ensure sentiment chips have correct colors - override any brand color CSS */
        span[style*="background-color: #ef4444"] {
            background-color: #ef4444 !important;
            color: white !important;
        }
        
        /* Style for Explore table - frozen first column effect */
        .explore-table-row {
            border-bottom: 1px solid #e5e7eb;
            padding: 8px 0;
        }
        /* Make topic column stand out (frozen effect) */
        div[data-testid="column"]:first-child {
            position: sticky;
            left: 0;
            background-color: white;
            z-index: 1;
            padding-right: 16px;
        }
        
        /* Sticky table header - make header row stick to top when scrolling */
        .explore-table-header,
        [data-testid="stHorizontalBlock"].explore-table-header {
            position: sticky !important;
            top: 0 !important;
            z-index: 100 !important;
            background-color: white !important;
            padding: 12px 0 !important;
            margin-bottom: 8px !important;
            border-bottom: 2px solid #e5e7eb !important;
            box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05) !important;
        }
        
        /* Ensure header columns have proper styling */
        .explore-table-header [data-testid="column"],
        [data-testid="stHorizontalBlock"].explore-table-header [data-testid="column"] {
            background-color: white !important;
            padding: 8px 12px !important;
        }
        
        /* Make header text bold and slightly larger */
        .explore-table-header [data-testid="stMarkdownContainer"],
        [data-testid="stHorizontalBlock"].explore-table-header [data-testid="stMarkdownContainer"] {
            font-weight: 600 !important;
            font-size: 0.95rem !important;
            color: #1f2937 !important;
        }
        
        /* Alternative selector for header row */
        [data-testid="stHorizontalBlock"]:has([data-testid="column"]:has-text("Topic")):has([data-testid="column"]:has-text("Importance")) {
            position: sticky !important;
            top: 0 !important;
            z-index: 100 !important;
            background-color: white !important;
            padding: 12px 0 !important;
            border-bottom: 2px solid #e5e7eb !important;
            box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05) !important;
        }
        
        /* Full-width expander when open - AGGRESSIVE approach */
        [data-testid="stExpander"].full-width-expander {
            position: fixed !important;
            left: 0 !important;
            right: 0 !important;
            width: 100vw !important;
            max-width: 100vw !important;
            margin-left: 0 !important;
            margin-right: 0 !important;
            padding-left: 2rem !important;
            padding-right: 2rem !important;
            z-index: 9999 !important;
            background-color: white !important;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1) !important;
        }
        
        /* Break out of ALL parent containers */
        [data-testid="column"]:has([data-testid="stExpander"].full-width-expander),
        [data-testid="stHorizontalBlock"]:has([data-testid="stExpander"].full-width-expander),
        [data-testid="stVerticalBlock"]:has([data-testid="stExpander"].full-width-expander) {
            width: 100vw !important;
            max-width: 100vw !important;
            padding-left: 0 !important;
            padding-right: 0 !important;
        }
        
        /* Ensure expander content is also full width */
        [data-testid="stExpander"].full-width-expander > div {
            max-width: 100% !important;
            width: 100% !important;
            padding-left: 0 !important;
            padding-right: 0 !important;
        }
        
        [data-testid="stExpander"].full-width-expander [data-testid="stExpanderContent"] {
            max-width: 100% !important;
            width: 100% !important;
            padding-left: 0 !important;
            padding-right: 0 !important;
        }
        
        /* Make all inner content full width */
        [data-testid="stExpander"].full-width-expander * {
            max-width: 100% !important;
        }
    </style>
    <script>
        // Make "Show full summary" expander full-width when open
        function makeExpanderFullWidth() {
            // Find all expanders
            document.querySelectorAll('[data-testid="stExpander"]').forEach(expander => {
                const button = expander.querySelector('button');
                if (!button) return;
                
                // Check if this is a "Show full summary" expander (with or without emoji)
                const buttonText = button.textContent || button.innerText || '';
                const isSummaryExpander = buttonText.includes('Show full summary') || 
                                         buttonText.includes('full summary') ||
                                         buttonText.includes('📖 Show full summary');
                
                if (!isSummaryExpander) {
                    // Reset any styles if not a summary expander
                    expander.classList.remove('full-width-expander');
                    expander.style.width = '';
                    expander.style.maxWidth = '';
                    expander.style.position = '';
                    expander.style.left = '';
                    expander.style.right = '';
                    expander.style.marginLeft = '';
                    expander.style.marginRight = '';
                    expander.style.zIndex = '';
                    return;
                }
                
                // Check if expanded
                const isExpanded = button.getAttribute('aria-expanded') === 'true' || 
                                   expander.classList.contains('streamlit-expanderHeader--is-open') ||
                                   button.classList.contains('streamlit-expanderHeader--is-open');
                
                if (isExpanded) {
                    // Make full width when expanded - use fixed positioning
                    expander.classList.add('full-width-expander');
                    
                    // Get current position to maintain vertical position
                    const rect = expander.getBoundingClientRect();
                    const scrollY = window.scrollY || window.pageYOffset;
                    
                    // Apply full-width fixed styles
                    expander.style.position = 'fixed';
                    expander.style.top = `${rect.top + scrollY}px`;
                    expander.style.left = '0';
                    expander.style.right = '0';
                    expander.style.width = '100vw';
                    expander.style.maxWidth = '100vw';
                    expander.style.marginLeft = '0';
                    expander.style.marginRight = '0';
                    expander.style.paddingLeft = '2rem';
                    expander.style.paddingRight = '2rem';
                    expander.style.zIndex = '9999';
                    expander.style.backgroundColor = 'white';
                    expander.style.boxShadow = '0 4px 6px rgba(0, 0, 0, 0.1)';
                    
                    // Make content full width
                    const content = expander.querySelector('[data-testid="stExpanderContent"]');
                    if (content) {
                        content.style.maxWidth = '100%';
                        content.style.width = '100%';
                        content.style.paddingLeft = '0';
                        content.style.paddingRight = '0';
                    }
                    
                    // Also expand parent containers
                    let parent = expander.parentElement;
                    while (parent && parent !== document.body) {
                        if (parent.hasAttribute('data-testid')) {
                            parent.style.width = '100vw';
                            parent.style.maxWidth = '100vw';
                        }
                        parent = parent.parentElement;
                    }
                } else {
                    // Reset to normal width when collapsed
                    expander.classList.remove('full-width-expander');
                    expander.style.width = '';
                    expander.style.maxWidth = '';
                    expander.style.position = '';
                    expander.style.left = '';
                    expander.style.right = '';
                    expander.style.marginLeft = '';
                    expander.style.marginRight = '';
                    expander.style.paddingLeft = '';
                    expander.style.paddingRight = '';
                    expander.style.zIndex = '';
                    
                    // Reset content
                    const content = expander.querySelector('[data-testid="stExpanderContent"]');
                    if (content) {
                        content.style.maxWidth = '';
                        content.style.width = '';
                        content.style.paddingLeft = '';
                        content.style.paddingRight = '';
                    }
                }
            });
        }
        
        // Run on load
        if (document.readyState === 'loading') {
            document.addEventListener('DOMContentLoaded', makeExpanderFullWidth);
        } else {
            makeExpanderFullWidth();
        }
        
        // Watch for changes (when expander is clicked or aria-expanded changes)
        const expanderObserver = new MutationObserver((mutations) => {
            let shouldUpdate = false;
            mutations.forEach(mutation => {
                if (mutation.type === 'attributes' && mutation.attributeName === 'aria-expanded') {
                    shouldUpdate = true;
                }
                if (mutation.type === 'childList') {
                    shouldUpdate = true;
                }
            });
            if (shouldUpdate) {
                setTimeout(makeExpanderFullWidth, 50);
            }
        });
        
        expanderObserver.observe(document.body, { 
            childList: true, 
            subtree: true, 
            attributes: true, 
            attributeFilter: ['aria-expanded', 'class'] 
        });
        
        // Listen for click events on expander buttons
        document.addEventListener('click', function(e) {
            const expanderButton = e.target.closest('[data-testid="stExpander"] button');
            if (expanderButton) {
                setTimeout(makeExpanderFullWidth, 150);
            }
        });
        
        // Also run on window resize
        let resizeTimeout;
        window.addEventListener('resize', function() {
            clearTimeout(resizeTimeout);
            resizeTimeout = setTimeout(makeExpanderFullWidth, 100);
        });
        
        // Make table header sticky
        function makeHeaderSticky() {
            // Find all horizontal blocks (columns containers)
            const horizontalBlocks = document.querySelectorAll('[data-testid="stHorizontalBlock"]');
            
            for (let block of horizontalBlocks) {
                const columns = block.querySelectorAll('[data-testid="column"]');
                if (columns.length < 6) continue;
                
                // Check if this block contains header text
                let hasTopic = false;
                let hasImportance = false;
                let hasCoverage = false;
                let hasMentions = false;
                let hasSentiment = false;
                let hasSummary = false;
                
                for (let col of columns) {
                    const text = (col.textContent || col.innerText || '').trim().toLowerCase();
                    if (text.includes('topic') && !text.includes('topic_id')) hasTopic = true;
                    if (text.includes('importance')) hasImportance = true;
                    if (text.includes('coverage')) hasCoverage = true;
                    if (text.includes('mentions')) hasMentions = true;
                    if (text.includes('sentiment')) hasSentiment = true;
                    if (text.includes('summary')) hasSummary = true;
                }
                
                // If it has all header columns, make it sticky
                if (hasTopic && hasImportance && hasCoverage && hasMentions && hasSentiment && hasSummary) {
                    block.classList.add('explore-table-header');
                    // Also ensure all child columns have white background
                    columns.forEach(col => {
                        col.style.backgroundColor = 'white';
                    });
                    break; // Found the header, stop searching
                }
            }
        }
        
        // Run header sticky on load
        if (document.readyState === 'loading') {
            document.addEventListener('DOMContentLoaded', makeHeaderSticky);
        } else {
            setTimeout(makeHeaderSticky, 100);
        }
        
        // Reapply on Streamlit updates
        const headerObserver = new MutationObserver(() => {
            setTimeout(makeHeaderSticky, 200);
        });
        headerObserver.observe(document.body, { childList: true, subtree: true });
    </script>
    """, unsafe_allow_html=True)
    
    # Table header
    header_cols = st.columns([2, 1.5, 1.5, 1.5, 1.5, 3])
    with header_cols[0]:
        st.markdown("**Topic**")
    with header_cols[1]:
        st.markdown("**Importance**")
    with header_cols[2]:
        st.markdown("**Coverage**")
    with header_cols[3]:
        st.markdown("**Mentions**")
    with header_cols[4]:
        st.markdown("**Sentiment**")
    with header_cols[5]:
        st.markdown("**Summary**")
    
    st.divider()
    
    # Render rows
    for idx, topic in enumerate(page_topics):
        rank = start_idx + idx + 1 if total_pages > 1 else idx + 1
        
        row_cols = st.columns([2, 1.5, 1.5, 1.5, 1.5, 3])
        
        with row_cols[0]:
            # Topic label (frozen column effect via styling)
            # topic_label is already capitalized in explore_model.from_topic_aggregate
            st.markdown(f"**{topic.topic_label}**")
            st.caption(topic.topic_id)
        
        with row_cols[1]:
            # Importance score with rank
            importance_text = _format_importance_with_rank(topic.importance_score, rank)
            st.markdown(importance_text)
            # Show signal bucket as subtle indicator
            bucket = topic.get_signal_bucket()
            st.caption(f"Signal: {bucket}")
        
        with row_cols[2]:
            # Coverage as progress bar
            coverage_html = _format_coverage_bar_html(topic.coverage_pct)
            st.markdown(coverage_html, unsafe_allow_html=True)
        
        with row_cols[3]:
            # Mentions count
            st.markdown(f"**{topic.mentions_count}**")
        
        with row_cols[4]:
            # Sentiment with icon
            sentiment_text = _format_sentiment_with_icon(topic.sentiment_label)
            
            # Check for sparse sentiment data (low confidence)
            confidence_data = _get_topic_confidence_data(topic, canonical_model)
            sentiment_dist = confidence_data.get('sentiment_distribution', {})
            total_sentiments = sum(sentiment_dist.values())
            unknown_count = sentiment_dist.get('unknown', 0)
            
            # Show "Low confidence" warning if sentiment data is sparse
            is_sparse = (
                total_sentiments == 0 or  # No sentiment data
                (total_sentiments > 0 and unknown_count / total_sentiments > 0.7) or  # >70% unknown
                (total_sentiments < 3 and topic.sentiment_label == 'unknown')  # Very few sentiments and all unknown
            )
            
            # Use st.html for proper HTML rendering
            st.html(sentiment_text)
            if is_sparse:
                st.caption("*<span style='color: #f59e0b;'>⚠️ Low confidence</span>*", unsafe_allow_html=True)
        
        with row_cols[5]:
            # Summary with truncation and expand toggle
            summary_full = topic.summary_text
            # Handle missing summaries with placeholder
            if not summary_full or summary_full.strip() == "":
                st.markdown("*<span style='color: #9ca3af; font-style: italic;'>No summary available</span>*", unsafe_allow_html=True)
            else:
                summary_truncated = render.truncate(summary_full, SUMMARY_TRUNCATE_LENGTH)
                is_truncated = len(summary_full) > SUMMARY_TRUNCATE_LENGTH
                
                if is_truncated:
                    # Show truncated text
                    st.markdown(summary_truncated)
                    # Expand/collapse using expander (local state only)
                    with st.expander("📖 Show full summary", expanded=False):
                        st.markdown(summary_full)
                else:
                    # Show full text if not truncated
                    st.markdown(summary_full)
        
        # Confidence section (collapsible)
        confidence_data = _get_topic_confidence_data(topic, canonical_model)
        with st.expander(f"🔍 Confidence Signals — {topic.topic_label}", expanded=False):
            # Mentions count
            st.markdown(f"**Mentions:** {confidence_data['mentions_count']}")
            
            # Source documents count
            st.markdown(f"**Source Documents:** {confidence_data['source_documents_count']}")
            
            # Sentiment distribution as chips/badges
            st.markdown("**Sentiment Distribution:**")
            sentiment_chips_html = render.format_sentiment_mix_html(confidence_data['sentiment_distribution'])
            st.html(sentiment_chips_html)
        
        st.divider()


def apply_brand_styles():
    """Apply brand color (#655CFE) to Streamlit UI elements."""
    brand_color = "#655CFE"
    brand_color_hover = "#5548E8"  # Slightly darker for hover states
    
    st.markdown(f"""
    <style>
        /* Brand color: #655CFE */
        :root {{
            --brand-color: {brand_color};
            --brand-color-hover: {brand_color_hover};
        }}
        
        /* Global override for Streamlit red colors */
        * {{
            --primary-color: {brand_color} !important;
        }}
        
        /* Override Streamlit's default red primary color - AGGRESSIVE */
        [data-baseweb="slider"] > div > div {{
            background-color: {brand_color} !important;
        }}
        
        [data-baseweb="slider-track"] {{
            background-color: {brand_color} !important;
        }}
        
        [data-baseweb="slider-handle"] {{
            background-color: {brand_color} !important;
            border-color: {brand_color} !important;
        }}
        
        /* Slider value display */
        .stSlider [data-testid="stMarkdownContainer"] {{
            color: {brand_color} !important;
        }}
        
        .stSlider label + div {{
            color: {brand_color} !important;
        }}
        
        /* Checkbox - override all red */
        [data-baseweb="checkbox"] input[type="checkbox"]:checked {{
            background-color: {brand_color} !important;
            border-color: {brand_color} !important;
        }}
        
        [data-baseweb="checkbox"] input[type="checkbox"]:checked + span {{
            background-color: {brand_color} !important;
            border-color: {brand_color} !important;
        }}
        
        [data-baseweb="checkbox"] svg {{
            color: {brand_color} !important;
        }}
        
        /* CRITICAL: Ensure sentiment chips keep their colors - MAXIMUM PRIORITY */
        .sentiment-chip-positive {{
            background-color: #22c55e !important;
            color: white !important;
            border: none !important;
        }}
        .sentiment-chip-negative {{
            background-color: #ef4444 !important;
            color: white !important;
            border: none !important;
        }}
        .sentiment-chip-neutral {{
            background-color: #6b7280 !important;
            color: white !important;
            border: none !important;
        }}
        .sentiment-chip-mixed {{
            background-color: #f59e0b !important;
            color: white !important;
            border: none !important;
        }}
        .sentiment-chip-unknown {{
            background-color: #9ca3af !important;
            color: white !important;
            border: none !important;
        }}
        
        /* Also target by data attribute */
        [data-sentiment="positive"] {{
            background-color: #22c55e !important;
            color: white !important;
        }}
        [data-sentiment="negative"] {{
            background-color: #ef4444 !important;
            color: white !important;
        }}
        [data-sentiment="neutral"] {{
            background-color: #6b7280 !important;
            color: white !important;
        }}
        [data-sentiment="mixed"] {{
            background-color: #f59e0b !important;
            color: white !important;
        }}
        [data-sentiment="unknown"] {{
            background-color: #9ca3af !important;
            color: white !important;
        }}
        
        [data-baseweb="tag"] {{
            background-color: {brand_color} !important;
            color: white !important;
        }}
        
        [data-baseweb="tag"] svg {{
            color: white !important;
        }}
        
        /* Ultra Compact Sidebar */
        .stSidebar {{
            font-size: 0.8rem;
        }}
        
        .stSidebar [data-testid="stHeader"] {{
            font-size: 1rem;
            margin-bottom: 0.25rem;
            padding-bottom: 0.15rem;
            margin-top: 0.25rem;
        }}
        
        .stSidebar [data-testid="stSubheader"] {{
            font-size: 0.9rem;
            margin-top: 0.4rem;
            margin-bottom: 0.3rem;
            padding-bottom: 0.15rem;
        }}
        
        .stSidebar [data-testid="stMarkdownContainer"] {{
            margin-bottom: 0.15rem;
            margin-top: 0.15rem;
        }}
        
        .stSidebar .stDivider {{
            margin: 0.3rem 0;
        }}
        
        /* Reduce spacing in all sidebar elements */
        .stSidebar > div {{
            padding-top: 0.3rem;
            padding-bottom: 0.3rem;
        }}
        
        .stSidebar [data-testid="stVerticalBlock"] {{
            gap: 0.3rem;
        }}
        
        .stSidebar [data-testid="stHorizontalBlock"] {{
            gap: 0.3rem;
        }}
        
        /* Prevent text wrapping in pagination buttons */
        .stButton > button {{
            white-space: nowrap !important;
        }}
        .stButton > button > div {{
            white-space: nowrap !important;
        }}
        
        /* Ultra compact buttons */
        .stSidebar .stButton > button {{
            background-color: {brand_color};
            color: white;
            border: none;
            border-radius: 0.2rem;
            padding: 0.25rem 0.6rem;
            font-size: 0.8rem;
            font-weight: 500;
            transition: all 0.3s;
            height: auto;
            min-height: 1.75rem;
            margin: 0.15rem 0;
        }}
        
        .stSidebar .stButton > button:hover {{
            background-color: {brand_color_hover};
            box-shadow: 0 2px 8px rgba(101, 92, 254, 0.3);
        }}
        
        /* Ultra compact download buttons */
        .stSidebar .stDownloadButton > button {{
            background-color: {brand_color};
            color: white;
            padding: 0.25rem 0.6rem;
            font-size: 0.8rem;
            height: auto;
            min-height: 1.75rem;
            margin: 0.15rem 0;
        }}
        
        .stSidebar .stDownloadButton > button:hover {{
            background-color: {brand_color_hover};
        }}
        
        /* Ultra compact inputs */
        .stSidebar input, .stSidebar select {{
            font-size: 0.8rem;
            padding: 0.25rem 0.4rem;
            margin: 0.15rem 0;
        }}
        
        .stSidebar [data-baseweb="select"] {{
            font-size: 0.8rem;
        }}
        
        /* Ultra compact multiselect */
        .stSidebar [data-baseweb="select"] > div {{
            padding: 0.25rem 0.4rem;
            font-size: 0.8rem;
            margin: 0.15rem 0;
        }}
        
        .stSidebar [data-baseweb="select"] [data-baseweb="input"] {{
            padding: 0.25rem 0.4rem;
            min-height: 1.75rem;
        }}
        
        /* Brand color checkboxes - MAXIMUM OVERRIDE */
        .stSidebar [data-baseweb="checkbox"] input[type="checkbox"]:checked,
        .stSidebar [data-baseweb="checkbox"] input[type="checkbox"][checked] {{
            background-color: {brand_color} !important;
            border-color: {brand_color} !important;
            accent-color: {brand_color} !important;
        }}
        
        .stSidebar [data-baseweb="checkbox"] input[type="checkbox"]:checked + span,
        .stSidebar [data-baseweb="checkbox"] input[type="checkbox"][checked] + span {{
            background-color: {brand_color} !important;
            border-color: {brand_color} !important;
        }}
        
        .stSidebar [data-baseweb="checkbox"] input[type="checkbox"]:focus {{
            box-shadow: 0 0 0 0.2rem rgba(101, 92, 254, 0.25) !important;
        }}
        
        .stSidebar [data-baseweb="checkbox"] svg {{
            color: {brand_color} !important;
            fill: {brand_color} !important;
        }}
        
        /* Override ANY red colors in checkbox - AGGRESSIVE */
        .stSidebar [data-baseweb="checkbox"] *[style*="rgb(255, 75, 75)"],
        .stSidebar [data-baseweb="checkbox"] *[style*="rgb(239, 68, 68)"],
        .stSidebar [data-baseweb="checkbox"] *[style*="rgb(255, 107, 107)"],
        .stSidebar [data-baseweb="checkbox"] *[style*="#ff4b4b"],
        .stSidebar [data-baseweb="checkbox"] *[style*="#ef4444"] {{
            background-color: {brand_color} !important;
            border-color: {brand_color} !important;
            color: {brand_color} !important;
            fill: {brand_color} !important;
        }}
        
        /* Override checkbox container */
        .stSidebar [data-baseweb="checkbox"] {{
            color: {brand_color} !important;
        }}
        
        /* Ultra compact slider with brand color */
        .stSidebar .stSlider {{
            margin: 0.25rem 0;
            padding: 0.15rem 0;
        }}
        
        .stSidebar .stSlider label {{
            font-size: 0.8rem;
            margin-bottom: 0.15rem;
        }}
        
        .stSidebar .stSlider > div > div > div {{
            background-color: {brand_color} !important;
        }}
        
        .stSidebar .stSlider [data-baseweb="slider"] > div > div {{
            background-color: {brand_color} !important;
        }}
        
        .stSidebar .stSlider [data-baseweb="slider"] > div > div > div {{
            background-color: {brand_color} !important;
        }}
        
        .stSidebar .stSlider [data-baseweb="slider-track"] {{
            background-color: {brand_color} !important;
        }}
        
        .stSidebar .stSlider [data-baseweb="slider-handle"] {{
            background-color: {brand_color} !important;
            border-color: {brand_color} !important;
        }}
        
        /* Multiselect tags - brand color */
        .stSidebar [data-baseweb="tag"] {{
            background-color: {brand_color} !important;
            color: white !important;
        }}
        
        .stSidebar [data-baseweb="tag"] svg {{
            color: white !important;
        }}
        
        .stSidebar [data-baseweb="select"] [data-baseweb="tag"] {{
            background-color: {brand_color} !important;
            color: white !important;
        }}
        
        .stSidebar [data-baseweb="select"] [data-baseweb="tag"] svg {{
            color: white !important;
        }}
        
        /* Ultra compact file uploader - fixed spacing */
        .stSidebar .stFileUploader {{
            font-size: 0.75rem;
            margin: 0.1rem 0;
        }}
        
        .stSidebar .stFileUploader > div {{
            min-height: auto !important;
            height: auto !important;
            display: flex !important;
            flex-direction: column !important;
            gap: 0.2rem !important;
        }}
        
        .stSidebar .stFileUploader > div > div {{
            border-color: {brand_color};
            padding: 0.3rem 0.4rem !important;
            min-height: 2.5rem !important;
            height: auto !important;
        }}
        
        .stSidebar .stFileUploader label {{
            font-size: 0.75rem;
            margin: 0.1rem 0 0.2rem 0;
            padding: 0;
        }}
        
        .stSidebar .stFileUploader [data-testid="stFileUploaderDropzone"] {{
            min-height: 2.5rem !important;
            height: auto !important;
            padding: 0.3rem 0.4rem !important;
            margin-bottom: 0.2rem !important;
        }}
        
        .stSidebar .stFileUploader [data-testid="stFileUploaderDropzoneInstructions"] {{
            font-size: 0.7rem;
            margin: 0.1rem 0;
            padding: 0;
        }}
        
        .stSidebar .stFileUploader [data-testid="stFileUploaderDropzoneInstructions"] p {{
            margin: 0.1rem 0;
            font-size: 0.7rem;
            line-height: 1.2;
        }}
        
        .stSidebar .stFileUploader button {{
            padding: 0.2rem 0.5rem !important;
            font-size: 0.7rem !important;
            min-height: 1.6rem !important;
            height: 1.6rem !important;
            margin: 0.1rem 0 !important;
        }}
        
        .stSidebar .stFileUploader [data-testid="stFileUploaderFileName"] {{
            font-size: 0.7rem;
            margin: 0.15rem 0;
            padding: 0.1rem 0;
            display: block;
        }}
        
        .stSidebar .stFileUploader [data-testid="stFileUploaderFileSize"] {{
            font-size: 0.65rem;
            margin: 0.05rem 0;
            padding: 0;
            display: block;
        }}
        
        /* Fix file uploader file info spacing */
        .stSidebar .stFileUploader [data-testid="stFileUploaderFileStatus"] {{
            margin-top: 0.2rem !important;
            margin-bottom: 0.1rem !important;
        }}
        
        /* Ultra compact captions */
        .stSidebar [data-testid="stCaption"] {{
            font-size: 0.7rem;
            margin-top: 0.15rem;
            margin-bottom: 0.15rem;
        }}
        
        /* Compact checkbox labels */
        .stSidebar [data-baseweb="checkbox"] label {{
            font-size: 0.8rem;
            margin: 0.15rem 0;
            padding: 0.15rem 0;
        }}
        
        /* Reduce spacing in columns */
        .stSidebar [data-testid="column"] {{
            padding: 0.15rem;
        }}
        
        /* Reduce info box spacing */
        .stSidebar .stAlert {{
            padding: 0.4rem 0.6rem;
            margin: 0.25rem 0;
            font-size: 0.8rem;
        }}
        
        /* Additional compact spacing */
        .stSidebar [data-baseweb="base-input"] {{
            padding: 0.25rem 0.4rem;
            min-height: 1.75rem;
        }}
        
        .stSidebar [data-baseweb="select"] [data-baseweb="popover"] {{
            max-height: 200px;
        }}
        
        /* Reduce line height */
        .stSidebar * {{
            line-height: 1.3;
        }}
        
        /* Compact multiselect container */
        .stSidebar [data-baseweb="select"] {{
            margin: 0.15rem 0;
        }}
        
        /* Remove extra padding from text inputs */
        .stSidebar input[type="text"] {{
            padding: 0.25rem 0.4rem !important;
            min-height: 1.75rem !important;
        }}
        
        /* Remove red error/warning colors */
        .stSidebar .stAlert {{
            border-left-color: {brand_color} !important;
        }}
        
        .stSidebar [data-baseweb="notification"] {{
            border-left-color: {brand_color} !important;
        }}
        
        /* Override Streamlit default red colors */
        .stSidebar [style*="rgb(255, 75, 75)"],
        .stSidebar [style*="rgb(239, 68, 68)"],
        .stSidebar [style*="#ef4444"],
        .stSidebar [style*="#ff4b4b"] {{
            background-color: {brand_color} !important;
            border-color: {brand_color} !important;
            color: white !important;
        }}
        
        /* Slider track and handle - override red */
        .stSidebar [data-baseweb="slider"] [style*="rgb(255, 75, 75)"],
        .stSidebar [data-baseweb="slider"] [style*="#ff4b4b"] {{
            background-color: {brand_color} !important;
        }}
        
        /* Checkbox - override red */
        .stSidebar [data-baseweb="checkbox"] [style*="rgb(255, 75, 75)"],
        .stSidebar [data-baseweb="checkbox"] [style*="#ff4b4b"] {{
            background-color: {brand_color} !important;
            border-color: {brand_color} !important;
        }}
        
        /* Tag - override red */
        .stSidebar [data-baseweb="tag"] [style*="rgb(255, 75, 75)"],
        .stSidebar [data-baseweb="tag"] [style*="#ff4b4b"] {{
            background-color: {brand_color} !important;
        }}
        
        /* Primary buttons (main area) */
        .stButton > button {{
            background-color: {brand_color};
            color: white;
            border: none;
            border-radius: 0.25rem;
            padding: 0.5rem 1rem;
            font-weight: 500;
            transition: all 0.3s;
        }}
        
        .stButton > button:hover {{
            background-color: {brand_color_hover};
            box-shadow: 0 2px 8px rgba(101, 92, 254, 0.3);
        }}
        
        /* Download buttons (main area) */
        .stDownloadButton > button {{
            background-color: {brand_color};
            color: white;
        }}
        
        .stDownloadButton > button:hover {{
            background-color: {brand_color_hover};
        }}
        
        /* Progress bar */
        .stProgress > div > div > div {{
            background-color: {brand_color};
        }}
        
        /* Tabs - ONLY ONE brand color underline, NO red/pink */
        .stTabs [data-baseweb="tab-list"] {{
            gap: 8px;
            border-bottom: 1px solid #e5e7eb;
        }}
        
        /* Remove ALL borders from all tabs */
        .stTabs [data-baseweb="tab"] {{
            color: #666;
            border: none !important;
            border-top: none !important;
            border-left: none !important;
            border-right: none !important;
            border-bottom: none !important;
            background: none !important;
        }}
        
        /* Remove ALL pseudo-elements */
        .stTabs [data-baseweb="tab"]::after,
        .stTabs [data-baseweb="tab"]::before,
        .stTabs [data-baseweb="tab"] *::after,
        .stTabs [data-baseweb="tab"] *::before {{
            display: none !important;
            content: none !important;
        }}
        
        /* Remove borders from inner divs */
        .stTabs [data-baseweb="tab"] > div,
        .stTabs [data-baseweb="tab"] > div > div {{
            border: none !important;
            border-top: none !important;
            border-left: none !important;
            border-right: none !important;
            border-bottom: none !important;
        }}
        
        /* Remove borders from buttons */
        .stTabs [data-baseweb="tab"] button {{
            border: none !important;
            border-top: none !important;
            border-left: none !important;
            border-right: none !important;
            border-bottom: none !important;
        }}
        
        /* Active tab - use Streamlit's default red color */
        .stTabs [aria-selected="true"] {{
            color: #ff4b4b !important;
        }}
        
        /* Remove ONLY brand color borders from tabs - keep Streamlit's default red */
        .stTabs [data-baseweb="tab"] [style*="{brand_color}"],
        .stTabs [data-baseweb="tab"] [style*="#655CFE"] {{
            border-bottom: none !important;
            background-color: transparent !important;
        }}
        
        /* Ensure only ONE underline - remove duplicate borders from children */
        .stTabs [aria-selected="true"] > * {{
            border-bottom: none !important;
        }}
        
        .stTabs [aria-selected="true"] > * > * {{
            border-bottom: none !important;
        }}
        
        /* Sidebar headers */
        .stSidebar [data-testid="stHeader"] {{
            color: {brand_color};
        }}
        
        /* Links */
        a {{
            color: {brand_color};
        }}
        
        a:hover {{
            color: {brand_color_hover};
        }}
        
        /* Selectbox/Multiselect focus */
        .stSelectbox > div > div {{
            border-color: {brand_color};
        }}
        
        /* Slider */
        .stSlider > div > div > div {{
            background-color: {brand_color};
        }}
        
        /* File uploader */
        .stFileUploader > div > div {{
            border-color: {brand_color};
        }}
        
        /* Metrics */
        [data-testid="stMetricValue"] {{
            color: {brand_color};
        }}
        
        /* Expander */
        .streamlit-expanderHeader {{
            color: {brand_color};
        }}
        
        /* Info boxes */
        .stInfo {{
            border-left: 4px solid {brand_color};
        }}
        
        /* Success messages */
        .stSuccess {{
            border-left: 4px solid {brand_color};
        }}
        
        /* Remove red from warnings/errors */
        .stWarning {{
            border-left-color: #f59e0b !important;
            background-color: rgba(245, 158, 11, 0.1) !important;
        }}
        
        .stError {{
            border-left-color: {brand_color} !important;
            background-color: rgba(101, 92, 254, 0.1) !important;
        }}
    </style>
    <script>
        // Force brand color on slider and checkbox - run after page load
        function applyBrandColors() {{
            const brandColor = '{brand_color}';
            
            // Override slider colors
            document.querySelectorAll('[data-baseweb="slider-handle"]').forEach(el => {{
                el.style.backgroundColor = brandColor;
                el.style.borderColor = brandColor;
            }});
            
            document.querySelectorAll('[data-baseweb="slider-track"]').forEach(el => {{
                el.style.backgroundColor = brandColor;
            }});
            
            // Override checkbox colors - AGGRESSIVE
            document.querySelectorAll('[data-baseweb="checkbox"]').forEach(container => {{
                const checkbox = container.querySelector('input[type="checkbox"]');
                if (checkbox && checkbox.checked) {{
                    checkbox.style.backgroundColor = brandColor;
                    checkbox.style.borderColor = brandColor;
                    checkbox.style.accentColor = brandColor;
                    
                    // Fix span wrapper
                    const span = checkbox.nextElementSibling;
                    if (span && span.tagName === 'SPAN') {{
                        span.style.backgroundColor = brandColor;
                        span.style.borderColor = brandColor;
                    }}
                }}
                
                // Fix SVG
                const svg = container.querySelector('svg');
                if (svg) {{
                    svg.style.color = brandColor;
                    svg.style.fill = brandColor;
                    svg.querySelectorAll('path, circle, rect').forEach(path => {{
                        path.style.fill = brandColor;
                        path.style.stroke = brandColor;
                    }});
                }}
            }});
            
            // Override slider value number color
            document.querySelectorAll('.stSlider [data-testid="stMarkdownContainer"] p').forEach(el => {{
                el.style.color = brandColor;
            }});
        }}
        
        // Run on load and after Streamlit reruns
        if (document.readyState === 'loading') {{
            document.addEventListener('DOMContentLoaded', applyBrandColors);
        }} else {{
            applyBrandColors();
        }}
        
        // Reapply after Streamlit updates
        const observer = new MutationObserver(applyBrandColors);
        observer.observe(document.body, {{ childList: true, subtree: true }});
        
        // Fix tabs - remove brand color, keep Streamlit's default red underline (only ONE)
        function fixTabs() {{
            const brandColor = '{brand_color}';
            const redColor = '#ff4b4b'; // Streamlit default red
            
            // Only remove brand color, don't touch Streamlit's default red borders
            document.querySelectorAll('[data-baseweb="tab"]').forEach(tab => {{
                // Set color for active tab
                if (tab.getAttribute('aria-selected') === 'true') {{
                    tab.style.color = redColor;
                }}
                
                // Remove ONLY brand color borders from tab and all children
                const removeBrandBorders = (element) => {{
                    const computed = window.getComputedStyle(element);
                    // Remove brand color borders only (keep red)
                    if (computed.borderBottom && (
                        computed.borderBottom.includes(brandColor) ||
                        computed.borderBottom.includes('#655CFE')
                    )) {{
                        element.style.borderBottom = 'none';
                        element.style.setProperty('border-bottom', 'none', 'important');
                    }}
                    
                    // Remove inline brand color
                    if (element.style.borderBottom && (
                        element.style.borderBottom.includes(brandColor) ||
                        element.style.borderBottom.includes('#655CFE')
                    )) {{
                        element.style.borderBottom = 'none';
                        element.style.setProperty('border-bottom', 'none', 'important');
                    }}
                    
                    Array.from(element.children).forEach(child => removeBrandBorders(child));
                }};
                
                removeBrandBorders(tab);
            }});
        }}
        
        // Fix sentiment chips - FORCE correct colors using classes and data attributes
        function fixSentimentChips() {{
            const colorMap = {{
                'positive': '#22c55e',
                'negative': '#ef4444',
                'neutral': '#6b7280',
                'mixed': '#f59e0b',
                'unknown': '#9ca3af'
            }};
            
            // Fix by data attribute
            document.querySelectorAll('[data-sentiment]').forEach(chip => {{
                const sentiment = chip.getAttribute('data-sentiment');
                if (sentiment && colorMap[sentiment]) {{
                    chip.style.setProperty('background-color', colorMap[sentiment], 'important');
                    chip.style.setProperty('color', 'white', 'important');
                    chip.style.setProperty('border', 'none', 'important');
                }}
            }});
            
            // Fix by class
            document.querySelectorAll('.sentiment-chip-positive').forEach(chip => {{
                chip.style.setProperty('background-color', '#22c55e', 'important');
                chip.style.setProperty('color', 'white', 'important');
            }});
            document.querySelectorAll('.sentiment-chip-negative').forEach(chip => {{
                chip.style.setProperty('background-color', '#ef4444', 'important');
                chip.style.setProperty('color', 'white', 'important');
            }});
            document.querySelectorAll('.sentiment-chip-neutral').forEach(chip => {{
                chip.style.setProperty('background-color', '#6b7280', 'important');
                chip.style.setProperty('color', 'white', 'important');
            }});
            document.querySelectorAll('.sentiment-chip-mixed').forEach(chip => {{
                chip.style.setProperty('background-color', '#f59e0b', 'important');
                chip.style.setProperty('color', 'white', 'important');
            }});
            document.querySelectorAll('.sentiment-chip-unknown').forEach(chip => {{
                chip.style.setProperty('background-color', '#9ca3af', 'important');
                chip.style.setProperty('color', 'white', 'important');
            }});
        }}
        
        // Run immediately and watch for changes
        if (document.readyState === 'loading') {{
            document.addEventListener('DOMContentLoaded', fixSentimentChips);
        }} else {{
            fixSentimentChips();
        }}
        
        const sentimentObserver = new MutationObserver(() => {{
            fixSentimentChips();
        }});
        sentimentObserver.observe(document.body, {{ childList: true, subtree: true }});
        
        // Fix checkboxes - force brand color
        function fixCheckboxes() {{
            const brandColor = '{brand_color}';
            
            document.querySelectorAll('[data-baseweb="checkbox"]').forEach(checkboxContainer => {{
                const checkbox = checkboxContainer.querySelector('input[type="checkbox"]');
                if (checkbox && checkbox.checked) {{
                    checkbox.style.backgroundColor = brandColor;
                    checkbox.style.borderColor = brandColor;
                    checkbox.style.accentColor = brandColor;
                    
                    // Fix the span wrapper
                    const span = checkbox.nextElementSibling;
                    if (span && span.tagName === 'SPAN') {{
                        span.style.backgroundColor = brandColor;
                        span.style.borderColor = brandColor;
                    }}
                    
                    // Fix SVG icon
                    const svg = checkboxContainer.querySelector('svg');
                    if (svg) {{
                        svg.style.color = brandColor;
                        svg.style.fill = brandColor;
                        // Also set stroke if exists
                        svg.querySelectorAll('path, circle, rect').forEach(path => {{
                            path.style.fill = brandColor;
                            path.style.stroke = brandColor;
                        }});
                    }}
                }}
            }});
        }}
        
        // Run tab fix
        if (document.readyState === 'loading') {{
            document.addEventListener('DOMContentLoaded', fixTabs);
        }} else {{
            fixTabs();
        }}
        
        // Reapply tab fix after Streamlit updates
        const tabObserver = new MutationObserver(() => {{
            fixTabs();
            fixCheckboxes();
            fixSentimentChips();
        }});
        tabObserver.observe(document.body, {{ childList: true, subtree: true }});
        
        // Also run fixes on load
        if (document.readyState === 'loading') {{
            document.addEventListener('DOMContentLoaded', () => {{
                fixCheckboxes();
                fixSentimentChips();
            }});
        }} else {{
            fixCheckboxes();
            fixSentimentChips();
        }}
    </script>
    """, unsafe_allow_html=True)


def main():
    """Main Streamlit application."""
    st.set_page_config(
        page_title="5-Minute Digest",
        page_icon="📊",
        layout="wide"
    )
    
    # Apply brand styles
    apply_brand_styles()
    
    st.title("📊 5-Minute Digest")
    
    # Initialize session state
    initialize_session_state()
    
    # Check if file processing just completed and trigger refresh
    if st.session_state.get('_processing_complete', False):
        st.session_state['_processing_complete'] = False
        # Use a small delay to ensure state is saved, then refresh
        st.rerun()
    
    # Get canonical model and topic aggregates from session state or process file
    canonical_model = st.session_state.get(SESSION_KEYS['canonical_model'])
    topic_aggregates = st.session_state.get(SESSION_KEYS['topic_aggregates'], [])
    validation_report = {}
    
    # Render sidebar
    canonical_model, topic_aggregates, validation_report = render_sidebar(canonical_model, topic_aggregates)
    
    if canonical_model is None:
        st.info("👈 Please upload an Excel file to begin")
        return
    
    # Apply filters
    filtered_aggregates = filter_topics(
        topic_aggregates,
        st.session_state[SESSION_KEYS['filters']],
        st.session_state[SESSION_KEYS['search_query']],
        canonical_model
    )
    
    # Store filtered aggregates in session state for export
    st.session_state['filtered_aggregates'] = filtered_aggregates
    
    # Get selected topic aggregates
    selected_topic_ids = st.session_state[SESSION_KEYS['selected_topics']]
    selected_aggregates = [t for t in filtered_aggregates if t['topic_id'] in selected_topic_ids]
    
    # Build digest
    digest_artifact = digest.build_digest(canonical_model, selected_aggregates, n_takeaways=5)
    
    # Main tabs
    tab1, tab2 = st.tabs(["Digest", "Explore"])
    
    with tab1:
        # Row 1: Validation + Stats
        if hasattr(canonical_model, 'participants'):
            render_validation_stats(validation_report, canonical_model)
        
        st.divider()
        
        # Row 2: Takeaways
        render_takeaways(digest_artifact, canonical_model)
        
        st.divider()
        
        # Row 3: Topic Cards
        render_topic_cards(digest_artifact, canonical_model)
    
    with tab2:
        render_explore_tab(topic_aggregates, canonical_model)


# Unit tests for topic selection behavior
if __name__ == "__main__" and False:  # Set to True to run tests
    import unittest
    
    class TestTopicSelection(unittest.TestCase):
        
        def test_compute_top_topics(self):
            """Test computing top topics from aggregates."""
            aggregates = [
                {'topic_id': 'topic1', 'topic_score': 0.9},
                {'topic_id': 'topic2', 'topic_score': 0.8},
                {'topic_id': 'topic3', 'topic_score': 0.7},
            ]
            top_topics = compute_top_topics(aggregates)
            self.assertEqual(top_topics, ['topic1', 'topic2', 'topic3'])
        
        def test_compute_selected_topics_initial(self):
            """Test initial selection when auto_select is True."""
            top_topics = ['t1', 't2', 't3', 't4', 't5']
            selected = compute_selected_topics(top_topics, [], 3, True, False, None)
            self.assertEqual(selected, ['t1', 't2', 't3'])
        
        def test_compute_selected_topics_n_increase_with_auto_add(self):
            """Test N increase with auto_add_on_change enabled."""
            top_topics = ['t1', 't2', 't3', 't4', 't5']
            current = ['t1', 't2', 't3']  # Top 3
            # Increase to 5 with auto_add enabled
            selected = compute_selected_topics(top_topics, current, 5, True, True, 3)
            # Should include t4 and t5
            self.assertIn('t4', selected)
            self.assertIn('t5', selected)
            self.assertEqual(len(selected), 5)
        
        def test_compute_selected_topics_n_increase_without_auto_add(self):
            """Test N increase without auto_add_on_change."""
            top_topics = ['t1', 't2', 't3', 't4', 't5']
            current = ['t1', 't2', 't3']  # Top 3
            # Increase to 5 without auto_add
            selected = compute_selected_topics(top_topics, current, 5, True, False, 3)
            # Should still only have top 3 (no auto-add)
            self.assertEqual(set(selected), {'t1', 't2', 't3'})
        
        def test_compute_selected_topics_n_decrease_preserves_manual(self):
            """Test N decrease preserves manually added topics."""
            top_topics = ['t1', 't2', 't3', 't4', 't5']
            # User has top 5 selected, plus manually added t10
            current = ['t1', 't2', 't3', 't4', 't5', 't10']
            # Decrease to 3
            selected = compute_selected_topics(top_topics, current, 3, True, False, 5)
            # Should preserve t10 (manually added)
            self.assertIn('t10', selected)
            # Should have top 3
            self.assertIn('t1', selected)
            self.assertIn('t2', selected)
            self.assertIn('t3', selected)
        
        def test_compute_selected_topics_manual_add_preserved(self):
            """Test that manually added topics are preserved."""
            top_topics = ['t1', 't2', 't3', 't4', 't5']
            # User manually added t10 (outside top N=3)
            current = ['t1', 't2', 't3', 't10']
            selected = compute_selected_topics(top_topics, current, 3, True, False, 3)
            # Should preserve t10
            self.assertIn('t10', selected)
            # Should have top 3
            self.assertEqual(set(selected[:3]), {'t1', 't2', 't3'})
        
        def test_reset_to_top_n(self):
            """Test reset to top N function."""
            top_topics = ['t1', 't2', 't3', 't4', 't5']
            result = reset_to_top_n(top_topics, 3)
            self.assertEqual(result, ['t1', 't2', 't3'])
        
        def test_clear_selection(self):
            """Test clear selection function."""
            result = clear_selection()
            self.assertEqual(result, [])
        
        def test_compute_selected_topics_auto_select_false(self):
            """Test that auto_select=False preserves current selection."""
            top_topics = ['t1', 't2', 't3']
            current = ['t2', 't3']  # User's manual selection
            selected = compute_selected_topics(top_topics, current, 3, False, False, None)
            # Should preserve user's selection
            self.assertEqual(selected, current)
        
        def test_compute_selected_topics_order(self):
            """Test that selection maintains proper order."""
            top_topics = ['t1', 't2', 't3', 't4', 't5']
            # Top 3 + manually added t10
            current = ['t1', 't2', 't3', 't10']
            selected = compute_selected_topics(top_topics, current, 3, True, False, 3)
            # Top N should come first, then manually added
            self.assertEqual(selected[:3], ['t1', 't2', 't3'])
            self.assertIn('t10', selected[3:])
    
    unittest.main()


if __name__ == "__main__":
    main()
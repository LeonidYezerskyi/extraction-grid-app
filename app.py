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
import render
import export


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
        st.session_state[SESSION_KEYS['top_n']] = 10
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
            filtered = [t for t in filtered if t['coverage_rate'] >= 0.7]
        elif coverage_tier == 'Medium':
            filtered = [t for t in filtered if 0.4 <= t['coverage_rate'] < 0.7]
        elif coverage_tier == 'Low':
            filtered = [t for t in filtered if t['coverage_rate'] < 0.4]
    
    # Tone rollup filter
    tone_rollup = filters.get('tone_rollup')
    if tone_rollup:
        # Need to compute sentiment mix for each topic
        filtered_by_tone = []
        for topic_agg in filtered:
            topic_id = topic_agg['topic_id']
            sentiment_mix = digest._compute_sentiment_mix(canonical_model.evidence_cells, topic_id)
            total = sum(sentiment_mix.values())
            if total > 0:
                dominant = max(sentiment_mix.items(), key=lambda x: x[1])[0]
                if dominant == tone_rollup.lower():
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


def render_sidebar(canonical_model, topic_aggregates: List[Dict[str, Any]]):
    """Render sidebar with all controls."""
    st.sidebar.header("Controls")
    
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
            st.session_state[SESSION_KEYS['uploaded_file']] = uploaded_file
            
            # Process with caching (config can be empty for now)
            config = {}
            dict_of_dfs, validation_report, canonical_model, topic_columns = process_uploaded_file(bytes_data, config)
            
            # Store in session state
            st.session_state[SESSION_KEYS['canonical_model']] = canonical_model
            st.session_state[SESSION_KEYS['validation_report']] = validation_report
            st.session_state['topic_columns'] = topic_columns
            
            # Compute scoring with cache
            topic_aggregates = compute_scoring_with_cache(file_hash, bytes_data, config)
            st.session_state[SESSION_KEYS['topic_aggregates']] = topic_aggregates
            
            # Reset selection
            st.session_state[SESSION_KEYS['selected_topics']] = []
        else:
            # Same file - use cached data from session state
            canonical_model = st.session_state.get(SESSION_KEYS['canonical_model'])
            topic_aggregates = st.session_state.get(SESSION_KEYS['topic_aggregates'], [])
            validation_report = st.session_state.get(SESSION_KEYS['validation_report'], {})
            
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
    top_n = st.sidebar.slider(
        "Top N Topics",
        min_value=4,
        max_value=20,
        value=st.session_state[SESSION_KEYS['top_n']],
        key='top_n_slider'
    )
    st.session_state[SESSION_KEYS['top_n']] = top_n
    
    # Auto-select Top N checkbox
    auto_select = st.sidebar.checkbox(
        "Auto-select Top N",
        value=st.session_state[SESSION_KEYS['auto_select_top_n']],
        key='auto_select_checkbox'
    )
    st.session_state[SESSION_KEYS['auto_select_top_n']] = auto_select
    
    if not topic_aggregates:
        st.sidebar.info("Upload a file to begin")
        return canonical_model, topic_aggregates, validation_report
    
    # Get available topic IDs
    available_topic_ids = [t['topic_id'] for t in topic_aggregates]
    
    # Auto-select Top N if enabled
    if auto_select:
        top_n_topics = [t['topic_id'] for t in topic_aggregates[:top_n]]
        if set(st.session_state[SESSION_KEYS['selected_topics']]) != set(top_n_topics):
            st.session_state[SESSION_KEYS['selected_topics']] = top_n_topics
    
    # Multi-select for selected topics
    selected = st.sidebar.multiselect(
        "Selected Topics",
        options=available_topic_ids,
        default=st.session_state[SESSION_KEYS['selected_topics']],
        key='topic_multiselect'
    )
    st.session_state[SESSION_KEYS['selected_topics']] = selected
    
    # Search-add dropdown for topics outside Top N
    top_n_ids = set([t['topic_id'] for t in topic_aggregates[:top_n]])
    other_topics = [tid for tid in available_topic_ids if tid not in top_n_ids]
    
    if other_topics:
        st.sidebar.subheader("Add Topic")
        add_topic = st.sidebar.selectbox(
            "Search and add topic",
            options=[''] + other_topics,
            key='add_topic_selectbox'
        )
        if add_topic and add_topic not in st.session_state[SESSION_KEYS['selected_topics']]:
            st.session_state[SESSION_KEYS['selected_topics']].append(add_topic)
            st.rerun()
    
    # Reset and Clear buttons
    col1, col2 = st.sidebar.columns(2)
    with col1:
        if st.button("Reset to Top N", key='reset_button'):
            top_n_topics = [t['topic_id'] for t in topic_aggregates[:top_n]]
            st.session_state[SESSION_KEYS['selected_topics']] = top_n_topics
            st.rerun()
    
    with col2:
        if st.button("Clear Selection", key='clear_button'):
            st.session_state[SESSION_KEYS['selected_topics']] = []
            st.rerun()
    
    st.sidebar.divider()
    
    # Filters
    st.sidebar.subheader("Filters")
    
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
    
    st.sidebar.divider()
    
    # Export buttons
    st.sidebar.subheader("Export")
    
    # Build digest for export
    selected_topic_ids = st.session_state.get(SESSION_KEYS['selected_topics'], [])
    if canonical_model and topic_aggregates:
        selected_aggregates = [t for t in topic_aggregates if t['topic_id'] in selected_topic_ids]
        digest_artifact = digest.build_digest(canonical_model, selected_aggregates, n_takeaways=5)
        
        html_content = export.export_to_html(digest_artifact)
        st.sidebar.download_button(
            label="📥 Export HTML",
            data=html_content,
            file_name="digest.html",
            mime="text/html",
            use_container_width=True
        )
        
        md_content = export.export_to_markdown(digest_artifact)
        st.sidebar.download_button(
            label="📄 Export Markdown",
            data=md_content,
            file_name="digest.md",
            mime="text/markdown",
            use_container_width=True
        )
    
    return canonical_model, topic_aggregates, validation_report


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
    
    # Validation warnings
    warnings = validation_report.get('warnings', [])
    if warnings:
        with st.expander("⚠️ Validation Warnings", expanded=False):
            for warning in warnings:
                st.warning(warning)


def render_takeaways(digest_artifact: Dict[str, Any]):
    """Render takeaways row."""
    st.subheader("Key Takeaways")
    takeaways = digest_artifact.get('takeaways', [])
    
    if not takeaways:
        st.info("No takeaways available. Select topics to generate takeaways.")
        return
    
    for takeaway in takeaways:
        takeaway_index = takeaway.get('takeaway_index', 0)
        takeaway_text = render.format_takeaway_text(takeaway.get('takeaway_text', ''))
        source_topic_id = takeaway.get('source_topic_id', '')
        
        # Find corresponding topic card
        topic_cards = digest_artifact.get('topic_cards', [])
        topic_card = next((tc for tc in topic_cards if tc['topic_id'] == source_topic_id), None)
        
        col1, col2 = st.columns([3, 1])
        
        with col1:
            st.write(f"**{takeaway_index}.** {takeaway_text}")
            if source_topic_id:
                st.caption(f"From: {source_topic_id}")
        
        with col2:
            if topic_card:
                evidence_count = topic_card.get('evidence_count', 0)
                st.metric("Evidence", evidence_count)
                
                proof_quote_preview = render.format_quote_preview(topic_card.get('proof_quote_preview', ''))
                if proof_quote_preview:
                    with st.expander("Proof Quote"):
                        st.write(proof_quote_preview)
                        # Show receipt links
                        receipt_links = topic_card.get('receipt_links', [])
                        if receipt_links:
                            st.caption(f"Receipts: {', '.join(receipt_links[:5])}")


def render_topic_cards(digest_artifact: Dict[str, Any], canonical_model):
    """Render topic cards row."""
    st.subheader("Topic Cards")
    topic_cards = digest_artifact.get('topic_cards', [])
    
    if not topic_cards:
        st.info("No topic cards available. Select topics to view details.")
        return
    
    for card in topic_cards:
        topic_id = card.get('topic_id', '')
        topic_oneliner = render.format_topic_oneliner(card.get('topic_one_liner', ''))
        coverage_rate = card.get('coverage_rate', 0.0)
        evidence_count = card.get('evidence_count', 0)
        sentiment_mix = card.get('sentiment_mix', {})
        proof_quote_preview = render.format_quote_preview(card.get('proof_quote_preview', ''))
        receipt_links = card.get('receipt_links', [])
        
        with st.container():
            st.markdown(f"### {topic_id}")
            
            if topic_oneliner:
                st.write(topic_oneliner)
            
            # Coverage and evidence
            col1, col2 = st.columns(2)
            with col1:
                st.markdown(render.format_coverage_bar_html(coverage_rate))
            with col2:
                st.metric("Evidence Count", evidence_count)
            
            # Sentiment mix
            st.markdown(render.format_sentiment_mix_html(sentiment_mix), unsafe_allow_html=True)
            
            # Proof quote
            if proof_quote_preview:
                with st.expander("Proof Quote"):
                    st.write(proof_quote_preview)
            
            # Receipts expander
            if receipt_links:
                with st.expander(f"Receipts ({len(receipt_links)})"):
                    # Group by participant
                    participant_receipts = {}
                    for receipt_ref in receipt_links:
                        if ':' in receipt_ref:
                            participant_id, quote_index = receipt_ref.split(':', 1)
                            if participant_id not in participant_receipts:
                                participant_receipts[participant_id] = []
                            participant_receipts[participant_id].append(quote_index)
                    
                    for participant_id, quote_indices in participant_receipts.items():
                        st.markdown(f"**Participant:** {participant_id}")
                        
                        # Get evidence cell for this participant and topic
                        for evidence_cell in canonical_model.evidence_cells:
                            if evidence_cell.participant_id == participant_id and evidence_cell.topic_id == topic_id:
                                if evidence_cell.summary_text:
                                    st.write(f"*Summary:* {evidence_cell.summary_text}")
                                
                                if evidence_cell.quotes_raw:
                                    quote_blocks = parse_quotes.parse_quotes(evidence_cell.quotes_raw)
                                    sentiment_blocks = []
                                    if evidence_cell.sentiments_raw:
                                        sentiment_blocks = parse_sentiment.parse_and_align_sentiments(
                                            evidence_cell.sentiments_raw, quote_blocks
                                        )
                                    
                                    for quote_block in quote_blocks:
                                        quote_index = quote_block.get('quote_index', 0)
                                        # Match both string and int representations
                                        if str(quote_index) in quote_indices or quote_index in [int(qi) for qi in quote_indices if qi.isdigit()]:
                                            st.markdown(f"**Quote {quote_index}:**")
                                            st.write(quote_block.get('quote_text', ''))
                                            
                                            # Find sentiment
                                            sentiment_block = next(
                                                (sb for sb in sentiment_blocks if sb['quote_index'] == quote_index),
                                                None
                                            )
                                            if sentiment_block:
                                                tone = sentiment_block.get('tone_rollup', 'unknown')
                                                st.markdown(render.format_sentiment_mix_html({tone: 1}), unsafe_allow_html=True)
                        
                        st.divider()


def render_explore_tab(topic_aggregates: List[Dict[str, Any]], canonical_model):
    """Render Explore tab with table view."""
    if not topic_aggregates:
        st.info("Upload a file and compute scores to view data.")
        return
    
    # Build dataframe
    df_data = []
    for topic_agg in topic_aggregates:
        topic_id = topic_agg['topic_id']
        sentiment_mix = digest._compute_sentiment_mix(canonical_model.evidence_cells, topic_id)
        total_sentiment = sum(sentiment_mix.values())
        dominant_tone = 'unknown'
        if total_sentiment > 0:
            dominant_tone = max(sentiment_mix.items(), key=lambda x: x[1])[0]
        
        df_data.append({
            'Topic ID': topic_id,
            'Score': round(topic_agg.get('topic_score', 0), 3),
            'Coverage %': round(topic_agg.get('coverage_rate', 0) * 100, 1),
            'Evidence': topic_agg.get('evidence_count', 0),
            'Intensity': round(topic_agg.get('intensity_rate', 0), 2),
            'Tone': dominant_tone,
            'One-liner': render.format_topic_oneliner(topic_agg.get('topic_one_liner', ''))
        })
    
    df = pd.DataFrame(df_data)
    st.dataframe(df, use_container_width=True, hide_index=True)


def main():
    """Main Streamlit application."""
    st.set_page_config(
        page_title="5-Minute Digest",
        page_icon="📊",
        layout="wide"
    )
    
    st.title("📊 5-Minute Digest")
    
    # Initialize session state
    initialize_session_state()
    
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
        render_takeaways(digest_artifact)
        
        st.divider()
        
        # Row 3: Topic Cards
        render_topic_cards(digest_artifact, canonical_model)
    
    with tab2:
        render_explore_tab(topic_aggregates, canonical_model)


if __name__ == "__main__":
    main()
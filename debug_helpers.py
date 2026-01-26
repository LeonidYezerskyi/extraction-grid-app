"""Debugging helpers for manual QA and troubleshooting."""

import sys
import os

# Add current directory to path
if os.path.dirname(os.path.abspath(__file__)) not in sys.path:
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import ingest
import normalize
import score
import digest
import render
import export


def print_validation_report(validation_report: dict):
    """Print validation report in readable format."""
    print("\n" + "=" * 60)
    print("VALIDATION REPORT")
    print("=" * 60)
    
    print(f"\nReadable: {validation_report.get('is_readable', 'Unknown')}")
    print(f"Valid: {validation_report.get('is_valid', 'Unknown')}")
    
    if validation_report.get('error'):
        print(f"\n❌ ERROR: {validation_report['error']}")
    
    print(f"\nMatched Sheets:")
    for role, sheet_name in validation_report.get('matched_sheets', {}).items():
        print(f"  - {role}: {sheet_name}")
    
    missing = validation_report.get('missing_sheets', [])
    if missing:
        print(f"\n⚠️  Missing Sheets: {', '.join(missing)}")
    
    unmatched = validation_report.get('unmatched_sheets', [])
    if unmatched:
        print(f"\nUnmatched Sheets: {', '.join(unmatched)}")
    
    topic_columns = validation_report.get('topic_columns', set())
    print(f"\nTopic Columns ({len(topic_columns)}):")
    for col in sorted(topic_columns):
        print(f"  - {col}")
    
    warnings = validation_report.get('warnings', [])
    if warnings:
        print(f"\n⚠️  Warnings ({len(warnings)}):")
        for warning in warnings:
            print(f"  - {warning}")
    
    coverage_stats = validation_report.get('coverage_stats', {})
    if coverage_stats:
        print(f"\nCoverage Stats:")
        for role, stats in coverage_stats.items():
            print(f"  {role}:")
            print(f"    - Total columns: {stats.get('total_columns', 0)}")
            print(f"    - Metadata columns: {stats.get('metadata_columns', 0)}")
            print(f"    - Topic columns: {stats.get('topic_columns', 0)}")
            print(f"    - Rows: {stats.get('rows', 0)}")
    
    print("=" * 60)


def print_canonical_model_summary(canonical_model):
    """Print canonical model summary."""
    print("\n" + "=" * 60)
    print("CANONICAL MODEL SUMMARY")
    print("=" * 60)
    
    print(f"\nParticipants: {len(canonical_model.participants)}")
    for i, p in enumerate(canonical_model.participants[:5], 1):
        print(f"  {i}. {p.participant_id} ({p.participant_label})")
    if len(canonical_model.participants) > 5:
        print(f"  ... and {len(canonical_model.participants) - 5} more")
    
    print(f"\nTopics: {len(canonical_model.topics)}")
    for i, t in enumerate(canonical_model.topics[:10], 1):
        print(f"  {i}. {t.topic_id}")
    if len(canonical_model.topics) > 10:
        print(f"  ... and {len(canonical_model.topics) - 10} more")
    
    print(f"\nEvidence Cells: {len(canonical_model.evidence_cells)}")
    
    # Sample evidence cells
    print(f"\nSample Evidence Cells (first 3):")
    for i, ec in enumerate(canonical_model.evidence_cells[:3], 1):
        has_summary = bool(ec.summary_text)
        has_quotes = bool(ec.quotes_raw)
        has_sentiments = bool(ec.sentiments_raw)
        print(f"  {i}. {ec.participant_id} / {ec.topic_id}")
        print(f"     Summary: {'✓' if has_summary else '✗'}, "
              f"Quotes: {'✓' if has_quotes else '✗'}, "
              f"Sentiments: {'✓' if has_sentiments else '✗'}")
    
    print("=" * 60)


def debug_run(file_path: str):
    """
    Run full pipeline (ingest->normalize->score->digest) and print debug info.
    
    Args:
        file_path: Path to Excel workbook file
    """
    print(f"\n{'=' * 60}")
    print(f"DEBUG RUN: {file_path}")
    print(f"{'=' * 60}\n")
    
    try:
        # Read file
        with open(file_path, 'rb') as f:
            excel_bytes = f.read()
        
        print("✓ File read successfully")
        
        # Ingest
        print("\n[1/4] Ingesting workbook...")
        dict_of_dfs, validation_report = ingest.read_workbook(excel_bytes)
        print_validation_report(validation_report)
        
        if not validation_report.get('is_valid', False):
            print("\n❌ Validation failed. Stopping.")
            return
        
        # Normalize
        print("\n[2/4] Normalizing to canonical model...")
        topic_columns = list(validation_report.get('topic_columns', []))
        canonical_model = normalize.wide_to_canonical(dict_of_dfs, topic_columns)
        print_canonical_model_summary(canonical_model)
        
        # Score
        print("\n[3/4] Computing topic aggregates and scores...")
        topic_aggregates = score.compute_topic_aggregates(canonical_model)
        print(f"\n✓ Computed {len(topic_aggregates)} topic aggregates")
        
        print(f"\nTop Topics (by score):")
        for i, agg in enumerate(topic_aggregates[:10], 1):
            print(f"  {i}. {agg['topic_id']}: "
                  f"score={agg['topic_score']:.3f}, "
                  f"coverage={agg['coverage_rate']:.2%}, "
                  f"evidence={agg['evidence_count']}")
        
        # Digest
        print("\n[4/4] Building digest...")
        top_n = 5
        selected_aggregates = topic_aggregates[:top_n]
        digest_artifact = digest.build_digest(canonical_model, selected_aggregates, n_takeaways=5)
        
        print(f"\n✓ Digest built successfully")
        print(f"\nDigest Summary:")
        print(f"  - Takeaways: {len(digest_artifact.get('takeaways', []))}")
        print(f"  - Topic Cards: {len(digest_artifact.get('topic_cards', []))}")
        
        print(f"\nTakeaways:")
        for takeaway in digest_artifact.get('takeaways', [])[:5]:
            text = render.format_takeaway_text(takeaway.get('takeaway_text', ''))
            print(f"  {takeaway.get('takeaway_index')}. {text[:80]}...")
        
        print(f"\nTopic Cards (first 3):")
        for card in digest_artifact.get('topic_cards', [])[:3]:
            topic_id = card.get('topic_id', '')
            evidence_count = card.get('evidence_count', 0)
            coverage = card.get('coverage_rate', 0.0)
            receipt_count = len(card.get('receipt_links', []))
            print(f"  - {topic_id}: {evidence_count} quotes, "
                  f"{coverage:.1%} coverage, {receipt_count} receipts")
        
        # Export check
        print(f"\n[Export Check]")
        html_content = export.export_to_html(digest_artifact)
        md_content = export.export_to_markdown(digest_artifact)
        print(f"  - HTML export: {len(html_content)} chars")
        print(f"  - Markdown export: {len(md_content)} chars")
        
        # Check if exports contain selected topics
        selected_topic_ids = {agg['topic_id'] for agg in selected_aggregates}
        html_has_all = all(tid in html_content for tid in selected_topic_ids)
        md_has_all = all(tid in md_content for tid in selected_topic_ids)
        
        print(f"  - HTML contains all topics: {'✓' if html_has_all else '✗'}")
        print(f"  - Markdown contains all topics: {'✓' if md_has_all else '✗'}")
        
        print(f"\n{'=' * 60}")
        print("✓ Debug run completed successfully")
        print(f"{'=' * 60}\n")
        
    except FileNotFoundError:
        print(f"\n❌ Error: File not found: {file_path}")
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()


def quick_check(file_path: str):
    """
    Quick check: just validate and show top topics.
    
    Args:
        file_path: Path to Excel workbook file
    """
    print(f"\nQuick Check: {file_path}\n")
    
    try:
        with open(file_path, 'rb') as f:
            excel_bytes = f.read()
        
        dict_of_dfs, validation_report = ingest.read_workbook(excel_bytes)
        
        if validation_report.get('is_valid'):
            topic_columns = list(validation_report.get('topic_columns', []))
            canonical_model = normalize.wide_to_canonical(dict_of_dfs, topic_columns)
            aggregates = score.compute_topic_aggregates(canonical_model)
            
            print(f"✓ Valid workbook")
            print(f"  Topics: {len(aggregates)}")
            print(f"  Participants: {len(canonical_model.participants)}")
            print(f"\nTop 5 Topics:")
            for i, agg in enumerate(aggregates[:5], 1):
                print(f"  {i}. {agg['topic_id']} (score: {agg['topic_score']:.3f})")
        else:
            print(f"✗ Invalid workbook: {validation_report.get('error', 'Unknown error')}")
    
    except Exception as e:
        print(f"✗ Error: {str(e)}")


if __name__ == '__main__':
    import sys
    
    if len(sys.argv) > 1:
        file_path = sys.argv[1]
        if '--quick' in sys.argv:
            quick_check(file_path)
        else:
            debug_run(file_path)
    else:
        print("Usage:")
        print("  python debug_helpers.py <file_path>          # Full debug run")
        print("  python debug_helpers.py <file_path> --quick  # Quick check")

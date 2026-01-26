"""Acceptance test script for verifying acceptance criteria."""

import sys
import os

# Add parent directory to path
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

import ingest
import normalize
import score
import digest
import render
import export
import pandas as pd
import io
from typing import Dict, List, Any


def create_example_workbook() -> bytes:
    """
    Create an example workbook in the expected shape for acceptance testing.
    
    Returns:
        Bytes of Excel workbook
    """
    # Create sample data
    summary_df = pd.DataFrame({
        'participant_id': ['p1', 'p2', 'p3', 'p4', 'p5'],
        'topic_innovation': [
            'Innovation summary from p1',
            'Innovation summary from p2',
            'Innovation summary from p3',
            'Innovation summary from p4',
            'Innovation summary from p5'
        ],
        'topic_collaboration': [
            'Collaboration summary from p1',
            'Collaboration summary from p2',
            'Collaboration summary from p3',
            None,
            None
        ],
        'topic_leadership': [
            'Leadership summary from p1',
            'Leadership summary from p2',
            None,
            None,
            None
        ]
    })
    
    quotes_df = pd.DataFrame({
        'participant_id': ['p1', 'p2', 'p3', 'p4', 'p5'],
        'topic_innovation': [
            '1. First innovation quote. 2. Second innovation quote.',
            '1. Innovation quote from p2.',
            '1. Innovation quote from p3.',
            '1. Innovation quote from p4.',
            '1. Innovation quote from p5.'
        ],
        'topic_collaboration': [
            '1. Collaboration quote from p1.',
            '1. Collaboration quote from p2.',
            '1. Collaboration quote from p3.',
            None,
            None
        ],
        'topic_leadership': [
            '1. Leadership quote from p1.',
            '1. Leadership quote from p2.',
            None,
            None,
            None
        ]
    })
    
    sentiments_df = pd.DataFrame({
        'participant_id': ['p1', 'p2', 'p3', 'p4', 'p5'],
        'topic_innovation': [
            '1: positive; 2: positive',
            '1: positive',
            '1: positive',
            '1: positive',
            '1: positive'
        ],
        'topic_collaboration': [
            '1: positive',
            '1: positive',
            '1: negative',
            None,
            None
        ],
        'topic_leadership': [
            '1: positive',
            '1: neutral',
            None,
            None,
            None
        ]
    })
    
    # Create Excel file
    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
        summary_df.to_excel(writer, sheet_name='summary', index=False)
        quotes_df.to_excel(writer, sheet_name='quotes', index=False)
        sentiments_df.to_excel(writer, sheet_name='sentiments', index=False)
    buffer.seek(0)
    return buffer.getvalue()


def check_deterministic_ranking() -> tuple[bool, str]:
    """
    Check that top N ranking is deterministic (same order on multiple runs).
    
    Returns:
        Tuple of (passed, message)
    """
    try:
        workbook_bytes = create_example_workbook()
        dict_of_dfs, validation_report = ingest.read_workbook(workbook_bytes)
        topic_columns = list(validation_report.get('topic_columns', []))
        canonical_model = normalize.wide_to_canonical(dict_of_dfs, topic_columns)
        
        # Run twice
        aggregates_1 = score.compute_topic_aggregates(canonical_model)
        aggregates_2 = score.compute_topic_aggregates(canonical_model)
        
        # Check order is identical
        topic_ids_1 = [a['topic_id'] for a in aggregates_1]
        topic_ids_2 = [a['topic_id'] for a in aggregates_2]
        
        if topic_ids_1 == topic_ids_2:
            return True, "Ranking is deterministic"
        else:
            return False, f"Ranking differs: {topic_ids_1} vs {topic_ids_2}"
    except Exception as e:
        return False, f"Error checking ranking: {str(e)}"


def check_topic_selection_changes_digest() -> tuple[bool, str]:
    """
    Check that selecting different topics changes digest output.
    
    Returns:
        Tuple of (passed, message)
    """
    try:
        workbook_bytes = create_example_workbook()
        dict_of_dfs, validation_report = ingest.read_workbook(workbook_bytes)
        topic_columns = list(validation_report.get('topic_columns', []))
        canonical_model = normalize.wide_to_canonical(dict_of_dfs, topic_columns)
        aggregates = score.compute_topic_aggregates(canonical_model)
        
        # Select first topic
        selected_1 = [aggregates[0]]
        digest_1 = digest.build_digest(canonical_model, selected_1, n_takeaways=1)
        
        # Select different topic
        if len(aggregates) > 1:
            selected_2 = [aggregates[1]]
            digest_2 = digest.build_digest(canonical_model, selected_2, n_takeaways=1)
            
            # Digests should differ
            topic_ids_1 = [tc['topic_id'] for tc in digest_1['topic_cards']]
            topic_ids_2 = [tc['topic_id'] for tc in digest_2['topic_cards']]
            
            if topic_ids_1 != topic_ids_2:
                return True, "Digest changes with topic selection"
            else:
                return False, "Digest does not change with different topic selection"
        else:
            return False, "Not enough topics to test selection change"
    except Exception as e:
        return False, f"Error checking topic selection: {str(e)}"


def check_digest_skimmable() -> tuple[bool, str]:
    """
    Check that digest is skimmable (truncated fields present, receipts exist).
    
    Returns:
        Tuple of (passed, message)
    """
    try:
        workbook_bytes = create_example_workbook()
        dict_of_dfs, validation_report = ingest.read_workbook(workbook_bytes)
        topic_columns = list(validation_report.get('topic_columns', []))
        canonical_model = normalize.wide_to_canonical(dict_of_dfs, topic_columns)
        aggregates = score.compute_topic_aggregates(canonical_model)
        
        # Build digest
        digest_artifact = digest.build_digest(canonical_model, aggregates[:3], n_takeaways=3)
        
        issues = []
        
        # Check takeaways are truncated
        for takeaway in digest_artifact.get('takeaways', []):
            takeaway_text = takeaway.get('takeaway_text', '')
            if takeaway_text:
                truncated = render.format_takeaway_text(takeaway_text)
                if len(truncated) > render.TAKEAWAY_MAX:
                    issues.append(f"Takeaway exceeds {render.TAKEAWAY_MAX} chars")
        
        # Check topic cards have truncated one-liners
        for card in digest_artifact.get('topic_cards', []):
            one_liner = card.get('topic_one_liner', '')
            if one_liner:
                truncated = render.format_topic_oneliner(one_liner)
                if len(truncated) > render.TOPIC_ONELINER_MAX:
                    issues.append(f"One-liner exceeds {render.TOPIC_ONELINER_MAX} chars")
            
            # Check proof quote preview is truncated
            proof_preview = card.get('proof_quote_preview', '')
            if proof_preview:
                truncated = render.format_quote_preview(proof_preview)
                if len(truncated) > render.QUOTE_PREVIEW_MAX:
                    issues.append(f"Quote preview exceeds {render.QUOTE_PREVIEW_MAX} chars")
            
            # Check receipts exist
            receipt_links = card.get('receipt_links', [])
            if not receipt_links:
                issues.append(f"No receipts for topic {card.get('topic_id')}")
        
        if issues:
            return False, f"Issues found: {', '.join(issues)}"
        else:
            return True, "Digest is skimmable (truncated fields and receipts present)"
    except Exception as e:
        return False, f"Error checking skimmability: {str(e)}"


def check_export_reflects_selection() -> tuple[bool, str]:
    """
    Check that export (HTML/MD) reflects current topic selection.
    
    Returns:
        Tuple of (passed, message)
    """
    try:
        workbook_bytes = create_example_workbook()
        dict_of_dfs, validation_report = ingest.read_workbook(workbook_bytes)
        topic_columns = list(validation_report.get('topic_columns', []))
        canonical_model = normalize.wide_to_canonical(dict_of_dfs, topic_columns)
        aggregates = score.compute_topic_aggregates(canonical_model)
        
        # Select specific topics
        selected_topics = aggregates[:2]  # Select first 2
        selected_topic_ids = {a['topic_id'] for a in selected_topics}
        
        digest_artifact = digest.build_digest(canonical_model, selected_topics, n_takeaways=2)
        
        # Export to HTML
        html_content = export.export_to_html(digest_artifact)
        
        # Export to Markdown
        md_content = export.export_to_markdown(digest_artifact)
        
        issues = []
        
        # Check HTML contains selected topics
        for topic_id in selected_topic_ids:
            if topic_id not in html_content:
                issues.append(f"Topic {topic_id} missing from HTML export")
        
        # Check Markdown contains selected topics
        for topic_id in selected_topic_ids:
            if topic_id not in md_content:
                issues.append(f"Topic {topic_id} missing from Markdown export")
        
        # Check HTML contains takeaways
        if not digest_artifact.get('takeaways'):
            issues.append("No takeaways in HTML export")
        else:
            takeaway_text = digest_artifact['takeaways'][0].get('takeaway_text', '')
            if takeaway_text and takeaway_text not in html_content:
                issues.append("Takeaway text missing from HTML export")
        
        # Check Markdown contains takeaways
        if not digest_artifact.get('takeaways'):
            issues.append("No takeaways in Markdown export")
        else:
            takeaway_text = digest_artifact['takeaways'][0].get('takeaway_text', '')
            if takeaway_text and takeaway_text not in md_content:
                issues.append("Takeaway text missing from Markdown export")
        
        if issues:
            return False, f"Issues found: {', '.join(issues)}"
        else:
            return True, "Exports reflect current topic selection"
    except Exception as e:
        return False, f"Error checking exports: {str(e)}"


def check_top_n_stable_ordering() -> tuple[bool, str]:
    """
    Check that Top N maintains stable ordering.
    
    Returns:
        Tuple of (passed, message)
    """
    try:
        workbook_bytes = create_example_workbook()
        dict_of_dfs, validation_report = ingest.read_workbook(workbook_bytes)
        topic_columns = list(validation_report.get('topic_columns', []))
        canonical_model = normalize.wide_to_canonical(dict_of_dfs, topic_columns)
        aggregates = score.compute_topic_aggregates(canonical_model)
        
        # Get top N (e.g., top 3)
        top_n = 3
        top_topics_1 = [a['topic_id'] for a in aggregates[:top_n]]
        
        # Run again
        aggregates_2 = score.compute_topic_aggregates(canonical_model)
        top_topics_2 = [a['topic_id'] for a in aggregates_2[:top_n]]
        
        if top_topics_1 == top_topics_2:
            # Check scores are also stable
            scores_1 = [a['topic_score'] for a in aggregates[:top_n]]
            scores_2 = [a['topic_score'] for a in aggregates_2[:top_n]]
            
            if scores_1 == scores_2:
                return True, f"Top {top_n} maintains stable ordering"
            else:
                return False, "Top N scores differ between runs"
        else:
            return False, f"Top {top_n} ordering differs: {top_topics_1} vs {top_topics_2}"
    except Exception as e:
        return False, f"Error checking Top N ordering: {str(e)}"


def check_basic_pipeline_functionality() -> tuple[bool, str]:
    """
    Check that basic pipeline (ingest -> normalize -> score -> digest) works.
    
    Returns:
        Tuple of (passed, message)
    """
    try:
        workbook_bytes = create_example_workbook()
        
        # Ingest
        dict_of_dfs, validation_report = ingest.read_workbook(workbook_bytes)
        if not validation_report.get('is_valid', False):
            return False, f"Workbook validation failed: {validation_report.get('error', 'Unknown error')}"
        
        # Normalize
        topic_columns = list(validation_report.get('topic_columns', []))
        if not topic_columns:
            return False, "No topic columns identified"
        
        canonical_model = normalize.wide_to_canonical(dict_of_dfs, topic_columns)
        if not canonical_model.participants:
            return False, "No participants in canonical model"
        if not canonical_model.topics:
            return False, "No topics in canonical model"
        
        # Score
        aggregates = score.compute_topic_aggregates(canonical_model)
        if not aggregates:
            return False, "No topic aggregates computed"
        
        # Digest
        digest_artifact = digest.build_digest(canonical_model, aggregates[:2], n_takeaways=2)
        if not digest_artifact.get('topic_cards'):
            return False, "No topic cards in digest"
        
        return True, "Basic pipeline functionality works"
    except Exception as e:
        return False, f"Error in basic pipeline: {str(e)}"


def run_acceptance_checks() -> Dict[str, Dict[str, Any]]:
    """
    Run all acceptance checks and return a report.
    
    Returns:
        Dictionary with pass/fail status for each criterion
    """
    print("Running acceptance checks...")
    print("=" * 60)
    
    report = {}
    
    # Check 0: Basic pipeline functionality
    print("\n[0/6] Checking basic pipeline functionality...")
    passed, message = check_basic_pipeline_functionality()
    report['basic_pipeline'] = {
        'passed': passed,
        'message': message
    }
    print(f"  {'✓ PASS' if passed else '✗ FAIL'}: {message}")
    
    if not passed:
        print("\n⚠️  Basic pipeline failed. Skipping other checks.")
        return report
    
    # Check 1: Deterministic ranking
    print("\n[1/6] Checking deterministic ranking...")
    passed, message = check_deterministic_ranking()
    report['deterministic_ranking'] = {
        'passed': passed,
        'message': message
    }
    print(f"  {'✓ PASS' if passed else '✗ FAIL'}: {message}")
    
    # Check 2: Top N stable ordering
    print("\n[2/6] Checking Top N stable ordering...")
    passed, message = check_top_n_stable_ordering()
    report['top_n_stable_ordering'] = {
        'passed': passed,
        'message': message
    }
    print(f"  {'✓ PASS' if passed else '✗ FAIL'}: {message}")
    
    # Check 3: Topic selection changes digest
    print("\n[3/6] Checking topic selection changes digest...")
    passed, message = check_topic_selection_changes_digest()
    report['topic_selection_changes_digest'] = {
        'passed': passed,
        'message': message
    }
    print(f"  {'✓ PASS' if passed else '✗ FAIL'}: {message}")
    
    # Check 4: Digest is skimmable
    print("\n[4/6] Checking digest is skimmable...")
    passed, message = check_digest_skimmable()
    report['digest_skimmable'] = {
        'passed': passed,
        'message': message
    }
    print(f"  {'✓ PASS' if passed else '✗ FAIL'}: {message}")
    
    # Check 5: Export reflects selection
    print("\n[5/6] Checking export reflects selection...")
    passed, message = check_export_reflects_selection()
    report['export_reflects_selection'] = {
        'passed': passed,
        'message': message
    }
    print(f"  {'✓ PASS' if passed else '✗ FAIL'}: {message}")
    
    # Summary
    print("\n" + "=" * 60)
    total_checks = len(report)
    passed_checks = sum(1 for r in report.values() if r['passed'])
    print(f"\nSummary: {passed_checks}/{total_checks} checks passed")
    
    # Detailed report
    report['summary'] = {
        'total_checks': total_checks,
        'passed_checks': passed_checks,
        'failed_checks': total_checks - passed_checks,
        'all_passed': passed_checks == total_checks
    }
    
    if passed_checks == total_checks:
        print("✓ All acceptance criteria met!")
    else:
        print("✗ Some acceptance criteria failed")
        print("\nFailed checks:")
        for check_name, result in report.items():
            if not result['passed']:
                print(f"  - {check_name}: {result['message']}")
    
    return report


if __name__ == '__main__':
    report = run_acceptance_checks()
    
    # Exit with appropriate code
    all_passed = all(r['passed'] for r in report.values())
    sys.exit(0 if all_passed else 1)

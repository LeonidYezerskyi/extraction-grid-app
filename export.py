"""Module for exporting processed data to various formats."""

from typing import Dict, List, Any
import render


def _escape_html(text: str) -> str:
    """
    Escape HTML special characters.
    
    Args:
        text: Text to escape
    
    Returns:
        Escaped text safe for HTML
    """
    if not text:
        return ''
    
    return (text
            .replace('&', '&amp;')
            .replace('<', '&lt;')
            .replace('>', '&gt;')
            .replace('"', '&quot;')
            .replace("'", '&#39;'))


def _format_sentiment_mix_for_html(sentiment_mix: Dict[str, int]) -> str:
    """
    Format sentiment mix as HTML chips for export.
    
    Args:
        sentiment_mix: Dictionary with sentiment counts
    
    Returns:
        HTML string with sentiment chips
    """
    return render.format_sentiment_mix_html(sentiment_mix)


def _format_coverage_for_html(coverage_rate: float) -> str:
    """
    Format coverage rate as HTML bar for export.
    
    Args:
        coverage_rate: Coverage rate (0.0 to 1.0)
    
    Returns:
        HTML string with coverage bar
    """
    return render.format_coverage_bar_html(coverage_rate)


def export_to_html(digest_artifact: Dict[str, Any], canonical_model=None) -> str:
    """
    Export digest artifact to self-contained HTML.
    
    Produces a complete HTML document with:
    - Takeaways section
    - Topic cards with coverage, sentiment mix, proof quotes
    - Full receipts with quotes
    
    Args:
        digest_artifact: Digest artifact from build_digest()
        canonical_model: Optional CanonicalModel for building full receipts with quotes
    
    Returns:
        Complete self-contained HTML string with inline styles
    
    Example:
        >>> digest = {'takeaways': [], 'topic_cards': [], 'metadata': {}}
        >>> html = export_to_html(digest)
        >>> '<html>' in html.lower()
        True
    """
    takeaways = digest_artifact.get('takeaways', [])
    topic_cards = digest_artifact.get('topic_cards', [])
    metadata = digest_artifact.get('metadata', {})
    
    # Build HTML
    html_parts = []
    
    # HTML header with minimal inline styles
    html_parts.append('''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>5-Minute Digest</title>
    <style>
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif;
            line-height: 1.6;
            color: #333;
            max-width: 900px;
            margin: 0 auto;
            padding: 20px;
            background-color: #f9fafb;
        }
        h1 {
            color: #1f2937;
            border-bottom: 3px solid #655CFE;
            padding-bottom: 10px;
            margin-bottom: 30px;
        }
        h2 {
            color: #374151;
            margin-top: 40px;
            margin-bottom: 20px;
            font-size: 1.5em;
        }
        .takeaway {
            background-color: white;
            border-left: 4px solid #655CFE;
            padding: 15px;
            margin-bottom: 15px;
            border-radius: 4px;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        }
        .takeaway-index {
            font-weight: bold;
            color: #655CFE;
            margin-right: 10px;
        }
        .topic-card {
            background-color: white;
            padding: 20px;
            margin-bottom: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            border: 1px solid #e5e7eb;
        }
        .topic-header {
            display: flex;
            justify-content: space-between;
            align-items: start;
            margin-bottom: 15px;
        }
        .topic-id {
            font-size: 0.9em;
            color: #6b7280;
            font-weight: normal;
        }
        .topic-oneliner {
            font-size: 1.1em;
            color: #1f2937;
            margin-bottom: 15px;
            font-weight: 500;
        }
        .proof-quote {
            background-color: #f3f4f6;
            border-left: 3px solid #10b981;
            padding: 12px;
            margin-top: 15px;
            border-radius: 4px;
            font-style: italic;
            color: #374151;
        }
        .receipt-links {
            margin-top: 15px;
            padding-top: 15px;
            border-top: 1px solid #e5e7eb;
        }
        .receipt-link {
            display: inline-block;
            margin: 5px 10px 5px 0;
            padding: 4px 8px;
            background-color: rgba(101, 92, 254, 0.1);
            color: #655CFE;
            text-decoration: none;
            border-radius: 4px;
            font-size: 0.85em;
        }
        .receipt-link:hover {
            background-color: rgba(101, 92, 254, 0.2);
        }
        .metadata {
            margin-top: 40px;
            padding: 15px;
            background-color: #f3f4f6;
            border-radius: 4px;
            font-size: 0.9em;
            color: #6b7280;
        }
    </style>
</head>
<body>
    <h1>5-Minute Digest</h1>
''')
    
    # Takeaways section
    if takeaways:
        html_parts.append('    <h2>Key Takeaways</h2>\n')
        for takeaway in takeaways:
            takeaway_text = render.format_takeaway_text(takeaway.get('takeaway_text', ''))
            takeaway_index = takeaway.get('takeaway_index', 0)
            source_topic_id = takeaway.get('source_topic_id', '')
            evidence_count = takeaway.get('evidence_count', 0)
            
            html_parts.append(f'    <div class="takeaway">\n')
            html_parts.append(f'        <span class="takeaway-index">{takeaway_index}.</span>')
            html_parts.append(f'        <span>{_escape_html(takeaway_text)}</span>')
            if source_topic_id:
                html_parts.append(f'        <span class="topic-id"> (from {_escape_html(source_topic_id)})</span>')
            if evidence_count > 0:
                html_parts.append(f'        <span style="margin-left: 10px; color: #6b7280; font-size: 0.9em;">({evidence_count} supporting excerpts)</span>')
            html_parts.append(f'    </div>\n')
    
    # Topic cards section
    if topic_cards:
        html_parts.append('    <h2>Topic Details</h2>\n')
        for card in topic_cards:
            topic_id = card.get('topic_id', '')
            topic_oneliner = render.format_topic_oneliner(card.get('topic_one_liner', ''))
            coverage_rate = card.get('coverage_rate', 0.0)
            evidence_count = card.get('evidence_count', 0)
            sentiment_mix = card.get('sentiment_mix', {})
            proof_quote_preview = render.format_quote_preview(card.get('proof_quote_preview', ''))
            proof_quote_ref = card.get('proof_quote_ref', '')
            receipt_links = card.get('receipt_links', [])
            
            html_parts.append(f'    <div class="topic-card" id="topic-{_escape_html(topic_id)}">\n')
            html_parts.append(f'        <div class="topic-header">\n')
            html_parts.append(f'            <h3>{_escape_html(topic_id)} <span class="topic-id">(Topic)</span></h3>\n')
            html_parts.append(f'        </div>\n')
            
            if topic_oneliner:
                html_parts.append(f'        <div class="topic-oneliner">{_escape_html(topic_oneliner)}</div>\n')
            
            # Coverage and evidence count
            html_parts.append(f'        <div style="margin-bottom: 10px;">\n')
            html_parts.append(f'            <strong>Coverage:</strong> {_format_coverage_for_html(coverage_rate)}\n')
            html_parts.append(f'            <span style="margin-left: 15px;"><strong>Evidence:</strong> {evidence_count} quotes</span>\n')
            html_parts.append(f'        </div>\n')
            
            # Sentiment mix
            html_parts.append(f'        <div style="margin-bottom: 10px;">\n')
            html_parts.append(f'            <strong>Sentiment:</strong> {_format_sentiment_mix_for_html(sentiment_mix)}\n')
            html_parts.append(f'        </div>\n')
            
            # Proof quote
            if proof_quote_preview:
                html_parts.append(f'        <div class="proof-quote">\n')
                html_parts.append(f'            <strong>Proof Quote:</strong> {_escape_html(proof_quote_preview)}')
                if proof_quote_ref:
                    html_parts.append(f' <a href="#receipt-{_escape_html(proof_quote_ref)}" class="receipt-link">[View Receipt]</a>')
                html_parts.append(f'\n        </div>\n')
            
            # Receipts section with full quotes
            if receipt_links:
                html_parts.append(f'        <div class="receipt-links">\n')
                html_parts.append(f'            <strong>Receipts ({len(receipt_links)} total):</strong>\n')
                html_parts.append(f'        </div>\n')
                
                # Build full receipts if canonical_model is available
                if canonical_model:
                    import render
                    receipt_displays = []
                    for receipt_ref in receipt_links:
                        receipt_display = render.build_receipt_display(
                            receipt_ref, canonical_model, topic_id=topic_id
                        )
                        if receipt_display.get('quote_full'):
                            receipt_displays.append(receipt_display)
                    
                    # Display receipts with full quotes
                    for receipt in receipt_displays:
                        participant_label = receipt.get('participant_label', 'Unknown')
                        quote_full = receipt.get('quote_full', '')
                        sentiment = receipt.get('sentiment', 'unknown')
                        
                        html_parts.append(f'        <div class="proof-quote" style="margin-top: 10px; margin-bottom: 10px;">\n')
                        html_parts.append(f'            <strong>{_escape_html(participant_label)}:</strong> ')
                        if sentiment and sentiment != 'unknown':
                            sentiment_color = {
                                'positive': '#10b981',
                                'negative': '#ef4444',
                                'neutral': '#6b7280',
                                'mixed': '#f59e0b'
                            }.get(sentiment, '#6b7280')
                            html_parts.append(f'<span style="background-color: {sentiment_color}20; color: {sentiment_color}; padding: 2px 6px; border-radius: 3px; font-size: 0.85em; margin-right: 8px;">{_escape_html(sentiment.title())}</span>')
                        html_parts.append(f'{_escape_html(quote_full)}\n')
                        html_parts.append(f'        </div>\n')
                else:
                    # Fallback: just show receipt references as links
                    receipt_html_links = []
                    for receipt_ref in receipt_links:
                        receipt_html_links.append(
                            f'<a href="#receipt-{_escape_html(receipt_ref)}" class="receipt-link" id="receipt-{_escape_html(receipt_ref)}">{_escape_html(receipt_ref)}</a>'
                        )
                    html_parts.append(' '.join(receipt_html_links))
                    html_parts.append(f'\n')
            
            html_parts.append(f'    </div>\n')
    
    # Metadata footer
    if metadata:
        html_parts.append('    <div class="metadata">\n')
        html_parts.append(f'        <strong>Digest Metadata:</strong><br>\n')
        html_parts.append(f'        Takeaways: {metadata.get("n_takeaways", 0)} | ')
        html_parts.append(f'Topics: {metadata.get("n_topics", 0)} | ')
        html_parts.append(f'Participants: {metadata.get("total_participants", 0)} | ')
        html_parts.append(f'Evidence Cells: {metadata.get("total_evidence_cells", 0)}\n')
        html_parts.append('    </div>\n')
    
    # Close HTML
    html_parts.append('</body>\n</html>')
    
    return ''.join(html_parts)


def export_to_markdown(digest_artifact: Dict[str, Any], canonical_model=None) -> str:
    """
    Export digest artifact to Markdown format.
    
    Produces Markdown with:
    - Takeaways section
    - Topic summaries with one-liners
    - Proof quotes for each topic
    - Full receipts with quotes
    
    Args:
        digest_artifact: Digest artifact from build_digest()
        canonical_model: Optional CanonicalModel for building full receipts with quotes
    
    Returns:
        Markdown string
    
    Example:
        >>> digest = {'takeaways': [], 'topic_cards': [], 'metadata': {}}
        >>> md = export_to_markdown(digest)
        >>> '# ' in md
        True
    """
    takeaways = digest_artifact.get('takeaways', [])
    topic_cards = digest_artifact.get('topic_cards', [])
    metadata = digest_artifact.get('metadata', {})
    
    md_parts = []
    
    # Title
    md_parts.append('# 5-Minute Digest\n\n')
    
    # Takeaways section
    if takeaways:
        md_parts.append('## Key Takeaways\n\n')
        for takeaway in takeaways:
            takeaway_index = takeaway.get('takeaway_index', 0)
            takeaway_text = render.format_takeaway_text(takeaway.get('takeaway_text', ''))
            source_topic_id = takeaway.get('source_topic_id', '')
            evidence_count = takeaway.get('evidence_count', 0)
            
            md_parts.append(f'{takeaway_index}. {takeaway_text}')
            if source_topic_id:
                md_parts.append(f' _(from {source_topic_id})_')
            if evidence_count > 0:
                md_parts.append(f' _({evidence_count} supporting excerpts)_')
            md_parts.append('\n\n')
    
    # Topic summaries section
    if topic_cards:
        md_parts.append('## Topic Summaries\n\n')
        for card in topic_cards:
            topic_id = card.get('topic_id', '')
            topic_oneliner = render.format_topic_oneliner(card.get('topic_one_liner', ''))
            coverage_rate = card.get('coverage_rate', 0.0)
            evidence_count = card.get('evidence_count', 0)
            sentiment_mix = card.get('sentiment_mix', {})
            proof_quote_preview = render.format_quote_preview(card.get('proof_quote_preview', ''))
            proof_quote_ref = card.get('proof_quote_ref', '')
            
            # Topic header
            md_parts.append(f'### {topic_id}\n\n')
            
            # One-liner
            if topic_oneliner:
                md_parts.append(f'{topic_oneliner}\n\n')
            
            # Stats
            coverage_pct = round(coverage_rate * 100, 1)
            md_parts.append(f'- **Coverage:** {coverage_pct}%\n')
            md_parts.append(f'- **Evidence:** {evidence_count} quotes\n')
            
            # Sentiment mix
            sentiment_parts = []
            for sentiment, count in sentiment_mix.items():
                if count > 0:
                    sentiment_parts.append(f'{sentiment}: {count}')
            if sentiment_parts:
                md_parts.append(f'- **Sentiment:** {", ".join(sentiment_parts)}\n')
            
            md_parts.append('\n')
            
            # Proof quote
            if proof_quote_preview:
                md_parts.append(f'> **Proof Quote:** {proof_quote_preview}\n')
                if proof_quote_ref:
                    md_parts.append(f'> _Reference: {proof_quote_ref}_\n')
                md_parts.append('\n')
            
            # Receipts section with full quotes
            receipt_links = card.get('receipt_links', [])
            if receipt_links:
                md_parts.append(f'**Receipts ({len(receipt_links)} total):**\n\n')
                
                # Build full receipts if canonical_model is available
                if canonical_model:
                    import render
                    receipt_displays = []
                    for receipt_ref in receipt_links:
                        receipt_display = render.build_receipt_display(
                            receipt_ref, canonical_model, topic_id=topic_id
                        )
                        if receipt_display.get('quote_full'):
                            receipt_displays.append(receipt_display)
                    
                    # Display receipts with full quotes
                    for receipt in receipt_displays:
                        participant_label = receipt.get('participant_label', 'Unknown')
                        quote_full = receipt.get('quote_full', '')
                        sentiment = receipt.get('sentiment', 'unknown')
                        
                        sentiment_marker = ''
                        if sentiment and sentiment != 'unknown':
                            sentiment_marker = f' [{sentiment.upper()}]'
                        
                        md_parts.append(f'- **{participant_label}**{sentiment_marker}:\n')
                        md_parts.append(f'  > {quote_full}\n\n')
                else:
                    # Fallback: just show receipt references
                    for receipt_ref in receipt_links:
                        md_parts.append(f'- `{receipt_ref}`\n')
                    md_parts.append('\n')
            
            md_parts.append('---\n\n')
    
    # Metadata footer
    if metadata:
        md_parts.append('## Metadata\n\n')
        md_parts.append(f'- Takeaways: {metadata.get("n_takeaways", 0)}\n')
        md_parts.append(f'- Topics: {metadata.get("n_topics", 0)}\n')
        md_parts.append(f'- Participants: {metadata.get("total_participants", 0)}\n')
        md_parts.append(f'- Evidence Cells: {metadata.get("total_evidence_cells", 0)}\n')
    
    return ''.join(md_parts)


# Unit tests
if __name__ == '__main__':
    import unittest
    
    class TestExport(unittest.TestCase):
        
        def test_export_to_html_structure(self):
            """Test that HTML export has correct structure."""
            digest = {
                'takeaways': [],
                'topic_cards': [],
                'metadata': {}
            }
            html = export_to_html(digest)
            
            self.assertIn('<!DOCTYPE html>', html)
            self.assertIn('<html', html)
            self.assertIn('</html>', html)
            self.assertIn('5-Minute Digest', html)
        
        def test_export_to_html_takeaways(self):
            """Test HTML export includes takeaways."""
            digest = {
                'takeaways': [
                    {'takeaway_index': 1, 'takeaway_text': 'First takeaway', 'source_topic_id': 'topic1'}
                ],
                'topic_cards': [],
                'metadata': {}
            }
            html = export_to_html(digest)
            
            self.assertIn('Key Takeaways', html)
            self.assertIn('First takeaway', html)
            self.assertIn('topic1', html)
        
        def test_export_to_html_topic_cards(self):
            """Test HTML export includes topic cards."""
            digest = {
                'takeaways': [],
                'topic_cards': [
                    {
                        'topic_id': 'topic1',
                        'topic_one_liner': 'Topic summary',
                        'coverage_rate': 0.75,
                        'evidence_count': 5,
                        'sentiment_mix': {'positive': 3, 'negative': 1, 'neutral': 1},
                        'proof_quote_preview': 'Proof quote text',
                        'proof_quote_ref': 'p1:1',
                        'receipt_links': ['p1:1', 'p1:2']
                    }
                ],
                'metadata': {}
            }
            html = export_to_html(digest)
            
            self.assertIn('Topic Details', html)
            self.assertIn('topic1', html)
            self.assertIn('Topic summary', html)
            self.assertIn('Proof quote text', html)
        
        def test_export_to_html_self_contained(self):
            """Test that HTML is self-contained (no external links)."""
            digest = {
                'takeaways': [],
                'topic_cards': [],
                'metadata': {}
            }
            html = export_to_html(digest)
            
            # Should have inline styles
            self.assertIn('<style>', html)
            # Should not have external CSS links
            self.assertNotIn('<link', html)
            # Should not have external script tags
            self.assertNotIn('<script src=', html)
        
        def test_export_to_markdown_structure(self):
            """Test that Markdown export has correct structure."""
            digest = {
                'takeaways': [],
                'topic_cards': [],
                'metadata': {}
            }
            md = export_to_markdown(digest)
            
            self.assertIn('# 5-Minute Digest', md)
        
        def test_export_to_markdown_takeaways(self):
            """Test Markdown export includes takeaways."""
            digest = {
                'takeaways': [
                    {'takeaway_index': 1, 'takeaway_text': 'First takeaway', 'source_topic_id': 'topic1'}
                ],
                'topic_cards': [],
                'metadata': {}
            }
            md = export_to_markdown(digest)
            
            self.assertIn('## Key Takeaways', md)
            self.assertIn('First takeaway', md)
        
        def test_export_to_markdown_topic_summaries(self):
            """Test Markdown export includes topic summaries."""
            digest = {
                'takeaways': [],
                'topic_cards': [
                    {
                        'topic_id': 'topic1',
                        'topic_one_liner': 'Topic summary',
                        'coverage_rate': 0.75,
                        'evidence_count': 5,
                        'sentiment_mix': {'positive': 3},
                        'proof_quote_preview': 'Proof quote',
                        'proof_quote_ref': 'p1:1',
                        'receipt_links': []
                    }
                ],
                'metadata': {}
            }
            md = export_to_markdown(digest)
            
            self.assertIn('## Topic Summaries', md)
            self.assertIn('### topic1', md)
            self.assertIn('Topic summary', md)
            self.assertIn('Proof quote', md)
        
        def test_export_deterministic(self):
            """Test that exports are deterministic."""
            digest = {
                'takeaways': [
                    {'takeaway_index': 1, 'takeaway_text': 'Test', 'source_topic_id': 't1'}
                ],
                'topic_cards': [],
                'metadata': {}
            }
            
            html1 = export_to_html(digest)
            html2 = export_to_html(digest)
            self.assertEqual(html1, html2)
            
            md1 = export_to_markdown(digest)
            md2 = export_to_markdown(digest)
            self.assertEqual(md1, md2)
        
        def test_html_escaping(self):
            """Test that HTML special characters are escaped."""
            digest = {
                'takeaways': [
                    {'takeaway_index': 1, 'takeaway_text': 'Test <script>alert("xss")</script>', 'source_topic_id': ''}
                ],
                'topic_cards': [],
                'metadata': {}
            }
            html = export_to_html(digest)
            
            # Should escape HTML
            self.assertNotIn('<script>', html)
            self.assertIn('&lt;script&gt;', html)
        
        def test_markdown_proof_quotes(self):
            """Test that proof quotes are included in Markdown."""
            digest = {
                'takeaways': [],
                'topic_cards': [
                    {
                        'topic_id': 'topic1',
                        'topic_one_liner': 'Summary',
                        'coverage_rate': 0.5,
                        'evidence_count': 1,
                        'sentiment_mix': {},
                        'proof_quote_preview': 'This is a proof quote',
                        'proof_quote_ref': 'p1:1',
                        'receipt_links': []
                    }
                ],
                'metadata': {}
            }
            md = export_to_markdown(digest)
            
            self.assertIn('Proof Quote', md)
            self.assertIn('This is a proof quote', md)
    
    unittest.main()
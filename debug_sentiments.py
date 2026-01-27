"""Debug script to see what sentiments are being parsed from Excel."""
import sys
import pandas as pd
import openpyxl
from io import BytesIO

# Import our modules
import ingest
import normalize
import parse_quotes
import parse_sentiment

# Read the Excel file
# Usage: python debug_sentiments.py [path_to_excel_file]
# If no file path provided, will look for common file names
import os

if len(sys.argv) > 1:
    file_path = sys.argv[1]
else:
    # Try to find Excel file in current directory
    excel_files = [f for f in os.listdir('.') if f.endswith(('.xlsx', '.xls'))]
    if excel_files:
        file_path = excel_files[0]
        print(f"Using file: {file_path}")
    else:
        print("ERROR: No Excel file found. Please provide file path as argument.")
        print("Usage: python debug_sentiments.py <path_to_excel_file>")
        sys.exit(1)

try:
    with open(file_path, 'rb') as f:
        excel_bytes = f.read()
    
    print("=" * 80)
    print("STEP 1: Reading workbook...")
    print("=" * 80)
    
    dict_of_dfs, validation_report = ingest.read_workbook(excel_bytes)
    
    print(f"Matched sheets: {validation_report.get('matched_sheets', {})}")
    print(f"Missing sheets: {validation_report.get('missing_sheets', [])}")
    
    sentiments_df = dict_of_dfs.get('sentiments')
    if sentiments_df is None:
        print("ERROR: No sentiments sheet found!")
        sys.exit(1)
    
    print(f"\nSentiments DataFrame shape: {sentiments_df.shape}")
    print(f"Columns: {list(sentiments_df.columns)}")
    print(f"\nFirst few rows of sentiments:")
    print(sentiments_df.head(10))
    
    print("\n" + "=" * 80)
    print("STEP 2: Normalizing to canonical model...")
    print("=" * 80)
    
    topic_columns = list(validation_report.get('topic_columns', []))
    canonical_model = normalize.wide_to_canonical(dict_of_dfs, topic_columns)
    
    print(f"Total evidence cells: {len(canonical_model.evidence_cells)}")
    
    # Find cells with sentiments
    cells_with_sentiments = [ec for ec in canonical_model.evidence_cells if ec.sentiments_raw]
    print(f"Cells with sentiments_raw: {len(cells_with_sentiments)}")
    
    if cells_with_sentiments:
        print("\nSample sentiments_raw values:")
        for i, cell in enumerate(cells_with_sentiments[:5]):
            print(f"\n  Cell {i+1} (participant={cell.participant_id}, topic={cell.topic_id}):")
            print(f"    sentiments_raw: {repr(cell.sentiments_raw[:200])}")  # First 200 chars
            if cell.quotes_raw:
                print(f"    quotes_raw: {repr(cell.quotes_raw[:100])}")  # First 100 chars
    
    print("\n" + "=" * 80)
    print("STEP 3: Parsing sentiments...")
    print("=" * 80)
    
    # Test parsing on a few cells
    for i, cell in enumerate(cells_with_sentiments[:3]):
        print(f"\n--- Cell {i+1} ---")
        print(f"Participant: {cell.participant_id}, Topic: {cell.topic_id}")
        print(f"Sentiments raw: {repr(cell.sentiments_raw)}")
        
        if cell.quotes_raw:
            quote_blocks = parse_quotes.parse_quotes(cell.quotes_raw)
            print(f"Quote blocks found: {len(quote_blocks)}")
            if quote_blocks:
                print(f"  Quote indices: {[q.get('quote_index') for q in quote_blocks[:5]]}")
        else:
            quote_blocks = []
            print("No quotes_raw")
        
        sentiment_blocks = parse_sentiment.parse_and_align_sentiments(
            cell.sentiments_raw, quote_blocks
        )
        print(f"Sentiment blocks found: {len(sentiment_blocks)}")
        
        for j, sb in enumerate(sentiment_blocks[:5]):
            print(f"  Block {j+1}:")
            print(f"    quote_index: {sb.get('quote_index')}")
            print(f"    labels: {sb.get('labels')}")
            print(f"    tone_rollup: {sb.get('tone_rollup')}")
            print(f"    alignment_confidence: {sb.get('alignment_confidence')}")
    
    print("\n" + "=" * 80)
    print("STEP 4: Computing sentiment mix for a topic...")
    print("=" * 80)
    
    # Get first topic
    if canonical_model.topics:
        first_topic = canonical_model.topics[0]
        print(f"Testing with topic: {first_topic.topic_id}")
        
        import digest
        sentiment_mix = digest._compute_sentiment_mix(
            canonical_model.evidence_cells, first_topic.topic_id
        )
        print(f"Sentiment mix: {sentiment_mix}")
    
except Exception as e:
    import traceback
    print(f"ERROR: {e}")
    traceback.print_exc()


# 5-Minute Digest - Streamlit Application

A Streamlit application for ingesting, analyzing, and digesting qualitative research data from Excel workbooks.

## Features

- **Workbook Ingestion**: Fuzzy-matching sheet detection (summary, quotes, sentiments)
- **Data Normalization**: Wide-to-long format conversion with participant/topic identification
- **Quote Parsing**: Supports numbered, bullet, and unnumbered quote formats
- **Sentiment Analysis**: Alignment of sentiments to quotes with tone rollup
- **Topic Scoring**: Coverage, evidence, and intensity metrics with ranking
- **Digest Generation**: 5-minute skimmable digest with takeaways and topic cards
- **Export**: Self-contained HTML and Markdown exports

## Installation

```bash
# Create virtual environment
python -m venv .venv

# Activate virtual environment
# Windows:
.venv\Scripts\activate
# Linux/Mac:
source .venv/bin/activate

# Install dependencies
python -m pip install -r requirements.txt
```

## Running the App

```bash
streamlit run app.py
```

## Project Structure

```
.
├── app.py                 # Main Streamlit application
├── ingest.py              # Workbook ingestion and sheet detection
├── normalize.py            # Wide-to-canonical normalization
├── parse_quotes.py         # Quote parsing (numbered, bullet, single)
├── parse_sentiment.py      # Sentiment parsing and alignment
├── score.py                # Topic scoring and aggregates
├── digest.py               # Digest building
├── render.py               # Rendering helpers and truncation
├── export.py               # HTML and Markdown export
├── edge_cases.py           # Edge case handling
├── requirements.txt        # Python dependencies
├── tests/                  # Test suite
│   ├── conftest.py        # Pytest fixtures
│   ├── test_*.py          # Unit tests
│   └── acceptance_check.py # Acceptance tests
├── debug_helpers.py        # Debugging utilities
└── *.md                    # Documentation
```

## Dependencies

- `streamlit` - Web application framework
- `pandas` - Data manipulation
- `openpyxl` - Excel file reading

## Verification

### Quick Check:

```bash
python test_app_startup.py
```

### Full Verification:

```bash
python verify_deployment.py
```

### Run Tests:

```bash
pytest tests/
```

### Acceptance Tests:

```bash
python tests/acceptance_check.py
```

## Usage

1. **Upload Excel File**: Use sidebar to upload workbook with `summary`, `quotes`, `sentiments` sheets
2. **Select Topics**: Use Top N slider or manual selection
3. **View Digest**: See takeaways and topic cards in main area
4. **Explore**: Switch to Explore tab for table view
5. **Export**: Download HTML or Markdown export

## Deployment

### Streamlit Cloud Deployment

- `QUICK_DEPLOY.md` - Quick deployment guide (3 steps)
- `DEPLOY_STREAMLIT_CLOUD.md` - Detailed deployment instructions

**Important**: Streamlit Cloud deploys through the web interface at https://share.streamlit.io/, not through the local Streamlit interface.

## Documentation

- `QUICK_START.md` - Quick start guide
- `DEPLOYMENT_VERIFICATION.md` - Deployment verification steps
- `MANUAL_QA_CHECKLIST.md` - Manual QA checklist
- `tests/README.md` - Test suite documentation

## License

This project is provided as-is for the 5-minute digest application.

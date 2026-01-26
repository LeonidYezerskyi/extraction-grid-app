# Test Suite

This directory contains unit tests for the 5-minute digest application.

## Running Tests

Run all tests:

```bash
pytest tests/
```

Run specific test file:

```bash
pytest tests/test_ingest.py
```

Run with verbose output:

```bash
pytest tests/ -v
```

Run with coverage:

```bash
pytest tests/ --cov=. --cov-report=html
```

## Test Structure

- `conftest.py`: Pytest fixtures for test data
- `test_ingest.py`: Tests for workbook ingestion and sheet detection
- `test_normalize.py`: Tests for wide-to-canonical normalization
- `test_parse_quotes.py`: Tests for quote parsing
- `test_parse_sentiment.py`: Tests for sentiment parsing and alignment
- `test_score.py`: Tests for topic scoring and aggregates
- `test_digest.py`: Tests for digest building
- `test_edge_cases.py`: Tests for edge case handling

## Fixtures

The test suite uses programmatic DataFrame fixtures rather than actual Excel files for:

- Deterministic testing
- Faster test execution
- Easy modification of test scenarios

Fixtures are defined in `conftest.py` and include:

- `sample_summary_df`: Summary sheet data
- `sample_quotes_df`: Quotes sheet data
- `sample_sentiments_df`: Sentiments sheet data
- `sample_dfs`: Combined dictionary of all sheets
- `sample_excel_bytes`: Excel file bytes created from DataFrames
- Edge case fixtures: sparse topics, single-sheet topics, etc.

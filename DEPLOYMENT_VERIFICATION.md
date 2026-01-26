# Deployment Verification Guide

## Quick Verification

Run the verification script:

```bash
python verify_deployment.py
```

## Manual Verification Steps

### 1. Check Requirements

Verify `requirements.txt` contains:

```
streamlit
pandas
openpyxl
```

### 2. Install Dependencies

In a virtual environment:

```bash
# Create virtual environment (if not exists)
python -m venv .venv

# Activate virtual environment
# On Windows:
.venv\Scripts\activate
# On Linux/Mac:
source .venv/bin/activate

# Install dependencies
python -m pip install -r requirements.txt
```

Expected output should show successful installation of:

- streamlit
- pandas
- openpyxl

### 3. Verify Imports

Test that all modules can be imported:

```bash
python -c "import ingest, normalize, parse_quotes, parse_sentiment, score, digest, render, export, edge_cases; print('✓ All imports successful')"
```

### 4. Run Streamlit App

Start the app:

```bash
streamlit run app.py
```

Expected:

- App starts without errors
- Browser opens at `http://localhost:8501`
- Sidebar shows "Upload Excel File"
- Main area shows "Please upload an Excel file to begin"

### 5. Test Basic Functionality

1. **Upload test file:**

   - Use `debug_helpers.py` to create a test workbook, or
   - Use any Excel file with `summary`, `quotes`, `sentiments` sheets
   - Verify file uploads without errors

2. **Verify processing:**

   - Check validation stats appear
   - Check topics appear in sidebar
   - Check takeaways appear in main area

3. **Test export:**
   - Select some topics
   - Click export buttons
   - Verify files download

## Common Issues and Fixes

### Issue: ModuleNotFoundError

**Fix:** Ensure all Python files are in the same directory as `app.py`

### Issue: Import errors for local modules

**Fix:** Check that module files exist and have correct names:

- `ingest.py`
- `normalize.py`
- `parse_quotes.py`
- `parse_sentiment.py`
- `score.py`
- `digest.py`
- `render.py`
- `export.py`
- `edge_cases.py`

### Issue: Streamlit not found

**Fix:**

```bash
pip install streamlit
```

### Issue: openpyxl not found

**Fix:**

```bash
pip install openpyxl
```

### Issue: App starts but shows errors

**Fix:** Check console/logs for specific error messages. Common issues:

- Missing sheets in workbook
- Invalid file format
- Import errors in modules

## Deployment to Streamlit Community Cloud

1. **Push to GitHub:**

   - Ensure all files are committed
   - Push to GitHub repository

2. **Deploy on Streamlit Cloud:**

   - Go to https://share.streamlit.io
   - Connect GitHub repository
   - Set main file path: `app.py`
   - Deploy

3. **Verify deployment:**
   - App should start automatically
   - Check logs for any errors
   - Test file upload functionality

## File Structure

Required files for deployment:

```
.
├── app.py                 # Main Streamlit application
├── requirements.txt       # Dependencies
├── ingest.py             # Workbook ingestion
├── normalize.py           # Data normalization
├── parse_quotes.py        # Quote parsing
├── parse_sentiment.py     # Sentiment parsing
├── score.py               # Topic scoring
├── digest.py              # Digest building
├── render.py              # Rendering helpers
├── export.py              # Export functions
└── edge_cases.py          # Edge case handling
```

Optional files (for development):

```
├── tests/                 # Test suite
├── debug_helpers.py       # Debugging utilities
└── *.md                   # Documentation
```

## Verification Checklist

- [ ] `requirements.txt` exists and contains streamlit, pandas, openpyxl
- [ ] All module files exist in root directory
- [ ] `python -m pip install -r requirements.txt` succeeds
- [ ] All modules can be imported without errors
- [ ] `streamlit run app.py` starts without errors
- [ ] App UI loads correctly
- [ ] File upload works
- [ ] Topics are computed and displayed
- [ ] Export functions work

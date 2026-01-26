# Quick Start Guide

## Installation

1. **Create virtual environment (recommended):**
   ```bash
   python -m venv .venv
   
   # Activate on Windows:
   .venv\Scripts\activate
   
   # Activate on Linux/Mac:
   source .venv/bin/activate
   ```

2. **Install dependencies:**
   ```bash
   python -m pip install -r requirements.txt
   ```

3. **Verify installation:**
   ```bash
   python test_app_startup.py
   ```

## Running the App

```bash
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`

## Verification Commands

### Check deployment readiness:
```bash
python verify_deployment.py
```

### Test app startup:
```bash
python test_app_startup.py
```

### Debug with sample data:
```bash
# Create a test workbook first, then:
python debug_helpers.py path/to/workbook.xlsx
```

## Requirements

- Python 3.8+
- streamlit
- pandas
- openpyxl

All dependencies are listed in `requirements.txt`

## Troubleshooting

If `streamlit run app.py` fails:

1. **Check dependencies are installed:**
   ```bash
   pip list | grep -E "streamlit|pandas|openpyxl"
   ```

2. **Verify all module files exist:**
   ```bash
   ls *.py
   ```

3. **Check for import errors:**
   ```bash
   python -c "import ingest, normalize, score, digest"
   ```

4. **Run verification script:**
   ```bash
   python verify_deployment.py
   ```
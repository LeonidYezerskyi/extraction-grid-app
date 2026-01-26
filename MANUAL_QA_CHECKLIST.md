# Manual QA Checklist

## Prerequisites

1. Install dependencies:

   ```bash
   pip install streamlit pandas openpyxl pytest
   ```

2. Ensure all modules are in the same directory as `app.py`

## Running the App Locally

1. Start Streamlit app:

   ```bash
   streamlit run app.py
   ```

2. App should open in browser at `http://localhost:8501`

## Manual QA Steps

### 1. File Upload Test

- [ ] **Upload sample workbook**

  - Click "Upload Excel File" in sidebar
  - Select a valid Excel file with `summary`, `quotes`, `sentiments` sheets
  - Verify file uploads without errors

- [ ] **Check validation report**
  - Look for validation stats in Row 1 (Participants, Topics, Evidence Cells, Matched Sheets)
  - Verify no error messages appear
  - Check warnings expander if present

### 2. Top N Computation

- [ ] **Verify Top N is computed**

  - Check that topic aggregates are computed after upload
  - Topics should appear in "Selected Topics" multiselect
  - Default should show Top 10 (or configured N)

- [ ] **Test Top N slider**
  - Change Top N slider (4-20)
  - Verify selection updates when "Auto-select Top N" is enabled
  - Verify selection is stable (doesn't change on every interaction)

### 3. Topic Selection

- [ ] **Test manual selection**

  - Uncheck "Auto-select Top N"
  - Manually select/deselect topics in multiselect
  - Verify selection persists

- [ ] **Test "Reset to Top N" button**

  - Click "Reset to Top N"
  - Verify selection resets to top N topics

- [ ] **Test "Clear Selection" button**

  - Click "Clear Selection"
  - Verify all topics are deselected

- [ ] **Test "Add Topic" dropdown**
  - Select a topic from "Search and add topic" dropdown
  - Verify it's added to selection

### 4. Takeaways

- [ ] **Verify takeaways appear**

  - Select 3-5 topics
  - Check "Key Takeaways" section shows takeaways
  - Verify each takeaway has:
    - Index number
    - Truncated text (max 180 chars visible)
    - "From: topic_id" caption
    - Evidence count metric
    - Proof quote expander

- [ ] **Test proof quote expander**
  - Click "Proof Quote" expander
  - Verify quote preview is shown (truncated to 320 chars)
  - Verify receipt links are shown

### 5. Topic Cards

- [ ] **Verify topic cards render**

  - Check "Topic Cards" section shows cards for selected topics
  - Each card should have:
    - Topic ID as header
    - One-liner (truncated to 240 chars, expandable)
    - Coverage bar with percentage
    - Evidence count metric
    - Sentiment mix chips
    - Proof quote expander
    - "Show Receipts" expander

- [ ] **Test receipts expander**
  - Click "Show Receipts" expander
  - Verify receipts are grouped by participant
  - Verify each receipt shows:
    - Participant ID
    - Summary (if available)
    - Quote blocks with sentiment chips
    - Receipt references

### 6. Filters

- [ ] **Test Coverage Tier filter**

  - Select "High" coverage tier
  - Verify only high-coverage topics shown
  - Try "Medium" and "Low"

- [ ] **Test Tone Roll-up filter**

  - Select a tone (Positive, Negative, etc.)
  - Verify filtered topics match selected tone

- [ ] **Test High Emotion toggle**

  - Enable "High Emotion Only"
  - Verify only high-intensity topics shown

- [ ] **Test Search box**
  - Enter search term in "Search (summaries + quotes)"
  - Verify topics are filtered by search term
  - Try searching in topic IDs, summaries, and quotes

### 7. Participant Filter

- [ ] **Test regex participant filter**
  - Enter pattern like `moderator|admin` in "Regex patterns"
  - Verify matching participants are filtered out
  - Check info message shows filtered participant count

### 8. Export

- [ ] **Test HTML export**

  - Click "📥 Export HTML" button
  - Verify file downloads
  - Open downloaded HTML file
  - Verify it contains:
    - Selected topics
    - Takeaways
    - Topic cards
    - Receipt links
  - Verify HTML is self-contained (no external links)

- [ ] **Test Markdown export**
  - Click "📄 Export Markdown" button
  - Verify file downloads
  - Open downloaded Markdown file
  - Verify it contains selected topics and takeaways

### 9. Explore Tab

- [ ] **Switch to "Explore" tab**

  - Verify table view shows all topics
  - Check columns: Topic ID, Score, Coverage %, Evidence, Intensity, Tone, One-liner
  - Verify topics are sorted by score (descending)

- [ ] **Check sparse topics grouping**
  - If sparse topics exist, verify they appear in separate section
  - Verify single-sheet topics appear in separate section

### 10. Edge Cases

- [ ] **Test missing sheet**

  - Upload workbook with missing `summary` sheet
  - Verify error message appears
  - Verify app stops processing

- [ ] **Test empty/sparse topics**

  - Verify sparse topics don't appear in Top N
  - Check they appear in Explore tab under "Sparse Topics"

- [ ] **Test sentiment without quotes**
  - If any evidence cells have sentiments but no quotes
  - Verify they're handled gracefully
  - Check "No quote text available" message appears

## Debugging Helpers

### Using debug_helpers.py

1. **Full debug run:**

   ```bash
   python debug_helpers.py path/to/workbook.xlsx
   ```

2. **Quick check:**

   ```bash
   python debug_helpers.py path/to/workbook.xlsx --quick
   ```

3. **In Python console:**

   ```python
   from debug_helpers import debug_run, print_validation_report, print_canonical_model_summary
   import ingest

   # Load and validate
   with open('workbook.xlsx', 'rb') as f:
       bytes_data = f.read()
   dict_of_dfs, validation_report = ingest.read_workbook(bytes_data)
   print_validation_report(validation_report)
   ```

### Checking Cache Behavior

1. **Clear Streamlit cache:**

   - In Streamlit UI: Settings → Clear cache
   - Or restart app

2. **Verify parse runs only on upload:**

   - Upload file → check console/logs for parse activity
   - Change selection/filters → verify no re-parsing
   - Upload same file again → verify cache is used

3. **Verify digest updates on selection change:**
   - Select different topics
   - Verify takeaways and topic cards update
   - Check digest content changes

### Common Issues

- **File not uploading:** Check file format (.xlsx), file size, permissions
- **No topics showing:** Check sheet names match (summary, quotes, sentiments)
- **Selection not updating:** Check "Auto-select Top N" checkbox state
- **Export empty:** Verify topics are selected before exporting
- **Cache issues:** Clear cache and restart app

## Expected Behavior

- ✅ Parse runs only when file is uploaded (cached)
- ✅ Scoring runs only when file changes (cached)
- ✅ Digest updates when selection/filters change (not cached)
- ✅ Top N maintains stable ordering
- ✅ All truncation budgets enforced
- ✅ Exports reflect current selection
- ✅ Edge cases handled gracefully with clear messages

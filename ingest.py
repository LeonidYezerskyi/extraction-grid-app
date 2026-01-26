"""Module for ingesting data from various sources."""

import io
from typing import Dict, Tuple, Optional, List, Set
import pandas as pd
import openpyxl
from difflib import get_close_matches


# Denylist of typical ID/profile column names that should be treated as metadata
METADATA_COLUMN_DENYLIST = {
    'id', 'ids', 'identifier', 'identifiers',
    'name', 'names', 'title', 'titles',
    'date', 'dates', 'time', 'timestamp', 'created', 'updated',
    'user', 'users', 'author', 'authors', 'person', 'people',
    'email', 'emails', 'phone', 'address', 'location',
    'profile', 'profiles', 'account', 'accounts',
    'status', 'type', 'category', 'tags',
    'row', 'rows', 'index', 'key', 'keys'
}


def _fuzzy_match_sheet_name(sheet_name: str, target_roles: List[str]) -> Optional[str]:
    """
    Fuzzy-match a sheet name to one of the target roles.
    
    Uses both difflib.get_close_matches and lowercase substring heuristics.
    
    Args:
        sheet_name: The name of the sheet to match
        target_roles: List of target role names ('summary', 'quotes', 'sentiments')
    
    Returns:
        Matched role name if found, None otherwise
    """
    sheet_lower = sheet_name.lower().strip()
    
    # First try exact lowercase match
    for role in target_roles:
        if sheet_lower == role:
            return role
    
    # Try substring match (target role in sheet name or vice versa)
    for role in target_roles:
        if role in sheet_lower or sheet_lower in role:
            return role
    
    # Try fuzzy matching with difflib
    matches = get_close_matches(sheet_lower, target_roles, n=1, cutoff=0.6)
    if matches:
        return matches[0]
    
    return None


def _identify_metadata_columns(df: pd.DataFrame) -> Set[str]:
    """
    Identify metadata columns using denylist and heuristics.
    
    Metadata columns are those that match the denylist or follow common
    ID/profile naming patterns.
    
    Args:
        df: DataFrame to analyze
    
    Returns:
        Set of column names that are metadata columns
    """
    metadata_cols = set()
    
    for col in df.columns:
        col_lower = str(col).lower().strip()
        
        # Check denylist
        if col_lower in METADATA_COLUMN_DENYLIST:
            metadata_cols.add(col)
            continue
        
        # Heuristic: columns ending with _id, _ids, _name, _names
        if any(col_lower.endswith(suffix) for suffix in ['_id', '_ids', '_name', '_names']):
            metadata_cols.add(col)
            continue
        
        # Heuristic: columns starting with id_, name_, user_, profile_
        if any(col_lower.startswith(prefix) for prefix in ['id_', 'name_', 'user_', 'profile_']):
            metadata_cols.add(col)
    
    return metadata_cols


def _get_topic_columns(df: pd.DataFrame) -> Set[str]:
    """
    Get topic columns (all columns except metadata columns).
    
    Args:
        df: DataFrame to analyze
    
    Returns:
        Set of column names that are topic columns
    """
    metadata_cols = _identify_metadata_columns(df)
    all_cols = set(df.columns)
    topic_cols = all_cols - metadata_cols
    return topic_cols


def read_workbook(excel_bytes: bytes) -> Tuple[Dict[str, Optional[pd.DataFrame]], Dict]:
    """
    Read an Excel workbook into pandas DataFrames and match sheets to roles.
    
    This function:
    - Reads all sheets from the workbook using openpyxl
    - Fuzzy-matches sheet names to roles: 'summary', 'quotes', 'sentiments'
    - Identifies metadata columns using a denylist and heuristics
    - Computes topic columns as the intersection across matched sheets
    - Returns DataFrames and a validation report
    
    Args:
        excel_bytes: Binary content of the Excel workbook (.xlsx file)
    
    Returns:
        Tuple of:
        - Dictionary mapping role names ('summary', 'quotes', 'sentiments') to
          DataFrames (None if sheet not found)
        - Validation report dictionary containing:
          * 'matched_sheets': dict mapping roles to matched sheet names
          * 'missing_sheets': list of roles that weren't matched
          * 'unmatched_sheets': list of sheet names that didn't match any role
          * 'topic_columns': set of topic column names (intersection across sheets)
          * 'warnings': list of warning messages
          * 'coverage_stats': dict with stats about column coverage per sheet
    
    Example:
        >>> with open('data.xlsx', 'rb') as f:
        ...     dfs, report = read_workbook(f.read())
        >>> summary_df = dfs['summary']
        >>> print(report['topic_columns'])
    """
    target_roles = ['summary', 'quotes', 'sentiments']
    result_dfs = {role: None for role in target_roles}
    validation_report = {
        'matched_sheets': {},
        'missing_sheets': [],
        'unmatched_sheets': [],
        'topic_columns': set(),
        'warnings': [],
        'coverage_stats': {}
    }
    
    # Read workbook from bytes
    try:
        workbook = openpyxl.load_workbook(io.BytesIO(excel_bytes), data_only=True)
    except Exception as e:
        error_msg = f"Failed to load workbook: {str(e)}"
        validation_report['warnings'].append(error_msg)
        validation_report['error'] = error_msg
        validation_report['is_readable'] = False
        return result_dfs, validation_report
    
    validation_report['is_readable'] = True
    
    # Read all sheets into DataFrames and match to roles
    sheet_to_role = {}
    all_sheet_names = workbook.sheetnames
    
    for sheet_name in all_sheet_names:
        matched_role = _fuzzy_match_sheet_name(sheet_name, target_roles)
        if matched_role:
            sheet_to_role[sheet_name] = matched_role
            validation_report['matched_sheets'][matched_role] = sheet_name
        else:
            validation_report['unmatched_sheets'].append(sheet_name)
    
    # Load matched sheets into DataFrames
    topic_columns_by_sheet = {}
    for sheet_name, role in sheet_to_role.items():
        try:
            df = pd.read_excel(io.BytesIO(excel_bytes), sheet_name=sheet_name, engine='openpyxl')
            result_dfs[role] = df
            topic_cols = _get_topic_columns(df)
            topic_columns_by_sheet[role] = topic_cols
            
            # Coverage stats
            validation_report['coverage_stats'][role] = {
                'total_columns': len(df.columns),
                'metadata_columns': len(_identify_metadata_columns(df)),
                'topic_columns': len(topic_cols),
                'rows': len(df)
            }
        except Exception as e:
            validation_report['warnings'].append(f"Failed to read sheet '{sheet_name}': {str(e)}")
            result_dfs[role] = None
    
    # Identify missing sheets
    for role in target_roles:
        if role not in validation_report['matched_sheets']:
            validation_report['missing_sheets'].append(role)
            validation_report['warnings'].append(f"Required sheet '{role}' not found or could not be matched")
    
    # Check if core sheets are missing
    core_sheets = ['summary']  # At minimum, summary is required
    missing_core = [s for s in core_sheets if s in validation_report['missing_sheets']]
    if missing_core:
        validation_report['error'] = f"Core required sheets missing: {', '.join(missing_core)}"
        validation_report['is_valid'] = False
    else:
        validation_report['is_valid'] = True
    
    # Compute topic columns intersection across all matched sheets
    if topic_columns_by_sheet:
        # Start with the first sheet's topic columns
        topic_columns_intersection = set(next(iter(topic_columns_by_sheet.values())))
        
        # Intersect with all other sheets
        for topic_cols in topic_columns_by_sheet.values():
            topic_columns_intersection &= topic_cols
        
        validation_report['topic_columns'] = topic_columns_intersection
        
        # Warning if intersection is empty
        if not topic_columns_intersection and len(topic_columns_by_sheet) > 1:
            validation_report['warnings'].append(
                "No common topic columns found across matched sheets"
            )
    else:
        validation_report['warnings'].append("No sheets were matched to roles")
    
    return result_dfs, validation_report

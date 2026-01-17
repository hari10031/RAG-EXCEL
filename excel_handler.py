"""
Excel Handler - Manages Excel file operations including:
- Upload and parsing
- Sheet merging
- Column normalization
- Data persistence
- Row deduplication
"""
import pandas as pd
import hashlib
import json
import re
from pathlib import Path
from typing import Optional, Dict, List, Tuple
import config


def normalize_column_name(col: str) -> str:
    """
    Normalize a column name:
    - Strip leading/trailing whitespace
    - Convert to lowercase
    - Replace spaces with underscores
    - Apply typo fixes
    """
    # Strip and lowercase
    normalized = col.strip().lower()
    
    # Replace spaces with underscores
    normalized = re.sub(r'\s+', '_', normalized)
    
    # Remove any trailing underscores
    normalized = normalized.rstrip('_')
    
    # Apply typo fixes
    if normalized in config.COLUMN_TYPO_FIXES:
        normalized = config.COLUMN_TYPO_FIXES[normalized]
    
    return normalized


def normalize_dataframe_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize all column names in a dataframe."""
    df = df.copy()
    df.columns = [normalize_column_name(col) for col in df.columns]
    return df


def compute_row_id(row: pd.Series, pub_type_col: str = 'pub_type') -> str:
    """
    Compute a stable row_id using SHA256 of normalized row JSON + pub_type.
    This ensures the same row always gets the same ID for deduplication.
    """
    # Convert row to dict, handling NaN values
    row_dict = {}
    for key, value in row.items():
        if pd.isna(value):
            row_dict[key] = None
        elif isinstance(value, (int, float)):
            row_dict[key] = str(value)
        else:
            row_dict[key] = str(value).strip()
    
    # Sort keys for consistent ordering
    sorted_dict = dict(sorted(row_dict.items()))
    
    # Create JSON string
    row_json = json.dumps(sorted_dict, sort_keys=True, ensure_ascii=False)
    
    # Add pub_type to the hash if it exists
    pub_type = row_dict.get(pub_type_col, '')
    hash_input = f"{row_json}|{pub_type}"
    
    # Compute SHA256
    return hashlib.sha256(hash_input.encode('utf-8')).hexdigest()[:16]


def add_row_ids(df: pd.DataFrame) -> pd.DataFrame:
    """Add row_id column to dataframe if not present."""
    df = df.copy()
    if 'row_id' not in df.columns:
        df['row_id'] = df.apply(compute_row_id, axis=1)
    return df


def read_excel_file(file_path_or_buffer) -> Dict[str, pd.DataFrame]:
    """
    Read an Excel file and return all sheets as a dictionary.
    
    Args:
        file_path_or_buffer: File path or file-like object
        
    Returns:
        Dictionary mapping sheet names to DataFrames
    """
    try:
        excel_file = pd.ExcelFile(file_path_or_buffer)
        sheets = {}
        for sheet_name in excel_file.sheet_names:
            df = pd.read_excel(excel_file, sheet_name=sheet_name)
            # Normalize columns
            df = normalize_dataframe_columns(df)
            sheets[sheet_name] = df
        return sheets
    except Exception as e:
        raise ValueError(f"Error reading Excel file: {str(e)}")


def merge_sheets(sheets: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    Merge all sheets into one canonical dataframe.
    Handles sheets with different columns by doing an outer join.
    
    Args:
        sheets: Dictionary of sheet names to DataFrames
        
    Returns:
        Merged DataFrame with all data
    """
    if not sheets:
        return pd.DataFrame()
    
    # Get all unique columns across all sheets
    all_columns = set()
    for df in sheets.values():
        all_columns.update(df.columns)
    
    # Reindex each dataframe to have all columns, then concat
    dfs_to_merge = []
    for sheet_name, df in sheets.items():
        df = df.copy()
        # Add source_sheet column to track origin
        df['source_sheet'] = sheet_name
        # Reindex to include all columns
        for col in all_columns:
            if col not in df.columns:
                df[col] = None
        dfs_to_merge.append(df)
    
    # Concatenate all dataframes
    merged = pd.concat(dfs_to_merge, ignore_index=True)
    
    return merged


def process_uploaded_file(file_buffer) -> Tuple[Dict[str, pd.DataFrame], pd.DataFrame]:
    """
    Process an uploaded Excel file:
    1. Read all sheets
    2. Normalize columns
    3. Merge into canonical dataframe
    4. Add row_ids
    
    Args:
        file_buffer: Uploaded file buffer
        
    Returns:
        Tuple of (original sheets dict, merged canonical dataframe)
    """
    # Read and normalize sheets
    sheets = read_excel_file(file_buffer)
    
    # Merge all sheets
    merged_df = merge_sheets(sheets)
    
    # Add row_ids for deduplication
    merged_df = add_row_ids(merged_df)
    
    return sheets, merged_df


def load_existing_store() -> Optional[pd.DataFrame]:
    """
    Load existing data store from Excel file.
    
    Returns:
        DataFrame if file exists, None otherwise
    """
    if config.EXCEL_STORE_PATH.exists():
        try:
            df = pd.read_excel(config.EXCEL_STORE_PATH, sheet_name=config.CANONICAL_SHEET_NAME)
            df = normalize_dataframe_columns(df)
            return df
        except Exception as e:
            print(f"Error loading existing store: {e}")
            return None
    return None


def save_to_excel_store(df: pd.DataFrame, keep_original_sheets: bool = False, 
                        original_sheets: Optional[Dict[str, pd.DataFrame]] = None) -> bool:
    """
    Save the canonical dataframe to the Excel store.
    Handles deduplication based on row_id.
    
    Args:
        df: DataFrame to save
        keep_original_sheets: Whether to keep original sheets in addition to 'All'
        original_sheets: Original sheets dictionary if keep_original_sheets is True
        
    Returns:
        True if successful, False otherwise
    """
    try:
        # Ensure data directory exists
        config.DATA_DIR.mkdir(exist_ok=True)
        
        # Load existing data if present
        existing_df = load_existing_store()
        
        if existing_df is not None:
            # Ensure row_id column exists in new data
            df = add_row_ids(df)
            
            # Merge with deduplication
            # Keep existing rows, add new ones based on row_id
            existing_ids = set(existing_df['row_id'].tolist())
            new_rows = df[~df['row_id'].isin(existing_ids)]
            
            # Handle concat properly to avoid FutureWarning
            if new_rows.empty:
                combined_df = existing_df.copy()
            else:
                # Align columns and convert to same dtypes to avoid warnings
                all_cols = list(set(existing_df.columns) | set(new_rows.columns))
                for col in all_cols:
                    if col not in existing_df.columns:
                        existing_df[col] = pd.NA
                    if col not in new_rows.columns:
                        new_rows[col] = pd.NA
                # Reorder columns to match
                new_rows = new_rows[existing_df.columns]
                combined_df = pd.concat([existing_df, new_rows], ignore_index=True)
        else:
            combined_df = add_row_ids(df)
        
        # Write to Excel
        with pd.ExcelWriter(config.EXCEL_STORE_PATH, engine='openpyxl') as writer:
            # Write canonical 'All' sheet
            combined_df.to_excel(writer, sheet_name=config.CANONICAL_SHEET_NAME, index=False)
            
            # Optionally keep original sheets
            if keep_original_sheets and original_sheets:
                for sheet_name, sheet_df in original_sheets.items():
                    if sheet_name != config.CANONICAL_SHEET_NAME:
                        # Truncate sheet name if too long (Excel limit is 31 chars)
                        safe_name = sheet_name[:31] if len(sheet_name) > 31 else sheet_name
                        sheet_df.to_excel(writer, sheet_name=safe_name, index=False)
        
        return True
    except Exception as e:
        print(f"Error saving to Excel store: {e}")
        raise e


def append_row_to_store(row_data: Dict) -> Tuple[bool, str, pd.DataFrame]:
    """
    Append a single row to the Excel store.
    
    Args:
        row_data: Dictionary of column values for the new row
        
    Returns:
        Tuple of (success, row_id, updated_dataframe)
    """
    try:
        # Load existing store
        existing_df = load_existing_store()
        
        if existing_df is None:
            # Create new dataframe with the row
            new_df = pd.DataFrame([row_data])
            new_df = normalize_dataframe_columns(new_df)
        else:
            # Normalize the row data keys
            normalized_row = {normalize_column_name(k): v for k, v in row_data.items()}
            
            # Ensure all existing columns are present
            for col in existing_df.columns:
                if col not in normalized_row and col != 'row_id':
                    normalized_row[col] = None
            
            # Create row dataframe
            row_df = pd.DataFrame([normalized_row])
            new_df = pd.concat([existing_df, row_df], ignore_index=True)
        
        # Compute row_id for the new row
        new_df = add_row_ids(new_df)
        
        # Get the row_id of the newly added row
        new_row_id = new_df.iloc[-1]['row_id']
        
        # Check for duplicates (same row_id)
        if existing_df is not None and new_row_id in existing_df['row_id'].values:
            return False, new_row_id, existing_df
        
        # Save to Excel
        with pd.ExcelWriter(config.EXCEL_STORE_PATH, engine='openpyxl') as writer:
            new_df.to_excel(writer, sheet_name=config.CANONICAL_SHEET_NAME, index=False)
        
        return True, new_row_id, new_df
        
    except Exception as e:
        print(f"Error appending row: {e}")
        raise e


def get_row_text_for_embedding(row: pd.Series) -> str:
    """
    Convert a row to text suitable for embedding.
    Combines all column values into a readable string.
    
    Args:
        row: Pandas Series representing a row
        
    Returns:
        Text representation of the row
    """
    parts = []
    for col, value in row.items():
        if col == 'row_id':  # Skip row_id in embedding text
            continue
        if pd.notna(value) and str(value).strip():
            # Clean column name for display
            display_col = col.replace('_', ' ').title()
            parts.append(f"{display_col}: {value}")
    
    return " | ".join(parts)


def dataframe_to_documents(df: pd.DataFrame) -> List[Dict]:
    """
    Convert a dataframe to a list of documents for embedding.
    
    Args:
        df: DataFrame to convert
        
    Returns:
        List of dictionaries with 'row_id', 'text', and 'metadata'
    """
    documents = []
    df = add_row_ids(df)
    
    for idx, row in df.iterrows():
        text = get_row_text_for_embedding(row)
        
        # Create metadata from row (excluding row_id)
        metadata = {}
        for col, value in row.items():
            if pd.notna(value):
                # ChromaDB requires string, int, float, or bool values
                if isinstance(value, (int, float, bool)):
                    metadata[col] = value
                else:
                    metadata[col] = str(value)
        
        documents.append({
            'row_id': row['row_id'],
            'text': text,
            'metadata': metadata
        })
    
    return documents


def delete_row_from_store(row_id: str) -> Tuple[bool, Optional[pd.DataFrame]]:
    """
    Delete a row from the Excel store by row_id.
    
    Args:
        row_id: The row_id to delete
        
    Returns:
        Tuple of (success, updated_dataframe)
    """
    try:
        existing_df = load_existing_store()
        
        if existing_df is None:
            return False, None
        
        # Filter out the row with the given row_id
        new_df = existing_df[existing_df['row_id'] != row_id]
        
        if len(new_df) == len(existing_df):
            # No row was deleted
            return False, existing_df
        
        # Save updated dataframe
        with pd.ExcelWriter(config.EXCEL_STORE_PATH, engine='openpyxl') as writer:
            new_df.to_excel(writer, sheet_name=config.CANONICAL_SHEET_NAME, index=False)
        
        return True, new_df
        
    except Exception as e:
        print(f"Error deleting row: {e}")
        return False, None

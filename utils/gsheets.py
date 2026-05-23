"""
Google Sheets Ingestion Utility.
Supports keyless public CSV export and authenticated gspread service-account parsing.
"""

import logging
import os
import re
import json
import pandas as pd

logger = logging.getLogger(__name__)

def load_google_sheet(url: str) -> pd.DataFrame:
    """
    Fetches a Google Sheet and returns it as a Pandas DataFrame.
    Tries direct public export first, then falls back to authenticated gspread.
    """
    # Clean URL whitespace
    url = url.strip()
    
    # 1. Try public export method first (keyless, fast, no credentials needed)
    match = re.search(r"/spreadsheets/d/([a-zA-Z0-9-_]+)", url)
    if match:
        spreadsheet_id = match.group(1)
        export_url = f"https://docs.google.com/spreadsheets/d/{spreadsheet_id}/export?format=csv"
        try:
            df = pd.read_csv(export_url)
            if not df.empty:
                logger.info("Successfully fetched public Google Sheet via CSV export URL.")
                return df
        except Exception as e:
            logger.info(f"Public Google Sheet export failed or sheet is private: {e}. Trying gspread...")
            
    # 2. Try gspread using service account credentials if available
    try:
        import gspread
        
        # Look for credentials JSON in environment or local file
        creds_json = os.getenv("GOOGLE_SERVICE_ACCOUNT_JSON", "")
        if creds_json:
            creds_dict = json.loads(creds_json)
            gc = gspread.service_account_from_dict(creds_dict)
        elif os.path.exists("service_account.json"):
            gc = gspread.service_account(filename="service_account.json")
        else:
            raise ValueError(
                "Google Sheets credentials not found. For private sheets, please provide a "
                "local 'service_account.json' file or set the GOOGLE_SERVICE_ACCOUNT_JSON "
                "environment variable."
            )
            
        sh = gc.open_by_url(url)
        worksheet = sh.get_worksheet(0)  # Fetch first sheet
        records = worksheet.get_all_records()
        df = pd.DataFrame(records)
        if df.empty:
            raise ValueError("Worksheet has no rows of data.")
        logger.info("Successfully fetched Google Sheet using gspread credentials.")
        return df
    except Exception as e:
        raise ValueError(f"Failed to load Google Sheet: {e}")

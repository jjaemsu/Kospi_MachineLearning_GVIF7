import pandas as pd
import os
import sys

# Add project root to path to allow importing from src
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from src.data.fetch_price import fetch_price_data
from src.data.fetch_volume import fetch_volume_data
from src.data.fetch_market import fetch_market_data
from src.data.fetch_macro import fetch_macro_data

def run_pipeline(merged_output_path="data/merged_data.csv"):
    print("=== Starting Unified Data Pipeline ===\n")
    
    # 1. Fetch All Data
    print("[1/4] Fetching all data sources...")
    df_price = fetch_price_data()
    df_volume = fetch_volume_data()
    df_market = fetch_market_data()
    df_macro = fetch_macro_data()
    
    print("\n[2/4] Aligning dates to KOSPI trading days...")
    # Base Index: KOSPI Trading Days
    kospi_dates = df_price.index
    print(f"Base KOSPI trading days: {len(kospi_dates)} days")
    
    base_idx = pd.DataFrame(index=kospi_dates)

    # 2. Align individual datasets and Fill Missing Values
    # volume_data
    df_volume_aligned = base_idx.join(df_volume, how='left').ffill().bfill()
    
    # market_data
    df_market_aligned = base_idx.join(df_market, how='left').ffill().bfill()
    
    # macro_data
    # Ensure macro index is datetime
    df_macro.index = pd.to_datetime(df_macro.index)
    df_macro_aligned = base_idx.join(df_macro, how='left').ffill().bfill()

    # 3. Merge All into one Unified Dataset
    print("[3/4] Merging into unified dataset...")
    # Remove 'Volume' from price_data if it exists to prevent overlap with volume_data
    df_price_clean = df_price.copy()
    if 'Volume' in df_price_clean.columns:
        df_price_clean = df_price_clean.drop(columns=['Volume'])

    df_merged = df_price_clean.join(df_volume_aligned)
    df_merged = df_merged.join(df_market_aligned)
    df_merged = df_merged.join(df_macro_aligned)

    # 4. Save and Overwrite All Files
    print("[4/4] Saving all files...")
    
    # Define paths
    paths = {
        "data/price_data.csv": df_price_clean,
        "data/volume_data.csv": df_volume_aligned,
        "data/market_data.csv": df_market_aligned,
        "data/macro_data.csv": df_macro_aligned,
        merged_output_path: df_merged
    }
    
    for path, df in paths.items():
        try:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            df.to_csv(path)
            print(f"Successfully saved: {path}")
        except PermissionError:
            print(f"Permission denied for {path}. File might be open. Saving to {path.replace('.csv', '_temp.csv')} instead.")
            df.to_csv(path.replace('.csv', '_temp.csv'))

    print("\n=== Pipeline Completed Successfully ===")
    return df_merged

if __name__ == "__main__":
    run_pipeline()

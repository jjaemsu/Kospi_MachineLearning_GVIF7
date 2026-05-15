import pandas as pd
import os
import sys

# Add project root to path to allow importing from src
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from src.data.fetch_price import fetch_price_data
from src.data.fetch_volume import fetch_volume_data
from src.data.fetch_market import fetch_market_data
from src.data.fetch_macro import fetch_macro_data

def run_pipeline(merged_output_path="data/merged_data.csv", target_output_path="data/target_data.csv"):
    print("=== Starting Unified Data Pipeline ===\n")
    
    # 1. Fetch All Data
    print("[1/5] Fetching all data sources...")
    df_price = fetch_price_data()
    df_volume = fetch_volume_data()
    df_market = fetch_market_data()
    df_macro = fetch_macro_data()
    
    print("\n[2/5] Aligning dates to KOSPI trading days...")
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

    # 3. Create Target Variable (y) as a Separate DataFrame
    print("[3/5] Creating separate target variable (y)...")
    # Target: 1 if today's Close > yesterday's Close, else 0
    df_target = pd.DataFrame(index=kospi_dates)
    df_target['Target'] = (df_price['Close'] > df_price['Close'].shift(1)).astype(int)
    # The first row will be 0 as there is no previous day to compare with.

    # 4. Merge All X Data into one Unified Dataset (excluding Target)
    print("[4/5] Merging X features into unified dataset...")
    # Remove 'Volume' from price_data if it exists to prevent overlap with volume_data
    df_price_clean = df_price.copy()
    if 'Volume' in df_price_clean.columns:
        df_price_clean = df_price_clean.drop(columns=['Volume'])

    df_merged = df_price_clean.join(df_volume_aligned)
    df_merged = df_merged.join(df_market_aligned)
    df_merged = df_merged.join(df_macro_aligned)

    # 5. Save and Overwrite All Files
    print("[5/5] Saving all files...")
    # Define paths
    paths = {
        "data/price_data.csv": df_price_clean,
        "data/volume_data.csv": df_volume_aligned,
        "data/market_data.csv": df_market_aligned,
        "data/macro_data.csv": df_macro_aligned,
        target_output_path: df_target,
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
    return df_merged, df_target

if __name__ == "__main__":
    run_pipeline()

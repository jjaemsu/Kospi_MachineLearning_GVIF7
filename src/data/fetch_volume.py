import pandas as pd
from datetime import datetime
from dateutil.relativedelta import relativedelta
import FinanceDataReader as fdr
from pykrx import stock
import os
from dotenv import load_dotenv

# Load environment variables for pykrx login
load_dotenv(dotenv_path='krx_info.env')

def fetch_volume_data(save_path=None):
    end_date = datetime.today()
    start_date = end_date - relativedelta(years=5)
    
    start_str = start_date.strftime('%Y-%m-%d')
    start_str_pykrx = start_date.strftime('%Y%m%d')
    end_str = end_date.strftime('%Y-%m-%d')
    end_str_pykrx = end_date.strftime('%Y%m%d')
    
    print(f"Fetching Combined Volume & Supply/Demand data from {start_str} to {end_str}...")
    
    # 1. Fetch Volume/Transaction Value via FDR
    df_fdr = fdr.DataReader('KS11', start_str, end_str)
    
    # FDR KS11 returns: Close, UpDown, Comp, Change, Open, High, Low, Volume, Amount, MarCap
    if 'Amount' in df_fdr.columns:
        df_fdr.rename(columns={'Amount': 'Transaction_Value'}, inplace=True)
    else:
        df_fdr['Transaction_Value'] = 0 
        
    df_fdr['Volume_ROC'] = df_fdr['Volume'].pct_change()
    
    df_vol = df_fdr[['Volume', 'Transaction_Value', 'Volume_ROC']]
    
    # 2. Fetch Supply/Demand via pykrx
    try:
        # Try a shorter range (last 30 days) to see if it works
        test_start = (datetime.today() - relativedelta(days=30)).strftime('%Y%m%d')
        df_sd_raw = stock.get_market_trading_value_by_date(test_start, end_str_pykrx, "KOSPI", detail=True)
        
        # If short range works, try the full range
        if not df_sd_raw.empty:
            df_sd_raw = stock.get_market_trading_value_by_date(start_str_pykrx, end_str_pykrx, "KOSPI", detail=True)
        
        investor_map = {
            '외국인': 'Foreigner',
            '금융투자': 'Financial_Investment',
            '투신': 'Investment_Trust',
            '연기금': 'Pension_Fund'
        }
        
        sd_list = []
        for kor, eng in investor_map.items():
            col_target = None
            for col in df_sd_raw.columns:
                if kor in col:
                    col_target = col
                    break
            if col_target:
                net_purchase = df_sd_raw[col_target].copy()
                net_purchase.name = f'Net_Purchase_{eng}'
                sd_list.append(net_purchase)
            else:
                print(f"Warning: Could not find column for {kor} ({eng})")
        
        if sd_list:
            df_sd = pd.concat(sd_list, axis=1)
            # Merge with volume data
            df_combined = pd.merge(df_vol, df_sd, left_index=True, right_index=True, how='left')
        else:
            df_combined = df_vol
            
    except Exception as e:
        print(f"Failed to fetch supply/demand. Error: {e}")
        df_combined = df_vol

    if save_path:
        try:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            df_combined.to_csv(save_path)
            print(f"Saved Combined Volume data to {save_path}")
        except PermissionError:
            print(f"Could not save to {save_path}. File might be open in another program.")
            temp_path = save_path.replace(".csv", "_temp.csv")
            df_combined.to_csv(temp_path)
            print(f"Saved to {temp_path} instead.")
        
    return df_combined

if __name__ == "__main__":
    data = fetch_volume_data(save_path="data/volume_data.csv")
    print(data.tail())

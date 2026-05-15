import yfinance as yf
import pandas as pd
from datetime import datetime
from dateutil.relativedelta import relativedelta
import os

def fetch_market_data(save_path=None):
    end_date = datetime.today()
    start_date = end_date - relativedelta(years=5)
    
    print(f"Fetching Market data from {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}...")
    
    tickers = {
        'S&P500': '^GSPC',
        'Nikkei225': '^N225',
        'TaiwanWeighted': '^TWII',
        'USD_KRW': 'USDKRW=X',
        'Dollar_Index': 'DX-Y.NYB',
        'VIX': '^VIX'
    }
    
    df_list = []
    for name, ticker in tickers.items():
        data = yf.download(ticker, start=start_date.strftime('%Y-%m-%d'), end=end_date.strftime('%Y-%m-%d'))['Close']
        if isinstance(data, pd.DataFrame):
            data = data.squeeze()
        data.name = name
        df_list.append(data)
        
    market_df = pd.concat(df_list, axis=1)
    # Forward fill missing values (due to timezone/holiday differences)
    market_df = market_df.ffill().dropna()
    
    if save_path:
        try:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            market_df.to_csv(save_path)
            print(f"Saved Market data to {save_path}")
        except PermissionError:
            print(f"Could not save to {save_path}. File might be open in another program.")
            temp_path = save_path.replace(".csv", "_temp.csv")
            market_df.to_csv(temp_path)
            print(f"Saved to {temp_path} instead.")
        
    return market_df

if __name__ == "__main__":
    data = fetch_market_data(save_path="data/market_data.csv")
    print(data.tail())

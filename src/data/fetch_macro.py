import pandas_datareader.data as web
import pandas as pd
from datetime import datetime
from dateutil.relativedelta import relativedelta
import requests
import warnings
import os
from pykrx import bond
from dotenv import load_dotenv

# 환경 변수 로드 (pykrx가 KRX_ID, KRX_PW를 자동으로 인식하여 로그인하도록 설정)
load_dotenv('krx_info.env')

warnings.filterwarnings('ignore', category=FutureWarning)

def fetch_fred_macro(start_date, end_date):
    fred_series = {
        'US_GDP': 'GDP',
        'US_M2': 'M2SL',
        'US_Interest_Rate_10Y': 'DGS10',
        'US_Interest_Rate_3Y': 'DGS3', # 미국 3년물
        'US_CPI': 'CPIAUCSL',
        'US_UNRATE': 'UNRATE',
        'KR_GDP': 'KORGDPNQDSMEI',  
        'KR_CPI': 'KORCPIALLMINMEI'
    }
    try:
        df_fred = web.DataReader(list(fred_series.values()), 'fred', start_date, end_date)
        df_fred.rename(columns={v: k for k, v in fred_series.items()}, inplace=True)
        return df_fred
    except Exception as e:
        print(f"Error fetching FRED data: {e}")
        return pd.DataFrame()

def fetch_krx_bonds(start_date, end_date):
    """pykrx를 통해 한국 국채(10년, 3년) 장외 채권 수익률을 가져옵니다."""
    start_str = start_date.strftime('%Y%m%d')
    end_str = end_date.strftime('%Y%m%d')
    
    df_list = []
    
    try:
        # 국고채 10년물
        df_10y = bond.get_otc_treasury_yields(start_str, end_str, "국고채10년")
        if not df_10y.empty and '수익률' in df_10y.columns:
            kr_10y = df_10y['수익률']
            kr_10y.name = 'KR_Interest_Rate_10Y'
            df_list.append(kr_10y)
            
        # 국고채 3년물
        df_3y = bond.get_otc_treasury_yields(start_str, end_str, "국고채3년")
        if not df_3y.empty and '수익률' in df_3y.columns:
            kr_3y = df_3y['수익률']
            kr_3y.name = 'KR_Interest_Rate_3Y'
            df_list.append(kr_3y)
            
        if df_list:
            df_bonds = pd.concat(df_list, axis=1)
            # 인덱스가 날짜 형식이 되도록 보장
            df_bonds.index = pd.to_datetime(df_bonds.index)
            return df_bonds
        else:
            print("No KRX bond data returned (possibly blocked).")
            return pd.DataFrame()
            
    except Exception as e:
        print(f"Error fetching KRX Bond data: {e}")
        return pd.DataFrame()

def fetch_ecos_macro(start_date, end_date):
    # ECOS API 연동을 위한 자리 (추후 유동성 지표 등 추가 가능)
    return pd.DataFrame()

def fetch_macro_data(save_path=None):
    end_date = datetime.today()
    start_date = end_date - relativedelta(years=5)
    
    print(f"Fetching Macro data from {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}...")
    
    df_fred = fetch_fred_macro(start_date, end_date)
    df_ecos = fetch_ecos_macro(start_date, end_date)
    df_krx_bonds = fetch_krx_bonds(start_date, end_date)
    
    # 데이터 결합 (FRED + ECOS + KRX)
    df_macro = pd.concat([df_fred, df_ecos, df_krx_bonds], axis=1)
    
    if not df_macro.empty:
        daily_idx = pd.date_range(start=start_date.date(), end=end_date.date(), freq='D')
        
        # 휴장일 및 주말 결측치를 이전 값으로 채우기
        df_macro = df_macro.ffill().bfill()
        df_macro = df_macro.reindex(daily_idx)
        df_macro = df_macro.ffill().bfill()
        
        # [파생변수] 미국 장단기 금리차 (Spread 10Y - 3Y) 추가
        if 'US_Interest_Rate_10Y' in df_macro.columns and 'US_Interest_Rate_3Y' in df_macro.columns:
            df_macro['US_Spread_10Y_3Y'] = df_macro['US_Interest_Rate_10Y'] - df_macro['US_Interest_Rate_3Y']
            
        # [파생변수] 한국 장단기 금리차 (Spread 10Y - 3Y) 추가
        if 'KR_Interest_Rate_10Y' in df_macro.columns and 'KR_Interest_Rate_3Y' in df_macro.columns:
            df_macro['KR_Spread_10Y_3Y'] = df_macro['KR_Interest_Rate_10Y'] - df_macro['KR_Interest_Rate_3Y']
        
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        df_macro.to_csv(save_path)
        print(f"Saved Macro data to {save_path}")
        
    return df_macro

if __name__ == "__main__":
    data = fetch_macro_data(save_path="data/macro_data.csv")
    
    # 출력할 컬럼 필터링 (존재하는 컬럼만)
    display_cols = [c for c in ['US_Interest_Rate_10Y', 'US_Interest_Rate_3Y', 'US_Spread_10Y_3Y', 
                                'KR_Interest_Rate_10Y', 'KR_Interest_Rate_3Y', 'KR_Spread_10Y_3Y'] if c in data.columns]
    print(data[display_cols].tail())

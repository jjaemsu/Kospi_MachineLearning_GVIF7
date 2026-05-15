# 데이터 정의서 (Data Dictionary)

이 문서는 `src/data/`의 수집 스크립트를 통해 생성되는 데이터셋의 각 변수(Feature)에 대한 설명을 담고 있습니다.

## 1. 가격 데이터 (`price_data.csv`)
코스피(KOSPI) 지수의 기초 가격 및 기술적 지표입니다.
- **Open / High / Low / Close**: 코스피 지수의 시가, 고가, 저가, 종가.
- **Return**: 전일 대비 종가 수익률.
- **Log_Return**: 로그 수익률 (변동성 분석 및 통계적 처리에 용이).
- **Volatility_20**: 최근 20거래일간의 로그 수익률 표준편차 (연율화되지 않음).
- **ATR_14**: Average True Range (14일). 시장의 평균적인 변동 폭.
- **RSI_14**: Relative Strength Index (14일). 과매수/과매도 지표.
- **MACD / MACD_Signal**: 이동평균 수렴 확산 지수 및 시그널 선.

## 2. 거래량 및 수급 데이터 (`volume_data.csv`)
시장의 에너지와 주요 투자 주체의 자금 흐름입니다.
- **Volume**: 코스피 시장 전체 거래량.
- **Transaction_Value**: 코스피 시장 전체 거래대금.
- **Volume_ROC**: 거래량 변화율 (Rate of Change).
- **Net_Purchase_Foreigner**: 외국인 순매수 대금.
- **Net_Purchase_Financial_Investment**: 금융투자(증권사 등) 순매수 대금.
- **Net_Purchase_Investment_Trust**: 투신(자산운용사 펀드 등) 순매수 대금.
- **Net_Purchase_Pension_Fund**: 연기금 순매수 대금.

## 3. 시장 데이터 (`market_data.csv`)
코스피와 상관관계가 높은 글로벌 증시 및 대외 경제 지표입니다.
- **S&P500**: 미국 대형주 지수 종가.
- **Nikkei225**: 일본 니케이 225 지수 종가.
- **TaiwanWeighted**: 대만 가권 지수 종가.
- **USD_KRW**: 원/달러 환율 종가.
- **Dollar_Index**: 주요 6개국 통화 대비 달러 가치 (DXY).
- **VIX**: S&P 500 옵션 기반 변동성 지수 (공포 지수).

## 4. 거시경제 데이터 (`macro_data.csv`)
중장기적 시장 환경을 나타내는 경제 지표입니다. (FRED 및 KRX 채권 데이터 기반)
- **US_GDP**: 미국 실질 GDP (분기별 데이터, 일간 확장).
- **US_M2**: 미국 통화량 M2 (월간 데이터, 일간 확장).
- **US_Interest_Rate_10Y / 3Y**: 미국 국채 10년물 및 3년물 수익률.
- **US_Spread_10Y_3Y**: 미국 장단기 금리차.
- **US_CPI / US_UNRATE**: 미국 소비자 물가 지수 및 실업률.
- **KR_Interest_Rate_10Y / 3Y**: 한국 국고채 10년물 및 3년물 수익률.
- **KR_Spread_10Y_3Y**: 한국 장단기 금리차.
- **KR_GDP / KR_CPI**: 한국 GDP 및 소비자 물가 지수.

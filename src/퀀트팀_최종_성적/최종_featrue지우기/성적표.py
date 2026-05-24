import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# 1. 데이터 불러오기
df = pd.read_csv('kospi_final_daily_results_1차.csv')

# 2. 날짜 데이터를 datetime 형식으로 변환 및 정렬
df['Date'] = pd.to_datetime(df['Date'])
df = df.sort_values('Date').reset_index(drop=True)

# 3. 누적 예측률 계산
# Is_Correct 컬럼(1: 맞음, 0: 틀림)의 누적합을 현재까지의 데이터 개수로 나눔
df['Cumulative_Accuracy'] = df['Is_Correct'].cumsum() / (df.index + 1)

# 4. 그래프 그리기 (스크린샷 스타일 반영)
plt.figure(figsize=(12, 6))
plt.plot(df['Date'], df['Cumulative_Accuracy'], color='#1f77b4', linewidth=2, label='Cumulative Accuracy')

# 5. Y축 범위 및 눈금 설정 (0.0 부터 1.0)
plt.ylim(0.0, 1.0)
plt.yticks([i/10 for i in range(11)]) # 0.0, 0.1, ..., 1.0

# 6. X축 날짜 포맷 설정
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
plt.gca().xaxis.set_major_locator(mdates.MonthLocator(interval=1))
plt.xticks(rotation=45)

# 7. 디자인 및 레이블링
plt.title('Cumulative Prediction Accuracy Over Time', fontsize=14, fontweight='bold')
plt.xlabel('Date', fontsize=12)
plt.ylabel('Cumulative Accuracy', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.6)
plt.axhline(y=0.5, color='red', linestyle='--', linewidth=1, alpha=0.5) # 기준선 (50%)
plt.legend(loc='lower right')
plt.tight_layout()

# 그래프 출력
plt.show()
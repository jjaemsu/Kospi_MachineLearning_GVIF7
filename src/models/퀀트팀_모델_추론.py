import pandas as pd
import xgboost as xgb
import numpy as np

class KospiTradingBot:
    def __init__(self, model_path): # 필수
        """
        1. 저장된 XGBoost 모델을 불러와 초기화합니다.
        """
        self.model = xgb.XGBClassifier()
        self.model.load_model(model_path)
        
        # 특성 기여도(Feature Contribution) 추출을 위해 내장 Booster 객체를 가져옵니다.
        self.booster = self.model.get_booster()
        print(f"[{model_path}] 모델 로드 완료.")

    def load_daily_data(self, csv_path): # 필수
        """
        3. 당일자 기준 새로운 입력 데이터를 불러옵니다.
        """
        # 가장 최근 시점(당일)의 데이터 한 줄만 예측에 사용한다고 가정합니다.
        df = pd.read_csv(csv_path, index_col=0)
        return df.tail(1) # 가장 마지막 행(당일 데이터)만 추출

    def analyze_local_importance(self, input_data): # 필수
        """
        2. 해당 시점(당일) 예측에 가장 큰 영향을 준 변수들을 분석합니다.
        """
        # DMatrix 변환 후 pred_contribs=True를 설정하면 각 변수의 영향력을 알 수 있습니다.
        dmatrix = xgb.DMatrix(input_data)
        
        # 반환값은 (샘플 수, 변수 개수 + 1(편향)) 형태의 배열입니다.
        contributions = self.booster.predict(dmatrix, pred_contribs=True)[0]
        feature_names = input_data.columns.tolist()
        
        # 편향(Bias) 항목을 제외한 각 변수의 기여도를 딕셔너리로 묶고, 영향력(절대값) 순으로 정렬합니다.
        importance_dict = {
            feature_names[i]: contributions[i] 
            for i in range(len(feature_names))
        }
        
        # 양수면 상승(1)에 기여, 음수면 하락(0)에 기여함을 의미합니다.
        sorted_importance = sorted(importance_dict.items(), key=lambda x: abs(x[1]), reverse=True)
        return sorted_importance

    def predict_market(self, input_data): # 필수
        """
        1 & 5. 모델을 실행하여 이진 결과(1/0)와 확신도(Probability)를 반환합니다.
        """
        # 이진 결과 (1: 상승/매수, 0: 하락/매도)
        signal = self.model.predict(input_data)[0]
        
        # 예측 확률 (클래스 0일 확률, 클래스 1일 확률)
        probabilities = self.model.predict_proba(input_data)[0]
        
        # 모델이 '상승(1)'이라고 확신하는 정도를 변수로 저장
        confidence_up = probabilities[1]
        
        return signal, confidence_up

    def execute_trade_strategy(self, signal, confidence_up): # 매수/매도 선택에 대한 알고리즘 (예시)
        """
        4. 특정 조건 및 확신도에 따라 선별적 매수/매도 비율을 결정하는 룰(Rule) 기반 메서드입니다.
        """
        position_size = 0.0 # 베팅 비율 (0.0 ~ 1.0)
        action = "HOLD"

        if signal == 1:
            if confidence_up >= 0.70:
                action = "STRONG BUY"
                position_size = 1.0   # 확신이 70% 이상이면 100% 비중 매수
            elif confidence_up >= 0.55:
                action = "BUY"
                position_size = 0.5   # 55~70% 사이면 50% 비중만 매수
            else:
                action = "WEAK BUY (HOLD)"
                position_size = 0.0   # 55% 미만이면 시그널이 1이어도 관망
        else:
            # signal == 0 (하락 예측)인 경우
            action = "SELL / SHORT"
            position_size = 0.0       # 매수 비중 0

        return action, position_size

# ==========================================
# 실제 실행 시나리오 (Main Execution)
# ==========================================
if __name__ == "__main__":
    # 1. 봇 객체 생성 및 모델 로드
    bot = KospiTradingBot("kospi_xgboost_model.json") # 학습 코드에서 저장한 모델과 같은 이름임 (수정 불필요)
    
    # 2. 당일 새로운 데이터 로드
    # (장 마감 전/후 생성된 당일 CSV 파일을 입력)
    today_data = bot.load_daily_data("today_kospi_inputs.csv") # 당일 입력 데이터는 꼭 이 이름으로 작성해줄 것
    
    # 3. 예측 및 확신도 추출
    signal, confidence = bot.predict_market(today_data)
    
    # 4. 전략 실행 (비율 산정) ** 선택적 기능이므로 수정 가능 **
    action, size = bot.execute_trade_strategy(signal, confidence)
    
    # 5. 당일 예측을 주도한 핵심 변수 분석 
    local_importances = bot.analyze_local_importance(today_data)

    # --- 결과 출력 ---
    print("\n[ 익일 코스피 방향성 예측 결과 ]")
    print(f"▶ 원시 시그널: {signal} (1: 상승, 0: 하락)")
    print(f"▶ 모델 추정 상승 확률(Confidence): {confidence * 100:.2f}%")
    
    print("\n[ 트레이딩 전략 판단 ]")
    print(f"▶ 권장 행동: {action}")
    print(f"▶ 자산 투입 비율: {size * 100}%")
    
    print("\n[ 당일 예측 결정적 요인 Top 5 ]")
    # 가장 영향력이 컸던 상위 5개 변수만 출력
    for rank, (feature, score) in enumerate(local_importances[:5], 1):
        direction = "상승 압력" if score > 0 else "하락 압력"
        print(f"{rank}. {feature} (기여도 점수: {score:+.4f} -> {direction})")

# 코드 설명
# class 객체로 선언해둔 상태이므로 기능 추가를 method 꼴로 자유롭게 할 수 있음
# xgboost틀 자체는 완료된 상태임
# 그 결과를 토대로 어떤 투자전략을 수행하도록 할 지 이 file에서 결정하면 됨
# 매수/매도 , 배팅 비율 등을 다음 코드에 넘겨 주면 됨

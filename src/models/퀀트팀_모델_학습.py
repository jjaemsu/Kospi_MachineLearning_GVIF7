import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score
import optuna
from sklearn.metrics import confusion_matrix

class KospiModelTrainer:
    def __init__(self, input_csv_path, output_csv_path):
        """
        [모델 트레이너 초기화]
        데이터를 불러오고 모델 학습에 필요한 기본 세팅을 하는 곳.
        """
        self.X = pd.read_csv(input_csv_path, index_col=0)
        self.y = pd.read_csv(output_csv_path, index_col=0)
        self.fallback_count = 0 # 차선책 모델 저장 카운터
        
        # 클래스 불균형(Class Imbalance) 가중치 계산
        num_neg = (self.y == 0).sum().values[0] 
        num_pos = (self.y == 1).sum().values[0] 
        self.scale_pos_weight = num_neg / num_pos
        
        print(f"[초기화 완료] 데이터 불균형 가중치(scale_pos_weight) 세팅 됨: {self.scale_pos_weight:.4f}")

    def _get_sample_weights(self, num_samples, decay_intensity):
        """
        [시간 지수 감쇠 가중치 생성]
        최근 데이터일수록 가중치가 1.0에 가깝고, 과거로 갈수록 가중치가 0에 가깝게 떨어집니다.
        decay_intensity가 클수록 과거 데이터를 강하게 무시합니다.
        """
        return np.exp(np.linspace(-decay_intensity, 0, num_samples))

    def objective(self, trial):
        """
        [Optuna 최적화 목적 함수]
        """
        
        # 여기를 수정하여 Optuna가 탐색할 하이퍼파라미터 공간을 정의합니다. (e.g. n_estimators의 경우 100~1000 사이의 정수를 탐색)
        param = {
                    # 1. 모델의 복잡도와 학습량 결정
                    'n_estimators': trial.suggest_int('n_estimators', 100, 1000),             # [나무의 개수] 모델이 몇 번의 시행착오를 거쳐 학습할지 결정. 너무 많으면 과거 데이터 패턴을 통째로 외워버림(과적합).
                    'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.1, log=True),# [학습 속도] 각 나무의 영향력을 얼마나 반영할지 결정. 작을수록 모델이 꼼꼼하게 학습하지만, 나무 개수(n_estimators)가 많이 필요함.
                    'max_depth': trial.suggest_int('max_depth', 3, 5),                        # [나무의 깊이] 질문을 얼마나 복잡하게 던질지 결정. KOSPI 3년 데이터는 양이 적으므로 깊이를 낮게(3~5) 가져가야 억지 패턴을 안 만듦.

                    # 2. 과적합 방지 및 일반화 (랜덤성 부여)
                    'subsample': trial.suggest_float('subsample', 0.5, 1.0),                  # [데이터 샘플링] 각 나무를 만들 때 전체 데이터 중 몇 %만 쓸지 결정. 데이터에 노이즈가 많을 때 일부만 써서 '운 좋게 맞은 패턴'을 걸러냄.
                    'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),    # [변수 샘플링] 각 나무를 만들 때 전체 변수(Feature) 중 몇 %만 무작위로 고를지 결정. 특정 지표(예: 이동평균선 하나)에만 의존하는 모델이 되는 것을 방지.
                    'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),         # [관측치 최소 합] 새로운 가지를 칠 때 필요한 데이터의 최소 무게. 이 값이 크면 아주 확실하고 굵직한 패턴만 학습하고 자잘한 잡음은 무시함.
                    'reg_lambda': trial.suggest_float('reg_lambda', 0.1, 10.0, log=True),     # [L2 정규화] 모델이 예측을 너무 극단적으로(강한 확신) 하지 않도록 변동성을 눌러주는 '진정제' 역할.

                    # 3. 모델 설정 고정값
                    'scale_pos_weight': self.scale_pos_weight,                                # [클래스 가중치] 
                    'eval_metric': 'logloss',                                                 # [평가 지표] 모델이 정답(상승/하락)을 맞추는 확률 자체가 얼마나 틀렸는지 '로그 손실'로 계산하여 정밀하게 다듬음.
                    'random_state': 42                                                        # [랜덤 시드] 실험을 할 때마다 결과가 바뀌지 않도록 고정하는 기준점.
                }
        
        # 핵심 추가 포인트: 과거 데이터 무시 강도도 Optuna가 찾아내게 만듦! (0.0 이면 감쇠 없음)
        decay_intensity = trial.suggest_float('decay_intensity', 0.0, 5.0)

        tscv = TimeSeriesSplit(n_splits=5)
        accuracies = []

        # 전체 데이터 길이에 맞는 가중치 배열 생성
        full_weights = self._get_sample_weights(len(self.X), decay_intensity)

        for train_index, val_index in tscv.split(self.X):
            X_train, X_val = self.X.iloc[train_index], self.X.iloc[val_index]
            y_train, y_val = self.y.iloc[train_index], self.y.iloc[val_index]
            
            # 훈련용 데이터에 맞는 가중치만 슬라이싱
            w_train = full_weights[train_index]

            model = xgb.XGBClassifier(**param)
            
            # 가중치(sample_weight)를 부여하여 학습 진행
            model.fit(
                X_train, y_train,
                sample_weight=w_train, 
                eval_set=[(X_val, y_val)],
                verbose=False 
            )

            preds = model.predict(X_val)
            acc = accuracy_score(y_val, preds)
            accuracies.append(acc)
        
        mean_acc = np.mean(accuracies)
        
        # 차선책 저장 함수에 decay_intensity도 함께 넘겨줌 (임계값은 테스트용 0.55로 세팅)
        self.save_fallback_model(param, mean_acc, decay_intensity, threshold=0.55)
        
        return mean_acc

    def save_fallback_model(self, params, current_score, decay_intensity, threshold=0.55):
        """
        [차선책 보관 로직]
        """
        if current_score >= threshold:
            self.fallback_count += 1
            filename = f"fallback_model_{self.fallback_count}_score{current_score:.4f}.json"
            print(f"  💡 [차선책 확보] 점수 {current_score:.4f} (기준 {threshold} 이상)! '{filename}' 저장 중...")

            fallback_params = params.copy()
            fallback_params['scale_pos_weight'] = self.scale_pos_weight
            fallback_params['eval_metric'] = 'logloss'
            fallback_params['random_state'] = 42

            fallback_model = xgb.XGBClassifier(**fallback_params)
            
            # 차선책 모델을 전체 데이터로 재학습할 때도 가중치 부여
            full_weights = self._get_sample_weights(len(self.X), decay_intensity)
            fallback_model.fit(self.X, self.y, sample_weight=full_weights, verbose=False)
            fallback_model.save_model(filename)

    def optimize_and_train(self, n_trials=50, save_path="best_kospi_model.json"):
        """
        [전체 파이프라인 실행]
        """
        print(f"\n--- Optuna 하이퍼파라미터 튜닝 시작 (총 {n_trials}회 시도) ---")
        study = optuna.create_study(direction='maximize')
        study.optimize(self.objective, n_trials=n_trials)

        print("\n[ 파라미터 튜닝 결과 ]")
        print(f"발견된 최고 교차검증 정확도: {study.best_value * 100:.2f}%")
        print("최적의 파라미터:", study.best_params)

        # 1. 딕셔너리에서 Optuna가 찾은 최고 파라미터를 복사
        best_params = study.best_params.copy()
        
        # 2. 모델 파라미터가 아닌 '가중치 강도(decay_intensity)'는 따로 분리 (XGBoost에 들어가면 에러남)
        best_decay = best_params.pop('decay_intensity')
        
        # 3. 고정 파라미터 재세팅
        best_params['scale_pos_weight'] = self.scale_pos_weight
        best_params['eval_metric'] = 'logloss'
        best_params['random_state'] = 42

        # 4. 전체 가중치 생성 및 최종 모델 학습
        final_model = xgb.XGBClassifier(**best_params)
        full_weights = self._get_sample_weights(len(self.X), best_decay)
        
        print(f"\n--- 전체 데이터를 사용해 최종 모델 학습 시작 (최적 Decay 강도: {best_decay:.2f}) ---")
        final_model.fit(self.X, self.y, sample_weight=full_weights, verbose=10)

        final_model.save_model(save_path)
        print(f"\n최종 모델 저장 완료: '{save_path}'")
        
        return final_model, study

def print_simple_matrix(model, X_test, y_test):
    predictions = model.predict(X_test)
    tn, fp, fn, tp = confusion_matrix(y_test, predictions).ravel()
    
    print(f"True Negatives (실제 하락장 - 하락 예측 방어): {tn}")
    print(f"False Positives (가짜 상승 - 손실 유발): {fp}")
    print(f"False Negatives (가짜 하락 - 기회 비용): {fn}")
    print(f"True Positives (실제 상승장 - 상승 예측 수익): {tp}")
    
    return tn, fp, fn, tp

# ==========================================
# 실제 실행 시나리오 (Main)
# ==========================================
if __name__ == "__main__":
    # 1. 트레이너 세팅 (전처리가 완료된 학습용 KOSPI CSV 파일 경로)
    # trainer = KospiModelTrainer('kospi_inputs.csv', 'kospi_outputs.csv')
    
    # 2. 하이퍼파라미터 튜닝 및 모델 저장 (총 50회 시도)
    # final_model, study_results = trainer.optimize_and_train(n_trials=50, save_path="best_kospi_model.json")
    
    # 3. 테스트 데이터(Out-of-Sample)로 최종 모델 평가
    # X_test = pd.read_csv('kospi_test_inputs.csv', index_col=0)
    # y_test = pd.read_csv('kospi_test_outputs.csv', index_col=0)
    
    # 4. 오차 행렬 출력 및 성과 확인
    # print("\n--- OOS(Out-of-Sample) 테스트 데이터 평가 ---")
    # tn, fp, fn, tp = print_simple_matrix(final_model, X_test, y_test)
    
    # 5. 핵심 지표 분석 ('잃지 않는 원칙' 확인)
    # total_actual_negatives = tn + fp
    # if total_actual_negatives > 0:
    #     tn_rate = (tn / total_actual_negatives) * 100
    #     print(f"\n[핵심 지표] 실제 하락장 중 성공적으로 회피한 비율 (TN Rate): {tn_rate:.2f}%")
    #     print("※ 이 수치가 높을수록 '가짜 상승'에 속아 자본을 잃을 확률이 낮아집니다.")
    pass


# 코드 설명
# 하이퍼파라미터를 수작업으로 찾지 않고, Optuna가 알아서 최적값을 찾도록 objective 함수 작성
# 시계열 데이터이므로 일반적인 KFold가 아니라 TimeSeriesSplit으로 교차
# 검증하는 방식 사용 (TF, FN, FP, TN 등도 계산해서 출력하는 함수도 추가 이에 맞춘 모델을 따로 저장해 둘 수도 있음)
# ex) 우리가 정한 case를 초과한 성과를 낸 모델을 차선책으로 저장해두는 코드를 작성할 수 있음.
# 최적의 하이퍼파라미터로 전체 데이터 학습 후 모델 저장
# save_fallback_model() 메소드 호출의 조건은 상당히 까다로울 필요가 있음. 너무 자주 저장되면, 알아보기도 어렵고 굉장히 느려짐
# 따라서, 단순한 정확도 기준 보다는, 우리가 보고싶은 다른 지표를 기준으로 삼아야만함
# threshold는 기준치임. 모델을 직접 돌려보며 어느정도가 일반적인지 직접 탐색해볼 필요가 있음

# 하이퍼파라미터의 강점
# 현행 방식은 하이퍼파라미터 선정 이유로 대부분의 하이퍼파라미터를 직접 수행해보았다고 주장할 수 있음

# 교수님 피드백 반영
# 3년치 데이터 반영시 과거 데이터의 영향력을 줄이는 방법으로, 시간 지수 감쇠 가중치(Time-Decay Sample Weights)를 도입함
# 다만, 이를 완전히 줄이는 것(3개월치만 사용 등)은 제(송경환) 입장에서는 과거 데이터의 흐름에서 얻는 바가 완전히 배제될 수 있으며
# 이는 과적합의 위험이 있고, 모델의 시장의 패턴을 학습하기에 부족하다고 생각하여 시장 지수 감쇠 가중치를 도입했습니다.
# 지수 감쇠와 시간사이의 관계는 설득력 있는 논리적 기반이 있음

# 아쉬운 점 : 현재 위 코드에서 seed는 42로 고정되어있음. 시드 강건성에 대해서 추가적 검증이 필요할 수 있음

# 우리 모델은 True Negatives가 가장 중요할 것으로 생각됌. 수익률도 중요하지만, 잃지 않는다는 원칙을 강조하는 것이 어떨지?

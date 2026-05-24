import os
import argparse
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, mean_squared_error
import time
from tqdm import tqdm

# ==========================================
# ⚙️ HYPERPARAMETER CONFIGURATION (하이퍼파라미터 설정)
# 여기서 모델의 주요 파라미터를 쉽게 수정할 수 있습니다.
# ==========================================
CONFIG = {
    'data_dir': 'data',
    'lookback': 10,                 # 시퀀스 길이 (과거 며칠의 데이터를 보고 예측할 것인가)
    'initial_train_days': 120,      # 최초 모델을 학습시킬 기초 데이터 기간 (일)
    'initial_epochs': 50,           # 최초 학습 시 반복할 에포크 수
    'finetune_epochs': 1,           # 이후 매일 1일치 전진하면서 추가로 학습(파인튜닝)할 에포크 수
    'batch_size': 32,               # 배치 사이즈
    'lr': 0.001,                    # 학습률 (Learning Rate)
    'hidden_dim': 32,               # LSTM 은닉층 노드 수
    'num_layers': 1,                # LSTM 레이어 층 수
    'dropout': 0.3,                 # 드롭아웃 비율 (과적합 방지, 0.0 ~ 1.0)
    'weight_decay': 1e-5            # L2 정규화 (과적합 방지)
}
# ==========================================

# Define LSTM Model
class KospiLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, dropout):
        super(KospiLSTM, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, 
                            batch_first=True, dropout=dropout if num_layers > 1 else 0.0)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :] 
        out = self.dropout(out)
        out = self.fc(out)
        return self.sigmoid(out)

def create_sequences(features, targets, seq_length):
    xs = []
    ys = []
    for i in range(len(features) - seq_length + 1):
        x = features[i:(i + seq_length)]
        y = targets[i + seq_length - 1]
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', type=str, default=CONFIG['data_dir'])
    parser.add_argument('--lookback', type=int, default=CONFIG['lookback'])
    parser.add_argument('--initial-train-days', type=int, default=CONFIG['initial_train_days'])
    parser.add_argument('--initial-epochs', type=int, default=CONFIG['initial_epochs'])
    parser.add_argument('--finetune-epochs', type=int, default=CONFIG['finetune_epochs'])
    parser.add_argument('--batch-size', type=int, default=CONFIG['batch_size'])
    parser.add_argument('--lr', type=float, default=CONFIG['lr'])
    parser.add_argument('--hidden-dim', type=int, default=CONFIG['hidden_dim'])
    args = parser.parse_args()

    print(f"Loading datasets from {args.data_dir}...")
    df_price = pd.read_csv(os.path.join(args.data_dir, 'price_data_stationary_lag1.csv'))
    df_macro = pd.read_csv(os.path.join(args.data_dir, 'macro_data_stationary_lag1.csv'))
    df_market = pd.read_csv(os.path.join(args.data_dir, 'market_data_stationary_lag1.csv'))
    df_volume = pd.read_csv(os.path.join(args.data_dir, 'volume_data_stationary_lag1.csv'))
    
    for df_ in [df_macro, df_market, df_volume]:
        if 'Target' in df_.columns:
            df_.drop(columns=['Target'], inplace=True)
            
    df = df_price.merge(df_macro, on='Date', how='inner')\
                 .merge(df_market, on='Date', how='inner')\
                 .merge(df_volume, on='Date', how='inner')

    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date').reset_index(drop=True)
    df = df.dropna().reset_index(drop=True)

    feature_cols = [c for c in df.columns if c not in ['Date', 'Target']]
    
    features_raw = df[feature_cols].values
    targets_raw = df['Target'].values
    dates_raw = df['Date'].values
    
    total_days = len(df)
    print(f"Total trading days available: {total_days}")
    
    if args.initial_train_days >= total_days:
        print("Error: initial-train-days must be less than total trading days.")
        return
    
    predictions = []
    
    model = KospiLSTM(input_dim=len(feature_cols), 
                      hidden_dim=args.hidden_dim, 
                      num_layers=CONFIG['num_layers'], 
                      dropout=CONFIG['dropout'])
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=CONFIG['weight_decay'])

    print(f"Starting Walk-Forward Validation (Fine-tuning approach)...")
    print(f"Initial training window: {args.initial_train_days} days.")
    print(f"Walk-forward begins from day {args.initial_train_days + 1}.")
    
    start_time = time.time()
    
    for t in tqdm(range(args.initial_train_days, total_days)):
        
        scaler = StandardScaler()
        scaler.fit(features_raw[:t])  # t 이전만으로 fit
        features_scaled = scaler.transform(features_raw[:t+1])  # 한번에 transform
      
        X, y = create_sequences(features_scaled, targets_raw[:t+1], args.lookback)
        
        X_train = X[:-1]
        y_train = y[:-1]
        X_test = X[-1:]
        y_test_actual = y[-1]
        
        X_train_t = torch.tensor(X_train, dtype=torch.float32)
        y_train_t = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
        X_test_t = torch.tensor(X_test, dtype=torch.float32)
        
        train_loader = DataLoader(TensorDataset(X_train_t, y_train_t), batch_size=args.batch_size, shuffle=True)
        
        epochs_to_run = args.initial_epochs if t == args.initial_train_days else args.finetune_epochs
            
        model.train()
        for epoch in range(epochs_to_run):
            for batch_x, batch_y in train_loader:
                optimizer.zero_grad()
                outputs = model(batch_x)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                
        model.eval()
        with torch.no_grad():
            pred_prob = model(X_test_t).item()
            pred_class = 1 if pred_prob > 0.5 else 0
            
        predictions.append({
            'Date': dates_raw[t],
            'Actual_Target': y_test_actual,
            'Predicted_Target': pred_class,
            'Probability_Up': pred_prob
        })

    end_time = time.time()
    print(f"\nWalk-forward validation completed in {end_time - start_time:.2f} seconds.")
    
    results_df = pd.DataFrame(predictions)
    
    acc = accuracy_score(results_df['Actual_Target'], results_df['Predicted_Target'])
    f1 = f1_score(results_df['Actual_Target'], results_df['Predicted_Target'])
    cm = confusion_matrix(results_df['Actual_Target'], results_df['Predicted_Target'])
    rmse = np.sqrt(mean_squared_error(results_df['Actual_Target'], results_df['Probability_Up']))

    print("\n--- Walk-Forward Validation Results ---")
    print(f"Total Days Predicted: {len(results_df)}")
    print(f"Accuracy: {acc:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print("Confusion Matrix:")
    print(cm)
    
    os.makedirs('outputs', exist_ok=True)
    output_csv = 'outputs/walk_forward_lstm.csv'
    results_df.to_csv(output_csv, index=False)
    print(f"Saved detailed walk-forward predictions to {output_csv}")

if __name__ == "__main__":
    main()


import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import os

# ==========================================
# 페이지 기본 설정
# ==========================================
st.set_page_config(
    page_title="KOSPI AI Trading Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==========================================
# 데이터 로드 함수 (캐싱 적용)
# ==========================================
@st.cache_data
def load_summary_data():
    summary_path = "outputs/데이터 시각화/models_summary_report.csv"
    if os.path.exists(summary_path):
        return pd.read_csv(summary_path)
    return None

@st.cache_data
def load_model_data(model_name):
    # 파일 이름 매핑 (최종 3차 결과물 반영)
    file_map = {
        "LSTM": "outputs/lstm_walk_forward_predictions.csv",
        "Random_Forest": "outputs/rf_walk_forward_results.csv",
        "XGBoost": "src/퀀트팀_최종_성적/최종_featrue지우기/kospi_final_daily_results_3차.csv",
        "Logistic_Regression": "outputs/logistic_regression_walk_forward.csv"
    }
    
    path = file_map.get(model_name)
    if path and os.path.exists(path):
        df = pd.read_csv(path)
        # 컬럼 표준화
        if 'actual' in df.columns: df.rename(columns={'actual': 'Actual_Target'}, inplace=True)
        if 'Actual' in df.columns: df.rename(columns={'Actual': 'Actual_Target'}, inplace=True)
        if 'predicted' in df.columns: df.rename(columns={'predicted': 'Predicted_Target'}, inplace=True)
        if 'Predicted' in df.columns: df.rename(columns={'Predicted': 'Predicted_Target'}, inplace=True)
        return df
    return None

# ==========================================
# 사이드바 네비게이션
# ==========================================
st.sidebar.title("📈 KOSPI AI Trading")
st.sidebar.markdown("---")
page = st.sidebar.radio(
    "Navigation",
    ["1. Executive Summary", "2. Model Performance", "3. Backtest Analysis", "4. Feature Insights"]
)
st.sidebar.markdown("---")
st.sidebar.info("Developed with Streamlit\n\nGVIF Project")

# 데이터 로드
summary_df = load_summary_data()

# ==========================================
# Page 1: Executive Summary
# ==========================================
if page == "1. Executive Summary":
    st.title("🌟 Executive Summary")
    st.markdown("### 머신러닝 기반 코스피 방향성 예측 및 투자 전략 성과")
    st.markdown("데이터 수집부터 모델 학습, 그리고 실제 매매 시뮬레이션까지의 최종 결과 요약입니다.")
    
    if summary_df is not None:
        # 가장 높은 수익률 모델
        best_model_idx = summary_df['Strategy_Return(%)'].idxmax()
        best_model = summary_df.loc[best_model_idx]
        
        st.subheader(f"🏆 최우수 모델: **{best_model['Model']}**")
        
        # 지표 카드
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("최종 수익률", f"{best_model['Strategy_Return(%)']:.2f}%", 
                      f"{best_model['Strategy_Return(%)'] - best_model['KOSPI_Return(%)']:.2f}% vs KOSPI")
        with col2:
            st.metric("KOSPI (B&H)", f"{best_model['KOSPI_Return(%)']:.2f}%")
        with col3:
            st.metric("최대 낙폭 (MDD)", f"{best_model['MDD(%)']:.2f}%")
        with col4:
            st.metric("샤프 지수 (Sharpe)", f"{best_model['Sharpe_Ratio']:.2f}")

        st.markdown("---")
        st.subheader("📊 전체 모델 성과 비교 리더보드")
        
        st.dataframe(
            summary_df.style.highlight_max(subset=['Strategy_Return(%)', 'Sharpe_Ratio', 'Accuracy'], color='rgba(0, 255, 0, 0.2)')
                             .highlight_min(subset=['MDD(%)'], color='rgba(255, 0, 0, 0.2)')
                             .format({
                                 'Strategy_Return(%)': '{:.2f}%',
                                 'KOSPI_Return(%)': '{:.2f}%',
                                 'MDD(%)': '{:.2f}%',
                                 'Sharpe_Ratio': '{:.2f}',
                                 'Accuracy': '{:.2%}'
                             }),
            use_container_width=True
        )
    else:
        st.warning("요약 데이터(csv)를 찾을 수 없습니다. 백테스트를 먼저 실행해 주세요.")

# ==========================================
# Page 2: Model Performance
# ==========================================
elif page == "2. Model Performance":
    st.title("🧠 Machine Learning Performance")
    st.markdown("모델들이 시장의 방향성을 얼마나 정확하게 맞췄는지 비교합니다.")
    
    if summary_df is not None:
        st.subheader("Accuracy & F1-Score 비교")
        fig = go.Figure()
        fig.add_trace(go.Bar(x=summary_df['Model'], y=summary_df['Accuracy'], name='Accuracy', marker_color='skyblue'))
        fig.add_trace(go.Bar(x=summary_df['Model'], y=summary_df['F1-Score'], name='F1-Score', marker_color='lightgreen'))
        fig.update_layout(barmode='group', template='plotly_white', yaxis_tickformat='.1%')
        st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("---")
        st.subheader("📈 모델별 누적 정확도 추이 (Cumulative Accuracy)")
        st.markdown("예측이 시간이 지남에 따라 얼마나 안정적으로 유지되는지 보여줍니다.")
        
        img_cols = st.columns(2)
        models_to_show = [
            ("LSTM", "lstm"), 
            ("Random Forest", "randomforest"), 
            ("XGBoost", "xgboost"), 
            ("Logistic Regression", "logisticregression")
        ]
        
        for i, (disp_name, file_name) in enumerate(models_to_show):
            img_path = f"outputs/데이터 시각화/cumulative_accuracy_{file_name}.png"
            with img_cols[i % 2]:
                if os.path.exists(img_path):
                    st.image(img_path, caption=f"{disp_name} 정확도 추이", use_container_width=True)
                else:
                    st.info(f"{disp_name} 정확도 이미지 파일 없음")

        st.markdown("---")
        st.subheader("🎯 모델별 상세 예측 분포 (Confusion Matrix)")
        selected_cm = st.selectbox("분석할 모델을 선택하세요:", summary_df['Model'].tolist())
        model_df = load_model_data(selected_cm)
        
        if model_df is not None:
            cm = pd.crosstab(model_df['Actual_Target'], model_df['Predicted_Target'], 
                             rownames=['실제(Actual)'], colnames=['예측(Predicted)'], margins=True)
            
            col_a, col_b = st.columns([1, 2])
            with col_a:
                st.write("**Confusion Matrix Table**")
                st.dataframe(cm)
            with col_b:
                st.markdown(f"""
                **{selected_cm} 분석:**
                - 실제 상승(1)을 맞춘 횟수: `{cm.loc[1, 1] if 1 in cm.index and 1 in cm.columns else 0}`
                - 실제 하락(0)을 맞춘 횟수: `{cm.loc[0, 0] if 0 in cm.index and 0 in cm.columns else 0}`
                - 전체 예측 수: `{len(model_df)}`
                """)
        else:
            st.error("상세 데이터를 불러올 수 없습니다.")

# ==========================================
# Page 3: Backtest Analysis
# ==========================================
elif page == "3. Backtest Analysis":
    st.title("📈 Backtest Analysis")
    st.markdown("구현된 매매 로직에 따른 모델별 상세 수익률 곡선과 매매 지점(Buy/Sell)입니다.")
    
    models = ["XGBoost", "Random_Forest", "LSTM", "Logistic_Regression"]
    selected_backtest = st.selectbox("분석할 모델을 선택하세요:", models)
    
    html_path = f"outputs/데이터 시각화/backtest_chart_{selected_backtest.lower()}.html"
    
    if os.path.exists(html_path):
        with open(html_path, 'r', encoding='utf-8') as f:
            html_data = f.read()
            st.components.v1.html(html_data, height=700, scrolling=True)
    else:
        st.error(f"백테스트 HTML 파일({html_path})이 없습니다.")

# ==========================================
# Page 4: Feature Insights
# ==========================================
elif page == "4. Feature Insights":
    st.title("🔍 Feature Insights")
    st.markdown("인공지능이 코스피 예측을 위해 가장 중요하게 참고한 지표들입니다.")
    
    fi_options = {
        "Random Forest": "outputs/rf_feature_importance.csv",
        "XGBoost (Final 3차)": "src/퀀트팀_최종_성적/최종_featrue지우기/kospi_final_feature_importances_3차.csv"
    }
    
    selected_fi = st.selectbox("분석할 모델을 선택하세요:", list(fi_options.keys()))
    fi_path = fi_options[selected_fi]
    
    if os.path.exists(fi_path):
        fi_df = pd.read_csv(fi_path)
        # XGBoost는 컬럼명이 'feature', 'importance' 또는 다른 이름일 수 있으므로 유연하게 처리
        if 'importance' not in fi_df.columns and len(fi_df.columns) >= 2:
            fi_df.columns = ['feature', 'importance']
            
        st.subheader(f"{selected_fi} 중요 지표 TOP 15")
        
        top_features = fi_df.head(15).sort_values(by='importance', ascending=True)
        fig = go.Figure(go.Bar(
            x=top_features['importance'],
            y=top_features['feature'],
            orientation='h',
            marker=dict(color='orange')
        ))
        fig.update_layout(template='plotly_white', height=500, xaxis_title="Importance")
        st.plotly_chart(fig, use_container_width=True)
        
        st.markdown(f"""
        **💡 {selected_fi} 분석 결과 요약:**
        - 이 모델은 거시경제 및 기술적 지표 중 특정 변수들에 가중치를 높게 두었습니다.
        - 상위 지표들의 변동을 통해 향후 코스피의 방향성을 가늠해 볼 수 있습니다.
        """)
    else:
        st.info(f"선택한 모델의 변수 중요도 파일({fi_path})이 없습니다.")

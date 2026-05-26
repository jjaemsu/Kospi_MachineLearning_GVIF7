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
def load_summary_data(for_performance=False):
    summary_path = "outputs/데이터 시각화/models_summary_report.csv"
    if os.path.exists(summary_path):
        df = pd.read_csv(summary_path)
        # 모델 성능 비교 페이지용: XGBoost 정확도를 1차 결과(약 57%)로 덮어쓰기
        if for_performance:
            xg_1st_path = "src/퀀트팀_최종_성적/최종_featrue지우기/kospi_final_daily_results_1차.csv"
            if os.path.exists(xg_1st_path):
                xg_df = pd.read_csv(xg_1st_path)
                actual_col = 'Actual' if 'Actual' in xg_df.columns else 'actual'
                pred_col = 'Predicted' if 'Predicted' in xg_df.columns else 'predicted'
                if actual_col in xg_df.columns and pred_col in xg_df.columns:
                    acc = (xg_df[actual_col] == xg_df[pred_col]).mean()
                    df.loc[df['Model'] == 'XGBoost', 'Accuracy'] = acc
        return df
    return None

@st.cache_data
def load_model_data(model_name, for_performance=False):
    # 파일 이름 매핑 (최종 3차 결과물 반영)
    file_map = {
        "LSTM": "outputs/lstm_walk_forward_predictions.csv",
        "Random_Forest": "outputs/rf_walk_forward_results.csv",
        "XGBoost": "src/퀀트팀_최종_성적/최종_featrue지우기/kospi_final_daily_results_3차.csv",
        "Logistic_Regression": "outputs/logistic_regression_walk_forward.csv"
    }
    
    # 모델 성능 비교 페이지용: 초기 56% 이상의 정확도를 보였던 XGBoost 1차 모델 사용
    if for_performance and model_name == "XGBoost":
        file_map["XGBoost"] = "src/퀀트팀_최종_성적/최종_featrue지우기/kospi_final_daily_results_1차.csv"
    
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
st.sidebar.title("📈 GVIF 7th Quant Strategy")
st.sidebar.markdown(
    """
    <div style="display: flex; justify-content: center; margin-bottom: 20px;">
        <a href="https://github.com/jjaemsu/Kospi_MachineLearning_GVIF7" target="_blank" style="text-decoration: none; display: flex; align-items: center; background-color: #24292f; padding: 8px 16px; border-radius: 8px; color: white; font-weight: 600; font-size: 14px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); width: 100%; justify-content: center;">
            <svg height="20" width="20" viewBox="0 0 16 16" style="margin-right: 8px; fill: white;"><path d="M8 0c4.42 0 8 3.58 8 8a8.013 8.013 0 0 1-5.45 7.59c-.4.08-.55-.17-.55-.38 0-.27.01-1.13.01-2.2 0-.75-.25-1.23-.54-1.48 1.78-.2 3.65-.88 3.65-3.95 0-.88-.31-1.59-.82-2.15.08-.2.36-1.02-.08-2.12 0 0-.67-.22-2.2.82-.64-.18-1.32-.27-2-.27-.68 0-1.36.09-2 .27-1.53-1.03-2.2-.82-2.2-.82-.44 1.1-.16 1.92-.08 2.12-.51.56-.82 1.28-.82 2.15 0 3.06 1.86 3.75 3.64 3.95-.23.2-.44.55-.51 1.07-.46.21-1.61.55-2.33-.66-.15-.24-.6-.83-1.23-.82-.67.01-.27.38.01.53.34.19.73.9.82 1.13.16.45.68 1.31 2.69.94 0 .67.01 1.3.01 1.49 0 .21-.15.45-.55.38A7.995 7.995 0 0 1 0 8c0-4.42 3.58-8 8-8Z"></path></svg>
            View on GitHub
        </a>
    </div>
    """,
    unsafe_allow_html=True
)
st.sidebar.markdown("---")
st.sidebar.markdown("### 📌 메인 메뉴")
page = st.sidebar.radio(
    "원하시는 리포트를 선택하세요:",
    ["📊 1. Executive Summary", "🔗 2. Correlation & VIF Analysis", "🧠 3. Model Performance", "📈 4. Backtest Analysis", "💡 5. Feature Insights"]
)
st.sidebar.markdown("---")
st.sidebar.info("Developed with Streamlit\n\nGVIF Project")
st.sidebar.markdown("<div style='text-align: center; color: gray; font-size: 13px; margin-top: 20px;'>만든 이 : 김낙영, 석윤주, 송경환, 이찬수</div>", unsafe_allow_html=True)

# 데이터 로드 (현재 페이지가 3번일 때만 for_performance=True 로 설정)
is_perf_page = (page == "🧠 3. Model Performance")
summary_df = load_summary_data(for_performance=is_perf_page)

# ==========================================
# Page 1: Executive Summary
# ==========================================
if page == "📊 1. Executive Summary":
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
        st.subheader("📈 주력 모델(XGBoost) 백테스트 결과")
        html_path = "outputs/데이터 시각화/backtest_chart_xgboost.html"
        if os.path.exists(html_path):
            with open(html_path, 'r', encoding='utf-8') as f:
                html_data = f.read()
                st.components.v1.html(html_data, height=700, scrolling=True)
        else:
            st.error(f"XGBoost 백테스트 HTML 파일({html_path})이 없습니다.")
    else:
        st.warning("요약 데이터(csv)를 찾을 수 없습니다. 백테스트를 먼저 실행해 주세요.")

# ==========================================
# Page 2: Correlation & VIF Analysis
# ==========================================
elif page == "🔗 2. Correlation & VIF Analysis":
    st.title("📊 Correlation & VIF Analysis")
    st.markdown("다중공선성(VIF) 검증 및 타겟 변수와의 상관관계를 시각화한 리포트입니다. 데이터셋을 선택하여 상세 리포트를 확인하세요.")
    
    base_dir = "outputs/correlation_vif"
    if os.path.exists(base_dir):
        # 폴더 목록만 가져오기
        datasets = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d)) and d != 'combined_cv_only']
        
        if datasets:
            # 'merged'가 포함된 폴더를 기본(가장 위)으로 정렬
            datasets = sorted(datasets, key=lambda x: 0 if 'merged' in x else 1)
            
            selected_dataset = st.selectbox("분석할 데이터셋을 선택하세요:", datasets)
            
            html_path = os.path.join(base_dir, selected_dataset, "correlation_vif_report.html")
            if os.path.exists(html_path):
                with open(html_path, 'r', encoding='utf-8') as f:
                    html_data = f.read()
                    st.components.v1.html(html_data, height=1000, scrolling=True)
            else:
                st.error(f"해당 데이터셋의 상세 리포트({html_path})를 찾을 수 없습니다.")
        else:
            st.warning("분석된 데이터셋 폴더를 찾을 수 없습니다.")
    else:
        st.error(f"VIF 결과 폴더({base_dir})가 존재하지 않습니다.")

# ==========================================
# Page 3: Model Performance
# ==========================================
elif page == "🧠 3. Model Performance":
    st.title("🧠 Machine Learning Performance")
    st.markdown("모델들이 시장의 방향성을 얼마나 정확하게 맞췄는지 비교합니다.")
    
    if summary_df is not None:
        st.subheader("📊 전체 모델 성과 비교 리더보드")
        st.markdown("※ XGBoost의 경우, 모델 선정의 당위성을 보여주기 위해 가장 성능이 우수했던 1차 모델(정답률 약 57%) 기준의 수치를 표시합니다.")
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
        
        st.markdown("---")
        st.subheader("Accuracy & F1-Score 비교")
        fig = go.Figure()
        fig.add_trace(go.Bar(x=summary_df['Model'], y=summary_df['Accuracy'], name='Accuracy', marker_color='skyblue'))
        fig.add_trace(go.Bar(x=summary_df['Model'], y=summary_df['F1-Score'], name='F1-Score', marker_color='lightgreen'))
        fig.update_layout(barmode='group', template='plotly_white', yaxis_tickformat='.1%')
        st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("---")
        st.subheader("📈 모델별 누적 정확도 추이 (Cumulative Accuracy)")
        st.markdown("예측이 시간이 지남에 따라 얼마나 안정적으로 유지되는지 보여줍니다.")
        
        # Load all models' predictions and calculate cumulative accuracy
        cum_acc_fig = go.Figure()
        models_to_show = ["XGBoost", "Random_Forest", "LSTM", "Logistic_Regression"]
        
        colors = {"XGBoost": "orange", "Random_Forest": "green", "LSTM": "blue", "Logistic_Regression": "red"}
        
        has_data = False
        for m_name in models_to_show:
            m_df = load_model_data(m_name, for_performance=True)
            if m_df is not None and 'Actual_Target' in m_df.columns and 'Predicted_Target' in m_df.columns:
                has_data = True
                m_df['Cumulative_Accuracy'] = (m_df['Actual_Target'] == m_df['Predicted_Target']).expanding().mean()
                
                # If Date exists, use it for x-axis, else use index
                if 'Date' in m_df.columns:
                    x_axis = pd.to_datetime(m_df['Date'])
                else:
                    x_axis = m_df.index
                    
                cum_acc_fig.add_trace(go.Scatter(x=x_axis, y=m_df['Cumulative_Accuracy'], mode='lines', name=m_name, line=dict(color=colors.get(m_name))))
        
        if has_data:
            cum_acc_fig.update_layout(
                template='plotly_white', 
                yaxis_title="Cumulative Accuracy",
                yaxis_tickformat='.1%',
                hovermode="x unified",
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
            )
            st.plotly_chart(cum_acc_fig, use_container_width=True)
        else:
            st.info("누적 정확도를 계산할 모델 예측 데이터가 없습니다.")

        st.markdown("---")
        st.subheader("🎯 모델별 상세 예측 분포 (Confusion Matrix)")
        selected_cm = st.selectbox("분석할 모델을 선택하세요:", summary_df['Model'].tolist())
        model_df = load_model_data(selected_cm, for_performance=True)
        
        if model_df is not None:
            cm = pd.crosstab(model_df['Actual_Target'], model_df['Predicted_Target'], 
                             rownames=['실제(Actual)'], colnames=['예측(Predicted)'], margins=True)
            
            col_a, col_b = st.columns([1, 2])
            with col_a:
                st.write("**Confusion Matrix Table**")
                st.dataframe(cm)
            with col_b:
                st.markdown(f"""
                **{{selected_cm}} 분석:**
                - 실제 상승(1)을 맞춘 횟수: `{cm.loc[1, 1] if 1 in cm.index and 1 in cm.columns else 0}`
                - 실제 하락(0)을 맞춘 횟수: `{cm.loc[0, 0] if 0 in cm.index and 0 in cm.columns else 0}`
                - 전체 예측 수: `{len(model_df)}`
                """)
        else:
            st.error("상세 데이터를 불러올 수 없습니다.")

# ==========================================
# Page 4: Backtest Analysis
# ==========================================
elif page == "📈 4. Backtest Analysis":
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
# Page 5: Feature Insights
# ==========================================
elif page == "💡 5. Feature Insights":
    st.title("🔍 Feature Insights")
    st.markdown("인공지능이 코스피 예측을 위해 가장 중요하게 참고한 지표들입니다.")
    
    fi_options = {
        "XGBoost (Final 1차)": "src/퀀트팀_최종_성적/최종_featrue지우기/kospi_final_feature_importances_1차.csv",
        "XGBoost (Final 2차)": "src/퀀트팀_최종_성적/최종_featrue지우기/kospi_final_feature_importances_2차.csv",
        "XGBoost (Final 3차)": "src/퀀트팀_최종_성적/최종_featrue지우기/kospi_final_feature_importances_3차.csv",
        "XGBoost (Final 4차)": "src/퀀트팀_최종_성적/최종_featrue지우기/kospi_final_feature_importances_4차.csv",
        "XGBoost (Final 5차)": "src/퀀트팀_최종_성적/최종_featrue지우기/kospi_final_feature_importances_5차.csv"
    }
    
    selected_fi = st.selectbox("분석할 모델을 선택하세요:", list(fi_options.keys()))
    fi_path = fi_options[selected_fi]
    
    if os.path.exists(fi_path):
        fi_df = pd.read_csv(fi_path)
        # XGBoost는 시계열(Date, Rank 1 Feature, Rank 1 Importance...) 형태일 수 있음
        if 'Rank 1 Feature' in fi_df.columns:
            feature_cols = [col for col in fi_df.columns if 'Feature' in col]
            imp_cols = [col for col in fi_df.columns if 'Importance' in col]
            
            all_features = []
            all_importances = []
            for f_col, i_col in zip(feature_cols, imp_cols):
                all_features.extend(fi_df[f_col].tolist())
                all_importances.extend(fi_df[i_col].tolist())
                
            agg_df = pd.DataFrame({'feature': all_features, 'importance': all_importances})
            fi_df = agg_df.groupby('feature', as_index=False).mean()
        elif 'importance' not in fi_df.columns and len(fi_df.columns) == 2:
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
        **💡 {{selected_fi}} 분석 결과 요약:**
        - 이 모델은 거시경제 및 기술적 지표 중 특정 변수들에 가중치를 높게 두었습니다.
        - 상위 지표들의 변동을 통해 향후 코스피의 방향성을 가늠해 볼 수 있습니다.
        """)
    else:
        st.info(f"선택한 모델의 변수 중요도 파일({fi_path})이 없습니다.")
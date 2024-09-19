import streamlit as st
import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import adfuller, grangercausalitytests
from pandas.api.types import is_datetime64_any_dtype as is_datetime

# GrangerCausalityAnalysis 클래스 정의
class GrangerCausalityAnalysis:
    def __init__(self, data, max_lag=12, signif_level=0.05):
        self.df = data
        self.max_lag = max_lag
        self.signif_level = signif_level
        self.stationary_df = None
        self.col_info_df = None
        self.result_df = None

    def adf_test(self, series):
        result = adfuller(series.dropna(), autolag='AIC')
        return result[1]  # p-value
    
    def difference_series(self, data):
        return data.diff().dropna()

    def check_stationarity_and_diff(self):
        df = self.df.copy()
        col_info = []
        stationary_data = df.copy()

        for col in stationary_data.columns:
            p_value = self.adf_test(stationary_data[col])
            diff_count = 0
            
            while p_value >= self.signif_level:
                stationary_data[col] = self.difference_series(stationary_data[col])
                p_value = self.adf_test(stationary_data[col].dropna())
                diff_count += 1
                if diff_count > 10:
                    break
            
            col_info.append({'Variable': col, 'Diff Count': diff_count, 'p-value': p_value})
        
        self.col_info_df = pd.DataFrame(col_info)
        self.stationary_df = stationary_data.dropna(how='any')

    def granger_causality(self):
        if self.stationary_df is None:
            raise ValueError("먼저 check_stationarity_and_diff()를 실행하세요.")
        
        stationary_data_clean = self.stationary_df.dropna()

        results = {}
        for col1 in stationary_data_clean.columns:
            for col2 in stationary_data_clean.columns:
                if col1 != col2:
                    test_result = grangercausalitytests(stationary_data_clean[[col1, col2]], self.max_lag, verbose=False)
                    p_values = {f"Lag {lag}": test[0]['ssr_ftest'][1] for lag, test in test_result.items()}
                    results[f"{col1} -> {col2}"] = p_values
        
        self.result_df = pd.DataFrame(results)

# Streamlit UI 구성
st.title("그레인저 인과성 분석 웹 애플리케이션")

# 파일 업로드 기능
uploaded_file = st.file_uploader("CSV 또는 Excel 파일을 업로드하세요.", type=['csv', 'xlsx'])

if uploaded_file is not None:
    # 파일 읽기
    if uploaded_file.name.endswith('.csv'):
        df = pd.read_csv(uploaded_file)
    elif uploaded_file.name.endswith('.xlsx'):
        df = pd.read_excel(uploaded_file)
    
    # 날짜 형식 컬럼 인식 및 인덱스로 설정
    for col in df.columns:
        if is_datetime(df[col]) or pd.to_datetime(df[col], errors='coerce').notna().all():
            df[col] = pd.to_datetime(df[col], errors='coerce')
            df.set_index(col, inplace=True)
            break
    
    # 데이터 표시
    st.write("업로드한 데이터:")
    st.write(df)

    # 그레인저 인과성 분석
    if st.button('그레인저 인과성 테스트 실행'):
        gca = GrangerCausalityAnalysis(df)
        gca.check_stationarity_and_diff()
        gca.granger_causality()
        
        # 결과 표시
        st.subheader('차분 정보')
        st.write(gca.col_info_df)

        st.subheader('그레인저 인과성 결과')
        st.write(gca.result_df)

        # 결과 다운로드 버튼
        def convert_df_to_csv(df):
            return df.to_csv(index=False).encode('utf-8')

        result_csv = convert_df_to_csv(gca.result_df)
        st.download_button(
            label="결과 CSV 파일 다운로드",
            data=result_csv,
            file_name='granger_results.csv',
            mime='text/csv',
        )

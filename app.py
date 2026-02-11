import streamlit as st
import pandas as pd
import numpy as np
import os
import joblib

# 프로젝트 경로 설정
BASE_PATH = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_PATH, "models")

def load_models():
    models = {}
    if not os.path.exists(MODEL_DIR):
        return models
    for file in os.listdir(MODEL_DIR):
        if file.endswith(".joblib"):
            target_name = file.replace("model_rf_", "").replace(".joblib", "")
            models[target_name] = joblib.load(os.path.join(MODEL_DIR, file))
    return models

def get_features():
    feature_path = os.path.join(MODEL_DIR, "feature_list.txt")
    if os.path.exists(feature_path):
        with open(feature_path, "r", encoding="utf-8-sig") as f:
            return [line.strip() for line in f.readlines()]
    return []

# 페이지 설정
st.set_page_config(page_title="고분자 물성 예측 시뮬레이터", layout="wide")

st.title("고분자 물성 예측 시뮬레이터")
st.markdown("---")
st.markdown("실험 조건과 재료 배합비를 입력하여 예상되는 고분자의 물성(점도, Tg, 수율)을 실시간으로 확인합니다.")

# 모델 로드
models = load_models()
all_features = get_features()

if not models:
    st.error("학습된 모델을 찾을 수 없습니다. 먼저 모델 학습을 진행해 주세요.")
else:
    # 레이아웃 구성
    col1, col2 = st.columns([1, 2])

    with col1:
        st.subheader("실험 조건 입력")
        
        # 기본 공정 조건
        temp = st.slider("반응 온도 (°C)", 50, 100, 80)
        time = st.number_input("반응 시간 (hr)", 0.0, 24.0, 4.5)
        solid_pct = st.number_input("이론 고형분 (wt%)", 0.0, 100.0, 40.0)
        scale = st.number_input("Scale (g)", 0.0, 2000.0, 900.0)

        st.subheader("모노머 배합비 (함량 입력)")
        st.info("모노머 함량의 합계가 100(phr)이 되도록 입력하는 것을 권장합니다.")
        
        # 모노머 입력 동적 생성
        monomer_inputs = {}
        for feat in all_features:
            if feat.startswith("monomer_"):
                name = feat.replace("monomer_", "")
                monomer_inputs[feat] = st.number_input(f"{name} 함량 (phr)", 0.0, 1000.0, 0.0)

    with col2:
        st.subheader("예측 결과 대시보드")
        
        # 추론 데이터 구성
        input_dict = {
            '온도': temp,
            '반응시간': time,
            '이론 고형분(%)': solid_pct,
            'Scale': scale
        }
        input_dict.update(monomer_inputs)
        
        input_df = pd.DataFrame([input_dict])
        for col in all_features:
            if col not in input_df.columns:
                input_df[col] = 0
        input_df = input_df[all_features]

        # 결과 출력
        res_cols = st.columns(len(models))
        for i, (target, model) in enumerate(models.items()):
            prediction = model.predict(input_df)[0]
            with res_cols[i]:
                st.metric(label=f"예상 {target}", value=f"{prediction:.2f}")

        # 분석용 차트 (예시)
        st.markdown("---")
        st.info("입력된 조건 기반으로 최적화된 물성 범위를 계산 중입니다.")
        
        # 여기에 추가적인 시각화 코드 삽입 가능
        st.write("입력 피처 요약:")
        st.dataframe(input_df.T.rename(columns={0: "값"}))

st.sidebar.markdown("### 프로젝트 정보")
st.sidebar.text("개발: 안현찬 (세계화학공업(주))")
st.sidebar.text(f"최종 업데이트: 2026-02-11")

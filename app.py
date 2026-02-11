import streamlit as st
import pandas as pd
import numpy as np
import os
import joblib

# 프로젝트 경로 설정 (상대 경로 적용)
BASE_PATH = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_PATH, "models")

def load_all_models():
    synthesis_models = {}
    coating_models = {}
    if not os.path.exists(MODEL_DIR):
        return synthesis_models, coating_models
        
    for file in os.listdir(MODEL_DIR):
        if not file.endswith(".joblib"):
            continue
        
        full_path = os.path.join(MODEL_DIR, file)
        if "model_rf_adhesion" in file:
            coating_models['점착력'] = joblib.load(full_path)
        else:
            target_name = file.replace("model_rf_", "").replace(".joblib", "")
            synthesis_models[target_name] = joblib.load(full_path)
            
    return synthesis_models, coating_models

def get_feature_list(filename):
    path = os.path.join(MODEL_DIR, filename)
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8-sig") as f:
            return [line.strip() for line in f.readlines()]
    return []

# 페이지 설정
st.set_page_config(page_title="Polymer Property Simulator", layout="wide")

# 모델 및 피처 로드
syn_models, coat_models = load_all_models()
syn_features = get_feature_list("feature_list.txt")
coat_features = get_feature_list("coating_feature_list.txt")

st.title("AI 고분자 물성 시뮬레이션 시스템")
st.markdown("---")

tab1, tab2 = st.tabs(["🧪 합성 시뮬레이터", "🏗️ 도포 시뮬레이터"])

with tab1:
    st.header("중합 공정 및 합성 물성 예측")
    if not syn_models:
        st.error("합성 모델을 찾을 수 없습니다.")
    else:
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.subheader("실험 조건 입력")
            
            # 기본 공정 조건
            temp = st.slider("반응 온도 (°C)", 50, 100, 83, key="syn_temp")
            time = st.number_input("반응 시간 (hr)", 0.0, 24.0, 4.75, key="syn_time")
            solid_pct = st.number_input("이론 고형분 (wt%)", 0.0, 100.0, 48.0, key="syn_solid")
            scale = st.number_input("Scale (g)", 0.0, 2000.0, 524.27, key="syn_scale")

            st.subheader("모노머 배합비 (phr)")
            st.info("합계가 100 phr이 되도록 입력을 권장합니다.")
            sum_placeholder = st.empty()
            
            default_monomers = {"monomer_BA": 89.7, "monomer_MMA": 9.0, "monomer_AA": 1.3}
            monomer_inputs = {}
            
            for feat in syn_features:
                if feat.startswith("monomer_"):
                    name = feat.replace("monomer_", "")
                    default_val = default_monomers.get(feat, 0.0)
                    monomer_inputs[feat] = st.number_input(f"{name} 함량", 0.0, 1000.0, default_val, key=f"syn_{feat}")
            
            total_phr = sum(monomer_inputs.values())
            if abs(total_phr - 100.0) > 0.01:
                sum_placeholder.warning(f"현재 합계: {total_phr:.2f} phr")
            else:
                sum_placeholder.success(f"현재 합계: {total_phr:.2f} phr (정상)")

        with col2:
            st.subheader("합성 결과 예측 대시보드")
            
            input_dict = {
                '온도': temp,
                '반응시간': time,
                '이론 고형분(%)': solid_pct / 100.0,
                'Scale': scale
            }
            input_dict.update(monomer_inputs)
            
            input_df = pd.DataFrame([input_dict])
            for col in syn_features:
                if col not in input_df.columns:
                    input_df[col] = 0
            input_df = input_df[syn_features]

            res_cols = st.columns(len(syn_models))
            for i, (target, model) in enumerate(syn_models.items()):
                prediction = model.predict(input_df)[0]
                with res_cols[i]:
                    st.metric(label=f"예상 {target}", value=f"{prediction:.2f}")
            
            st.markdown("---")
            st.write("입력 데이터 상세:")
            st.dataframe(input_df.T.rename(columns={0: "값"}))

with tab2:
    st.header("코팅 공정 및 도포 성능 예측")
    if not coat_models:
        st.error("도포 모델을 찾을 수 없습니다.")
    else:
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.subheader("도포 조건 입력")
            
            # 도포량
            coat_weight = st.number_input("도포량 (g/m² 또는 #bar 등 수치)", 0.0, 50.0, 2.7, key="coat_weight")
            
            # 원단 선택 (fabric_ 피처 기반)
            fabric_options = [f.replace("fabric_", "") for f in coat_features if f.startswith("fabric_")]
            selected_fabric = st.selectbox("기재(원단) 선택", fabric_options, key="coat_fabric")
            
            st.subheader("첨가제 및 경화제 (%)")
            
            additive_inputs = {}
            for feat in coat_features:
                if feat.startswith("hardener_") or feat.startswith("additive_"):
                    name = feat.replace("hardener_", "[경화제] ").replace("additive_", "[첨가제] ")
                    additive_inputs[feat] = st.number_input(f"{name} 함량", 0.0, 20.0, 0.0, key=f"coat_{feat}")

        with col2:
            st.subheader("도포 성능 예측 결과")
            
            # 입력 딕셔너리 구성
            coat_input_dict = {'도포량_num': coat_weight}
            coat_input_dict.update(additive_inputs)
            
            # 원단 원-핫 인코딩
            for fabric in fabric_options:
                coat_input_dict[f"fabric_{fabric}"] = 1.0 if fabric == selected_fabric else 0.0
            
            coat_input_df = pd.DataFrame([coat_input_dict])
            # 모든 학습 피처 존재 확인 후 정렬
            for col in coat_features:
                if col not in coat_input_df.columns:
                    coat_input_df[col] = 0.0
            coat_input_df = coat_input_df[coat_features]
            
            # 예측 수행
            adhesion_pred = coat_models['점착력'].predict(coat_input_df)[0]
            
            st.metric(label="예상 점착력 (gf/25mm)", value=f"{adhesion_pred:.2f}")
            
            st.markdown("---")
            st.info("도포 모델은 경화제 종류와 기재 타입에 따른 점착력 변동을 예측합니다.")
            st.write("입력 조건 요약:")
            st.dataframe(coat_input_df.T.rename(columns={0: "값"}))

st.sidebar.markdown("### 프로젝트 관리")
st.sidebar.text("담당: 안현찬 (세계화학공업(주))")
st.sidebar.text("최종 업데이트: 2026-02-12")
st.sidebar.info("사용자 피드백을 반영하여 경로 동적화 및 교차 검증 시스템이 적용되었습니다.")

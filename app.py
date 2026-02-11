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

tab1, tab2, tab3 = st.tabs(["🧪 합성 시뮬레이터", "🏗️ 도포 시뮬레이터", "🎯 역설계 시뮬레이터"])

with tab1:
    # ... (생략된 기존 tab1 로직은 유지됨)
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
            
            # 입력 데이터 구성
            input_dict = {
                '온도': temp,
                '반응시간': time,
                '이론 고형분(%)': solid_pct / 100.0,
                'Scale': scale
            }
            input_dict.update(monomer_inputs)
            
            # 화학적 도메인 피처 추가 (실시간 계산)
            from scripts.chemical_db import get_chemical_features
            chem_f = get_chemical_features(monomer_inputs)
            input_dict.update(chem_f)
            
            input_df = pd.DataFrame([input_dict])
            
            # 피처 목록 동기화 및 순서 고정
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
            selected_fabric = st.selectbox("기재(원단) 선택", fabric_options, index=fabric_options.index("T45") if "T45" in fabric_options else 0, key="coat_fabric")
            
            st.subheader("첨가제 및 경화제 (%)")
            st.info("첨가제 및 경화제의 투입 비율(%)을 입력합니다.")
            coat_sum_placeholder = st.empty()
            
            # 도포 예시값 (Row 0 데이터 기준)
            default_additives = {
                "hardener_CX100": 1.0,
                "hardener_SV02": 0.7,
                "additive_AF-10": 0.05
            }
            
            additive_inputs = {}
            for feat in coat_features:
                if feat.startswith("hardener_") or feat.startswith("additive_"):
                    name = feat.replace("hardener_", "[경화제] ").replace("additive_", "[첨가제] ")
                    default_val = default_additives.get(feat, 0.0)
                    additive_inputs[feat] = st.number_input(f"{name} 함량", 0.0, 20.0, default_val, key=f"coat_{feat}")
            
            # 합계 표시
            total_coat_pct = sum(additive_inputs.values())
            coat_sum_placeholder.info(f"현재 첨가제/경화제 합계: {total_coat_pct:.3f} %")

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

with tab3:
    st.header("목표 물성 기반 역설계 (Inverse Design)")
    st.markdown("---")
    
    if not syn_models:
        st.error("학습된 합성 모델이 없어 역설계 기능을 사용할 수 없습니다.")
    else:
        st.info("원하는 목표 물성($T_g$ 등)을 입력하면, AI가 최적의 모노머 배합비를 추천합니다.")
        
        opt_col1, opt_col2 = st.columns([1, 2])
        
        with opt_col1:
            st.subheader("다중 목표 물성 설정")
            available_targets = list(syn_models.keys())
            selected_properties = st.multiselect("최적화 대상 물성 선택 (복수 선택 가능)", 
                                                 available_targets, 
                                                 default=["Tg"] if "Tg" in available_targets else [])
            
            targets_dict = {}
            if selected_properties:
                for prop in selected_properties:
                    st.markdown(f"**{prop} 설정**")
                    col_val, col_weight = st.columns(2)
                    
                    with col_val:
                        if "Tg" in prop:
                            val = st.number_input(f"목표 {prop}", -80.0, 100.0, -30.0, step=0.5, key=f"opt_val_{prop}")
                        elif "점도" in prop:
                            val = st.number_input(f"목표 {prop}", 0, 50000, 5000, key=f"opt_val_{prop}")
                        elif "입도" in prop:
                            val = st.number_input(f"목표 {prop}", 0, 1000, 150, key=f"opt_val_{prop}")
                        else:
                            val = st.number_input(f"목표 {prop}", 0.0, 10000.0, 100.0, key=f"opt_val_{prop}")
                    
                    with col_weight:
                        weight = st.slider(f"{prop} 가중치 (중요도)", 0.0, 2.0, 1.0, step=0.1, key=f"opt_weight_{prop}")
                    
                    targets_dict[prop] = {'target': val, 'weight': weight}
            
            st.subheader("공정 제약 조건")
            opt_temp = st.number_input("중합 온도 (°C)", 50, 120, 80, key="opt_temp")
            opt_time = st.number_input("반응 시간 (hr)", 0.0, 24.0, 4.5, key="opt_time")
            opt_solid = st.number_input("이론 고형분 (%)", 10.0, 70.0, 48.0, key="opt_solid")
            
            if st.button("최적 배합비 산출 시작 🚀", use_container_width=True):
                if not targets_dict:
                    st.warning("최소 하나 이상의 목표 물성을 설정해 주세요.")
                else:
                    from scripts.optimize_recipe import optimize_recipe
                    
                    params = {
                        '온도': opt_temp,
                        '반응시간': opt_time,
                        '이론 고형분(%)': opt_solid / 100.0,
                        'Scale': 500
                    }
                    
                    with st.spinner("다중 목표를 최적화하는 배합비를 계산 중입니다..."):
                        recipe, err = optimize_recipe(targets_dict, params)
                        
                        if recipe:
                            st.session_state['opt_result'] = recipe
                            st.session_state['opt_targets_dict'] = targets_dict
                        else:
                            st.error(f"오류 발생: {err}")

        with opt_col2:
            st.subheader("AI 추천 최적 배합비")
            
            if 'opt_result' in st.session_state and 'opt_targets_dict' in st.session_state:
                res = st.session_state['opt_result']
                targets = st.session_state['opt_targets_dict']
                
                target_summary = ", ".join([f"{k}({v['target']})" for k, v in targets.items()])
                st.success(f"설정된 다중 목표 [{target_summary}] 달성을 위한 최적 조합을 찾았습니다.")
                
                # 결과 테이블 구성
                res_data = []
                for m, v in res.items():
                    res_data.append({"항목": m, "함량 (phr)": v})
                res_df = pd.DataFrame(res_data)
                
                col_table, col_chart = st.columns([1, 1.2])
                with col_table:
                    st.table(res_df.set_index("항목"))
                
                with col_chart:
                    import plotly.express as px
                    fig = px.pie(res_df, values='함량 (phr)', names='항목', 
                                 title='추천 레시피 구성비',
                                 color_discrete_sequence=px.colors.qualitative.Pastel)
                    fig.update_layout(margin=dict(t=40, b=0, l=0, r=0))
                    st.plotly_chart(fig, use_container_width=True)
                
                st.info("💡 위 배합비를 '합성 시뮬레이터' 탭에 입력하여 실제 예측치를 상세 검증해 보세요.")
                
                if st.button("합성 시뮬레이터로 배합비 전송 📤", use_container_width=True):
                    # 합성 시뮬레이터의 세션 상태 키값 업데이트
                    for m, v in res.items():
                        key = f"syn_monomer_{m}"
                        st.session_state[key] = float(v)
                    st.success("배합비가 '합성 시뮬레이터' 탭으로 전송되었습니다. 해당 탭으로 이동하여 확인하세요.")
                    st.rerun()
            else:
                st.write("왼쪽에서 목표 설정을 완료한 후 버튼을 클릭해 주세요.")

st.sidebar.markdown("### 프로젝트 관리")
st.sidebar.text("담당: 안현찬 (세계화학공업(주))")
st.sidebar.text("최종 업데이트: 2026-02-12")

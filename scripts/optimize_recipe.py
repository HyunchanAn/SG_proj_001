import numpy as np
import pandas as pd
from scipy.optimize import differential_evolution
import joblib
import os
try:
    from scripts.chemical_db import get_chemical_features
except ImportError:
    from chemical_db import get_chemical_features

# 경로 설정
current_dir = os.path.dirname(os.path.abspath(__file__))
base_dir = os.path.dirname(current_dir)
model_dir = os.path.join(base_dir, "models")

def load_property_model(target="Tg"):
    # 타겟 명칭 정제 (파일명 규칙에 맞게)
    name = target.replace('%', 'pct').replace('(', '').replace(')', '').replace(' ', '')
    model_path = os.path.join(model_dir, f"model_rf_{name}.joblib")
    if os.path.exists(model_path):
        return joblib.load(model_path)
    return None

def load_feature_list():
    path = os.path.join(model_dir, "feature_list.txt")
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8-sig") as f:
            return [line.strip() for line in f.readlines()]
    return []

def optimize_recipe(target_property, target_value, fixed_params=None):
    """
    target_property: 'Tg', '점도(cP)', '수율(%)', '입도(nm)' 등
    target_value: 목표 수치
    fixed_params: {'온도': 80, '반응시간': 4, ...}
    """
    model = load_property_model(target_property)
    features = load_feature_list()
    
    if not model or not features:
        return None, f"'{target_property}' 모델 또는 피처 목록을 불러올 수 없습니다."

    # 최적화 대상 모노머 (주요 모노머들 자동 탐지)
    monomer_cols = [f for f in features if f.startswith("monomer_")]
    # 데이터가 많은 주요 3인방 위주로 범위 설정 (사용자 피드백 반영하여 MMA, BA, AA 등 포함)
    target_monomers = ["monomer_BA", "monomer_MMA", "monomer_AA", "monomer_2-EHA"]
    target_monomers = [m for m in target_monomers if m in monomer_cols]
    
    # Differential Evolution을 위한 경계값 (Monomer PHR: 0~100)
    bounds = [(0, 100) for _ in target_monomers]
    
    def objective(x):
        # x: 각 target_monomers의Phr 수치
        monomer_inputs = {m: 0.0 for m in monomer_cols}
        for i, m_name in enumerate(target_monomers):
            monomer_inputs[m_name] = x[i]
        
        # 화학적 피처 계산
        chem_f = get_chemical_features(monomer_inputs)
        
        # 입력 데이터프레임 구성
        input_dict = {
            '온도': fixed_params.get('온도', 80),
            '반응시간': fixed_params.get('반응시간', 4.5),
            '이론 고형분(%)': fixed_params.get('이론 고형분(%)', 0.48),
            'Scale': fixed_params.get('Scale', 500)
        }
        input_dict.update(monomer_inputs)
        input_dict.update(chem_f)
        
        input_df = pd.DataFrame([input_dict])
        input_df = input_df[features] # 학습 시 사용된 피처 순서와 동일하게 유지
        
        prediction = model.predict(input_df)[0]
        
        # PHR 합계가 100이 아닐 경우 페널티 부여
        total_phr = np.sum(x)
        penalty = (total_phr - 100.0)**2 * 1000 # 합계 제약 조건 강화
        
        return (prediction - target_value)**2 + penalty

    # 전역 최적화 실행 (Differential Evolution)
    # 수렴 및 전역 탐색 성능을 위해 파라미터 튜닝
    res = differential_evolution(objective, bounds, strategy='best1bin', 
                                  maxiter=100, popsize=20, tol=0.01, mutation=(0.5, 1), 
                                  recombination=0.7, seed=42)
    
    if res.success or res.fun < 10.0: # 어느 정도 근접하면 성공으로 간주 (페널티 포함 손실 함수)
        optimized_phr = {}
        # 최적화된 결과값 (PHR) 추출
        for i, m_name in enumerate(target_monomers):
            optimized_phr[m_name.replace("monomer_", "")] = max(0, res.x[i]) # 음수 방지
        
        # 최종 합계 보정 (합계가 100.0이 되도록 정규화)
        total = sum(optimized_phr.values())
        if total > 0:
            for k in optimized_phr:
                optimized_phr[k] = (optimized_phr[k] / total) * 100.0
        else:
            # 모든 값이 0인 경우 기본값 부여
            optimized_phr = {"BA": 100.0}
                
        return optimized_phr, None
    else:
        return None, f"최적의 배합비를 찾는 데 실패했습니다. (Loss: {res.fun:.4f})"

if __name__ == "__main__":
    # 테스트 코드
    import pandas as pd
    test_target = "Tg"
    test_val = -35.0
    params = {'온도': 80, '반응시간': 4.5, '이론 고형분(%)': 0.48, 'Scale': 500}
    print(f"--- 역설계 고도화 엔진 테스트 (목표 {test_target}: {test_val}) ---")
    recipe, err = optimize_recipe(test_target, test_val, params)
    if recipe:
        print("최적 배합비 추천 결과 (phr):")
        for m, v in recipe.items():
            print(f"- {m}: {v:.2f}")
        print(f"합계: {sum(recipe.values()):.2f}")
    else:
        print(f"에러: {err}")

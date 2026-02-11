import numpy as np
import pandas as pd
from scipy.optimize import minimize
import joblib
import os
try:
    from scripts.chemical_db import get_chemical_features
except ImportError:
    from chemical_db import get_chemical_features

# 경로 설정
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
model_dir = os.path.join(base_dir, "models")

def load_syn_model(target="Tg"):
    model_path = os.path.join(model_dir, f"model_rf_{target}.joblib")
    if os.path.exists(model_path):
        return joblib.load(model_path)
    return None

def load_feature_list():
    path = os.path.join(model_dir, "feature_list.txt")
    with open(path, "r", encoding="utf-8-sig") as f:
        return [line.strip() for line in f.readlines()]

def optimize_recipe(target_tg, fixed_params=None):
    """
    target_tg: 목표 Tg (Celsius)
    fixed_params: {'온도': 80, '반응시간': 4, ...}
    """
    model = load_syn_model("Tg")
    features = load_feature_list()
    
    if not model or not features:
        return None, "모델 또는 피처 목록을 불러올 수 없습니다."

    # 최적화 대상 모노머 (주요 3종으로 제한하여 예시 구현)
    target_monomers = ["monomer_BA", "monomer_MMA", "monomer_AA"]
    initial_guess = [80.0, 15.0, 5.0] # BA, MMA, AA
    bounds = [(0, 100), (0, 100), (0, 100)]
    
    def objective(x):
        # x: [BA_phr, MMA_phr, AA_phr]
        monomer_inputs = {m: 0.0 for m in features if m.startswith("monomer_")}
        monomer_inputs["monomer_BA"] = x[0]
        monomer_inputs["monomer_MMA"] = x[1]
        monomer_inputs["monomer_AA"] = x[2]
        
        # 화학적 피처 계산
        chem_f = get_chemical_features(monomer_inputs)
        
        # 입력 데이터프레임 구성
        input_dict = {
            '온도': fixed_params.get('온도', 80),
            '반응시간': fixed_params.get('반응시간', 4),
            '이론 고형분(%)': fixed_params.get('이론 고형분(%)', 0.5),
            'Scale': fixed_params.get('Scale', 500)
        }
        input_dict.update(monomer_inputs)
        input_dict.update(chem_f)
        
        input_df = pd.DataFrame([input_dict])
        input_df = input_df[features] # 순서 고정
        
        pred_tg = model.predict(input_df)[0]
        return (pred_tg - target_tg)**2

    # 제약 조건: PHR 합계 = 100
    cons = ({'type': 'eq', 'fun': lambda x:  np.sum(x) - 100.0})
    
    res = minimize(objective, initial_guess, method='SLSQP', bounds=bounds, constraints=cons)
    
    if res.success:
        optimized_phr = {
            "BA": res.x[0],
            "MMA": res.x[1],
            "AA": res.x[2]
        }
        return optimized_phr, None
    else:
        return None, "최적화에 실패했습니다."

if __name__ == "__main__":
    # 테스트 코드
    target = -30.0
    params = {'온도': 80, '반응시간': 4.5, '이론 고형분(%)': 0.48, 'Scale': 500}
    recipe, err = optimize_recipe(target, params)
    if recipe:
        print(f"목표 Tg: {target}C")
        print("최적 배합비 추천 (phr):")
        for m, v in recipe.items():
            print(f"- {m}: {v:.2f}")
    else:
        print(f"Error: {err}")

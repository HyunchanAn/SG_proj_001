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

def optimize_recipe(targets_dict, fixed_params=None):
    """
    targets_dict: {'Tg': {'target': -30, 'weight': 1.0}, '점도(cP)': {'target': 5000, 'weight': 0.5}, ...}
    fixed_params: {'온도': 80, '반응시간': 4.5, ...}
    """
    if not targets_dict:
        return None, "최소 하나 이상의 목표 물성을 설정해야 합니다."

    features = load_feature_list()
    if not features:
        return None, "피처 목록을 불러올 수 없습니다."

    # 모델들을 미리 로드
    models = {}
    for target_name in targets_dict:
        model = load_property_model(target_name)
        if model:
            models[target_name] = model
        else:
            return None, f"'{target_name}' 모델을 불러올 수 없습니다."

    # 최적화 대상 모노머 자동 탐지
    monomer_cols = [f for f in features if f.startswith("monomer_")]
    target_monomers = ["monomer_BA", "monomer_MMA", "monomer_AA", "monomer_2-EHA"]
    target_monomers = [m for m in target_monomers if m in monomer_cols]
    
    bounds = [(0, 100) for _ in target_monomers]
    
    def objective(x):
        monomer_inputs = {m: 0.0 for m in monomer_cols}
        for i, m_name in enumerate(target_monomers):
            monomer_inputs[m_name] = x[i]
        
        chem_f = get_chemical_features(monomer_inputs)
        
        input_dict = {
            '온도': fixed_params.get('온도', 80),
            '반응시간': fixed_params.get('반응시간', 4.5),
            '이론 고형분(%)': fixed_params.get('이론 고형분(%)', 0.48),
            'Scale': fixed_params.get('Scale', 500)
        }
        input_dict.update(monomer_inputs)
        input_dict.update(chem_f)
        
        input_df = pd.DataFrame([input_dict])
        input_df = input_df[features]
        
        # 통합 손실 함수 계산 (가중치 적용)
        total_loss = 0.0
        for target_name, config in targets_dict.items():
            pred = models[target_name].predict(input_df)[0]
            target_val = config['target']
            weight = config.get('weight', 1.0)
            
            # 물성별 스케일 차이가 크므로 상대 오차 또는 로그 스케일 고려 가능하나 
            # 여기서는 단순 가중치 제곱 오차 적용 (사용자가 가중치로 조절)
            total_loss += weight * ((pred - target_val) / (abs(target_val) + 1e-6))**2
        
        # PHR 합계 페널티
        total_phr = np.sum(x)
        penalty = (total_phr - 100.0)**2 * 100.0
        
        return total_loss + penalty

    res = differential_evolution(objective, bounds, strategy='best1bin', 
                                  maxiter=100, popsize=20, tol=0.01, mutation=(0.5, 1), 
                                  recombination=0.7, seed=42)
    
    if res.success or res.fun < 10.0:
        optimized_phr = {}
        for i, m_name in enumerate(target_monomers):
            optimized_phr[m_name.replace("monomer_", "")] = max(0, res.x[i])
        
        total = sum(optimized_phr.values())
        if total > 0:
            for k in optimized_phr:
                optimized_phr[k] = (optimized_phr[k] / total) * 100.0
        else:
            optimized_phr = {"BA": 100.0}
                
        return optimized_phr, None
    else:
        return None, f"최적의 배합비를 찾는 데 실패했습니다. (Loss: {res.fun:.4f})"

if __name__ == "__main__":
    import pandas as pd
    # 다중 목표 테스트
    test_targets = {
        'Tg': {'target': -35.0, 'weight': 1.0},
        '점도(cP)': {'target': 5000, 'weight': 0.5}
    }
    params = {'온도': 80, '반응시간': 4.5, '이론 고형분(%)': 0.48, 'Scale': 500}
    print(f"--- 다중 목표 역설계 엔진 테스트 ---")
    recipe, err = optimize_recipe(test_targets, params)
    if recipe:
        print("최적 배합비 추천 결과 (phr):")
        for m, v in recipe.items():
            print(f"- {m}: {v:.2f}")
        print(f"합계: {sum(recipe.values()):.2f}")
    else:
        print(f"에러: {err}")

import pandas as pd
import numpy as np
import os
import re

# 현재 스크립트 위치 기준 상위 디렉토리 경로 설정
script_dir = os.path.dirname(os.path.abspath(__file__))
base_dir = os.path.dirname(script_dir)
input_path = os.path.join(base_dir, "data_cleaned", "cleaned_synthesis_data.csv")
output_path = os.path.join(base_dir, "data_cleaned", "model_features.csv")

def extract_monomer_features(text):
    if pd.isna(text) or not isinstance(text, str) or text.strip() == "":
        return {}
    
    # 1. 괄호 안의 숫자(함량)를 포함한 화학명 쌍 추출
    # 하이픈(-), 점(.)을 포함한 화학명 및 숫자 추출 보강
    matches = re.findall(r'([a-zA-Z가-힣0-9\-\.]+)\s*\(?([\d.]+)\)?', text)
    
    features = {}
    known_additives = ["NDM", "AIBN", "V-65", "LPO"] # 분자량 조절제, 개시제 등 제외

    for name, val in matches:
        try:
            clean_name = name.strip('-').strip('.')
            
            # 필터링 1: 순수 숫자만 있는 경우 무시 (예: "1", "100")
            if clean_name.isdigit():
                continue
                
            # 필터링 2: 1글자 이하 무시 (예: "-", ".")
            if len(clean_name) < 2:
                continue

            # 필터링 3: Known Additives 제외 (또는 별도 피처로 빼야 함)
            # 여기서는 모델이 '주 모노머'로 오인하지 않도록 일단 제외하거나 'additive_'로 접두어 변경
            if clean_name.upper() in known_additives:
                # feat_name = "additive_" + clean_name 
                # (일단 기존 로직 유지를 위해 제외 처리하거나, 별도로 관리. 
                #  사용자는 '모노머'라고 생각하고 넣었을 수 있으므로... 
                #  하지만 역설계 최적화에서 20 phr씩 넣는걸 막으려면 monomer_ 접두어를 안 쓰는 게 맞음)
                continue 

            feat_name = "monomer_" + clean_name
            features[feat_name] = float(val)
        except:
            continue
    return features

def preprocess_for_model():
    try:
        if not os.path.exists(input_path):
            print(f"Error: Input file not found at {input_path}")
            return

        df = pd.read_csv(input_path)
        print(f"Loaded {len(df)} rows from cleaned synthesis data.")
        
        # 1. Monomer Feature Extraction
        monomer_data = df['모노머'].apply(extract_monomer_features).tolist()
        monomer_df = pd.DataFrame(monomer_data).fillna(0)
        print(f"Extracted {len(monomer_df.columns)} monomer features.")
        
        # 1-1. Add Chemical Domain Knowledge Features
        from chemical_db import get_chemical_features
        chem_features_list = [get_chemical_features(m_dict) for m_dict in monomer_data]
        chem_df = pd.DataFrame(chem_features_list)
        print(f"Added {len(chem_df.columns)} chemical domain features.")
        
        # 2. Select numerical process features
        process_cols = ['온도', '반응시간', 'Scale', '이론 고형분(%)']
        # Filter existing columns only
        existing_process = [c for c in process_cols if c in df.columns]
        process_df = df[existing_process].copy()
        for col in existing_process:
            process_df[col] = pd.to_numeric(process_df[col], errors='coerce')
        process_df = process_df.fillna(process_df.mean())
        
        # 3. Targets
        target_cols = ['수율(%)', '점도(cP)', 'Tg', '입도(nm)']
        existing_targets = [c for c in target_cols if c in df.columns]
        target_df = df[existing_targets].copy()
        for col in existing_targets:
            target_df[col] = pd.to_numeric(target_df[col], errors='coerce')
        
        # Combine
        final_df = pd.concat([process_df, monomer_df, chem_df, target_df], axis=1)
        
        # Drop rows where all existing targets are null
        final_df = final_df.dropna(subset=existing_targets, how='all')
        
        final_df.to_csv(output_path, index=False, encoding='utf-8-sig')
        print(f"Feature dataset successfully created: {output_path} ({len(final_df)} rows)")
        
    except Exception as e:
        print(f"Critical Error in preprocessing: {e}")

if __name__ == "__main__":
    preprocess_for_model()

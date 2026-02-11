import pandas as pd
import numpy as np
import os
import re

# 현재 스크립트 위치 기준 경로 설정
script_dir = os.path.dirname(os.path.abspath(__file__))
input_path = os.path.join(script_dir, "data_cleaned", "cleaned_synthesis_data.csv")
output_path = os.path.join(script_dir, "data_cleaned", "model_features.csv")

def extract_monomer_features(text):
    if pd.isna(text) or not isinstance(text, str) or text.strip() == "":
        return {}
    
    # 정규표현식 강화: 'BA 40', 'BA(40)', 'BA 40/EA 10', 'BA 40.5 EA 10' 등 대응
    # 1. 괄호 제거 및 슬래시/쉼표를 공백으로 치환하여 통일
    clean_text = re.sub(r'[\(\),/]', ' ', text)
    # 2. 이름과 숫자 쌍 추출 (예: BA 40)
    matches = re.findall(r'([a-zA-Z가-힣0-9.-]+)\s*([\d.]+)', clean_text)
    
    features = {}
    for name, val in matches:
        try:
            # 특수문자로 시작하거나 끝나는 이름 정제
            feat_name = "monomer_" + name.strip('-').strip('.')
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
        final_df = pd.concat([process_df, monomer_df, target_df], axis=1)
        
        # Drop rows where all existing targets are null
        final_df = final_df.dropna(subset=existing_targets, how='all')
        
        final_df.to_csv(output_path, index=False, encoding='utf-8-sig')
        print(f"Feature dataset successfully created: {output_path} ({len(final_df)} rows)")
        
    except Exception as e:
        print(f"Critical Error in preprocessing: {e}")

if __name__ == "__main__":
    preprocess_for_model()

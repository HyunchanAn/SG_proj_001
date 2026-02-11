import pandas as pd
import numpy as np
import os
import re

base_path = r"e:\Github\SG_proj_001"
input_path = os.path.join(base_path, "data_cleaned", "cleaned_synthesis_data.csv")
output_path = os.path.join(base_path, "data_cleaned", "model_features.csv")

def extract_monomer_features(text):
    if pd.isna(text) or not isinstance(text, str):
        return {}
    
    # Improved regex to handle various formats: "2EHA 40", "EA(50)", "AA 1.5"
    matches = re.findall(r'([a-zA-Z가-힣0-9]+)\s*\(?([\d.]+)\)?', text)
    features = {}
    for name, val in matches:
        try:
            # Ensure name is string and val is float
            feat_name = "monomer_" + str(name)
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

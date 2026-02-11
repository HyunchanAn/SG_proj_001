import pandas as pd
import numpy as np
import os
import re

# 현재 스크립트 위치 기준 경로 설정
script_dir = os.path.dirname(os.path.abspath(__file__))
input_path = os.path.join(script_dir, "data_cleaned", "cleaned_coating_data.csv")
output_path = os.path.join(script_dir, "data_cleaned", "coating_model_features.csv")

def parse_ratios(text, prefix):
    if pd.isna(text) or not isinstance(text, str):
        return {}
    # 형식: (CX100/1%)(SV02/0.7%)
    matches = re.findall(r'([a-zA-Z0-9_\uAC00-\uD7A3]+)/([\d.]+)', text)
    return {f'{prefix}_{name}': float(val) for name, val in matches}

def parse_val_in_bracket(text):
    if pd.isna(text) or not isinstance(text, str):
        return None
    # 숫자가 포함된 첫 번째 괄호 내용 추출: (2.7)(#3) -> 2.7
    match = re.search(r'\(([\d.]+)\)', text)
    if match:
        return float(match.group(1))
    return None

def parse_adhesion(text):
    if pd.isna(text) or not isinstance(text, str):
        return None
    # 형식: "(51,55)*(초/90/BA)" -> 숫자의 평균값 추출
    first_part = text.split('*')[0]
    nums = re.findall(r'[\d.]+', first_part)
    if not nums:
        return None
    try:
        float_nums = [float(n) for n in nums]
        return sum(float_nums) / len(float_nums)
    except:
        return None

def preprocess_coating_data():
    try:
        if not os.path.exists(input_path):
            print(f"Error: Input file not found at {input_path}")
            return

        df = pd.read_csv(input_path)
        print(f"Loaded {len(df)} rows from cleaned coating data.")

        # 1. Feature: 경화제 및 첨가제 파싱
        hardener_data = df['경화제'].apply(lambda x: parse_ratios(x, 'hardener')).tolist()
        additive_data = df['첨가제'].apply(lambda x: parse_ratios(x, 'additive')).tolist()
        
        hardener_df = pd.DataFrame(hardener_data).fillna(0)
        additive_df = pd.DataFrame(additive_data).fillna(0)

        # 2. Feature: 도포량 (수치)
        df['도포량_num'] = df['도포량'].apply(parse_val_in_bracket)

        # 3. Feature: 원단 (카테고리 -> 겟 더미)
        fabric_df = pd.get_dummies(df['원단'], prefix='fabric').astype(float)

        # 4. Target: 점착력
        df['점착력_target'] = df['점착력'].apply(parse_adhesion)

        # Combine Features
        features_df = pd.concat([
            df[['도포량_num']], 
            hardener_df, 
            additive_df, 
            fabric_df
        ], axis=1)

        # Combine with Target
        final_df = pd.concat([features_df, df[['점착력_target']]], axis=1)

        # Drop rows where target is missing
        final_df = final_df.dropna(subset=['점착력_target'])
        
        # Fill missing feature values with mean
        final_df = final_df.fillna(final_df.mean())

        final_df.to_csv(output_path, index=False, encoding='utf-8-sig')
        print(f"Coating feature dataset successfully created: {output_path} ({len(final_df)} rows)")

    except Exception as e:
        print(f"Critical Error in coating preprocessing: {e}")

if __name__ == "__main__":
    preprocess_coating_data()

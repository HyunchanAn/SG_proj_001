import pandas as pd
import os
import re

# 현재 스크립트 위치 기준 상위 디렉토리 경로 설정
script_dir = os.path.dirname(os.path.abspath(__file__))
base_path = os.path.dirname(script_dir)
output_dir = os.path.join(base_path, "data_cleaned")
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

def clean_numeric(value):
    if pd.isna(value):
        return value
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        # Remove units and extract numbers
        value = re.sub(r'[^\d.]+', '', value)
        if value == '' or value == '.':
            return None
    try:
        return float(value)
    except:
        return None

def process_synthesis_data():
    file_path = os.path.join(base_path, "Lab 합성 총괄_250401부터241031까지.csv")
    try:
        # Synthesis data uses Tab
        df = pd.read_csv(file_path, encoding='cp949', sep='\t', on_bad_lines='skip')
        
        numeric_cols = [
            '이론 고형분(%)', '측정 고형분(%)', '수율(%)', '전환율(%)', 
            '응집량(%)', 'pH', 'Tg', '점도(cP)', '입도(nm)', 'Scale'
        ]
        
        for col in numeric_cols:
            if col in df.columns:
                df[col] = df[col].apply(clean_numeric)
        
        output_path = os.path.join(output_dir, "cleaned_synthesis_data.csv")
        df.to_csv(output_path, index=False, encoding='utf-8-sig')
        print(f"Success: Synthesis data cleaned ({len(df)} rows).")
        return True
    except Exception as e:
        print(f"Error processing synthesis data: {e}")
        return False

def process_coating_data():
    file_path = os.path.join(base_path, "Lab 도포 총괄_250401부터241031까지.csv")
    try:
        # Try Tab first, then Comma
        df = None
        for s in ['\t', ',']:
            try:
                temp_df = pd.read_csv(file_path, encoding='cp949', sep=s, on_bad_lines='skip')
                if len(temp_df.columns) > 5:
                    df = temp_df
                    print(f"Coating data parsed with separator: {repr(s)}")
                    break
            except:
                continue
        
        if df is not None:
            df.columns = [c.strip() for c in df.columns]
            # Simple numeric cleaning for common cols if they exist
            # (Coating data might have different cols, but let's try basic ones)
            potential_numeric = ['두께', 'Viscosity', 'Solid']
            for col in df.columns:
                if any(p in col for p in potential_numeric):
                    df[col] = df[col].apply(clean_numeric)
            
            output_path = os.path.join(output_dir, "cleaned_coating_data.csv")
            df.to_csv(output_path, index=False, encoding='utf-8-sig')
            print(f"Success: Coating data cleaned ({len(df)} rows).")
            return True
        else:
            print("Failed to parse coating data with any separator.")
            return False
    except Exception as e:
        print(f"Error processing coating data: {e}")
        return False

print("--- Data Cleansing Execution ---")
s_ok = process_synthesis_data()
c_ok = process_coating_data()
print("--- Execution Finished ---")

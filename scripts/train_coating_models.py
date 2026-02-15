import pandas as pd
import numpy as np
import os
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.metrics import mean_absolute_error, r2_score
import joblib

# 현재 스크립트 위치 기준 상위 디렉토리 경로 설정
script_dir = os.path.dirname(os.path.abspath(__file__))
base_dir = os.path.dirname(script_dir)
input_path = os.path.join(base_dir, "data_cleaned", "coating_model_features.csv")
model_dir = os.path.join(base_dir, "models")
report_dir = os.path.join(base_dir, "reports")
if not os.path.exists(model_dir):
    os.makedirs(model_dir)
if not os.path.exists(report_dir):
    os.makedirs(report_dir)

def train_coating_models():
    if not os.path.exists(input_path):
        print(f"Error: Coating feature dataset not found at {input_path}")
        return []

    df = pd.read_csv(input_path, encoding='utf-8-sig')
    
    # Target: 점착력_target
    target_col = '점착력_target'
    feature_cols = [c for c in df.columns if c != target_col]
    
    X = df[feature_cols]
    y = df[target_col]
    
    print(f"Dataset Size: {len(df)} rows")
    
    # 1. K-Fold Cross Validation
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(RandomForestRegressor(n_estimators=100, max_depth=7, random_state=42), 
                                X, y, cv=kf, scoring='r2')
    cv_r2_mean = np.mean(cv_scores)
    cv_r2_std = np.std(cv_scores)
    
    # 2. Final Training & Test Evaluation
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = RandomForestRegressor(n_estimators=100, max_depth=7, random_state=42)
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    test_mae = mean_absolute_error(y_test, y_pred)
    test_r2 = r2_score(y_test, y_pred)
    
    print(f"--- Coating Model Results: {target_col} ---")
    print(f"CV R2 Score: {cv_r2_mean:.4f} (+/- {cv_r2_std:.4f})")
    print(f"Test R2 Score: {test_r2:.4f}")
    print(f"Test MAE: {test_mae:.4f}")
    
    # Save model
    model_path = os.path.join(model_dir, "model_rf_adhesion.joblib")
    joblib.dump(model, model_path)
    
    # Save feature list for coating inference
    with open(os.path.join(model_dir, "coating_feature_list.txt"), "w", encoding="utf-8-sig") as f:
        f.write("\n".join(feature_cols))
        
    return [{
        'Target': target_col,
        'DataPoints': len(df),
        'CV_R2_Mean': cv_r2_mean,
        'CV_R2_Std': cv_r2_std,
        'Test_R2': test_r2,
        'Test_MAE': test_mae
    }]

if __name__ == "__main__":
    print("Starting Coating Model Training...")
    metrics = train_coating_models()
    print("\nCoating Training Complete.")
    
    # Save metrics to existing report path (added to reports folder)
    report_path = os.path.join(report_dir, "training_metrics_coating.txt")
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("# Coating Model Training Report\n\n")
        f.write("| Target | Data Points | CV R2 Mean | CV R2 Std | Test R2 | Test MAE |\n")
        f.write("| --- | --- | --- | --- | --- | --- |\n")
        for m in metrics:
            f.write(f"| {m['Target']} | {m['DataPoints']} | {m['CV_R2_Mean']:.4f} | {m['CV_R2_Std']:.4f} | {m['Test_R2']:.4f} | {m['Test_MAE']:.4f} |\n")

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
input_path = os.path.join(base_dir, "data_cleaned", "model_features.csv")
model_dir = os.path.join(base_dir, "models")
report_dir = os.path.join(base_dir, "reports")
if not os.path.exists(model_dir):
    os.makedirs(model_dir)
if not os.path.exists(report_dir):
    os.makedirs(report_dir)

def train_property_models_rf():
    if not os.path.exists(input_path):
        print(f"Error: Feature dataset not found at {input_path}")
        return []

    df = pd.read_csv(input_path, encoding='utf-8-sig')
    
    # Define features and targets
    target_cols = ['수율(%)', '점도(cP)', 'Tg', '입도(nm)']
    feature_cols = [c for c in df.columns if c not in target_cols]
    
    X = df[feature_cols]
    
    results = []
    
    for target in target_cols:
        if target not in df.columns:
            continue
            
        # Drop rows where target is NaN for this specific model
        y_temp = df[target]
        valid_idx = y_temp.dropna().index
        
        if len(valid_idx) < 10:
            print(f"Skipping {target}: Not enough data ({len(valid_idx)} rows)")
            continue
            
        X_target = X.loc[valid_idx]
        y_target = y_temp.loc[valid_idx]
        
        # 1. K-Fold Cross Validation (일반화 성능 검증)
        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        cv_scores = cross_val_score(RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42), 
                                    X_target, y_target, cv=kf, scoring='r2')
        cv_r2_mean = np.mean(cv_scores)
        cv_r2_std = np.std(cv_scores)
        
        # 2. Final Training & Test Split Evaluation
        X_train, X_test, y_train, y_test = train_test_split(X_target, y_target, test_size=0.2, random_state=42)
        
        model = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)
        model.fit(X_train, y_train)
        
        y_pred = model.predict(X_test)
        test_mae = mean_absolute_error(y_test, y_pred)
        test_r2 = r2_score(y_test, y_pred)
        
        print(f"--- RandomForest Model Results: {target} ---")
        print(f"Data Points: {len(X_target)}")
        print(f"CV R2 Score: {cv_r2_mean:.4f} (+/- {cv_r2_std:.4f})")
        print(f"Test R2 Score: {test_r2:.4f}")
        print(f"Test MAE: {test_mae:.4f}")
        
        # Save model using joblib
        model_name = target.replace('%', 'pct').replace('(', '').replace(')', '')
        model_path = os.path.join(model_dir, f"model_rf_{model_name}.joblib")
        joblib.dump(model, model_path)
        
        results.append({
            'Target': target,
            'DataPoints': len(X_target),
            'CV_R2_Mean': cv_r2_mean,
            'CV_R2_Std': cv_r2_std,
            'Test_R2': test_r2,
            'Test_MAE': test_mae
        })
        
    # Save feature list for inference (after loop)
    with open(os.path.join(model_dir, "feature_list.txt"), "w", encoding="utf-8-sig") as f:
        f.write("\n".join(feature_cols))
        
    return results

if __name__ == "__main__":
    print("Starting AI Model Training (RandomForest with K-Fold CV)...")
    metrics = train_property_models_rf()
    print("\nTraining Complete.")
    
    # Save training report (in reports folder)
    report_path = os.path.join(report_dir, "training_metrics.txt")
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("# Model Training Report (RandomForest with Cross-Validation)\n\n")
        f.write("| Target | Data Points | CV R2 Mean | CV R2 Std | Test R2 | Test MAE |\n")
        f.write("| --- | --- | --- | --- | --- | --- |\n")
        for m in metrics:
            f.write(f"| {m['Target']} | {m['DataPoints']} | {m['CV_R2_Mean']:.4f} | {m['CV_R2_Std']:.4f} | {m['Test_R2']:.4f} | {m['Test_MAE']:.4f} |\n")

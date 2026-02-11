import pandas as pd
import numpy as np
import os
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score

base_path = r"e:\Github\SG_proj_001"
input_path = os.path.join(base_path, "data_cleaned", "model_features.csv")
model_dir = os.path.join(base_path, "models")
if not os.path.exists(model_dir):
    os.makedirs(model_dir)

def train_property_models():
    if not os.path.exists(input_path):
        print("Feature dataset not found.")
        return

    df = pd.read_csv(input_path)
    
    # Define features and targets
    target_cols = ['수율(%)', '점도(cP)', 'Tg', '입도(nm)']
    feature_cols = [c for c in df.columns if c not in target_cols]
    
    X = df[feature_cols]
    
    results = []
    
    for target in target_cols:
        # Drop rows where target is NaN for this specific model
        y_temp = df[target]
        valid_idx = y_temp.dropna().index
        
        if len(valid_idx) < 10:
            print(f"Skipping {target}: Not enough data ({len(valid_idx)} rows)")
            continue
            
        X_target = X.loc[valid_idx]
        y_target = y_temp.loc[valid_idx]
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X_target, y_target, test_size=0.2, random_state=42)
        
        # Train XGBoost
        model = xgb.XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42)
        model.fit(X_train, y_train)
        
        # Evaluate
        y_pred = model.predict(X_test)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        print(f"--- Model Results: {target} ---")
        print(f"Data Points: {len(X_target)}")
        print(f"MAE: {mae:.4f}")
        print(f"R2 Score: {r2:.4f}")
        
        # Save model
        model_path = os.path.join(model_dir, f"model_{target.replace('%', 'pct').replace('(', '').replace(')', '')}.json")
        model.save_model(model_path)
        
        results.append({
            'Target': target,
            'DataPoints': len(X_target),
            'MAE': mae,
            'R2': r2
        })
        
    return results

if __name__ == "__main__":
    print("Starting AI Model Training...")
    metrics = train_property_models()
    print("\nTraining Complete.")
    
    # Save training report
    report_path = os.path.join(base_path, "training_metrics.txt")
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("# Model Training Report\n\n")
        for m in metrics:
            f.write(f"- Target: {m['Target']}\n")
            f.write(f"  - Data Points: {m['DataPoints']}\n")
            f.write(f"  - MAE: {m.get('MAE', 0):.4f}\n")
            f.write(f"  - R2: {m.get('R2', 0):.4f}\n\n")

import joblib
import pandas as pd
import numpy as np
import os

base_path = r"e:\Github\SG_proj_001"
model_dir = os.path.join(base_path, "models")
feature_path = os.path.join(model_dir, "feature_list.txt")

def test_sensitivity():
    if not os.path.exists(feature_path):
        print("Feature list not found.")
        return

    with open(feature_path, "r", encoding="utf-8-sig") as f:
        all_features = [line.strip() for line in f.readlines()]

    models = {}
    for file in os.listdir(model_dir):
        if file.endswith(".joblib"):
            target_name = file.replace("model_rf_", "").replace(".joblib", "")
            models[target_name] = joblib.load(os.path.join(model_dir, file))

    if not models:
        print("No models found.")
        return

    # Base input (from S250421A)
    base_input = {feat: 0.0 for feat in all_features}
    base_input.update({
        '온도': 83.0,
        '반응시간': 4.75,
        '이론 고형분(%)': 0.48, # Note: in CSV it was 0.48, in app it's 48.0. Wait.
        'Scale': 524.27,
        'monomer_BA': 89.7,
        'monomer_MMA': 9.0,
        'monomer_AA': 1.3
    })

    output = []
    output.append("--- Feature Importance (Top 5) ---")
    for target, model in models.items():
        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1]
        output.append(f"Target: {target}")
        for i in range(min(5, len(importances))):
            output.append(f"  {all_features[indices[i]]}: {importances[indices[i]]:.4f}")
        
        if '온도' in all_features:
            temp_idx = all_features.index('온도')
            output.append(f"  온도 Importance: {importances[temp_idx]:.4f}")

    output.append("\n--- Temperature Sensitivity Test ---")
    temps = [50.0, 70.0, 83.0, 100.0]
    for target, model in models.items():
        output.append(f"Target: {target}")
        for t in temps:
            test_input = base_input.copy()
            test_input['온도'] = t
            df_test = pd.DataFrame([test_input])[all_features]
            pred = model.predict(df_test)[0]
            output.append(f"  Temp {t}: Prediction = {pred:.4f}")

    with open(os.path.join(base_path, "sensitivity_report.txt"), "w", encoding="utf-8") as rf:
        rf.write("\n".join(output))
    print("Report saved to sensitivity_report.txt")

if __name__ == "__main__":
    test_sensitivity()

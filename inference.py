import pandas as pd
import numpy as np
import os
import joblib

base_path = r"e:\Github\SG_proj_001"
model_dir = os.path.join(base_path, "models")

def predict_property(features_dict):
    # Load feature list
    feature_list_path = os.path.join(model_dir, "feature_list.txt")
    if not os.path.exists(feature_list_path):
        return "Error: Feature list not found."
    
    with open(feature_list_path, "r") as f:
        all_features = [line.strip() for line in f.readlines()]
    
    # Create input vector
    input_df = pd.DataFrame([features_dict])
    # Fill missing features with 0
    for col in all_features:
        if col not in input_df.columns:
            input_df[col] = 0
    input_df = input_df[all_features]
    
    predictions = {}
    for model_file in os.listdir(model_dir):
        if model_file.endswith(".joblib"):
            target_name = model_file.replace("model_rf_", "").replace(".joblib", "")
            model = joblib.load(os.path.join(model_dir, model_file))
            pred_val = model.predict(input_df)[0]
            predictions[target_name] = pred_val
            
    return predictions

if __name__ == "__main__":
    # Example Inference for testing
    test_input = {
        '온도': 80,
        '반응시간': 4.5,
        '이론 고형분(%)': 40,
        'monomer_2EHA': 50,
        'monomer_EA': 45,
        'monomer_AA': 5
    }
    
    print("--- AI Property Prediction Simulator ---")
    print(f"Input Conditions: {test_input}")
    res = predict_property(test_input)
    print(f"\nPredicted Results: {res}")

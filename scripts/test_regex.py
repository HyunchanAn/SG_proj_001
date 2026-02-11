import re

def extract_monomer_features(text):
    clean_text = re.sub(r'[\(\),/]', ' ', text)
    matches = re.findall(r'([a-zA-Z가-힣0-9.-]+)\s*([\d.]+)', clean_text)
    features = {}
    for name, val in matches:
        feat_name = "monomer_" + name.strip('-').strip('.')
        features[feat_name] = float(val)
    return features

test_cases = [
    "2-EHA 39.35",
    "BA 40/EA 10",
    "MMA(15.5)",
    "SR-1025 2.0",
    "4-HBA 5.0"
]

for tc in test_cases:
    print(f"Input: {tc} -> Output: {extract_monomer_features(tc)}")

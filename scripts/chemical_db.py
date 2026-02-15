# 주요 모노머의 화학적 물성 데이터베이스 (Chemical Domain Knowledge)
# 자료원: Polymer Handbook, Sigma-Aldrich, 모노머 제조사 TDS 등 참조

MONOMER_PROPERTIES = {
    "BA": {"tg": 219.15, "mw": 128.17, "polarity": 0.15},       # Butyl Acrylate (-54C)
    "2-EHA": {"tg": 203.15, "mw": 184.28, "polarity": 0.10},    # 2-Ethylhexyl Acrylate (-70C)
    "MMA": {"tg": 378.15, "mw": 100.12, "polarity": 0.30},      # Methyl Methacrylate (105C)
    "AA": {"tg": 379.15, "mw": 72.06, "polarity": 0.90},        # Acrylic Acid (106C)
    "MA": {"tg": 281.15, "mw": 86.09, "polarity": 0.40},        # Methyl Acrylate (8C)
    "EA": {"tg": 249.15, "mw": 100.12, "polarity": 0.25},       # Ethyl Acrylate (-24C)
    "BMA": {"tg": 293.15, "mw": 142.20, "polarity": 0.12},      # Butyl Methacrylate (20C)
    "St": {"tg": 373.15, "mw": 104.15, "polarity": 0.05},       # Styrene (100C)
    "2-HEMA": {"tg": 328.15, "mw": 130.14, "polarity": 0.85},   # 2-HEMA (55C)
    "2-HEA": {"tg": 258.15, "mw": 116.12, "polarity": 0.80},    # 2-HEA (-15C)
    "GMA": {"tg": 319.15, "mw": 142.15, "polarity": 0.50},      # Glycidyl Methacrylate (46C)
    "VAc": {"tg": 303.15, "mw": 86.09, "polarity": 0.35},       # Vinyl Acetate (30C)
    "AN": {"tg": 378.15, "mw": 53.06, "polarity": 0.75},        # Acrylonitrile (105C)
    "LMA": {"tg": 208.15, "mw": 254.41, "polarity": 0.05},      # Lauryl Methacrylate (-65C)
    "EMA": {"tg": 338.15, "mw": 114.14, "polarity": 0.20},      # Ethyl Methacrylate (65C)
    "IBOA": {"tg": 367.15, "mw": 208.30, "polarity": 0.10},     # Isobornyl Acrylate (94C)
    "IBOMA": {"tg": 423.15, "mw": 222.32, "polarity": 0.08},    # Isobornyl Methacrylate (150C)
    "MAA": {"tg": 501.15, "mw": 86.09, "polarity": 0.85},       # Methacrylic Acid (228C)
    "CHMA": {"tg": 377.15, "mw": 168.23, "polarity": 0.10},     # Cyclohexyl Methacrylate (104C)
    "4-HBA": {"tg": 233.15, "mw": 144.17, "polarity": 0.75},    # 4-Hydroxybutyl Acrylate (-40C)
}

# 기본값 (데이터 부재 시 사용)
DEFAULT_PROPS = {"chem_avg_tg": 298.15, "chem_avg_mw": 100.0, "chem_avg_polarity": 0.20}

def get_chemical_features(monomer_ratios):
    """
    monomer_ratios: { 'monomer_BA': 80.0, 'monomer_MMA': 20.0 }
    반환: 가중 평균된 Tg, MW, Polarity 지표
    """
    total_phr = sum(monomer_ratios.values())
    if total_phr == 0:
        return DEFAULT_PROPS
    
    avg_tg = 0.0
    avg_mw = 0.0
    avg_polarity = 0.0
    
    # 루프 내에서 사용할 기본 물성값 (dict 형태)
    fallback_props = {"tg": 298.15, "mw": 100.0, "polarity": 0.20}

    for name, phr in monomer_ratios.items():
        base_name = name.replace("monomer_", "")
        props = MONOMER_PROPERTIES.get(base_name, fallback_props)
        
        weight = phr / total_phr
        avg_tg += props["tg"] * weight
        avg_mw += props["mw"] * weight
        avg_polarity += props["polarity"] * weight
        
    return {
        "chem_avg_tg": avg_tg,
        "chem_avg_mw": avg_mw,
        "chem_avg_polarity": avg_polarity
    }

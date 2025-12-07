import numpy as np
import pandas as pd

def prepare_subtype_vector(features: dict, feature_order: list):
    row = []
    for f in feature_order:
        row.append(features.get(f, 0))
    return np.array(row).reshape(1, -1)

def preprocess_psg_features(feature_dict):
    return pd.DataFrame([feature_dict])

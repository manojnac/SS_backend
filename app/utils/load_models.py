import joblib
import json
import os

BASE = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../models"))

psg_model = joblib.load(f"{BASE}/psg_model.pkl")
subtype_model = joblib.load(f"{BASE}/subtype_model.pkl")
subtype_scaler = joblib.load(f"{BASE}/subtype_scaler.pkl")

with open(f"{BASE}/subtype_features.json") as f:
    subtype_features = json.load(f)

with open(f"{BASE}/subtype_label_map.json") as f:
    subtype_label_map = json.load(f)

print("[INFO] Models loaded successfully.")

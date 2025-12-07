from app.utils.firestore import db
from app.utils.load_models import subtype_model, subtype_scaler, subtype_features, subtype_label_map
from app.utils.preprocess import prepare_subtype_vector

def classify_subtype(user_id, feature_dict):
    vector = prepare_subtype_vector(feature_dict, subtype_features)
    vector_scaled = subtype_scaler.transform(vector)
    cluster = int(subtype_model.predict(vector_scaled)[0])

    subtype_label = subtype_label_map.get(str(cluster), "Unknown Subtype")

    db.collection("users").document(user_id).collection("analysis").document("subtype").set({
        "cluster_id": cluster,
        "subtype": subtype_label,
        "features": feature_dict
    })

    return cluster, subtype_label

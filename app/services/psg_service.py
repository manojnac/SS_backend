from app.utils.firestore import db
import shap
from app.utils.load_models import psg_model
from app.utils.preprocess import preprocess_psg_features

def analyze_psg(user_id, psg_features):
    X = preprocess_psg_features(psg_features)
    prediction = float(psg_model.predict(X)[0])
    
    explainer = shap.TreeExplainer(psg_model)
    shap_values = explainer.shap_values(X)[0].tolist()

    db.collection("users").document(user_id).collection("analysis").document("psg").set({
        "features": psg_features,
        "isi_predicted": prediction,
        "shap": shap_values
    })

    return prediction, shap_values

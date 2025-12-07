from fastapi import APIRouter
from app.utils.firestore import db
import numpy as np

router = APIRouter()

@router.get("/")
def global_correlation():
    docs = db.collection("users").stream()

    isi_scores = []
    phq_scores = []

    for user in docs:
        user_id = user.id
        isi_ref = db.collection("users").document(user_id).collection("assessments").document("isi").get()
        phq_ref = db.collection("users").document(user_id).collection("assessments").document("phq9").get()

        if isi_ref.exists and phq_ref.exists:
            isi_scores.append(isi_ref.to_dict()["score"])
            phq_scores.append(phq_ref.to_dict()["score"])

    if len(isi_scores) < 2:
        return {"correlation": None, "message": "Not enough data"}

    corr = np.corrcoef(isi_scores, phq_scores)[0,1]

    return {"correlation": float(corr)}

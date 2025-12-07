from app.utils.scoring import get_severity, ISI_SEVERITY
from app.utils.firestore import db

def compute_isi(responses):
    total = sum(responses)
    severity = get_severity(total, ISI_SEVERITY)
    
     db.collection("users").document(user_id).collection("assessments").document("isi").set({
        "responses": responses,
        "score": total,
        "severity": severity
    })

    return total, severity

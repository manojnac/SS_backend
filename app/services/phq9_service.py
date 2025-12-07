from app.utils.scoring import get_severity, PHQ9_SEVERITY
from app.utils.firestore import db

def compute_phq9(responses):
    total = sum(responses)
    severity = get_severity(total, PHQ9_SEVERITY)
    
    db.collection("users").document(user_id).collection("assessments").document("phq9").set({
        "responses": responses,
        "score": total,
        "severity": severity
    })

    return total, severity

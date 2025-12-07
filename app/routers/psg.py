from fastapi import APIRouter
from app.models.request_models import PSGRequest
from app.models.response_models import PSGResponse
from app.services.psg_service import analyze_psg

router = APIRouter()

@router.post("/", response_model=PSGResponse)
def analyze_psg_signal(payload: PSGRequest):
    prediction, shap_values = analyze_psg(payload.subject_id)
    return PSGResponse(
        isi_score=prediction,
        features={"subject": payload.subject_id},
        shap_values=shap_values
    )

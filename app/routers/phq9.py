from fastapi import APIRouter
from app.models.request_models import PHQ9Request
from app.models.response_models import PHQ9Response
from app.services.phq9_service import compute_phq9

router = APIRouter()

@router.post("/", response_model=PHQ9Response)
def submit_phq9(payload: PHQ9Request):
    total, severity = compute_phq9(payload.responses)
    return PHQ9Response(total=total, severity=severity)

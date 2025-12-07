from fastapi import APIRouter
from app.models.request_models import ISIRequest
from app.models.response_models import ISIResponse
from app.services.isi_service import compute_isi

router = APIRouter()

@router.post("/", response_model=ISIResponse)
def submit_isi(payload: ISIRequest):
    total, severity = compute_isi(payload.responses, payload.user_id)
    return ISIResponse(total=total, severity=severity)

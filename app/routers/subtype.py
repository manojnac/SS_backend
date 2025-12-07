from fastapi import APIRouter
from app.models.request_models import SubtypeRequest
from app.models.response_models import SubtypeResponse
from app.services.subtype_service import classify_subtype

router = APIRouter()

@router.post("/", response_model=SubtypeResponse)
def predict_subtype(payload: SubtypeRequest):
    cluster, label = classify_subtype(payload.features)
    return SubtypeResponse(cluster_id=cluster, subtype=label)

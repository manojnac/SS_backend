from pydantic import BaseModel

class ISIResponse(BaseModel):
    total: int
    severity: str

class PHQ9Response(BaseModel):
    total: int
    severity: str

class PSGResponse(BaseModel):
    isi_score: float
    features: dict
    shap_values: list

class SubtypeResponse(BaseModel):
    subtype: str
    cluster_id: int

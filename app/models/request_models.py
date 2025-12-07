from pydantic import BaseModel
from typing import List

class ISIRequest(BaseModel):
    user_id: str
    responses: List[int]

class PHQ9Request(BaseModel):
    user_id: str
    responses: List[int]

class PSGRequest(BaseModel):
    subject_id: str

class SubtypeRequest(BaseModel):
    features: dict

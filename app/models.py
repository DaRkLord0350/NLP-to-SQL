from pydantic import BaseModel
from typing import Any, Optional, List

class PredictRequest(BaseModel):
    question: str

class PredictResponse(BaseModel):
    sql: str
    result: Any           # list/rows/string
    explanation: str
    error: Optional[str] = None

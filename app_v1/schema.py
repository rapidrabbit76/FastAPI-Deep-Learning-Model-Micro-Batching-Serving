from pydantic import BaseModel


class PredictTag(BaseModel):
    name: str
    prob: float

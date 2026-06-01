from pydantic import BaseModel


class RegressionPrediction(BaseModel):
    id: str | int | None
    price: float


class RegressionResponse(BaseModel):
    predictions: list[RegressionPrediction]


class ClassificationPrediction(BaseModel):
    id: str | int | None
    type: str


class ClassificationResponse(BaseModel):
    predictions: list[ClassificationPrediction]

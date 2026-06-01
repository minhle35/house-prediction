import pandas as pd
from fastapi import APIRouter, HTTPException

from app.io import get_classification, get_regression
from app.schemas.request import PredictRequest
from app.schemas.response import (
    ClassificationPrediction,
    ClassificationResponse,
    RegressionPrediction,
    RegressionResponse,
)

router = APIRouter(prefix="/predict", tags=["predict"])


@router.post("/regression", response_model=RegressionResponse)
def predict_regression(request: PredictRequest) -> RegressionResponse:
    try:
        pipeline = get_regression()
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    df = pd.DataFrame([item.model_dump(mode="json", exclude={"id"}) for item in request.data])
    prices: list[float] = pipeline.predict(df).tolist()

    return RegressionResponse(
        predictions=[
            RegressionPrediction(id=item.id, price=price)
            for item, price in zip(request.data, prices)
        ]
    )


@router.post("/classification", response_model=ClassificationResponse)
def predict_classification(request: PredictRequest) -> ClassificationResponse:
    try:
        pipeline, le = get_classification()
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    df = pd.DataFrame([item.model_dump(mode="json", exclude={"id"}) for item in request.data])
    encoded: list[int] = pipeline.predict(df).tolist()
    types: list[str] = le.inverse_transform(encoded).tolist()

    return ClassificationResponse(
        predictions=[
            ClassificationPrediction(id=item.id, type=t)
            for item, t in zip(request.data, types)
        ]
    )

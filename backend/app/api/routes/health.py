from fastapi import APIRouter, HTTPException

from app.io import get_classification, get_regression

router = APIRouter(prefix="/health", tags=["health"])


@router.get("/")
def health() -> dict:
    return {"status": "ok"}


@router.get("/ready")
def readiness() -> dict:
    """Returns 200 when both models are loaded, 503 otherwise."""
    try:
        get_regression()
        get_classification()
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    return {"status": "ready"}

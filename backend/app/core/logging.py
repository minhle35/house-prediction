import logging
import sys

from app.core.config import settings

_LOG_FORMAT_JSON = (
    '{"time": "%(asctime)s", "level": "%(levelname)s",'
    ' "logger": "%(name)s", "message": "%(message)s"}'
)
_LOG_FORMAT_DEV = "[%(asctime)s] %(levelname)s %(name)s - %(message)s"


def configure_logging() -> None:
    fmt = _LOG_FORMAT_DEV if settings.environment == "development" else _LOG_FORMAT_JSON
    logging.basicConfig(
        level=getattr(logging, settings.log_level.upper(), logging.INFO),
        format=fmt,
        stream=sys.stdout,
    )
    # Silence noisy third-party loggers
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
    logging.getLogger("lightgbm").setLevel(logging.WARNING)

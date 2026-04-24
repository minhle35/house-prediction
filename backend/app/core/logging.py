import logging
import sys

from app.core.config import settings

_LOG_FORMAT_JSON = (
    '{"time": "%(asctime)s", "level": "%(levelname)s",'
    ' "logger": "%(name)s", "message": "%(message)s"}'
)
_LOG_FORMAT_DEV = "[%(asctime)s] %(levelname)s %(name)s - %(message)s"


def configure_logging(verbosity: int = 0) -> None:
    """Configure logging with basic config.

    * When verbosity == 0: logging is disabled
    * When verbosity == 1 (`-v`): logging set to INFO
    * When verbosity > 1 (`-vv`): logging set to DEBUG
    """
    if verbosity <= 0:
        return

    level = logging.INFO if verbosity == 1 else logging.DEBUG
    fmt = _LOG_FORMAT_DEV if settings.environment == "development" else _LOG_FORMAT_JSON
    logging.basicConfig(level=level, format=fmt, stream=sys.stdout)
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
    logging.getLogger("lightgbm").setLevel(logging.WARNING)

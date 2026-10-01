"""Fixed public errors and structured events without exception/request content."""

import json
import logging

from starlette.responses import JSONResponse

logger = logging.getLogger("encoder")


def event(name: str) -> None:
    logger.log(logging.INFO if name == "model_loaded" else logging.WARNING, name)


def error(status: int, code: str) -> JSONResponse:
    headers = {"Cache-Control": "no-store"}
    if status == 401:
        headers["WWW-Authenticate"] = "Bearer"
    return JSONResponse({"error": code}, status_code=status, headers=headers)


class JsonFormatter(logging.Formatter):
    """Never render arbitrary third-party messages, arguments or tracebacks."""

    def format(self, record: logging.LogRecord) -> str:
        return json.dumps(
            {
                "level": record.levelname.lower(),
                "event": record.getMessage() if record.name == "encoder" else "http_server",
                "logger": record.name,
            },
            separators=(",", ":"),
        )

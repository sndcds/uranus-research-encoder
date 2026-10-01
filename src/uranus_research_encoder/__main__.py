"""Safe server defaults for local and container operation."""

import argparse
import logging

import uvicorn


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the internal encoder")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=6335)
    args = parser.parse_args()
    logging.captureWarnings(True)
    uvicorn.run(
        "uranus_research_encoder.app:app",
        host=args.host,
        port=args.port,
        workers=1,
        access_log=False,
        proxy_headers=False,
        timeout_keep_alive=5,
        backlog=32,
        limit_concurrency=32,
        log_config={
            "version": 1,
            "disable_existing_loggers": False,
            "formatters": {"json": {"()": "uranus_research_encoder.errors.JsonFormatter"}},
            "handlers": {"default": {"class": "logging.StreamHandler", "formatter": "json"}},
            "root": {"handlers": ["default"], "level": "INFO"},
            "loggers": {
                "uvicorn": {"handlers": ["default"], "level": "INFO", "propagate": False},
                "uvicorn.error": {"level": "INFO"},
                "uvicorn.access": {"handlers": [], "propagate": False},
            },
        },
    )


if __name__ == "__main__":
    main()

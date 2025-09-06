import json
import logging
import sys
from typing import Any, Dict


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload: Dict[str, Any] = {
            "level": record.levelname,
            "name": record.name,
            "message": record.getMessage(),
        }
        if record.exc_info:
            payload["exc_info"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=False)


def setup_logging(level: str = "INFO", json_output: bool = True) -> None:
    logging.root.handlers.clear()
    logging.root.setLevel(level.upper())
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(level.upper())
    handler.setFormatter(JsonFormatter() if json_output else logging.Formatter("%(levelname)s %(name)s: %(message)s"))
    logging.root.addHandler(handler)


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)



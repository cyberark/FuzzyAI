import logging
from typing import Any

DEFAULT_FORMAT = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"

class CustomFormatter(logging.Formatter):

    grey = "\x1b[38;20m"
    yellow = "\x1b[33;20m"
    red = "\x1b[31;20m"
    bold_red = "\x1b[31;1m"
    reset = "\x1b[0m"

    FORMATS = {
        logging.DEBUG: grey + DEFAULT_FORMAT + reset,
        logging.INFO: grey + DEFAULT_FORMAT + reset,
        logging.WARNING: yellow + DEFAULT_FORMAT + reset,
        logging.ERROR: red + DEFAULT_FORMAT + reset,
        logging.CRITICAL: bold_red + DEFAULT_FORMAT + reset
    }

    def __init__(self, fmt = None, datefmt = None, style = "%", validate = True, *, defaults = None) -> None:  # type: ignore
        fmt = DEFAULT_FORMAT
        super().__init__(fmt, datefmt, style, validate, defaults=defaults)

    def format(self, record: Any) -> Any:
        log_fmt = self.FORMATS.get(record.levelno)
        formatter = logging.Formatter(log_fmt)
        return formatter.format(record)
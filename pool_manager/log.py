from __future__ import annotations

import logging
import sys
from pathlib import Path

import colorlog

TRACE = logging.DEBUG - 5

LOG_MODES = {"stdout", "file", "both"}


class PoolManagerLogger(logging.Logger):
    def trace(self, msg, *args, **kwargs):
        if self.isEnabledFor(TRACE):
            self._log(TRACE, msg, args, **kwargs)


logging.setLoggerClass(PoolManagerLogger)
logging.addLevelName(TRACE, "TRACE")


def _colored_formatter():
    return colorlog.ColoredFormatter(
        "%(asctime)s %(log_color)s[%(levelname)s]%(reset)s %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        log_colors={
            "TRACE": "cyan",
            "DEBUG": "blue",
            "INFO": "green",
            "WARNING": "yellow",
            "ERROR": "red",
            "CRITICAL": "bold_red",
        },
    )


def _plain_formatter():
    return logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def setup_logging(level: str = "INFO", log_mode: str = "stdout", log_file: str = ""):
    if log_mode not in LOG_MODES:
        raise ValueError(f"Invalid log_mode: {log_mode!r} (choose {', '.join(sorted(LOG_MODES))})")

    level = level.upper()
    logger = logging.getLogger("pool_manager")
    logger.setLevel(level)

    logger.handlers.clear()

    if log_mode in ("stdout", "both"):
        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(level)
        handler.setFormatter(_colored_formatter())
        logger.addHandler(handler)

    if log_mode in ("file", "both"):
        path = Path(log_file) if log_file else Path("pool-manager.log")
        handler = logging.FileHandler(path)
        handler.setLevel(level)
        handler.setFormatter(_plain_formatter())
        logger.addHandler(handler)

    return logger

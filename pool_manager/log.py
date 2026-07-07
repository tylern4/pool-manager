from __future__ import annotations

import sys

from loguru import logger

LOG_MODES = {"stdout", "file", "both"}


def setup_logging(level: str = "INFO", log_mode: str = "stdout", log_file: str = ""):
    if log_mode not in LOG_MODES:
        raise ValueError(f"Invalid log_mode: {log_mode!r} (choose {', '.join(sorted(LOG_MODES))})")

    level = level.upper()
    logger.remove()

    colored_fmt = (
        "<green>{time:YYYY-MM-DD HH:mm:ss}</green> "
        "<level>{level: <8}</level> "
        "<cyan>{name}</cyan>: <level>{message}</level>"
    )
    plain_fmt = "{time:YYYY-MM-DD HH:mm:ss} [{level: <8}] {name}: {message}"

    if log_mode in ("stdout", "both"):
        logger.add(sys.stdout, format=colored_fmt, level=level, colorize=True)

    if log_mode in ("file", "both"):
        logger.add(
            log_file or "pool-manager.log",
            format=plain_fmt,
            level=level,
            colorize=False,
        )

    return logger

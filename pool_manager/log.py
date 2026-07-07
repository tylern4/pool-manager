from __future__ import annotations

import sys

from loguru import logger

LOG_MODES = {"stdout", "file", "both"}


def setup_logging(level: str = "INFO", log_mode: str = "stdout", log_file: str = ""):
    if log_mode not in LOG_MODES:
        raise ValueError(f"Invalid log_mode: {log_mode!r} (choose {', '.join(sorted(LOG_MODES))})")

    level = level.upper()
    logger.remove()

    if log_mode in ("stdout", "both"):
        logger.add(sys.stdout, level=level, colorize=True)

    if log_mode in ("file", "both"):
        logger.add(
            log_file or "pool-manager.log",
            level=level,
            colorize=False,
        )

    return logger

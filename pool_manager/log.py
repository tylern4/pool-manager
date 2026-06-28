import logging
import sys

TRACE = logging.DEBUG - 5


class PoolManagerLogger(logging.Logger):
    def trace(self, msg, *args, **kwargs):
        if self.isEnabledFor(TRACE):
            self._log(TRACE, msg, args, **kwargs)


logging.setLoggerClass(PoolManagerLogger)
logging.addLevelName(TRACE, "TRACE")


def setup_logging(level: str = "INFO"):
    logger = logging.getLogger("pool_manager")
    logger.setLevel(level.upper())

    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(level.upper())

    fmt = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    handler.setFormatter(fmt)

    logger.handlers.clear()
    logger.addHandler(handler)

    return logger

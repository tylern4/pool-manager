import logging

from pool_manager.log import TRACE, setup_logging


class TestLogging:
    def test_setup_info(self):
        logger = setup_logging("INFO")
        assert logger.level == logging.INFO

    def test_setup_debug(self):
        logger = setup_logging("DEBUG")
        assert logger.level == logging.DEBUG

    def test_trace_level_exists(self):
        assert TRACE == logging.DEBUG - 5

    def test_logger_has_trace_method(self):
        logger = setup_logging("TRACE")
        assert hasattr(logger, "trace")
        assert logger.isEnabledFor(TRACE)

    def test_handler_attached(self):
        logger = setup_logging("INFO")
        assert len(logger.handlers) == 1

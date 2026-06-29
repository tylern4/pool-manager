import logging
import os
import tempfile

from pool_manager.log import LOG_MODES, TRACE, setup_logging


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

    def test_stdout_mode_has_one_handler(self):
        logger = setup_logging("INFO", log_mode="stdout")
        assert len(logger.handlers) == 1

    def test_file_mode_has_one_handler(self):
        with tempfile.NamedTemporaryFile(suffix=".log", delete=False) as f:
            log_path = f.name
        try:
            logger = setup_logging("INFO", log_mode="file", log_file=log_path)
            assert len(logger.handlers) == 1
            assert os.path.exists(log_path)
            logger.info("file test")
            logger.handlers[0].flush()
            with open(log_path) as f2:
                content = f2.read()
            assert "[INFO]" in content
            assert "file test" in content
        finally:
            os.unlink(log_path)

    def test_both_mode_has_two_handlers(self):
        with tempfile.NamedTemporaryFile(suffix=".log", delete=False) as f:
            log_path = f.name
        try:
            logger = setup_logging("INFO", log_mode="both", log_file=log_path)
            assert len(logger.handlers) == 2
            assert os.path.exists(log_path)
        finally:
            os.unlink(log_path)

    def test_file_output_is_plain_not_colored(self):
        with tempfile.NamedTemporaryFile(suffix=".log", delete=False) as f:
            log_path = f.name
        try:
            logger = setup_logging("INFO", log_mode="file", log_file=log_path)
            logger.info("no color here")
            logger.handlers[0].flush()
            with open(log_path) as f2:
                content = f2.read()
            assert "\x1b[" not in content
        finally:
            os.unlink(log_path)

    def test_invalid_mode_raises(self):
        import pytest

        with pytest.raises(ValueError, match="Invalid log_mode"):
            setup_logging("INFO", log_mode="invalid")

    def test_default_log_file_name(self):
        logger = setup_logging("INFO", log_mode="file")
        assert len(logger.handlers) == 1
        handler = logger.handlers[0]
        assert "pool-manager.log" in handler.baseFilename

    def test_log_mode_constants(self):
        assert LOG_MODES == {"stdout", "file", "both"}

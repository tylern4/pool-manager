import contextlib
import io
import tempfile
from pathlib import Path

import pytest
from loguru import logger

from pool_manager.log import LOG_MODES, setup_logging


@pytest.fixture(autouse=True)
def _clean_loguru():
    logger.remove()


class TestLogging:
    def test_setup_info_level(self):
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            setup_logging("INFO")
            logger.info("hello info")
            logger.trace("hello trace")
        output = buf.getvalue()
        assert "hello info" in output
        assert "hello trace" not in output

    def test_setup_debug_level(self):
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            setup_logging("DEBUG")
            logger.debug("hello debug")
            logger.trace("hello trace")
        output = buf.getvalue()
        assert "hello debug" in output
        assert "hello trace" not in output

    def test_setup_trace_level(self):
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            setup_logging("TRACE")
            logger.trace("hello trace")
        output = buf.getvalue()
        assert "hello trace" in output

    def test_file_mode(self):
        with tempfile.NamedTemporaryFile(suffix=".log", delete=False) as f:
            log_path = f.name
        try:
            setup_logging("INFO", log_mode="file", log_file=log_path)
            logger.info("file test")
            content = Path(log_path).read_text()
            assert "file test" in content
        finally:
            Path(log_path).unlink()

    def test_file_output_is_plain(self):
        with tempfile.NamedTemporaryFile(suffix=".log", delete=False) as f:
            log_path = f.name
        try:
            setup_logging("INFO", log_mode="file", log_file=log_path)
            logger.info("no color here")
            content = Path(log_path).read_text()
            assert "\x1b[" not in content
        finally:
            Path(log_path).unlink()

    def test_invalid_mode_raises(self):
        with pytest.raises(ValueError, match="Invalid log_mode"):
            setup_logging("INFO", log_mode="invalid")

    def test_default_log_file_name(self):
        setup_logging("INFO", log_mode="file")
        logger.info("test default file")
        assert Path("pool-manager.log").exists()
        Path("pool-manager.log").unlink()

    def test_log_mode_constants(self):
        assert LOG_MODES == {"stdout", "file", "both"}

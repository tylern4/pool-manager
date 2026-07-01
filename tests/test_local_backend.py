import tempfile
import time
from pathlib import Path

from pool_manager.scheduler.base import JobState
from pool_manager.scheduler.local_subprocess import LocalSubprocessBackend


class TestLocalSubprocessBackend:
    def test_submit_and_list(self):
        backend = LocalSubprocessBackend()
        # Create a simple script that sleeps briefly
        with tempfile.NamedTemporaryFile(mode="w", suffix=".sh", delete=False) as f:
            f.write("#!/bin/bash\nsleep 5\n")
            script = f.name

        Path(script).chmod(0o755)
        try:
            job_id = backend.submit(script, {})
            assert job_id is not None
            assert int(job_id) > 0

            active = backend.list_active()
            ids = [j.job_id for j in active]
            assert job_id in ids
            assert all(j.state == JobState.RUNNING for j in active if j.job_id == job_id)
        finally:
            backend.cancel(job_id)
            Path(script).unlink(missing_ok=True)

    def test_cancel(self):
        backend = LocalSubprocessBackend()
        with tempfile.NamedTemporaryFile(mode="w", suffix=".sh", delete=False) as f:
            f.write("#!/bin/bash\nsleep 30\n")
            script = f.name

        Path(script).chmod(0o755)
        try:
            job_id = backend.submit(script, {})
            assert job_id in [j.job_id for j in backend.list_active()]

            backend.cancel(job_id)
            time.sleep(0.2)
            assert job_id not in [j.job_id for j in backend.list_active()]
        finally:
            Path(script).unlink(missing_ok=True)

    def test_signal(self):
        backend = LocalSubprocessBackend()
        with tempfile.NamedTemporaryFile(mode="w", suffix=".sh", delete=False) as f:
            f.write("#!/bin/bash\nsleep 30\n")
            script = f.name

        Path(script).chmod(0o755)
        try:
            job_id = backend.submit(script, {})
            assert job_id in [j.job_id for j in backend.list_active()]

            backend.signal(job_id, "SIGTERM")
            time.sleep(0.2)
            assert job_id not in [j.job_id for j in backend.list_active()]
        finally:
            Path(script).unlink(missing_ok=True)

    def test_list_active_cleans_dead(self):
        backend = LocalSubprocessBackend()
        with tempfile.NamedTemporaryFile(mode="w", suffix=".sh", delete=False) as f:
            f.write("#!/bin/bash\nexit 0\n")
            script = f.name

        Path(script).chmod(0o755)
        try:
            job_id = backend.submit(script, {})
            time.sleep(0.3)
            active = backend.list_active()
            assert job_id not in [j.job_id for j in active]
        finally:
            Path(script).unlink(missing_ok=True)

    def test_name(self):
        backend = LocalSubprocessBackend()
        assert "local_subprocess" in backend.name()

    def test_test_mode_submit(self):
        backend = LocalSubprocessBackend(test_mode=True)
        job_id = backend.submit("/fake/script.sh", {"arg": "val"})
        assert job_id.startswith("test_")

    def test_submit_with_env_args(self):
        backend = LocalSubprocessBackend()
        import tempfile
        from pathlib import Path

        with tempfile.NamedTemporaryFile(mode="w", suffix=".sh", delete=False) as f:
            f.write("#!/bin/bash\nsleep 0.01\n")
            script = f.name
        Path(script).chmod(0o755)
        try:
            job_id = backend.submit(script, {"custom_arg": "custom_val"})
            assert job_id is not None
        finally:
            backend.cancel(job_id)
            Path(script).unlink(missing_ok=True)

    def test_test_mode_cancel(self):
        backend = LocalSubprocessBackend(test_mode=True)
        backend.cancel("test_123")
        # Should not raise

    def test_test_mode_signal(self):
        backend = LocalSubprocessBackend(test_mode=True)
        backend.signal("test_123", "SIGTERM")
        # Should not raise

    def test_cancel_nonexistent(self):
        backend = LocalSubprocessBackend()
        backend.cancel("nonexistent")
        # Should not raise

    def test_signal_nonexistent(self):
        backend = LocalSubprocessBackend()
        backend.signal("nonexistent", "SIGTERM")
        # Should not raise

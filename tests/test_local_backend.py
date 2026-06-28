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

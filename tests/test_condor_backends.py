"""
Tests for Condor backends that don't require a real HTCondor installation.

CondorPythonBackend is tested with a mock, since it requires htcondor.
CondorSubprocessBackend and CondorRESTAPIBackend can be tested by mocking
subprocess/httpx.
"""

from unittest.mock import MagicMock, patch

from pool_manager.work_queue.condor_rest import CondorRESTAPIBackend
from pool_manager.work_queue.condor_subprocess import CondorSubprocessBackend


class TestCondorSubprocessBackend:
    def test_parses_idle_count(self):
        backend = CondorSubprocessBackend()
        mock_output = "-- 0 jobs; 0 completed, 0 removed, 0 idle, 0 running, 0 held, 0 suspended"
        with patch("pool_manager.work_queue.condor_subprocess.subprocess.run") as mock_run:
            mock_run.return_value.returncode = 0
            mock_run.return_value.stdout = mock_output
            mock_run.return_value.stderr = ""
            assert backend.count_idle() == 0

    def test_parses_nonzero_idle(self):
        backend = CondorSubprocessBackend()
        mock_output = "-- 10 jobs; 0 completed, 0 removed, 3 idle, 7 running, 0 held, 0 suspended"
        with patch("pool_manager.work_queue.condor_subprocess.subprocess.run") as mock_run:
            mock_run.return_value.returncode = 0
            mock_run.return_value.stdout = mock_output
            mock_run.return_value.stderr = ""
            assert backend.count_idle() == 3

    def test_nonzero_exit_returns_zero(self):
        backend = CondorSubprocessBackend()
        with patch("pool_manager.work_queue.condor_subprocess.subprocess.run") as mock_run:
            mock_run.return_value.returncode = 1
            mock_run.return_value.stdout = ""
            mock_run.return_value.stderr = "error"
            assert backend.count_idle() == 0

    def test_name(self):
        backend = CondorSubprocessBackend()
        assert "condor_subprocess" in backend.name()

    def test_name_with_schedd(self):
        backend = CondorSubprocessBackend(schedd_name="schedd.example.com")
        assert "schedd.example.com" in backend.name()


class TestCondorRESTAPIBackend:
    def test_count_idle(self):
        backend = CondorRESTAPIBackend(url="http://htcondor:8080")
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"data": [{"ClusterId": 1}, {"ClusterId": 2}]}
        with patch("httpx.get", return_value=mock_response):
            count = backend.count_idle()
            assert count == 2

    def test_count_idle_empty(self):
        backend = CondorRESTAPIBackend(url="http://htcondor:8080")
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"data": []}
        with patch("httpx.get", return_value=mock_response):
            count = backend.count_idle()
            assert count == 0

    def test_name(self):
        backend = CondorRESTAPIBackend(url="http://htcondor:8080")
        assert "htcondor:8080" in backend.name()

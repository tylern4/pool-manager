"""
Tests for Condor backends that don't require a real HTCondor installation.

CondorPythonBackend is tested with a mock, since it requires htcondor.
CondorSubprocessBackend and CondorRESTAPIBackend can be tested by mocking
subprocess/httpx.
"""

import json
from unittest.mock import MagicMock, patch

from pool_manager.placement import TaskResources
from pool_manager.scheduler.htcondor_rest import CondorRestClient, HTCondorRESTAPIBackend
from pool_manager.work_queue.condor_rest import CondorRESTAPIBackend
from pool_manager.work_queue.condor_subprocess import CondorSubprocessBackend


def _mock_json_jobs(jobs: list[dict]) -> str:
    return json.dumps(jobs)


class TestCondorSubprocessBackend:
    def test_parses_empty_idle_count(self):
        backend = CondorSubprocessBackend()
        with patch("pool_manager.work_queue.condor_subprocess.subprocess.run") as mock_run:
            mock_run.return_value.returncode = 0
            mock_run.return_value.stdout = _mock_json_jobs([])
            mock_run.return_value.stderr = ""
            assert backend.count_idle() == 0

    def test_parses_nonzero_idle_count(self):
        backend = CondorSubprocessBackend()
        mock_jobs = [
            {"ClusterId": 1, "JobStatus": 1, "RequestCpus": 1, "RequestMemory": 2000},
            {"ClusterId": 2, "JobStatus": 1, "RequestCpus": 2, "RequestMemory": 4000},
            {"ClusterId": 3, "JobStatus": 1, "RequestCpus": 4, "RequestMemory": 8000},
        ]
        with patch("pool_manager.work_queue.condor_subprocess.subprocess.run") as mock_run:
            mock_run.return_value.returncode = 0
            mock_run.return_value.stdout = _mock_json_jobs(mock_jobs)
            mock_run.return_value.stderr = ""
            assert backend.count_idle() == 3

    def test_nonzero_exit_returns_zero(self):
        backend = CondorSubprocessBackend()
        with patch("pool_manager.work_queue.condor_subprocess.subprocess.run") as mock_run:
            mock_run.return_value.returncode = 1
            mock_run.return_value.stdout = ""
            mock_run.return_value.stderr = "error"
            assert backend.count_idle() == 0

    def test_list_idle_parses_task_resources(self):
        backend = CondorSubprocessBackend()
        mock_jobs = [
            {
                "ClusterId": 1,
                "JobStatus": 1,
                "RequestCpus": 2.0,
                "RequestMemory": 4096,
                "RequestGpus": 0,
            },
            {
                "ClusterId": 2,
                "JobStatus": 1,
                "RequestCpus": 4.0,
                "RequestMemory": 8192,
                "RequestGpus": 1,
            },
        ]
        with patch("pool_manager.work_queue.condor_subprocess.subprocess.run") as mock_run:
            mock_run.return_value.returncode = 0
            mock_run.return_value.stdout = _mock_json_jobs(mock_jobs)
            mock_run.return_value.stderr = ""
            tasks = backend.list_idle()
            assert len(tasks) == 2
            assert tasks[0] == TaskResources(cpus=2.0, memory_mb=4096, gpus=0)
            assert tasks[1] == TaskResources(cpus=4.0, memory_mb=8192, gpus=1)

    def test_list_idle_defaults_missing_attrs(self):
        backend = CondorSubprocessBackend()
        mock_jobs = [
            {"ClusterId": 1},
        ]
        with patch("pool_manager.work_queue.condor_subprocess.subprocess.run") as mock_run:
            mock_run.return_value.returncode = 0
            mock_run.return_value.stdout = _mock_json_jobs(mock_jobs)
            mock_run.return_value.stderr = ""
            tasks = backend.list_idle()
            assert len(tasks) == 1
            assert tasks[0] == TaskResources(cpus=1.0, memory_mb=1024, gpus=0)

    def test_list_idle_zero_exit_empty(self):
        backend = CondorSubprocessBackend()
        with patch("pool_manager.work_queue.condor_subprocess.subprocess.run") as mock_run:
            mock_run.return_value.returncode = 1
            mock_run.return_value.stdout = ""
            mock_run.return_value.stderr = "error"
            assert backend.list_idle() == []

    def test_invalid_json_returns_empty(self):
        backend = CondorSubprocessBackend()
        with patch("pool_manager.work_queue.condor_subprocess.subprocess.run") as mock_run:
            mock_run.return_value.returncode = 0
            mock_run.return_value.stdout = "not json"
            mock_run.return_value.stderr = ""
            assert backend.list_idle() == []

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
        mock_response.json.return_value = {
            "data": [
                {"ClusterId": 1, "RequestCpus": 1},
                {"ClusterId": 2, "RequestCpus": 2},
            ]
        }
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

    def test_list_idle_parses_task_resources(self):
        backend = CondorRESTAPIBackend(url="http://htcondor:8080")
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "data": [
                {"ClusterId": 1, "RequestCpus": 2.0, "RequestMemory": 4096, "RequestGpus": 0},
                {"ClusterId": 2, "RequestCpus": 4.0, "RequestMemory": 8192, "RequestGpus": 1},
            ]
        }
        with patch("httpx.get", return_value=mock_response):
            tasks = backend.list_idle()
            assert len(tasks) == 2
            assert tasks[0] == TaskResources(cpus=2.0, memory_mb=4096, gpus=0)
            assert tasks[1] == TaskResources(cpus=4.0, memory_mb=8192, gpus=1)

    def test_list_idle_defaults_missing_attrs(self):
        backend = CondorRESTAPIBackend(url="http://htcondor:8080")
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"data": [{"ClusterId": 1}]}
        with patch("httpx.get", return_value=mock_response):
            tasks = backend.list_idle()
            assert len(tasks) == 1
            assert tasks[0] == TaskResources(cpus=1.0, memory_mb=1024, gpus=0)

    def test_name(self):
        backend = CondorRESTAPIBackend(url="http://htcondor:8080")
        assert "htcondor:8080" in backend.name()


class TestHTCondorRESTAPISchedulerBackend:
    def test_submit(self):
        backend = HTCondorRESTAPIBackend(url="http://htcondor:8080", token="test-token")
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"cluster": 456, "first_proc": 0, "num_procs": 1}
        with patch("httpx.post", return_value=mock_response) as mock_post:
            job_id = backend.submit("/fake/worker.sh", {"request_cpus": "1"})
            assert job_id == "456"
            call_url = mock_post.call_args[0][0]
            assert call_url == "http://htcondor:8080/condor_submit"
            call_kwargs = mock_post.call_args.kwargs
            assert call_kwargs["headers"]["Authorization"] == "Bearer test-token"
            assert call_kwargs["json"]["executable"] == "/fake/worker.sh"
            assert call_kwargs["json"]["request_cpus"] == "1"

    def test_submit_defaults_executable(self):
        client = CondorRestClient(url="http://htcondor:8080")
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"cluster": 789}
        with patch("httpx.post", return_value=mock_response) as mock_post:
            job_id = client.submit("/some/script.sh", {})
            assert job_id == "789"
            assert mock_post.call_args.kwargs["json"]["executable"] == "/some/script.sh"

    def test_cancel(self):
        backend = HTCondorRESTAPIBackend(url="http://htcondor:8080")
        mock_response = MagicMock()
        mock_response.status_code = 200
        with patch("httpx.delete", return_value=mock_response) as mock_delete:
            backend.cancel("456")
            assert "condor_rm/456" in mock_delete.call_args[0][0]

    def test_cancel_non_200_logs_warning(self):
        backend = HTCondorRESTAPIBackend(url="http://htcondor:8080")
        mock_response = MagicMock()
        mock_response.status_code = 404
        with patch("httpx.delete", return_value=mock_response):
            backend.cancel("999")

    def test_list_active(self):
        backend = HTCondorRESTAPIBackend(url="http://htcondor:8080")
        mock_jobs = [
            {"ClusterId": 1, "JobStatus": 1},
            {"ClusterId": 2, "JobStatus": 2},
            {"ClusterId": 3, "JobStatus": 3},
            {"ClusterId": 4, "JobStatus": 4},
            {"ClusterId": 5, "JobStatus": 5},
        ]
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = mock_jobs
        with patch("httpx.get", return_value=mock_response):
            active = backend.list_active()
            ids = [j.job_id for j in active]
            assert "1" in ids
            assert "2" in ids
            assert "5" in ids
            assert "3" not in ids
            assert "4" not in ids

    def test_list_active_empty(self):
        backend = HTCondorRESTAPIBackend(url="http://htcondor:8080")
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = []
        with patch("httpx.get", return_value=mock_response):
            assert backend.list_active() == []

    def test_signal_calls_remove(self):
        backend = HTCondorRESTAPIBackend(url="http://htcondor:8080")
        mock_response = MagicMock()
        mock_response.status_code = 200
        with patch("httpx.delete", return_value=mock_response) as mock_delete:
            backend.signal("456", "SIGTERM")
            assert "condor_rm/456" in mock_delete.call_args[0][0]

    def test_name(self):
        backend = HTCondorRESTAPIBackend(url="http://htcondor:8080", token="tok")
        assert "htcondor:8080" in backend.name()

    def test_client_name(self):
        client = CondorRestClient(url="http://htcondor:8080")
        assert "htcondor:8080" in client.name()

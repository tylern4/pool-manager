"""
Tests for Condor backends that don't require a real HTCondor installation.

CondorPythonBackend is tested with a mock, since it requires htcondor.
CondorSubprocessBackend and CondorRESTAPIBackend can be tested by mocking
subprocess/httpx.
"""

from unittest.mock import patch

from pool_manager.placement import TaskResources
from pool_manager.scheduler.htcondor_rest import (
    HTCondorRESTAPIBackend,
    _parse_walltime_minutes,
    _translate_submit_args,
)
from pool_manager.work_queue.condor_rest import CondorRESTAPIBackend
from pool_manager.work_queue.condor_subprocess import CondorSubprocessBackend


def _mock_json_jobs(jobs: list[dict]) -> str:
    import json

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
            assert tasks[0] == TaskResources(cpus=2.0, memory_mb=4096, gpus=0, job_status=1)
            assert tasks[1] == TaskResources(cpus=4.0, memory_mb=8192, gpus=1, job_status=1)

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
            assert tasks[0] == TaskResources(cpus=1.0, memory_mb=1024, gpus=0, job_status=0)

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
        mock_jobs = [
            {"ClusterId": 1, "RequestCpus": 1},
            {"ClusterId": 2, "RequestCpus": 2},
        ]
        with patch("pool_manager.work_queue.condor_rest.CondorClient") as mock_client:
            mock_client.return_value.get_queue.return_value = mock_jobs
            backend = CondorRESTAPIBackend(url="http://htcondor:8080", token="tok")
            count = backend.count_idle()
            assert count == 2

    def test_count_idle_empty(self):
        with patch("pool_manager.work_queue.condor_rest.CondorClient") as mock_client:
            mock_client.return_value.get_queue.return_value = []
            backend = CondorRESTAPIBackend(url="http://htcondor:8080", token="tok")
            count = backend.count_idle()
            assert count == 0

    def test_list_idle_parses_task_resources(self):
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
                "JobStatus": 2,
                "RequestCpus": 4.0,
                "RequestMemory": 8192,
                "RequestGpus": 1,
            },
        ]
        with patch("pool_manager.work_queue.condor_rest.CondorClient") as mock_client:
            mock_client.return_value.get_queue.return_value = mock_jobs
            backend = CondorRESTAPIBackend(url="http://htcondor:8080", token="tok")
            tasks = backend.list_idle()
            assert len(tasks) == 2
            assert tasks[0] == TaskResources(cpus=2.0, memory_mb=4096, gpus=0, job_status=1)
            assert tasks[1] == TaskResources(cpus=4.0, memory_mb=8192, gpus=1, job_status=2)

    def test_list_idle_defaults_missing_attrs(self):
        with patch("pool_manager.work_queue.condor_rest.CondorClient") as mock_client:
            mock_client.return_value.get_queue.return_value = [{"ClusterId": 1}]
            backend = CondorRESTAPIBackend(url="http://htcondor:8080", token="tok")
            tasks = backend.list_idle()
            assert len(tasks) == 1
            assert tasks[0] == TaskResources(cpus=1.0, memory_mb=1024, gpus=0)

    def test_name(self):
        backend = CondorRESTAPIBackend(url="http://htcondor:8080", token="tok")
        assert "htcondor:8080" in backend.name()


class TestHTCondorRESTAPISchedulerBackend:
    def test_submit(self):
        with patch("pool_manager.scheduler.htcondor_rest.CondorClient") as mock_client:
            mock_client.return_value.submit.return_value = {
                "cluster": 456,
                "first_proc": 0,
                "num_procs": 1,
            }
            backend = HTCondorRESTAPIBackend(url="http://htcondor:8080", token="test-token")
            job_id = backend.submit("/fake/worker.sh", {"cpus-per-task": "1"})
            assert job_id == "456"
            payload = mock_client.return_value.submit.call_args.args[0]
            assert payload["executable"] == "/fake/worker.sh"
            assert payload["request_cpus"] == "1"

    def test_submit_translates_resources(self):
        with patch("pool_manager.scheduler.htcondor_rest.CondorClient") as mock_client:
            mock_client.return_value.submit.return_value = {"cluster": 789}
            backend = HTCondorRESTAPIBackend(url="http://htcondor:8080", token="tok")
            backend.submit(
                "/fake/worker.sh",
                {
                    "job-name": "htcondor_worker_small",
                    "cpus-per-task": "4",
                    "mem": "8000M",
                    "gpus": "1",
                    "time": "00:30:00",
                    "partition": "debug",
                },
            )
            payload = mock_client.return_value.submit.call_args.args[0]
            assert payload["request_cpus"] == "4"
            assert payload["request_memory"] == "8000M"
            assert payload["request_gpus"] == "1"
            assert payload["runtime_minutes"] == "30"
            assert "job-name" not in payload
            assert "partition" not in payload
            assert payload["executable"] == "/fake/worker.sh"

    def test_cancel(self):
        with patch("pool_manager.scheduler.htcondor_rest.CondorClient") as mock_client:
            backend = HTCondorRESTAPIBackend(url="http://htcondor:8080", token="tok")
            backend.cancel("456")
            mock_client.return_value.remove.assert_called_once_with(job_id="456")

    def test_list_active(self):
        mock_jobs = [
            {"ClusterId": 1, "JobStatus": 1},
            {"ClusterId": 2, "JobStatus": 2},
            {"ClusterId": 3, "JobStatus": 3},
            {"ClusterId": 4, "JobStatus": 4},
            {"ClusterId": 5, "JobStatus": 5},
        ]
        with patch("pool_manager.scheduler.htcondor_rest.CondorClient") as mock_client:
            mock_client.return_value.get_queue.return_value = mock_jobs
            backend = HTCondorRESTAPIBackend(url="http://htcondor:8080", token="tok")
            active = backend.list_active()
            ids = [j.job_id for j in active]
            assert "1" in ids
            assert "2" in ids
            assert "5" in ids
            assert "3" not in ids
            assert "4" not in ids

    def test_list_active_empty(self):
        with patch("pool_manager.scheduler.htcondor_rest.CondorClient") as mock_client:
            mock_client.return_value.get_queue.return_value = []
            backend = HTCondorRESTAPIBackend(url="http://htcondor:8080", token="tok")
            assert backend.list_active() == []

    def test_signal_calls_remove(self):
        with patch("pool_manager.scheduler.htcondor_rest.CondorClient") as mock_client:
            backend = HTCondorRESTAPIBackend(url="http://htcondor:8080", token="tok")
            backend.signal("456", "SIGTERM")
            mock_client.return_value.remove.assert_called_once_with(job_id="456")

    def test_name(self):
        backend = HTCondorRESTAPIBackend(url="http://htcondor:8080", token="tok")
        assert "htcondor:8080" in backend.name()

    def test_parse_walltime_minutes(self):
        assert _parse_walltime_minutes("00:30:00") == 30
        assert _parse_walltime_minutes("08:00:00") == 480
        assert _parse_walltime_minutes("120") == 120
        assert _parse_walltime_minutes("bogus") == 0

    def test_translate_submit_args_passthrough(self):
        translated = _translate_submit_args(
            {"request_cpus": "2", "request_memory": "4096", "environment": "FOO=1"}
        )
        assert translated == {"request_cpus": "2", "request_memory": "4096", "environment": "FOO=1"}

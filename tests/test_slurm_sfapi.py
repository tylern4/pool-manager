from unittest.mock import MagicMock, patch

import pytest

from pool_manager.scheduler.base import JobState
from pool_manager.scheduler.slurm_sfapi import SlurmSFAPIBackend, _sfapi_to_jobstate


class TestSFApiJobStateMapping:
    def test_pending(self):
        from sfapi_client.jobs import JobState as S

        assert _sfapi_to_jobstate(S.PENDING) == JobState.PENDING

    def test_running(self):
        from sfapi_client.jobs import JobState as S

        assert _sfapi_to_jobstate(S.RUNNING) == JobState.RUNNING

    def test_configuring(self):
        from sfapi_client.jobs import JobState as S

        assert _sfapi_to_jobstate(S.CONFIGURING) == JobState.PENDING

    def test_completing(self):
        from sfapi_client.jobs import JobState as S

        assert _sfapi_to_jobstate(S.COMPLETING) == JobState.RUNNING

    def test_signaling(self):
        from sfapi_client.jobs import JobState as S

        assert _sfapi_to_jobstate(S.SIGNALING) == JobState.RUNNING

    def test_cancelled_is_unknown(self):
        from sfapi_client.jobs import JobState as S

        assert _sfapi_to_jobstate(S.CANCELLED) == JobState.UNKNOWN

    def test_completed_is_unknown(self):
        from sfapi_client.jobs import JobState as S

        assert _sfapi_to_jobstate(S.COMPLETED) == JobState.UNKNOWN


class TestSlurmSFAPIBackend:
    def make_mock_job(self, job_id: str, state_str: str = "PENDING"):
        from sfapi_client.jobs import JobState as S

        job = MagicMock()
        job.jobid = job_id
        job.state = S(state_str)
        return job

    @pytest.fixture
    def mock_client(self):
        with patch("sfapi_client.Client") as mock_cls:
            mock_instance = MagicMock()
            mock_compute = MagicMock()
            mock_instance.compute.return_value = mock_compute
            mock_cls.return_value = mock_instance
            yield mock_cls, mock_instance, mock_compute

    def test_submit(self, mock_client):
        mock_cls, mock_instance, mock_compute = mock_client
        mock_job = self.make_mock_job("42", "PENDING")
        mock_compute.submit_job.return_value = mock_job

        backend = SlurmSFAPIBackend(machine="perlmutter", user="testuser")
        with patch("builtins.open") as mock_open:
            read_mock = mock_open.return_value.__enter__.return_value.read
            read_mock.return_value = "#!/bin/bash\necho hello"

            job_id = backend.submit("/fake/script.sh", {"partition": "debug"})

        assert job_id == "42"
        mock_compute.submit_job.assert_called_once()
        script_arg = mock_compute.submit_job.call_args[0][0]
        assert "#SBATCH --partition=debug" in script_arg

    def test_submit_injects_sbatch_args(self, mock_client):
        mock_cls, mock_instance, mock_compute = mock_client
        mock_job = self.make_mock_job("99", "PENDING")
        mock_compute.submit_job.return_value = mock_job

        backend = SlurmSFAPIBackend(machine="perlmutter")
        with patch("builtins.open") as mock_open:
            read_mock = mock_open.return_value.__enter__.return_value.read
            read_mock.return_value = "#!/bin/bash\necho hello"

            backend.submit("/x.sh", {"account": "myproject", "time": "01:00:00"})

        script_arg = mock_compute.submit_job.call_args[0][0]
        assert "#SBATCH --account=myproject" in script_arg
        assert "#SBATCH --time=01:00:00" in script_arg
        assert "echo hello" in script_arg

    def test_cancel(self, mock_client):
        mock_cls, mock_instance, mock_compute = mock_client
        mock_job = self.make_mock_job("42")
        mock_compute.job.return_value = mock_job

        backend = SlurmSFAPIBackend(machine="perlmutter")
        backend.cancel("42")

        mock_compute.job.assert_called_once_with(jobid="42")
        assert mock_job.cancel.called

    def test_list_active(self, mock_client):
        mock_cls, mock_instance, mock_compute = mock_client
        running = self.make_mock_job("1", "RUNNING")
        pending = self.make_mock_job("2", "PENDING")
        done = self.make_mock_job("3", "COMPLETED")
        mock_compute.jobs.return_value = [running, pending, done]

        backend = SlurmSFAPIBackend(machine="perlmutter", user="testuser")
        jobs = backend.list_active()

        assert len(jobs) == 2
        assert jobs[0].job_id == "1"
        assert jobs[0].state == JobState.RUNNING
        assert jobs[1].job_id == "2"
        assert jobs[1].state == JobState.PENDING

    def test_list_active_passes_user(self, mock_client):
        mock_cls, mock_instance, mock_compute = mock_client
        mock_compute.jobs.return_value = []

        backend = SlurmSFAPIBackend(machine="perlmutter", user="elvis")
        backend.list_active()

        mock_compute.jobs.assert_called_once_with(user="elvis")

    def test_signal(self, mock_client):
        mock_cls, mock_instance, mock_compute = mock_client
        mock_response = MagicMock()
        mock_compute.client.post.return_value = mock_response

        backend = SlurmSFAPIBackend(machine="perlmutter")
        backend.signal("42", "SIGTERM")

        mock_compute.client.post.assert_called_once_with(
            "compute/jobs/perlmutter/42/signal",
            data={"signal": "SIGTERM"},
        )
        assert mock_response.raise_for_status.called

    def test_name(self, mock_client):
        backend = SlurmSFAPIBackend(machine="perlmutter")
        assert "slurm_sfapi" in backend.name()
        assert "perlmutter" in backend.name()

    def test_constructor_with_key_path(self, mock_client):
        mock_cls, mock_instance, mock_compute = mock_client

        backend = SlurmSFAPIBackend(
            machine="perlmutter",
            key_path="/fake/key.pem",
        )

        assert backend._machine == "perlmutter"
        assert backend._client_kwargs["key"] is not None

    def test_constructor_with_user(self, mock_client):
        backend = SlurmSFAPIBackend(machine="perlmutter", user="elvis")
        assert backend._user == "elvis"

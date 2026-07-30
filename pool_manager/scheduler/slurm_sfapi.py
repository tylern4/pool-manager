import json
from pathlib import Path

from loguru import logger
from sfapi_client import Client
from sfapi_client._jobs import JobCommand
from sfapi_client.compute import Machine
from sfapi_client.jobs import JobState as SFApiJobState

from pool_manager.scheduler.base import JobInfo, JobState, SchedulerBackend, _test_job_id


class SlurmSFAPIBackend(SchedulerBackend):
    def __init__(
        self,
        machine: str,
        client_id: str = "",
        client_secret: str = "",
        key_path: str = "",
        user: str = "",
        job_name_prefix: str = "htcondor_worker_",
        test_mode: bool = False,
    ):
        self._machine = machine
        self._user = user
        self._job_name_prefix = job_name_prefix
        self._test_mode = test_mode
        self._client_kwargs: dict = {}

        if client_id and client_secret:
            self._client_kwargs["client_id"] = client_id
            self._client_kwargs["client_secret"] = json.loads(client_secret)
        elif key_path:
            self._client_kwargs["key"] = Path(key_path)

        self._client = Client(**self._client_kwargs)

    def _compute(self, reuse_client: bool = True):
        client = self._client if reuse_client else Client(**self._client_kwargs)
        machine = Machine(self._machine)
        return client.compute(machine)

    def submit(self, script_path: str, submit_args: dict[str, str]) -> str:
        if self._test_mode:
            job_id = _test_job_id()
            logger.info(
                "[TEST] Would submit job {} via SFAPI on {} (script={}, args={})",
                job_id,
                self._machine,
                script_path,
                submit_args,
            )
            return job_id

        compute = self._compute()

        script_path_p = Path(script_path)
        if not script_path_p.exists():
            raise FileNotFoundError(f"Worker script not found: {script_path}")
        script_content = script_path_p.read_text()

        sbatch_lines = []
        for key, val in submit_args.items():
            key = key.replace("_", "-")
            sbatch_lines.append(f"#SBATCH --{key}={val}")

        if sbatch_lines:
            if script_content.startswith("#!/"):
                first, rest = script_content.split("\n", 1)
                wrapped = first + "\n" + "\n".join(sbatch_lines) + "\n" + rest
            else:
                wrapped = "#!/bin/bash\n" + "\n".join(sbatch_lines) + "\n" + script_content
        else:
            wrapped = script_content

        logger.debug("Submitting script via SFAPI on {}", self._machine)
        logger.trace("Wrapped script:\n{}", wrapped)

        job = compute.submit_job(wrapped)
        job_id = str(job.jobid)
        logger.info("Submitted job {} on {} via SFAPI", job_id, self._machine)
        return job_id

    def cancel(self, job_id: str) -> None:
        if self._test_mode:
            logger.info("[TEST] Would cancel job {} via SFAPI on {}", job_id, self._machine)
            return

        client = Client(**self._client_kwargs)
        compute = client.compute(Machine(self._machine))
        job = compute.job(jobid=job_id)
        logger.debug("Cancelling job {} via SFAPI", job_id)
        job.cancel()
        logger.debug("Cancelled job {}", job_id)

    def list_active(self) -> list[JobInfo]:
        client = Client(**self._client_kwargs)
        compute = client.compute(Machine(self._machine))

        kwargs: dict = {}
        if self._user:
            kwargs["user"] = self._user

        logger.debug("Listing jobs on {} via SFAPI (sacct)", self._machine)
        jobs = compute.jobs(command=JobCommand.sacct, **kwargs)

        result: list[JobInfo] = []
        for j in jobs:
            if self._job_name_prefix and not (j.jobname or "").startswith(self._job_name_prefix):
                continue
            state = _sfapi_to_jobstate(j.state)
            if state in (JobState.RUNNING, JobState.PENDING):
                result.append(JobInfo(job_id=str(j.jobid), state=state, job_name=j.jobname or ""))

        logger.debug("Active jobs on {}: {}", self._machine, [j.job_id for j in result])
        return result

    def signal(self, job_id: str, sig: str) -> None:
        if self._test_mode:
            logger.info("[TEST] Would send signal {} to job {} via SFAPI", sig, job_id)
            return

        client = Client(**self._client_kwargs)
        compute = client.compute(Machine(self._machine))
        logger.debug("Sending signal {} to job {} via SFAPI", sig, job_id)
        resp = compute.client.post(
            f"compute/jobs/{self._machine}/{job_id}/signal",
            data={"signal": sig},
        )
        resp.raise_for_status()
        logger.debug("Sent signal {} to job {}", sig, job_id)

    def name(self) -> str:
        return f"slurm_sfapi({self._machine})"


def _sfapi_to_jobstate(state) -> JobState:
    active_states = {
        SFApiJobState.PENDING,
        SFApiJobState.CONFIGURING,
        SFApiJobState.RUNNING,
        SFApiJobState.COMPLETING,
        SFApiJobState.SIGNALING,
        SFApiJobState.STAGE_OUT,
        SFApiJobState.RESIZING,
        SFApiJobState.REQUEUED,
        SFApiJobState.SUSPENDED,
    }
    if state in (SFApiJobState.CANCELLED, SFApiJobState.COMPLETED):
        return JobState.COMPLETED
    if state in active_states:
        if state in (
            SFApiJobState.RUNNING,
            SFApiJobState.COMPLETING,
            SFApiJobState.SIGNALING,
            SFApiJobState.STAGE_OUT,
        ):
            return JobState.RUNNING
        return JobState.PENDING
    return JobState.UNKNOWN

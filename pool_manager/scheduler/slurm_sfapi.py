import logging
from pathlib import Path

from pool_manager.log import TRACE
from pool_manager.scheduler.base import JobInfo, JobState, SchedulerBackend

log = logging.getLogger("pool_manager.scheduler.slurm_sfapi")


class SlurmSFAPIBackend(SchedulerBackend):
    def __init__(
        self,
        machine: str,
        client_id: str = "",
        client_secret: str = "",
        key_path: str = "",
        user: str = "",
    ):
        from sfapi_client import Client

        self._machine = machine
        self._user = user
        self._client_kwargs: dict = {}

        if client_id and client_secret:
            import json

            from authlib.jose import JsonWebKey

            self._client_kwargs["client_id"] = client_id
            self._client_kwargs["client_secret"] = JsonWebKey.import_key(json.loads(client_secret))
        elif key_path:
            self._client_kwargs["key"] = Path(key_path)

        self._client = Client(**self._client_kwargs)

    def _compute(self, reuse_client: bool = True):
        from sfapi_client import Client
        from sfapi_client.compute import Machine

        client = self._client if reuse_client else Client(**self._client_kwargs)
        machine = Machine(self._machine)
        return client.compute(machine)

    def submit(self, script_path: str, submit_args: dict[str, str]) -> str:
        compute = self._compute()

        with open(script_path) as f:
            script_content = f.read()

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

        log.debug("Submitting script via SFAPI on %s", self._machine)
        log.log(TRACE, "Wrapped script:\n%s", wrapped)

        job = compute.submit_job(wrapped)
        job_id = str(job.jobid)
        log.info("Submitted job %s on %s via SFAPI", job_id, self._machine)
        return job_id

    def cancel(self, job_id: str) -> None:
        from sfapi_client import Client
        from sfapi_client.compute import Machine

        client = Client(**self._client_kwargs)
        compute = client.compute(Machine(self._machine))
        job = compute.job(jobid=job_id)
        log.debug("Cancelling job %s via SFAPI", job_id)
        job.cancel()
        log.debug("Cancelled job %s", job_id)

    def list_active(self) -> list[JobInfo]:
        from sfapi_client import Client
        from sfapi_client.compute import Machine

        client = Client(**self._client_kwargs)
        compute = client.compute(Machine(self._machine))

        kwargs = {}
        if self._user:
            kwargs["user"] = self._user

        log.debug("Listing jobs on %s via SFAPI", self._machine)
        jobs = compute.jobs(**kwargs)

        result: list[JobInfo] = []
        for j in jobs:
            state = _sfapi_to_jobstate(j.state)
            if state in (JobState.RUNNING, JobState.PENDING):
                result.append(JobInfo(job_id=str(j.jobid), state=state))

        log.debug("Active jobs on %s: %s", self._machine, [j.job_id for j in result])
        return result

    def signal(self, job_id: str, sig: str) -> None:
        from sfapi_client import Client
        from sfapi_client.compute import Machine

        client = Client(**self._client_kwargs)
        compute = client.compute(Machine(self._machine))
        log.debug("Sending signal %s to job %s via SFAPI", sig, job_id)
        resp = compute.client.post(
            f"compute/jobs/{self._machine}/{job_id}/signal",
            data={"signal": sig},
        )
        resp.raise_for_status()
        log.debug("Sent signal %s to job %s", sig, job_id)

    def name(self) -> str:
        return f"slurm_sfapi({self._machine})"


def _sfapi_to_jobstate(state) -> JobState:
    from sfapi_client.jobs import JobState as SFApiJobState

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

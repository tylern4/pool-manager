from __future__ import annotations

try:
    from htcondor_rest import CondorClient
except ImportError:
    CondorClient = None

from loguru import logger

from pool_manager.scheduler.base import JobInfo, JobState, SchedulerBackend

# Map the scheduler's generic submit_args onto HTCondor submit attributes.
_SUBMIT_ALIASES = {
    "cpus-per-task": "request_cpus",
    "mem": "request_memory",
    "gpus": "request_gpus",
}
# Keys that have no HTCondor equivalent (SLURM-style batch settings).
_SUBMIT_DROP = {"job-name", "partition", "account", "nodes", "ntasks", "ntasks-per-node"}

_HTSTATUS_MAP = {
    1: JobState.PENDING,
    2: JobState.RUNNING,
    5: JobState.PENDING,
    6: JobState.RUNNING,
}


def _translate_submit_args(submit_args: dict[str, str]) -> dict[str, str]:
    """Convert generic/slurm-style submit args into HTCondor submit attributes."""
    payload: dict[str, str] = {}
    for key, value in submit_args.items():
        if key in _SUBMIT_DROP:
            continue
        if key == "time":
            minutes = _parse_walltime_minutes(value)
            if minutes:
                payload["runtime_minutes"] = str(minutes)
            continue
        if key in _SUBMIT_ALIASES:
            payload[_SUBMIT_ALIASES[key]] = value
            continue
        payload[key] = value
    return payload


def _parse_walltime_minutes(value: str) -> int:
    parts = value.split(":")
    if len(parts) == 3:
        try:
            return int(parts[0]) * 60 + int(parts[1])
        except ValueError:
            return 0
    if len(parts) == 2:
        try:
            return int(parts[0]) * 60 + int(parts[1])
        except ValueError:
            return 0
    try:
        return int(value)
    except ValueError:
        return 0


class HTCondorRESTAPIBackend(SchedulerBackend):
    def __init__(
        self,
        url: str,
        token: str = "",
        owner: str = "",
        job_name_prefix: str = "htcondor_worker_",
    ):
        self._url = url.rstrip("/")
        self._client = (
            CondorClient(base_url=self._url, token=token)
            if token
            else CondorClient(base_url=self._url)
        )
        self._owner = owner
        self._job_name_prefix = job_name_prefix

    def submit(self, script_path: str, submit_args: dict[str, str]) -> str:
        payload = _translate_submit_args(submit_args)
        payload.setdefault("executable", script_path)

        logger.debug("Submitting HTCondor job via htcondor-rest: {}", payload)
        result = self._client.submit(payload)
        job_id = str(result["cluster"])
        logger.debug("Submitted HTCondor job {} via REST API", job_id)
        return job_id

    def cancel(self, job_id: str) -> None:
        logger.debug("Removing HTCondor job {} via REST API", job_id)
        self._client.remove(job_id=job_id)

    def list_active(self) -> list[JobInfo]:
        constraint = ""
        if self._owner:
            constraint = f'Owner == "{self._owner}"'
        jobs = self._client.get_queue(
            projection="ClusterId,JobStatus,Name", constraint=constraint or None
        )
        active: list[JobInfo] = []
        for raw in jobs:
            try:
                cluster_id = raw.get("ClusterId")
                if cluster_id is None:
                    continue
                state = _parse_htcondor_job_status(raw.get("JobStatus"))
                if state in (JobState.PENDING, JobState.RUNNING):
                    active.append(
                        JobInfo(
                            job_id=str(cluster_id),
                            state=state,
                            job_name=raw.get("Name", "") or "",
                        )
                    )
            except Exception:
                logger.exception("Failed to parse job from REST response: {}", raw)
        logger.debug("Active HTCondor jobs from REST: {}", [j.job_id for j in active])
        return active

    def signal(self, job_id: str, sig: str) -> None:
        logger.debug("Signalling job {} via removal (HTCondor REST has no signal endpoint)", job_id)
        self.cancel(job_id)

    def name(self) -> str:
        return f"htcondor_rest({self._url})"


def _parse_htcondor_job_status(raw) -> JobState:
    if raw is None:
        return JobState.UNKNOWN
    try:
        status = int(raw)
    except (ValueError, TypeError):
        return JobState.UNKNOWN
    return _HTSTATUS_MAP.get(status, JobState.UNKNOWN)

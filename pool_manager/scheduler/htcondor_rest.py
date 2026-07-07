try:
    import httpx
except ImportError:
    httpx = None

from loguru import logger

from pool_manager.scheduler.base import JobInfo, JobState, SchedulerBackend


class CondorRestClient:
    """Simple HTTP client for the condor_rest API."""

    def __init__(
        self, url: str, token: str = "", owner: str = "", job_name_prefix: str = "htcondor_worker_"
    ):
        self._url = url.rstrip("/")
        self._token = token
        self._owner = owner
        self._job_name_prefix = job_name_prefix

    def _headers(self) -> dict[str, str]:
        h = {"Content-Type": "application/json"}
        if self._token:
            h["Authorization"] = f"Bearer {self._token}"
        return h

    def submit(self, script_path: str, submit_args: dict[str, str]) -> str:
        payload = dict(submit_args)
        payload.setdefault("executable", script_path)

        url = f"{self._url}/condor_submit"
        logger.debug("POST {}", url)
        resp = httpx.post(url, json=payload, headers=self._headers(), timeout=30)
        logger.trace("submit response: status={} body={}", resp.status_code, resp.text[:2000])
        resp.raise_for_status()
        data = resp.json()
        job_id = str(data.get("cluster", ""))
        logger.debug("Submitted HTCondor job {} via REST API", job_id)
        return job_id

    def remove(self, job_id: str) -> None:
        url = f"{self._url}/condor_rm/{job_id}"
        logger.debug("DELETE {}", url)
        resp = httpx.delete(url, headers=self._headers(), timeout=30)
        logger.trace("remove response: status={}", resp.status_code)
        if resp.status_code not in (200, 204):
            logger.warning("Failed to remove job {} via REST: HTTP {}", job_id, resp.status_code)

    def list_jobs(self, constraint: str = "") -> list[dict]:
        url = f"{self._url}/condor_q"
        params: dict[str, str] = {}
        if constraint:
            params["constraint"] = constraint
        logger.debug("GET {} params={}", url, params)
        resp = httpx.get(url, headers=self._headers(), params=params, timeout=30)
        logger.trace("list_jobs response: status={}", resp.status_code)
        resp.raise_for_status()
        return resp.json()

    def name(self) -> str:
        return f"condor_rest({self._url})"


_HTSTATUS_MAP = {
    1: JobState.PENDING,
    2: JobState.RUNNING,
    5: JobState.PENDING,
    6: JobState.RUNNING,
}


class HTCondorRESTAPIBackend(SchedulerBackend):
    def __init__(
        self, url: str, token: str = "", owner: str = "", job_name_prefix: str = "htcondor_worker_"
    ):
        self._client = CondorRestClient(
            url=url, token=token, owner=owner, job_name_prefix=job_name_prefix
        )

    def submit(self, script_path: str, submit_args: dict[str, str]) -> str:
        return self._client.submit(script_path, submit_args)

    def cancel(self, job_id: str) -> None:
        self._client.remove(job_id)

    def list_active(self) -> list[JobInfo]:
        constraint = ""
        if self._client._owner:
            constraint = f'Owner == "{self._client._owner}"'
        if self._client._job_name_prefix:
            name_constraint = f'Name =?= "{self._client._job_name_prefix}*"'
            if constraint:
                constraint = f"({constraint}) && ({name_constraint})"
            else:
                constraint = name_constraint
        jobs: list[JobInfo] = []
        for raw in self._client.list_jobs(constraint=constraint):
            try:
                cluster_id = raw.get("ClusterId")
                job_status = raw.get("JobStatus")
                job_name = raw.get("Name", "")
                if cluster_id is None:
                    continue
                state = _parse_htcondor_job_status(job_status)
                if state in (JobState.PENDING, JobState.RUNNING):
                    jobs.append(JobInfo(job_id=str(cluster_id), state=state, job_name=job_name))
            except Exception:
                logger.exception("Failed to parse job from REST response: {}", raw)
        return jobs

    def signal(self, job_id: str, sig: str) -> None:
        logger.debug("Signalling job {} via removal (HTCondor REST has no signal endpoint)", job_id)
        self._client.remove(job_id)

    def name(self) -> str:
        return self._client.name()


def _parse_htcondor_job_status(raw) -> JobState:
    if raw is None:
        return JobState.UNKNOWN
    try:
        status = int(raw)
    except (ValueError, TypeError):
        return JobState.UNKNOWN
    return _HTSTATUS_MAP.get(status, JobState.UNKNOWN)

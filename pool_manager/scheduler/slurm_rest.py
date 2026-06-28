import logging

from pool_manager.log import TRACE
from pool_manager.scheduler.base import JobInfo, JobState, SchedulerBackend

log = logging.getLogger("pool_manager.scheduler.slurm_rest")


class SlurmRESTAPIBackend(SchedulerBackend):
    def __init__(self, url: str, token: str = ""):
        self._url = url.rstrip("/")
        self._token = token

    def _headers(self) -> dict[str, str]:
        h = {"Content-Type": "application/json"}
        if self._token:
            h["X-SLURM-USER-TOKEN"] = self._token
        return h

    def submit(self, script_path: str, submit_args: dict[str, str]) -> str:
        import httpx

        with open(script_path) as f:
            script_content = f.read()

        payload = {"script": script_content}
        if submit_args:
            payload["job"] = submit_args

        url = f"{self._url}/slurm/v0.0.38/job/submit"
        log.debug("POST %s", url)
        resp = httpx.post(url, json=payload, headers=self._headers(), timeout=30)
        log.log(TRACE, "submit response: status=%d body=%s", resp.status_code, resp.text[:2000])
        resp.raise_for_status()
        data = resp.json()
        job_id = str(data.get("job_id", data.get("job_id", "")))
        log.debug("Submitted Slurm job %s via REST API", job_id)
        return job_id

    def cancel(self, job_id: str) -> None:
        import httpx

        url = f"{self._url}/slurm/v0.0.38/job/{job_id}"
        log.debug("DELETE %s", url)
        resp = httpx.delete(url, headers=self._headers(), timeout=30)
        log.log(TRACE, "cancel response: status=%d", resp.status_code)
        if resp.status_code not in (200, 204):
            log.warning("Failed to cancel job %s via REST: HTTP %d", job_id, resp.status_code)

    def list_active(self) -> list[JobInfo]:
        import httpx

        url = f"{self._url}/slurm/v0.0.38/jobs"
        log.debug("GET %s", url)
        resp = httpx.get(url, headers=self._headers(), timeout=30)
        log.log(TRACE, "list_active response: status=%d", resp.status_code)
        resp.raise_for_status()
        data = resp.json()

        jobs: list[JobInfo] = []
        for job in data.get("jobs", []):
            job_id = str(job.get("job_id", ""))
            state_str = job.get("state", "").upper()
            state = _parse_slurm_rest_state(state_str)
            jobs.append(JobInfo(job_id=job_id, state=state))

        log.debug("Active Slurm jobs from REST: %s", [j.job_id for j in jobs])
        return jobs

    def signal(self, job_id: str, sig: str) -> None:
        self.cancel(job_id)

    def name(self) -> str:
        return f"slurm_rest({self._url})"


def _parse_slurm_rest_state(raw: str) -> JobState:
    mapping = {
        "PENDING": JobState.PENDING,
        "RUNNING": JobState.RUNNING,
        "CONFIGURING": JobState.PENDING,
        "COMPLETING": JobState.RUNNING,
    }
    return mapping.get(raw, JobState.UNKNOWN)

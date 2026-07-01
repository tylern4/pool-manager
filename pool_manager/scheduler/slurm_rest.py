try:
    import httpx
except ImportError:
    httpx = None

import logging
import os
from pathlib import Path

from pool_manager.log import TRACE
from pool_manager.scheduler.base import JobInfo, JobState, SchedulerBackend, _test_job_id

log = logging.getLogger("pool_manager.scheduler.slurm_rest")

slurm_api_ver = os.getenv("SLURM_API_VER", "v0.0.38")


class SlurmRESTAPIBackend(SchedulerBackend):
    def __init__(
        self,
        url: str,
        token: str = "",
        user: str = "",
        job_name_prefix: str = "htcondor_worker_",
        test_mode: bool = False,
    ):
        self._url = url.rstrip("/")
        self._token = token
        self._user = user
        self._job_name_prefix = job_name_prefix
        self._test_mode = test_mode

    def _headers(self) -> dict[str, str]:
        h = {"Content-Type": "application/json"}
        if self._token:
            h["X-SLURM-USER-TOKEN"] = self._token
        return h

    def submit(self, script_path: str, submit_args: dict[str, str]) -> str:
        if self._test_mode:
            job_id = _test_job_id()
            log.info(
                "[TEST] Would submit job %s via REST (script=%s, args=%s)",
                job_id,
                script_path,
                submit_args,
            )
            return job_id

        script_path_p = Path(script_path)
        if not script_path_p.exists():
            raise FileNotFoundError(f"Worker script not found: {script_path}")
        script_content = script_path_p.read_text()

        payload = {"script": script_content}
        if submit_args:
            payload["job"] = submit_args

        url = f"{self._url}/slurm/{slurm_api_ver}/job/submit"
        log.debug("POST %s", url)
        resp = httpx.post(url, json=payload, headers=self._headers(), timeout=30)
        log.log(TRACE, "submit response: status=%d body=%s", resp.status_code, resp.text[:2000])
        resp.raise_for_status()
        data = resp.json()
        job_id = str(data.get("job_id", data.get("job_id", "")))
        log.debug("Submitted Slurm job %s via REST API", job_id)
        return job_id

    def cancel(self, job_id: str) -> None:
        if self._test_mode:
            log.info("[TEST] Would cancel job %s via REST", job_id)
            return

        url = f"{self._url}/slurm/{slurm_api_ver}/job/{job_id}"
        log.debug("DELETE %s", url)
        resp = httpx.delete(url, headers=self._headers(), timeout=30)
        log.log(TRACE, "cancel response: status=%d", resp.status_code)
        if resp.status_code not in (200, 204):
            log.warning("Failed to cancel job %s via REST: HTTP %d", job_id, resp.status_code)

    def list_active(self) -> list[JobInfo]:
        url = f"{self._url}/slurm/{slurm_api_ver}/jobs"
        params: dict[str, str] = {}
        if self._user:
            params["user"] = self._user
        log.debug("GET %s params=%s", url, params)
        resp = httpx.get(url, headers=self._headers(), params=params, timeout=30)
        log.log(TRACE, "list_active response: status=%d", resp.status_code)
        resp.raise_for_status()
        data = resp.json()

        jobs: list[JobInfo] = []
        for job in data.get("jobs", []):
            job_name = job.get("name", "")
            if self._job_name_prefix and not job_name.startswith(self._job_name_prefix):
                continue
            job_id = str(job.get("job_id", ""))
            state_str = job.get("state", "").upper()
            state = _parse_slurm_rest_state(state_str)
            jobs.append(JobInfo(job_id=job_id, state=state, job_name=job_name))

        log.debug("Active Slurm jobs from REST: %s", [j.job_id for j in jobs])
        return jobs

    def signal(self, job_id: str, sig: str) -> None:
        if self._test_mode:
            log.info("[TEST] Would send signal %s to job %s", sig, job_id)
            return
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

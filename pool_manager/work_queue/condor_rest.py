import logging

from pool_manager.log import TRACE
from pool_manager.work_queue.base import CondorBackend

log = logging.getLogger("pool_manager.work_queue.condor_rest")


class CondorRESTAPIBackend(CondorBackend):
    def __init__(self, url: str, token: str = ""):
        self._url = url.rstrip("/")
        self._token = token

    def count_idle(self, constraint: str = "JobStatus == 1") -> int:
        import httpx

        headers = {}
        if self._token:
            headers["Authorization"] = f"Bearer {self._token}"

        params = {"constraint": constraint, "projection": "ClusterId"}
        url = f"{self._url}/v1/jobs"

        log.debug("GET %s with params: %s", url, params)
        resp = httpx.get(url, headers=headers, params=params, timeout=30)
        log.log(TRACE, "REST response status=%d body=%s", resp.status_code, resp.text[:2000])

        resp.raise_for_status()
        data = resp.json()
        count = len(data.get("data", data.get("jobs", [])))
        log.debug("HTCondor REST idle count: %d", count)
        return count

    def name(self) -> str:
        return f"condor_rest({self._url})"

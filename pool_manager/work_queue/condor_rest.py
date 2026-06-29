import logging

from pool_manager.log import TRACE
from pool_manager.placement import TaskResources
from pool_manager.work_queue.base import CondorBackend

log = logging.getLogger("pool_manager.work_queue.condor_rest")


class CondorRESTAPIBackend(CondorBackend):
    def __init__(self, url: str, token: str = ""):
        self._url = url.rstrip("/")
        self._token = token

    def count_idle(self, constraint: str = "") -> int:
        return len(self.list_idle(constraint=constraint))

    def list_idle(self, constraint: str = "") -> list[TaskResources]:
        import httpx

        headers = {}
        if self._token:
            headers["Authorization"] = f"Bearer {self._token}"

        params: dict[str, str] = {
            "projection": "ClusterId,RequestCpus,RequestMemory,RequestGpus",
        }
        if constraint:
            params["constraint"] = constraint
        url = f"{self._url}/v1/jobs"

        log.debug("GET %s with params: %s", url, params)
        resp = httpx.get(url, headers=headers, params=params, timeout=30)
        log.log(TRACE, "REST response status=%d body=%s", resp.status_code, resp.text[:2000])

        resp.raise_for_status()
        data = resp.json()
        jobs = data.get("data", data.get("jobs", []))
        tasks = [
            TaskResources(
                cpus=float(job.get("RequestCpus", 1) or 1),
                memory_mb=int(job.get("RequestMemory", 1024) or 1024),
                gpus=int(job.get("RequestGpus", 0) or 0),
            )
            for job in jobs
        ]
        log.debug("HTCondor REST idle count: %d", len(tasks))
        return tasks

    def name(self) -> str:
        return f"condor_rest({self._url})"

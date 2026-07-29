try:
    import httpx
except ImportError:
    httpx = None

from loguru import logger

from pool_manager.placement import TaskResources
from pool_manager.work_queue.base import CondorBackend


class CondorRESTAPIBackend(CondorBackend):
    def __init__(self, url: str, token: str = ""):
        self._url = url.rstrip("/")
        self._token = token

    def count_idle(self, constraint: str = "") -> int:
        return len(self.list_idle(constraint=constraint))

    def list_idle(self, constraint: str = "") -> list[TaskResources]:
        headers = {}
        if self._token:
            headers["Authorization"] = f"Bearer {self._token}"

        params: dict[str, str] = {
            "projection": "ClusterId,RequestCpus,RequestMemory,RequestGpus",
        }
        if constraint:
            params["constraint"] = constraint
        url = f"{self._url}/v1/jobs"

        logger.debug("GET {} with params: {}", url, params)
        resp = httpx.get(url, headers=headers, params=params, timeout=30)
        logger.trace("REST response status={} body={}", resp.status_code, resp.text[:2000])

        resp.raise_for_status()
        data = resp.json()
        jobs = data.get("data", data.get("jobs", []))
        tasks = []
        for job in jobs:
            job = {k.lower(): v for k, v in job.items()}
            tasks.append(
                TaskResources(
                    cpus=float(job.get("requestcpus", 1)),
                    memory_mb=int(job.get("requestmemory", 1024)),
                    gpus=int(job.get("requestgpus", 0)),
                    runtime_minutes=int(job.get("runtime_minutes", 0)),
                )
            )
        logger.debug("HTCondor REST idle count: {}", len(tasks))
        return tasks

    def name(self) -> str:
        return f"condor_rest({self._url})"

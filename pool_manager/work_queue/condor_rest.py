try:
    import httpx
except ImportError:
    httpx = None

from loguru import logger

from pool_manager.placement import TaskResources
from pool_manager.work_queue.base import CondorBackend, WorkerSlotStatus


class CondorRESTAPIBackend(CondorBackend):
    def __init__(self, url: str, token: str = ""):
        self._url = url.rstrip("/")
        self._token = token

    def _auth_headers(self) -> dict[str, str]:
        headers: dict[str, str] = {}
        if self._token:
            headers["Authorization"] = f"Bearer {self._token}"
        return headers

    def count_idle(self, constraint: str = "") -> int:
        return len(self.list_idle(constraint=constraint))

    def list_idle(self, constraint: str = "") -> list[TaskResources]:
        params: dict[str, str] = {
            "projection": "ClusterId,RequestCpus,RequestMemory,RequestGpus,RuntimeMinutes",
        }
        if constraint:
            params["constraint"] = constraint
        url = f"{self._url}/v1/jobs"

        logger.debug("GET {} with params: {}", url, params)
        resp = httpx.get(url, headers=self._auth_headers(), params=params, timeout=30)
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
                    runtime_minutes=float(job.get("runtimeminutes", 0)),
                )
            )
        logger.debug("HTCondor REST idle count: {}", len(tasks))
        return tasks

    def list_worker_status(self, constraint: str = "") -> list[WorkerSlotStatus]:
        params: dict[str, str] = {
            "projection": "Name,Owner,State",
        }
        if constraint:
            params["constraint"] = constraint
        url = f"{self._url}/condor_status"

        logger.debug("GET {} with params: {}", url, params)
        resp = httpx.get(url, headers=self._auth_headers(), params=params, timeout=30)
        logger.trace("condor_status response status={} body={}", resp.status_code, resp.text[:2000])

        resp.raise_for_status()
        data = resp.json()
        slots = data.get("data", data.get("slots", []))
        statuses = []
        for slot in slots:
            slot = {k.lower(): v for k, v in slot.items()}
            statuses.append(
                WorkerSlotStatus(
                    slot_name=slot.get("name", ""),
                    owner_job_id=slot.get("owner", None),
                    state=slot.get("state", "idle"),
                )
            )
        logger.debug("HTCondor REST worker slot count: {}", len(statuses))
        return statuses

    def name(self) -> str:
        return f"condor_rest({self._url})"

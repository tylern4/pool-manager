try:
    from htcondor_rest import CondorClient
except ImportError:
    CondorClient = None

from loguru import logger

from pool_manager.placement import TaskResources
from pool_manager.work_queue.base import CondorBackend


class CondorRESTAPIBackend(CondorBackend):
    def __init__(self, url: str, token: str = ""):
        self._url = url.rstrip("/")
        self._client = (
            CondorClient(base_url=self._url, token=token)
            if token
            else CondorClient(base_url=self._url)
        )

    def count_idle(self, constraint: str = "") -> int:
        return len(self.list_idle(constraint=constraint))

    def list_idle(self, constraint: str = "") -> list[TaskResources]:
        projection = "ClusterId,JobStatus,RequestCpus,RequestMemory,RequestGpus,runtime_minutes"
        logger.debug("Querying HTCondor via htcondor-rest with constraint: {}", constraint)
        jobs = self._client.get_queue(projection=projection, constraint=constraint or None)
        tasks = []
        for job in jobs:
            job = {k.lower(): v for k, v in job.items()}
            tasks.append(
                TaskResources(
                    cpus=float(job.get("requestcpus", 1) or 1),
                    memory_mb=int(job.get("requestmemory", 1024) or 1024),
                    gpus=int(job.get("requestgpus", 0) or 0),
                    runtime_minutes=int(job.get("runtime_minutes", 0) or 0),
                    job_status=int(job.get("jobstatus", 0) or 0),
                )
            )
        logger.debug("HTCondor REST idle count: {}", len(tasks))
        return tasks

    def name(self) -> str:
        return f"condor_rest({self._url})"

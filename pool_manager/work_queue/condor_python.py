try:
    import htcondor2 as htcondor
except ImportError:
    htcondor = None

from loguru import logger

from pool_manager.placement import TaskResources
from pool_manager.work_queue.base import CondorBackend


class CondorPythonBackend(CondorBackend):
    def __init__(self, schedd_name: str = ""):
        self._schedd_name = schedd_name

    def count_idle(self, constraint: str = "") -> int:
        return len(self.list_idle(constraint=constraint))

    def list_idle(self, constraint: str = "") -> list[TaskResources]:
        schedd = htcondor.Schedd(self._schedd_name) if self._schedd_name else htcondor.Schedd()
        projection = ["ClusterId", "RequestCpus", "RequestMemory", "RequestGpus"]
        logger.debug(
            "Querying HTCondor schedd '{}' with constraint: {}",
            self._schedd_name or "(default)",
            constraint,
        )
        kw = {"projection": projection}
        if constraint:
            kw["constraint"] = constraint
        result = schedd.query(**kw)
        tasks = []
        for job in result:
            tasks.append(
                TaskResources(
                    cpus=float(job.get("RequestCpus", 1) or 1),
                    memory_mb=int(job.get("RequestMemory", 1024) or 1024),
                    gpus=int(job.get("RequestGpus", 0) or 0),
                    runtime_minutes=int(job.get("runtime_minutes", 0) or 0),
                    job_status=int(job.get("JobStatus", 0) or 0),
                )
            )
        logger.debug("HTCondor idle job count: {}", len(tasks))
        return tasks

    def name(self) -> str:
        base = "condor_python"
        return f"{base}(schedd={self._schedd_name})" if self._schedd_name else base

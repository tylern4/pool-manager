import logging

from pool_manager.placement import TaskResources
from pool_manager.work_queue.base import CondorBackend

log = logging.getLogger("pool_manager.work_queue.condor_python")


class CondorPythonBackend(CondorBackend):
    def __init__(self, schedd_name: str = ""):
        self._schedd_name = schedd_name

    def count_idle(self, constraint: str = "JobStatus == 1") -> int:
        return len(self.list_idle(constraint=constraint))

    def list_idle(self, constraint: str = "JobStatus == 1") -> list[TaskResources]:
        import htcondor

        schedd = htcondor.Schedd(self._schedd_name) if self._schedd_name else htcondor.Schedd()
        projection = ["ClusterId", "RequestCpus", "RequestMemory", "RequestGpus"]
        log.debug(
            "Querying HTCondor schedd '%s' with constraint: %s",
            self._schedd_name or "(default)",
            constraint,
        )
        kw = {"projection": projection}
        if constraint:
            kw["constraint"] = constraint
        result = schedd.query(**kw)
        tasks = [
            TaskResources(
                cpus=float(job.get("RequestCpus", 1) or 1),
                memory_mb=int(job.get("RequestMemory", 1024) or 1024),
                gpus=int(job.get("RequestGpus", 0) or 0),
            )
            for job in result
        ]
        log.debug("HTCondor idle job count: %d", len(tasks))
        return tasks

    def name(self) -> str:
        base = "condor_python"
        return f"{base}(schedd={self._schedd_name})" if self._schedd_name else base

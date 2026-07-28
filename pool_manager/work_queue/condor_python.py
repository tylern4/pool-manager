try:
    import htcondor
except ImportError:
    htcondor = None

from loguru import logger

from pool_manager.placement import TaskResources
from pool_manager.work_queue.base import CondorBackend, WorkerSlotStatus


class CondorPythonBackend(CondorBackend):
    def __init__(self, schedd_name: str = ""):
        self._schedd_name = schedd_name

    def count_idle(self, constraint: str = "") -> int:
        return len(self.list_idle(constraint=constraint))

    def list_idle(self, constraint: str = "") -> list[TaskResources]:
        schedd = htcondor.Schedd(self._schedd_name) if self._schedd_name else htcondor.Schedd()
        projection = ["ClusterId", "RequestCpus", "RequestMemory", "RequestGpus", "RuntimeMinutes"]
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
            job = {k.lower(): v for k, v in job.items()}
            tasks.append(
                TaskResources(
                    cpus=float(job.get("requestcpus", 1)),
                    memory_mb=int(job.get("requestmemory", 1024)),
                    gpus=int(job.get("requestgpus", 0)),
                    runtime_minutes=float(job.get("runtimeminutes", 0)),
                )
            )
        logger.debug("HTCondor idle job count: {}", len(tasks))
        return tasks

    def list_worker_status(self, constraint: str = "") -> list[WorkerSlotStatus]:
        logger.debug(
            "Querying HTCondor collector for worker slots with constraint: {}",
            constraint,
        )
        kw: dict = {}
        if constraint:
            kw["constraint"] = constraint
        result = htcondor.Collector().query(
            htcondor.AdTypes.StartdAd,
            projection=["Name", "Owner", "State"],
            **kw,
        )
        statuses = []
        for slot in result:
            slot = {k.lower(): v for k, v in slot.items()}
            statuses.append(
                WorkerSlotStatus(
                    slot_name=slot.get("name", ""),
                    owner_job_id=slot.get("owner", None),
                    state=slot.get("state", "idle"),
                )
            )
        logger.debug("HTCondor Python worker slot count: {}", len(statuses))
        return statuses

    def name(self) -> str:
        base = "condor_python"
        return f"{base}(schedd={self._schedd_name})" if self._schedd_name else base

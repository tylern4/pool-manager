import logging

from pool_manager.work_queue.base import CondorBackend

log = logging.getLogger("pool_manager.work_queue.condor_python")


class CondorPythonBackend(CondorBackend):
    def __init__(self, schedd_name: str = ""):
        self._schedd_name = schedd_name

    def count_idle(self, constraint: str = "JobStatus == 1") -> int:
        import htcondor

        schedd = htcondor.Schedd(self._schedd_name) if self._schedd_name else htcondor.Schedd()
        projection = ["ClusterId"]
        log.debug(
            "Querying HTCondor schedd '%s' with constraint: %s",
            self._schedd_name or "(default)",
            constraint,
        )
        result = schedd.query(constraint=constraint, projection=projection)
        count = len(result)
        log.debug("HTCondor idle job count: %d", count)
        return count

    def name(self) -> str:
        base = "condor_python"
        return f"{base}(schedd={self._schedd_name})" if self._schedd_name else base

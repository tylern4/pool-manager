import logging

from pool_manager.placement import TaskResources
from pool_manager.work_queue.base import CondorBackend, WorkQueue

log = logging.getLogger("pool_manager.work_queue.condor")


class CondorWorkQueue(WorkQueue):
    def __init__(self, backend: CondorBackend, constraint: str = "JobStatus == 1"):
        self._backend = backend
        self._constraint = constraint

    def count_idle(self) -> int:
        log.debug("Counting idle jobs via %s", self._backend.name())
        return self._backend.count_idle(constraint=self._constraint)

    def list_idle(self) -> list[TaskResources]:
        log.debug("Listing idle jobs with resources via %s", self._backend.name())
        return self._backend.list_idle(constraint=self._constraint)

    def name(self) -> str:
        return self._backend.name()
